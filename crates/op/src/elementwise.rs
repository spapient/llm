//! 逐元素二元骨架：`y[n] = op(a[n], b[n])`。
//!
//! 与 [`crate::gemv`] 同构的一副骨架，差别只在没有规约轴：一次循环读两个数、做一次逐元素运算、
//! 写一个数。`add_f32` / `mul_f32` 是同一副骨架的两种 [`Binary`] 实现，骨架（表达式缓存、代价、
//! 样本、校验、遍历）只有一份，按实现类型单态化，`Model` 里仍是三个 `fn` 指针而不是热路径分发。

use crate::einsum::{Formula, FormulaCache, OperandSpec};
use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用一元骨架（与 dequant / softmax / rmsnorm / rope / scan 同一模式）。
pub use crate::unary::{AXES, project_shape};

/// 逐元素二元算子随语义变化的那部分。
pub trait Binary {
    /// 一条 einsum 表达式：箭头左三项（两个读、一个写），共享一条轴。
    ///
    /// 二元逐元素形状只有这一种，默认值即标准规格；需要别的表达式时在实现里覆盖。
    const EXPR: &'static str = "n,n,n->n";

    /// 表达式各项的位宽与方向，顺序与 [`Binary::EXPR`] 一一对应。
    ///
    /// 默认 f32 两读一写；位宽不同的格式在实现里覆盖。
    const OPERANDS: &'static [OperandSpec] = &crate::einsum::f32_binary_operands();

    /// 缓存单元格（每个实现一份），见 [`crate::gemv::Format::cache`]。
    fn cache() -> &'static FormulaCache;

    /// 表达式：第一次调用时解析并校验，之后复用。
    ///
    /// # Errors
    ///
    /// 表达式与操作数表不自洽时返回说明（带 `op::elementwise::formula` 帧）。
    fn formula() -> Result<&'static Formula, Error> {
        Self::cache().get(Self::EXPR, Self::OPERANDS, "op::elementwise::formula")
    }

    /// 逐元素运算：两个输入值，一个输出值。
    fn apply(x: f32, y: f32) -> f32;

    /// 第一路输入的确定性取值：默认线性夹具 `−1.0 + 0.5·i`；需要不同取值时覆盖。
    fn input_a(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.5, -1.0)
    }

    /// 第二路输入的确定性取值：默认线性夹具 `0.5 + 0.25·i`；需要不同取值时覆盖。
    fn input_b(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.25, 0.5)
    }
}

/// 标量代价：完全由表达式与操作数位宽推出，本函数不含流量常数。
pub fn cost<B: Binary>(shape: &ShapeContext) -> Result<Cost, Error> {
    B::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("op::elementwise::cost"))
}

/// 确定性样本：第一个参数是元素数（轴 `n`），第二个参数逐元素算子不用。
pub fn sample<B: Binary>(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "a",
                values: Values::F32(B::input_a(len)),
            },
            Buffer {
                name: "b",
                values: Values::F32(B::input_b(len)),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：逐元素应用 [`Binary::apply`]，不依赖任何设备。
pub fn reference<B: Binary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let a = sample.f32("a")?;
    let b = sample.f32("b")?;
    if a.len() != b.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("两个输入长度不一致：{} 与 {}", a.len(), b.len()),
        ));
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| B::apply(*x, *y))
        .collect())
}

/// 内核本体：结果写进调用方给的缓冲，不自带分配。
///
/// # Errors
///
/// 两个输入等长、且与输出等长，否则返回说明。
pub fn forward<B: Binary>(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "缓冲长度不一致：a {}、b {}、out {}",
                a.len(),
                b.len(),
                out.len()
            ),
        ));
    }
    // 三个切片 zip：连续内存、无下标边界检查，编译器最容易自动向量化。
    for ((out, x), y) in out.iter_mut().zip(a).zip(b) {
        *out = B::apply(*x, *y);
    }
    Ok(())
}

/// 生成一份 f32 二元逐元素算子的公开面：标记类型、[`Binary`] 实现、[`annotation::model`] 别名、
/// `forward` 入口与测试挂载。`add` / `mul` / `residual_add` 只差一行 `apply`（可选样本夹具）；
/// 新增一种二元运算加一次调用即可。
#[macro_export]
macro_rules! binary_f32 {
    (
        $name:ident, $alias:ident, $op:tt, $test:tt,
        $apply:expr
        $(, $input_a:expr, $input_b:expr)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：逐元素二元运算见 `apply`，骨架见 `crate::elementwise`。")]
        pub struct $name;

        impl $crate::elementwise::Binary for $name {
            fn cache() -> &'static $crate::einsum::FormulaCache {
                static CACHE: $crate::einsum::FormulaCache = $crate::einsum::FormulaCache::new();
                &CACHE
            }

            fn apply(x: f32, y: f32) -> f32 {
                const APPLY: fn(f32, f32) -> f32 = $apply;
                APPLY(x, y)
            }

            $( fn input_a(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input_a;
                INPUT(len)
            }

            fn input_b(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input_b;
                INPUT(len)
            } )?
        }

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 三个缓冲等长，否则返回说明。
        pub fn forward(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::elementwise::forward::<$name>(a, b, out)
        }

        #[cfg(test)]
        mod tests {
            include!($test);
        }
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    /// 生成 add/mul 等二元算子共有的三则测试：参考对拍、长度负例、饱和与重复边界。
    macro_rules! elementwise_binary_tests {
        ($ty:ty, $forward:ident) => {
            #[test]
            fn matches_the_scalar_reference() {
                let sample = crate::elementwise::sample::<$ty>(128, 1);
                let expected = crate::elementwise::reference::<$ty>(&sample).expect("参考可计算");
                let mut out = vec![0.0; 128];
                let a = sample.f32("a").expect("a 存在");
                let b = sample.f32("b").expect("b 存在");
                $forward(a, b, &mut out).expect("缓冲等长");
                assert_eq!(out, expected);
            }

            #[test]
            fn rejects_each_mismatched_pair_and_allows_empty() {
                let mut out = vec![0.0; 2];
                assert!(
                    $forward(&[1.0], &[1.0, 2.0], &mut out).is_err(),
                    "a 与 b 不等"
                );
                assert!(
                    $forward(&[1.0, 2.0], &[1.0], &mut out).is_err(),
                    "b 与 out 不等"
                );
                let mut empty = vec![];
                $forward(&[], &[], &mut empty).expect("空输入逐元素放行");
            }

            #[test]
            fn saturating_and_repeated_inputs_match_reference() {
                let cases: &[(&[f32], &[f32])] = &[
                    (&[0.0, 0.0], &[0.0, 0.0]),
                    (&[1.0, 1.0], &[2.0, 2.0]),
                    (&[f32::MAX, f32::MIN], &[0.0, 0.0]),
                    (&[f32::MAX, f32::MAX], &[f32::MIN, f32::MIN]),
                    (&[3.0, -1.0], &[-3.0, 1.0]),
                ];
                for (index, (a, b)) in cases.iter().enumerate() {
                    let mut sample = crate::elementwise::sample::<$ty>(a.len(), 1);
                    sample.buffers[0].values = crate::model::Values::F32(a.to_vec());
                    sample.buffers[1].values = crate::model::Values::F32(b.to_vec());
                    let expected =
                        crate::elementwise::reference::<$ty>(&sample).expect("参考可计算");
                    let mut out = vec![0.0; a.len()];
                    $forward(a, b, &mut out).expect("等长");
                    assert_eq!(out, expected, "用例 {index}");
                }
            }
        };
    }
    pub(crate) use elementwise_binary_tests;
}

// 声明式算子表：同一副二元骨架的七个算子，每个算子一个 `pub mod` 保持 `op::<name>::forward`
// 路径不变（`forward` 同名，必须隔在各自的模块里）。新增二元运算在此补一段。

pub mod add_f32 {
    //! 逐元素加法：`y[n] = a[n] + b[n]`。
    //!
    //! 语义表由 `#[annotation::model]` 生成（`tests/contracts.rs` 要求语义表只有这一个生产者），
    //! 算术与遍历由 [`crate::elementwise`] 骨架按 [`Add`] 类型单态化。

    crate::binary_f32!(
        Add,
        AddF32,
        "add_f32",
        "../tests/add_f32_tests.rs",
        |x, y| x + y
    );
}

pub mod sub_f32 {
    //! 逐元素减法：`y[n] = a[n] - b[n]`。
    //!
    //! 语义表由 `#[annotation::model]` 生成（`tests/contracts.rs` 要求语义表只有这一个生产者），
    //! 算术与遍历由 [`crate::elementwise`] 骨架按 [`Sub`] 类型单态化。

    crate::binary_f32!(
        Sub,
        SubF32,
        "sub_f32",
        "../tests/sub_f32_tests.rs",
        |x, y| x - y
    );
}

pub mod mul_f32 {
    //! 逐元素乘法：`y[n] = a[n] * b[n]`。
    //!
    //! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Mul`] 单态化。

    crate::binary_f32!(
        Mul,
        MulF32,
        "mul_f32",
        "../tests/mul_f32_tests.rs",
        |x, y| x * y
    );
}

pub mod div_f32 {
    //! 逐元素除法：`y[n] = a[n] / b[n]`。
    //!
    //! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Div`] 单态化。
    //! 除零按 IEEE-754 语义（`±inf` / `NaN`），不在算子层额外判错。

    crate::binary_f32!(
        Div,
        DivF32,
        "div_f32",
        "../tests/div_f32_tests.rs",
        |x, y| x / y
    );
}

pub mod max_f32 {
    //! 逐元素取大：`y[n] = max(a[n], b[n])`。
    //!
    //! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Max`] 单态化。

    crate::binary_f32!(
        Max,
        MaxF32,
        "max_f32",
        "../tests/max_f32_tests.rs",
        |x, y| x.max(y)
    );
}

pub mod min_f32 {
    //! 逐元素取小：`y[n] = min(a[n], b[n])`。
    //!
    //! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Min`] 单态化。

    crate::binary_f32!(
        Min,
        MinF32,
        "min_f32",
        "../tests/min_f32_tests.rs",
        |x, y| x.min(y)
    );
}

pub mod residual_add_f32 {
    //! 残差相加：`y[n] = a[n] + b[n]`。
    //!
    //! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Residual`]
    //! 单态化。与 `add` 同一段加法，独立成类型以对应网络残差层，样本夹具不同。

    crate::binary_f32!(
        Residual,
        ResidualAddF32,
        "residual_add_f32",
        "../tests/residual_add_f32_tests.rs",
        |x, y| x + y,
        |len| (0..len).map(|i| (i as f32) * 0.5).collect(),
        |len| (0..len).map(|i| (i as f32) * 0.25).collect()
    );
}
