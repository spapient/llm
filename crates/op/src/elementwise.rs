//! 逐元素二元骨架：`y[n] = op(a[n], b[n])`。
//!
//! 与 [`crate::gemv`] 同构的一副骨架，差别只在没有规约轴：一次循环读两个数、做一次逐元素运算、
//! 写一个数。`add_f32` / `mul_f32` 是同一副骨架的两种 [`Binary`] 实现，骨架（表达式缓存、代价、
//! 样本、校验、遍历）只有一份，按实现类型单态化，`Model` 里仍是三个 `fn` 指针而不是热路径分发。

use crate::einsum::{Bits, Formula, FormulaCache, OperandSpec};
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 逐元素骨架只有一条轴：它既是输入轴也是输出轴，按输入角色声明。
pub const AXES: &[AxisSpec] = &[AxisSpec {
    role: AxisRole::Input,
    label: 'n',
}];

/// 逐元素二元算子随语义变化的那部分。
pub trait Binary {
    /// 一条 einsum 表达式：箭头左三项（两个读、一个写），共享一条轴。
    ///
    /// 二元逐元素形状只有这一种，默认值即标准规格；需要别的表达式时在实现里覆盖。
    const EXPR: &'static str = "n,n,n->n";

    /// 表达式各项的位宽与方向，顺序与 [`Binary::EXPR`] 一一对应。
    ///
    /// 默认 f32 两读一写；位宽不同的格式在实现里覆盖。
    const OPERANDS: &'static [OperandSpec] = &f32_operands();

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
        linear(len, 0.5, -1.0)
    }

    /// 第二路输入的确定性取值：默认线性夹具 `0.5 + 0.25·i`；需要不同取值时覆盖。
    fn input_b(len: usize) -> Vec<f32> {
        linear(len, 0.25, 0.5)
    }
}

/// 线性确定性夹具：`base + step·i`，同一长度取值可复现。
#[must_use]
fn linear(len: usize, step: f32, base: f32) -> Vec<f32> {
    (0..len).map(|i| base + i as f32 * step).collect()
}

/// 逐元素单轴形状：轴 `n` 取第一个参数（元素数），第二个参数不用。
pub fn project_shape(len: usize, _: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('n', len)])
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

/// 两个 f32 操作数的标准规格：两个读、一个写。
#[must_use]
pub const fn f32_operands() -> [OperandSpec; 3] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::read("b", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
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
