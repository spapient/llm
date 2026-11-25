//! 逐元素一元骨架：`y[n] = f(a[n])`。

use crate::formula::FormulaOwner;
use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
use crate::scalar::{FloatSlice, FloatSliceMut, Scalar};
use diagnostic::{Code, Error};
use tensor::View;

pub(crate) mod ops {
    //! 一元算子目录：模块路径、语义名称、测试族与边界数据只声明一次。
    //!
    //! 除 `gelu` 的误差函数外，每个算子的差异都是表项；通用骨架与测试挂载由 `unary_f32!` 生成。

    /// 把一元算子表展开成独立模块，保持 `nn::<name>::MODEL` / `nn::<name>::forward` 路径不变。
    macro_rules! unary_catalog {
        ($(
            $module:ident => ($doc:literal, $name:ident, $alias:ident, $op:literal,
                $kind:ident, $edge:expr, $apply:expr $(, $rest:expr)* $(,)?)
        ),+ $(,)?) => {
            $(
                #[doc = $doc]
                pub mod $module {
                    $crate::unary_f32!($name, $alias, $op, $kind, $edge, $apply $(, $rest)*);
                }
            )+
        };
    }

    unary_catalog! {
        exp_f32 => ("逐元素指数：`y[n] = exp(a[n])`。", Exp, ExpF32, "exp_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, -1000.0, 1.0], &[1.0, 0.0, std::f32::consts::E], &[0.0, 0.0, 1e-6]),
            |x| x.exp(),
            |len| (0..len).map(|i| ((i % 65) as f32) * 0.25 - 8.0).collect()
        ),
        log_f32 => ("逐元素自然对数：`y[n] = ln(a[n])`。", Log, LogF32, "log_f32", unary_pair_tests,
            assert_unary_edge(forward, &[1.0, std::f32::consts::E], &[0.0, 1.0], &[0.0, 1e-6]),
            |x| x.ln(),
            |len| crate::model::linear(len, 0.25, 0.5)
        ),
        silu_f32 => ("逐元素 SiLU 激活：`y[n] = a[n] * sigmoid(a[n])`。", Silu, SiluF32, "silu_f32", unary_op_tests,
            assert_unary_edge(forward, &[1.0], &[0.731_058_6], &[1e-6]),
            |x| x / (K::ONE + (-x).exp())
        ),
        relu_f32 => ("逐元素 ReLU：`y[n] = max(a[n], 0)`。", Relu, ReluF32, "relu_f32", unary_op_tests,
            assert_unary_edge(forward, &[-2.0, 0.0, 2.0], &[0.0, 0.0, 2.0], &[0.0; 3]),
            |x| x.max(K::ZERO)
        ),
        sigmoid_f32 => ("逐元素 Sigmoid：`y[n] = 1 / (1 + exp(-a[n]))`。", Sigmoid, SigmoidF32, "sigmoid_f32", unary_op_tests,
            assert_unary_edge(forward, &[0.0, 1.0, -1000.0, 1000.0], &[0.5, 0.731_058_6, 0.0, 1.0], &[0.0, 1e-6, 0.0, 0.0]),
            |x| K::ONE / (K::ONE + (-x).exp())
        ),
        tanh_f32 => ("逐元素双曲正切：`y[n] = 2 / (1 + exp(-2·a[n])) - 1`。", Tanh, TanhF32, "tanh_f32", unary_op_tests,
            assert_unary_edge(forward, &[0.0, 1.0, -1000.0, 1000.0], &[0.0, 0.761_594_2, -1.0, 1.0], &[0.0, 1e-6, 0.0, 0.0]),
            |x| K::from_f64(2.0) / (K::ONE + (-K::from_f64(2.0) * x).exp()) - K::ONE
        ),
        hard_sigmoid_f32 => ("逐元素 HardSigmoid：`y[n] = clamp(a[n] / 6 + 0.5, 0, 1)`。", HardSigmoid, HardSigmoidF32, "hard_sigmoid_f32", unary_op_tests,
            assert_unary_edge(forward, &[-6.0, 0.0, 6.0, 3.0], &[0.0, 0.5, 1.0, 1.0], &[0.0, 0.0, 0.0, 1e-6]),
            |x| {
                let v = x / K::from_f64(6.0) + K::from_f64(0.5);
                v.max(K::ZERO).min(K::ONE)
            }
        ),
        abs_f32 => ("逐元素绝对值：`y[n] = |a[n]|`。", Abs, AbsF32, "abs_f32", unary_op_tests,
            assert_unary_edge(forward, &[-3.0, 0.0, 2.0], &[3.0, 0.0, 2.0], &[0.0; 3]),
            |x| x.abs(), false
        ),
        neg_f32 => ("逐元素取负：`y[n] = -a[n]`。", Neg, NegF32, "neg_f32", unary_op_tests,
            assert_unary_edge(forward, &[-2.0, 0.0, 3.0], &[2.0, 0.0, -3.0], &[0.0; 3]),
            |x| -x, false
        ),
        square_f32 => ("逐元素平方：`y[n] = a[n]²`。", Square, SquareF32, "square_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, 3.0, -2.0], &[0.0, 9.0, 4.0], &[0.0; 3]),
            |x| x * x, false
        ),
        sign_f32 => ("逐元素符号：正数取 1、负数取 −1、零取 0。", Sign, SignF32, "sign_f32", unary_op_tests,
            assert_unary_edge(forward, &[-3.0, 0.0, 2.0], &[-1.0, 0.0, 1.0], &[0.0; 3]),
            |x| if x > K::ZERO { K::ONE } else if x < K::ZERO { -K::ONE } else { K::ZERO }, false
        ),
        recip_f32 => ("逐元素倒数：`y[n] = 1 / a[n]`；零输入按 IEEE 得无穷。", Recip, RecipF32, "recip_f32", unary_pair_tests,
            assert_unary_edge(forward, &[2.0, 4.0, 0.5], &[0.5, 0.25, 2.0], &[0.0; 3]),
            |x| K::ONE / x, false
        ),
        cos_f32 => ("逐元素余弦：`y[n] = cos(a[n])`。", Cos, CosF32, "cos_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, std::f32::consts::FRAC_PI_2], &[1.0, 0.0], &[0.0, 1e-6]),
            |x| x.cos(), false
        ),
        sin_f32 => ("逐元素正弦：`y[n] = sin(a[n])`。", Sin, SinF32, "sin_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, std::f32::consts::FRAC_PI_2], &[0.0, 1.0], &[0.0, 1e-6]),
            |x| x.sin(), false
        ),
        sqrt_f32 => ("逐元素平方根：`y[n] = √a[n]`；负入按 IEEE 得 NaN。", Sqrt, SqrtF32, "sqrt_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, 4.0], &[0.0, 2.0], &[0.0; 2]),
            |x| x.sqrt(), false,
            |len| (0..len).map(|i| (i as f32) * 0.25).collect()
        ),
        rsqrt_f32 => ("逐元素倒数平方根：`y[n] = 1 / √a[n]`；样本取正且非零。", Rsqrt, RsqrtF32, "rsqrt_f32", unary_pair_tests,
            assert_unary_edge(forward, &[1.0, 4.0], &[1.0, 0.5], &[0.0; 2]),
            |x| K::ONE / x.sqrt(), false,
            |len| (0..len).map(|i| (i + 1) as f32 * 0.25).collect()
        ),
        ceil_f32 => ("逐元素向上取整：`y[n] = ceil(a[n])`。", Ceil, CeilF32, "ceil_f32", unary_op_tests,
            assert_unary_edge(forward, &[1.2, -1.2], &[2.0, -1.0], &[0.0; 2]),
            |x| x.ceil(), false
        ),
        floor_f32 => ("逐元素向下取整：`y[n] = floor(a[n])`。", Floor, FloorF32, "floor_f32", unary_op_tests,
            assert_unary_edge(forward, &[1.2, -1.2], &[1.0, -2.0], &[0.0; 2]),
            |x| x.floor(), false
        ),
        round_f32 => ("逐元素四舍五入：`y[n] = round(a[n])`，半值远离零。", Round, RoundF32, "round_f32", unary_op_tests,
            assert_unary_edge(forward, &[1.5, -1.5], &[2.0, -2.0], &[0.0; 2]),
            |x| x.round(), false
        ),
        trunc_f32 => ("逐元素向零截断：`y[n] = trunc(a[n])`。", Trunc, TruncF32, "trunc_f32", unary_op_tests,
            assert_unary_edge(forward, &[1.7, -1.7], &[1.0, -1.0], &[0.0; 2]),
            |x| x.trunc(), false
        ),
        log1p_f32 => ("逐元素 `ln(1+a[n])`：定义域 `a[n] > -1`。", Log1p, Log1pF32, "log1p_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, std::f32::consts::E - 1.0], &[0.0, 1.0], &[0.0, 1e-6]),
            |x| x.ln_1p(), false,
            |len| crate::model::linear(len, 0.25, 0.0)
        ),
        exp2_f32 => ("逐元素 `2^a[n]`：经 `exp(x·ln2)` 复用指数接口。", Exp2, Exp2F32, "exp2_f32", unary_pair_tests,
            assert_unary_edge(forward, &[0.0, 3.0], &[1.0, 8.0], &[0.0; 2]),
            |x| x.exp2(), false,
            |len| crate::model::linear(len, 0.125, -2.0)
        ),
    }

    pub mod gelu_f32 {
        //! 逐元素 GELU 激活（精确 erf 版）：`y[n] = 0.5*a[n]*(1 + erf(a[n]/sqrt(2)))`。

        use crate::scalar::Scalar;

        /// 标量误差函数：Abramowitz-Stegun 手册公式 7.1.26（`|误差| < 1.5e-7`），泛型到浮点标量档。
        pub fn erf<K: Scalar>(x: K) -> K {
            let sign = if x < K::ZERO { -K::ONE } else { K::ONE };
            let x = x.abs();
            let p = K::from_f64(0.327_591_1);
            let a1 = K::from_f64(0.254_829_6);
            let a2 = K::from_f64(-0.284_496_72);
            let a3 = K::from_f64(1.421_413_8);
            let a4 = K::from_f64(-1.453_152_1);
            let a5 = K::from_f64(1.061_405_4);
            let t = K::ONE / (K::ONE + p * x);
            let series = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
            let y = K::ONE - series * (-x * x).exp();
            sign * y
        }

        crate::unary_f32!(
            Gelu,
            GeluF32,
            "gelu_f32",
            unary_pair_tests,
            {
                assert!((erf(0.0_f32) - 0.0).abs() < 1e-6);
                assert!((erf(1.0_f32) - 0.842_700_8).abs() < 1.5e-6);
                assert!((erf(-1.0_f32) + 0.842_700_8).abs() < 1.5e-6);
            },
            |x| K::from_f64(0.5)
                * x
                * (K::ONE + erf(x * K::from_f64(std::f64::consts::FRAC_1_SQRT_2)))
        );
    }

    pub mod leaky_relu_f32 {
        //! 逐元素 LeakyReLU：非负恒等，负区按 `SLOPE` 缩放。

        /// 斜率常量：与内核模板、PyTorch 默认值一致。
        pub const SLOPE: f32 = 0.01;

        crate::unary_f32!(
            LeakyRelu,
            LeakyReluF32,
            "leaky_relu_f32",
            unary_op_tests,
            assert_unary_edge(
                forward,
                &[-2.0, 0.0, 2.0],
                &[-2.0 * SLOPE, 0.0, 2.0],
                &[1e-6, 0.0, 0.0]
            ),
            |x| if x >= K::ZERO {
                x
            } else {
                K::from_f64(SLOPE as f64) * x
            }
        );
    }

    pub mod scale {
        //! 标量缩放：`y[n] = alpha * a[n]`。

        /// 样本固定的缩放系数（内核模板里写同一常量）。
        pub const ALPHA: f32 = 0.5;

        crate::unary_f32!(
            Scale,
            ScaleModel,
            "scale",
            unary_op_tests,
            assert_unary_edge(forward, &[1.0], &[ALPHA], &[0.0]),
            |x| K::from_f64(ALPHA as f64) * x,
            |len| crate::model::linear(len, 0.25, -2.0)
        );

        #[cfg(test)]
        mod tier_tests {
            use super::*;
            use crate::scalar::Scalar;

            fn check<K: Scalar>(tol: f64) {
                let input = [-2.0, 0.0, 3.0].map(K::from_f64);
                let mut out = [K::ZERO; 3];
                forward(tensor::View::one(&input), &mut out).expect("等长");
                for (got, want) in out.iter().zip([-1.0, 0.0, 1.5]) {
                    assert!((got.to_f64() - want).abs() < tol, "{got:?} vs {want}");
                }
            }

            #[test]
            fn forward_supports_every_float_tier() {
                check::<f64>(1e-12);
                check::<half::f16>(1e-2);
                check::<half::bf16>(2e-2);
            }
        }
    }
}

/// 逐元素一元算子随语义变化的那部分。
///
/// 逐元素函数泛型到浮点标量档 `K: Scalar`（ADR 0054 §1 的「元素档 K」）：同一份闭包
/// 服务 `f32` / `f64` / `f16` / `bf16` 四档，无需为每档另写算子。对拍通道（样本 / 参考 /
/// 目录位宽）当前归一 f32，[`apply_f32`](Self::apply_f32) 即 `apply::<f32>` 的默认别名。
pub trait Unary: FormulaOwner {
    /// 逐元素函数：一个输入值，一个输出值，泛型到浮点标量档 `K`。
    fn apply<K: Scalar>(x: K) -> K;

    /// f32 标量核：对拍通道归一 f32 的入口，默认即 [`apply`](Self::apply) 的 f32 单态化。
    fn apply_f32(x: f32) -> f32 {
        Self::apply(x)
    }

    /// 输入的确定性取值：默认线性夹具 `−4.0 + 0.25·i`；需要不同取值时在实现里覆盖。
    fn input(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.25, -4.0)
    }
}

/// 一元骨架只有一条轴：既是输入轴也是输出轴。
pub const AXES: &[AxisSpec] = &[AxisSpec {
    role: AxisRole::Input,
    label: 'n',
}];

/// 单轴形状：轴 `n` 取第一个参数（元素数）。
///
/// # Errors
///
/// 形状上下文构造失败时返回说明（当前实现不会失败）。
pub fn project_shape(len: usize, _: usize) -> Result<ShapeCtx, Error> {
    ShapeCtx::new([('n', len)])
}

/// 标量代价：完全由表达式与操作数位宽推出。
pub fn cost<U: Unary>(shape: &ShapeCtx) -> Result<Cost, Error> {
    U::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("nn::unary::cost"))
}

/// 一档浮点样本与参考的完整声明：类型、`Values` 变体和读取器只配置一次。
/// 输入夹具先收窄到目标档，参考再应用同一份泛型 `apply`，对拍结果统一回到 f32。
macro_rules! sample_tier {
    ($sample:ident, $reference:ident, $k:ty, $variant:ident, $read:ident) => {
        /// 构造这一精度档的确定性一元样本；第一个参数是元素数。
        pub fn $sample<U: Unary>(len: usize, _: usize) -> Sample {
            fn values<K: Scalar>(input: Vec<f32>) -> Vec<K> {
                input
                    .into_iter()
                    .map(|x| K::from_f64(f64::from(x)))
                    .collect()
            }
            Sample {
                m: len,
                n: len,
                scalars: vec![("N", len as i64)],
                bufs: vec![
                    Buf {
                        name: "a",
                        values: Values::$variant(values::<$k>(U::input(len))),
                    },
                    Buf {
                        name: "y",
                        values: Values::$variant(values::<$k>(vec![0.0; len])),
                    },
                ],
            }
        }

        /// 逐元素执行这一档的标量核，并把结果转换成统一的 f32 对拍面。
        pub fn $reference<U: Unary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            Ok(sample
                .$read("a")?
                .iter()
                .map(|x| U::apply(*x).to_f64() as f32)
                .collect())
        }
    };
}

sample_tier!(sample, reference, f32, F32, f32);
sample_tier!(sample_f16, reference_f16, half::f16, F16, f16);
sample_tier!(sample_bf16, reference_bf16, half::bf16, Bf16, bf16);

/// 内核本体：结果写进调用方给的缓冲，泛型到浮点标量档 `K`。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward<K: Scalar, U: Unary>(a: View<'_, K>, out: &mut [K]) -> diagnostic::Result<()> {
    let a = a.as_slice();
    if a.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：a {}、out {}", a.len(), out.len()),
        ));
    }
    // 切片 zip：连续内存、无下标边界检查，便于自动向量化。
    for (out, x) in out.iter_mut().zip(a) {
        *out = U::apply(*x);
    }
    Ok(())
}

/// 运行时精度入口：一次 match 把 tagged 缓冲分派到对应档的单态化 [`forward`]。
///
/// 调用方持运行时精度（如配置驱动的 dtype）时走这里，无需在调用点编译期单态化；
/// 输入与输出必须同档，错档挡回。这是「枚举身份 + 入口一次分派」的执行缝。
///
/// # Errors
///
/// 输入输出档不一致，或缓冲长度不一致时返回说明。
pub fn dispatch<U: Unary>(a: FloatSlice<'_>, out: FloatSliceMut<'_>) -> diagnostic::Result<()> {
    match (a, out) {
        (FloatSlice::F32(a), FloatSliceMut::F32(o)) => forward::<f32, U>(View::one(a), o),
        (FloatSlice::F64(a), FloatSliceMut::F64(o)) => forward::<f64, U>(View::one(a), o),
        (FloatSlice::F16(a), FloatSliceMut::F16(o)) => forward::<half::f16, U>(View::one(a), o),
        (FloatSlice::Bf16(a), FloatSliceMut::Bf16(o)) => forward::<half::bf16, U>(View::one(a), o),
        _ => Err(Error::new(Code::Invalid, "输入与输出精度档不一致")),
    }
}

/// 两档一元算子共用的 [`Unary`](crate::unary::Unary) 实现：闭包先绑定到 `fn(K) -> K` 函数指针，
/// 参数类型由签名反推为泛型档 `K`（调用点还进不了方法泛型作用域，不在方法上标注 `K`）；可选
/// 输入样本槽原样转发。f32 档与精度档（`unary_tier`）只共享这一段，其余档差异各自展开。
#[macro_export]
macro_rules! unary_impl {
    ($name:ident, $apply:expr $(, $input:expr)?) => {
        impl $crate::unary::Unary for $name {
            fn apply<K: $crate::scalar::Scalar>(x: K) -> K {
                let apply: fn(K) -> K = $apply;
                apply(x)
            }

            $( fn input(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input;
                INPUT(len)
            } )?
        }
    };
}

/// 生成一份 f32 一元逐元素算子的公开面：标记类型、[`Unary`] 实现、[`annotation::model`] 别名、
/// `forward` 入口与测试挂载。`scale` / `gelu` / `silu` 只差一行 `apply`（可选样本夹具）；
/// 新增一种一元运算加一次调用即可。
#[macro_export]
macro_rules! unary_f32 {
    (
        $name:ident, $alias:ident, $op:tt,
        $kind:ident, $edge:expr,
        $apply:expr, $diff:literal
        $(, $input:expr)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：逐元素一元运算见 `apply`，骨架见 `crate::unary`。")]
        pub struct $name;

        $crate::formula_owner!(
            $name,
            "n,n->n",
            $crate::einsum::f32_io_operands(),
            "nn::unary::formula"
        );

        $crate::unary_impl! { $name, $apply $(, $input)? }

        // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术；骨架走绝对路径，算子模块
        // 嵌在哪一层都不依赖 `super` 的父模块解析。
        $crate::operator_via!(
            @impl $name, $op, 1, (4096, 4096), $diff, 1, $crate::unary::AXES,
            $crate::unary::project_shape,
            $crate::unary::cost::<$name>,
            $crate::unary::sample::<$name>,
            $crate::unary::reference::<$name>
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 输入与输出等长，否则返回说明。
        pub fn forward<K: $crate::scalar::Scalar>(
            a: tensor::View<'_, K>,
            out: &mut [K],
        ) -> diagnostic::Result<()> {
            $crate::unary::forward::<K, $name>(a, out)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            #[allow(unused_imports)]
            use $crate::unary::test_support::assert_unary_edge;
            $crate::unary::test_support::$kind!($name, forward);

            #[test]
            fn edge_cases_match_reference() {
                $edge
            }
        }
    };
    // 缺省可微：既有算子沿用 `true`；不可微调子在显式位传 `false`。
    (
        $name:ident, $alias:ident, $op:tt,
        $kind:ident, $edge:expr,
        $apply:expr
        $(, $input:expr)?
    ) => {
        $crate::unary_f32! {
            $name, $alias, $op, $kind, $edge, $apply, true $(, $input)?
        }
    };
}

/// 生成一份**指定精度档**的一元逐元素算子公开面：与 [`unary_f32`] 同一份泛型 `apply` 骨架，
/// 只差档配置（dtype 字符串 / [`DType`](crate::model::DType) / 标量类型 / 操作数位宽 / 样本 / 参考）。
/// 「加一档=加一次调用」：算法闭包只写一遍，档差异全是数据参数。
#[macro_export]
macro_rules! unary_tier {
    (
        $name:ident, $alias:ident, $op:tt,
        $dtype:literal, $dt:expr, $k:ty, $operands:expr, $sample:path, $reference:path,
        $apply:expr, $diff:literal
        $(, $input:expr)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：逐元素一元运算见 `apply`，骨架见 `crate::unary`。")]
        pub struct $name;

        $crate::formula_owner!($name, "n,n->n", $operands, "nn::unary::formula");

        $crate::unary_impl! { $name, $apply $(, $input)? }

        $crate::operator_via!(
            @impl $name, $op, 1, (4096, 4096), $diff, 1, $crate::unary::AXES,
            $crate::unary::project_shape,
            $crate::unary::cost::<$name>,
            $sample, $reference, $dt
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = $dtype, skeleton = "unary")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 输入与输出等长，否则返回说明。
        pub fn forward(a: tensor::View<'_, $k>, out: &mut [$k]) -> diagnostic::Result<()> {
            $crate::unary::forward::<$k, $name>(a, out)
        }
    };
}

pub mod abs_f16 {
    //! f16 一元算子共享泛型 `apply` 与档配置；本项为逐元素绝对值 `y[n] = |a[n]|`。
    crate::unary_tier!(
        Abs,
        AbsF16,
        "abs_f16",
        "f16",
        crate::model::DType::F16,
        half::f16,
        crate::einsum::f16_io_operands(),
        crate::unary::sample_f16::<Abs>,
        crate::unary::reference_f16::<Abs>,
        |x| x.abs(),
        false
    );

    #[cfg(test)]
    mod tests {
        use super::*;
        crate::unary::test_support::unary_tier_tests!(
            crate::unary::sample_f16::<Abs>,
            crate::unary::reference_f16::<Abs>,
            half::f16,
            f16,
            forward
        );
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    include!("../tests/kernel_unary_test_support.rs");
}
