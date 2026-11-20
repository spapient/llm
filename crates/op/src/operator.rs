//! 算子基座：语义变化点的唯一契约（ADR 0050）。
//!
//! 六族骨架（[`Reduce`](crate::reduce::Reduce)、`Unary`、`Binary`、`Format`、`Gemm`）仍各自声明
//! 族内装配配置；[`Operator`] 是它们之上的统一语义契约，也是 [`Model`] 的行为侧投影。每个具体
//! 算子同时实现族 trait 与本 trait，方法体只转发，不写第二份算术。
//!
//! `#[annotation::model]` 生成的 `MODEL` 是这一契约的一份实例。迁移期两份并存，由
//! 本文件末尾的投影一致性测试锁住漂移；每族迁移完成后 `MODEL` 改由 [`model_of`] 生成。

use crate::model::{AxisSpec, Cost, DType, Model, Sample, ShapeContext};

/// 一个算子的全部语义：名字、生产形状、精度、轴、可微性，以及代价/样本/参考三条行为。
///
/// 后端不进入本契约：执行变化点归 `cutile::backend`，跨层即破坏分层（ADR 0050）。
pub trait Operator {
    /// 注册名：与 `ops/<name>.toml` 的文件名、`REGISTRY` 的一行一致。
    const NAME: &'static str;
    /// 生产形状 `(m, n)`：必须是本算子的合法输入。
    const SHAPE: (usize, usize);
    /// 输出元素类型。
    const DTYPE: DType;
    /// 基本轴（角色 + 表达式字母）。
    const AXES: &'static [AxisSpec];
    /// 训练反向图里是否可微（`REGISTRY` 的同名列）。
    const DIFFERENTIABLE: bool;
    /// 形状粒度：`m` 必须是它的整数倍；稠密算子为 1。
    const GROUP: u32 = 1;
    /// 输入缓冲个数（不含输出）：类型化 API 用它把输入写成定长数组。
    ///
    /// 由算子定义处的装配行给出；`arity_matches_the_samples` 再把它与样本里非 `y` 缓冲数对拍。
    const ARITY: usize;

    /// 把形状对 `(m, n)` 翻译成算子自己的具名轴上下文。
    fn shape_ctx(m: usize, n: usize) -> Result<ShapeContext, diagnostic::Error>;
    /// 给定形状上下文的标量代价；不自洽时报错，不静默截断因子。
    fn cost(shape: &ShapeContext) -> Result<Cost, diagnostic::Error>;
    /// 给定 `(m, n)` 的确定性样本。
    fn sample(m: usize, n: usize) -> Sample;
    /// 标量参考：不依赖设备，也不依赖片段。
    fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>>;
}

/// 把 [`Operator`] 投影成语义表 [`Model`]。
///
/// 这是注解与基座之间唯一的翻译点：`#[annotation::model]` 生成
/// `const MODEL: Model = model_of::<T>();` 之后，`MODEL` 不再是第二份事实。
/// `const` 使表仍是编译期常量（ADR 0005），不引入运行期注册。
pub const fn model_of<T: Operator>() -> Model {
    Model {
        group: T::GROUP,
        shape: T::SHAPE,
        dtype: T::DTYPE,
        axes: T::AXES,
        shape_ctx: T::shape_ctx,
        cost: T::cost,
        sample: T::sample,
        reference: T::reference,
    }
}

/// `const` 字符串逐字节相等：注册表清单与 `Operator::NAME` 的编译期对拍用。
pub const fn same_str(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// 为「模块自带具体入口」的算子装配 [`Operator`]（ADR 0050 §3）：`AXES` / `project_shape` /
/// `cost` / `sample` / `reference` 都取当前模块的同名项，装配行只剩类型、注册名、生产形状、
/// 可微性与参考入口。族骨架的泛型入口（reduce / unary / binary / gemv / gemm）走
/// [`operator_via!`](crate::operator_via)。
#[macro_export]
macro_rules! operator_here {
    ($ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $reference:path) => {
        $crate::operator_via!(
            @impl $ty,
            $op,
            $arity,
            $shape,
            $differentiable,
            1,
            AXES,
            project_shape,
            self::cost,
            self::sample,
            $reference
        );
    };
}

/// [`Operator`] 的唯一生成点：名字、生产形状、精度、轴与三条行为全部由四个入口路径给出。
///
/// 两种形态：`grained` 把粒度对齐到骨架的 [`Format::GRAIN`](crate::gemv::Format::GRAIN)，
/// 其余按稠密算子取 `1`；入口路径由调用方给（族骨架取 `$crate::$skel::*` 的泛型入口，
/// 模块自带入口时取 `self::*`，见 [`operator_here!`](crate::operator_here)）。装配宏因此都不必再写
/// 第二份 impl。
#[macro_export]
macro_rules! operator_via {
    ($ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $skel:ident, $axes:expr) => {
        $crate::operator_via!(
            @impl $ty,
            $op,
            $arity,
            $shape,
            $differentiable,
            1,
            $axes,
            $crate::$skel::project_shape,
            $crate::$skel::cost::<$ty>,
            $crate::$skel::sample::<$ty>,
            $crate::$skel::reference::<$ty>
        );
    };
    (grained $ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $skel:ident, $axes:expr) => {
        $crate::operator_via!(
            @impl $ty,
            $op,
            $arity,
            $shape,
            $differentiable,
            <$ty as $crate::$skel::Format>::GRAIN as u32,
            $axes,
            $crate::$skel::project_shape,
            $crate::$skel::cost::<$ty>,
            $crate::$skel::sample::<$ty>,
            $crate::$skel::reference::<$ty>
        );
    };
    (
        @impl $ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $group:expr,
        $axes:expr, $project:path, $cost:path, $sample:path, $reference:path
    ) => {
        impl $crate::operator::Operator for $ty {
            const NAME: &'static str = $op;
            const SHAPE: (usize, usize) = $shape;
            const DTYPE: $crate::model::DType = $crate::model::DType::F32;
            const AXES: &'static [$crate::model::AxisSpec] = $axes;
            const DIFFERENTIABLE: bool = $differentiable;
            const GROUP: u32 = $group;
            const ARITY: usize = $arity;

            fn shape_ctx(
                m: usize,
                n: usize,
            ) -> ::core::result::Result<$crate::model::ShapeContext, ::diagnostic::Error> {
                $project(m, n)
            }

            fn cost(
                shape: &$crate::model::ShapeContext,
            ) -> ::core::result::Result<$crate::model::Cost, ::diagnostic::Error> {
                $cost(shape)
            }

            fn sample(m: usize, n: usize) -> $crate::model::Sample {
                $sample(m, n)
            }

            fn reference(sample: &$crate::model::Sample) -> ::diagnostic::Result<Vec<f32>> {
                $reference(sample)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    // [`Operator`](crate::operator::Operator) 投影与注册表的一致性（ADR 0050 §2）。
    //
    // 过渡期 `MODEL` 仍由 `#[model]` 注解生成；本测试逐字段比对 `model_of::<T>()` 与注册表里的
    // 那一份，任何一侧漂移都在这里失败，而不是等到 `verify` 用错语义表。

    use crate::model::Model;
    use crate::operator::{Operator, model_of};

    /// 投影与注册表逐字段一致：常量字段直接比，行为字段在同一个形状上比输出。
    fn assert_projection_matches_registry<T: Operator>() {
        let projected = model_of::<T>();
        let registered = crate::model(T::NAME).expect("算子必须已注册");
        assert_eq!(projected.group, registered.group, "{}: group", T::NAME);
        assert_eq!(projected.shape, registered.shape, "{}: shape", T::NAME);
        assert_eq!(projected.dtype, registered.dtype, "{}: dtype", T::NAME);
        assert_eq!(projected.axes, registered.axes, "{}: axes", T::NAME);
        assert_behavior_matches(projected, registered, T::NAME);
    }

    /// 代价、样本与参考：同一形状上下文的输出逐位相等。
    fn assert_behavior_matches(projected: Model, registered: Model, name: &str) {
        let shape = (projected.shape_ctx)(64, 8).expect("形状上下文可构造");
        let sample = (projected.sample)(64, 8);
        assert_eq!(sample, (registered.sample)(64, 8), "{name}: sample");
        assert_eq!(
            (projected.cost)(&shape).expect("投影代价可计算"),
            (registered.cost)(&shape).expect("注册表代价可计算"),
            "{name}: cost"
        );
        assert_eq!(
            (projected.reference)(&sample).expect("投影参考可计算"),
            (registered.reference)(&sample).expect("注册表参考可计算"),
            "{name}: reference"
        );
    }

    #[test]
    fn reduce_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::reduce_sum_f32::Sum>();
        assert_projection_matches_registry::<crate::reduce_max_f32::Max>();
    }

    #[test]
    fn unary_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::unary::exp_f32::Exp>();
        assert_projection_matches_registry::<crate::unary::log_f32::Log>();
        assert_projection_matches_registry::<crate::unary::gelu_f32::Gelu>();
        assert_projection_matches_registry::<crate::unary::silu_f32::Silu>();
        assert_projection_matches_registry::<crate::unary::scale_f32::Scale>();
    }

    #[test]
    fn binary_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::elementwise::add_f32::Add>();
        assert_projection_matches_registry::<crate::elementwise::sub_f32::Sub>();
        assert_projection_matches_registry::<crate::elementwise::mul_f32::Mul>();
        assert_projection_matches_registry::<crate::elementwise::div_f32::Div>();
        assert_projection_matches_registry::<crate::elementwise::max_f32::Max>();
        assert_projection_matches_registry::<crate::elementwise::min_f32::Min>();
        assert_projection_matches_registry::<crate::elementwise::residual_add_f32::Residual>();
    }

    #[test]
    fn dequant_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::dequant_i2::DequantI2>();
        assert_projection_matches_registry::<crate::dequant_i4::DequantI4>();
        assert_projection_matches_registry::<crate::dequant_i8::DequantI8>();
    }

    #[test]
    fn gemv_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::gemv::gemv_i2::I2>();
        assert_projection_matches_registry::<crate::gemv::gemv_i4::I4>();
        assert_projection_matches_registry::<crate::gemv::gemv_i8::I8>();
        assert_projection_matches_registry::<crate::gemv::gemv_f32::F32>();
    }

    #[test]
    fn gemm_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::gemm::gemm_i2::GemmI2>();
        assert_projection_matches_registry::<crate::gemm::gemm_i4::GemmI4>();
        assert_projection_matches_registry::<crate::gemm::gemm_i8::GemmI8>();
        assert_projection_matches_registry::<crate::gemm::gemm_f32::Dense>();
    }

    #[test]
    fn normalization_and_shape_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::softmax::softmax_f32::Softmax>();
        assert_projection_matches_registry::<crate::softmax::rmsnorm_f32::RmsNorm>();
        assert_projection_matches_registry::<crate::softmax::rope_f32::Rope>();
        assert_projection_matches_registry::<crate::softmax::embed_f32::Embed>();
        assert_projection_matches_registry::<crate::scan::scan_f32::Scan>();
    }

    #[test]
    fn attention_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::attention::Attention>();
        assert_projection_matches_registry::<crate::attention::FlashAttention>();
        assert_projection_matches_registry::<crate::attention::decode_attention::DecodeAttention>();
        assert_projection_matches_registry::<
            crate::attention::decode_attention::DecodeFlashAttention,
        >();
    }

    #[test]
    fn matmul_projection_matches_the_registered_model() {
        assert_projection_matches_registry::<crate::matmul::Matmul>();
        assert_projection_matches_registry::<crate::matmul_i8::MatmulI8Op>();
    }
}
