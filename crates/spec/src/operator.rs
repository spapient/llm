//! 算子基座：语义变化点的唯一契约（ADR 0050）。

use crate::model::{AxisSpec, Cost, DType, Model, Sample, ShapeContext};

/// 一个算子的全部语义：名字、生产形状、精度、轴、可微性，以及代价/样本/参考三条行为。
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

/// 为「模块自带具体入口」的算子装配 [`Operator`]（ADR 0050 §3）：`AXES` / `project_shape` /
/// `cost` / `sample` / `reference` 都取当前模块的同名项，装配行只剩类型、注册名、生产形状、
/// 可微性与参考入口。族骨架的泛型入口（reduce / unary / binary / gemv / gemm）走
/// [`operator_via!`]。
#[macro_export]
macro_rules! operator_here {
    ($ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $reference:path) => {
        ::llm_spec::operator_via!(
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
#[macro_export]
macro_rules! operator_via {
    ($ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $skel:ident, $axes:expr) => {
        ::llm_spec::operator_via!(
            @impl $ty,
            $op,
            $arity,
            $shape,
            $differentiable,
            1,
            $axes,
            $skel::project_shape,
            $skel::cost::<$ty>,
            $skel::sample::<$ty>,
            $skel::reference::<$ty>
        );
    };
    (grained $ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $skel:ident, $axes:expr) => {
        ::llm_spec::operator_via!(
            @impl $ty,
            $op,
            $arity,
            $shape,
            $differentiable,
            <$ty as $skel::Format>::GRAIN as u32,
            $axes,
            $skel::project_shape,
            $skel::cost::<$ty>,
            $skel::sample::<$ty>,
            $skel::reference::<$ty>
        );
    };
    (
        @impl $ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $group:expr,
        $axes:expr, $project:path, $cost:path, $sample:path, $reference:path
    ) => {
        impl ::llm_spec::operator::Operator for $ty {
            const NAME: &'static str = $op;
            const SHAPE: (usize, usize) = $shape;
            const DTYPE: ::llm_spec::model::DType = ::llm_spec::model::DType::F32;
            const AXES: &'static [::llm_spec::model::AxisSpec] = $axes;
            const DIFFERENTIABLE: bool = $differentiable;
            const GROUP: u32 = $group;
            const ARITY: usize = $arity;

            fn shape_ctx(
                m: usize,
                n: usize,
            ) -> ::core::result::Result<::llm_spec::model::ShapeContext, ::diagnostic::Error> {
                $project(m, n)
            }

            fn cost(
                shape: &::llm_spec::model::ShapeContext,
            ) -> ::core::result::Result<::llm_spec::model::Cost, ::diagnostic::Error> {
                $cost(shape)
            }

            fn sample(m: usize, n: usize) -> ::llm_spec::model::Sample {
                $sample(m, n)
            }

            fn reference(sample: &::llm_spec::model::Sample) -> ::diagnostic::Result<Vec<f32>> {
                $reference(sample)
            }
        }
    };
}
