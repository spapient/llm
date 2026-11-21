//! 算子基座：语义变化点的唯一契约（ADR 0050）。
//!
//! 六族骨架（`Reduce`、`Unary`、`Binary`、`Format`、`Gemm`，实现都在 `op-kernel`）仍各自声明
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

/// 为「模块自带具体入口」的算子装配 [`Operator`]（ADR 0050 §3）：`AXES` / `project_shape` /
/// `cost` / `sample` / `reference` 都取当前模块的同名项，装配行只剩类型、注册名、生产形状、
/// 可微性与参考入口。族骨架的泛型入口（reduce / unary / binary / gemv / gemm）走
/// [`operator_via!`]。
#[macro_export]
macro_rules! operator_here {
    ($ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $reference:path) => {
        ::llm_op_spec::operator_via!(
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
/// 两种形态：`grained` 把粒度对齐到骨架的 `Format::GRAIN`，
/// 其余按稠密算子取 `1`；入口路径由调用方给（族骨架取 `$skel::*` 的泛型入口，
/// 模块自带入口时取 `self::*`，见 [`operator_here!`]）。装配宏因此都不必再写
/// 第二份 impl。
#[macro_export]
macro_rules! operator_via {
    ($ty:ty, $op:tt, $arity:literal, $shape:expr, $differentiable:literal, $skel:ident, $axes:expr) => {
        ::llm_op_spec::operator_via!(
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
        ::llm_op_spec::operator_via!(
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
        impl ::llm_op_spec::operator::Operator for $ty {
            const NAME: &'static str = $op;
            const SHAPE: (usize, usize) = $shape;
            const DTYPE: ::llm_op_spec::model::DType = ::llm_op_spec::model::DType::F32;
            const AXES: &'static [::llm_op_spec::model::AxisSpec] = $axes;
            const DIFFERENTIABLE: bool = $differentiable;
            const GROUP: u32 = $group;
            const ARITY: usize = $arity;

            fn shape_ctx(
                m: usize,
                n: usize,
            ) -> ::core::result::Result<::llm_op_spec::model::ShapeContext, ::diagnostic::Error> {
                $project(m, n)
            }

            fn cost(
                shape: &::llm_op_spec::model::ShapeContext,
            ) -> ::core::result::Result<::llm_op_spec::model::Cost, ::diagnostic::Error> {
                $cost(shape)
            }

            fn sample(m: usize, n: usize) -> ::llm_op_spec::model::Sample {
                $sample(m, n)
            }

            fn reference(sample: &::llm_op_spec::model::Sample) -> ::diagnostic::Result<Vec<f32>> {
                $reference(sample)
            }
        }
    };
}
