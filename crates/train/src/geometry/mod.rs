//! 算子实例几何：把窗口、步长、分组、批空间这些**运行时参数**从算子名里拆出来。
//!
//! `op::classic` 的声明档（固定的 `k/s/p`、`4×4 → 2×2`、`C=4, spatial=2, n=2`）在这里是
//! [`Geometry::declared`] 的一个实例：对拍基准仍然是它，但训练侧可以按任意几何前向，反向也
//! 按**同一个实例**回传。前向与反向共用这一个枚举，是"两套几何静默错配"的根治办法——算术只有
//! 一份，声明档只是它的一个取值。
//!
//! 布局沿用 `op::conv` / `op::norm` 的口径：池化与归一化都是 NHWC（通道在最内维），
//! 层归一化是行主序 `rows × width`。

mod declared;
mod dispatch;
mod kernel;
mod shape;
mod spec;

pub(crate) use spec::MOMENTUM;
pub use spec::{EPS, Geometry, Running};

#[cfg(test)]
#[path = "../../tests/geometry_tests.rs"]
mod tests;
