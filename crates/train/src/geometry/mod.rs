//! 算子实例几何：把窗口、步长、分组、批空间这些**运行时参数**从算子名里拆出来。

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
