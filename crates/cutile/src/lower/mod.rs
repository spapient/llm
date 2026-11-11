//! lowering：五步、无 IR（DESIGN §10）。
//!
//! 中间产物全部是可见的数据结构：`TileCombo`（① expand）、`Path`（② negotiate，见
//! [`crate::cap`]）、`Plan`（③ select，见 [`crate::plan`]）、[`substitute::Rendered`]（④）、
//! [`Source`]（⑤）。没有任何隐藏的优化阶段。

pub mod emit;
pub mod expand;
pub mod substitute;

pub use emit::Source;
pub use expand::{Axis, OpDecl, Skeleton, TileCombo, Variant, expand};
pub use substitute::{FragmentSet, Rendered};
