//! 能力词汇表与参数化签名。
//!
//! 分层：`vocab` 是闭集与词汇表，`attr` 是区间与属性，`signature` 是签名与协商判定。

mod attr;
mod signature;
mod vocab;

pub use attr::{Attr, Span};
pub use signature::{KernelKey, Mismatch, Need, Provide, Witness};
pub use vocab::{AttrKey, Closed, DType, DTypeSet, Grade, Prim, Set, Space, SpaceSet};

#[cfg(test)]
#[path = "../../tests/atom.rs"]
mod tests;
