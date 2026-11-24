//! 结构化相等断言：失败时按行级 diff 标出首个差异，而不是只打印两段 Debug。
//!
//! 三条断言口径分工，避免用错判据：
//!
//! - [`assert_eq`]：集合 / 结构体 / 字节片等**精确相等**的对比，需要一眼定位首个差异时用；
//! - [`crate::compare`]：浮点对拍（逐元素容差、整批相对误差、方向余弦），数值不许用精确相等；
//! - [`crate::hash`]：位级 / 字节快照的逐字节锁定，走 CRC 参考值，不做 Debug diff。
//!
//! 这是 std `assert_eq!` 的诊断增强替代，**判定语义仍是 `PartialEq`**：两边不等即失败，
//! 不引入任何浮点容差。宏由 `similar-asserts` 提供，测试库随 testkit 只进 dev 构建。

pub use similar_asserts::assert_eq;
