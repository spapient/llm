//! 归一化族：批 / 组 / 实例 / 层，四种统计范围共用 [`crate::classic`] 的 `norm_op!` 装配。

/// 批归一化（声明档 `C=4, spatial=2, n=2`，训练口径：批统计自上而下、滑动统计按动量混入）。
pub mod batch;
/// 组归一化（声明档 `G=2, C=4, spatial=2, n=2`）。
pub mod group;
/// 实例归一化（声明档 `C=4, spatial=2, n=2`）：等价于 `GroupNorm(G = C)`。
pub mod instance;
/// 层归一化（声明档 `3 行 × 4 宽`）：逐行在最后一维统计。
pub mod layer;
