//! 窗口族：最大池化（1d/2d）、平均池化与自适应池化，共用 [`crate::classic`] 的 `window_op!` 装配。

/// 自适应平均池化（声明档 4×4 → 2×2）。
pub mod adaptive;
/// 二维平均池化（声明档几何 `3×3, s=1, p=1`，补零不计入分母）。
pub mod avg2d;
/// 一维最大池化（声明档几何 `k=2, s=2, p=0`，单通道）。
pub mod max1d;
/// 二维最大池化（声明档几何 `3×3, s=1, p=1`，4×4 输入、same 输出）。
pub mod max2d;
