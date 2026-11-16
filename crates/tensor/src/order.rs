//! 排列顺序：同一组元素在扁平缓冲里沿哪个轴相邻。
//!
//! 这是纯形状元数据，不涉及缓冲与分配，因此归 `tensor`；`storage` 把它随布局再导出，
//! 不另持一份枚举。

/// 元素在扁平缓冲里的排列顺序。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// 行主序：最后一轴相邻（C / numpy 默认）。
    RowMajor,
    /// 列主序：第一轴相邻（Fortran / cuBLAS 的 `T` 位）。
    ColMajor,
}
