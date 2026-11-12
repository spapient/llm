//! 物理布局与搬运：[`Order`] + [`Layout`] + [`remap`]。
//!
//! **拥有**：`Layout`、`Order`、`remap`。**不拥有**：分配（归 `mem`）、逻辑张量元数据（归
//! `tensor`）、地址空间（`GM` / `FAST` / `REG` 归 `cutile` / `prof`）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`weights`。
//!
//! 布局是"同一组元素在扁平缓冲里的排列方式"，所以它只需要形状、不需要数据类型：搬运在这里
//! 是元素级别的重排，按位打包（int4）由 `tensor::DType` 与具体算子负责。
//!
//! # 例子
//!
//! ```
//! use llm_storage::{remap, Layout, Order};
//! use tensor::Shape;
//!
//! let shape = Shape::new([2, 3]);
//! let row_major = Layout::new(shape.clone(), Order::RowMajor);
//! let col_major = Layout::new(shape, Order::ColMajor);
//! let source = [0u8, 1, 2, 3, 4, 5];
//! let moved = remap(&source, &row_major, &col_major).expect("形状一致");
//! assert_eq!(moved, [0, 3, 1, 4, 2, 5]);
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};
use tensor::{Shape, Strides};

/// 元素在扁平缓冲里的排列顺序。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// 行主序：最后一轴相邻（C / numpy 默认）。
    RowMajor,
    /// 列主序：第一轴相邻（Fortran / cuBLAS 的 `T` 位）。
    ColMajor,
}

/// 形状 + 排列顺序；步长由它推出，避免两处各写一份。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    order: Order,
}

impl Layout {
    /// 由形状与顺序构造。
    pub fn new(shape: Shape, order: Order) -> Self {
        Self { shape, order }
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 排列顺序。
    pub fn order(&self) -> Order {
        self.order
    }

    /// 按顺序推出的步长。
    pub fn strides(&self) -> Strides {
        let dims = self.shape.dims();
        let mut strides = vec![0; dims.len()];
        let mut acc = 1usize;
        match self.order {
            Order::RowMajor => {
                for axis in (0..dims.len()).rev() {
                    strides[axis] = acc;
                    acc = acc.saturating_mul(dims[axis]);
                }
            }
            Order::ColMajor => {
                for axis in 0..dims.len() {
                    strides[axis] = acc;
                    acc = acc.saturating_mul(dims[axis]);
                }
            }
        }
        Strides::new(strides)
    }
}

/// 把 `src` 从 `from` 布局重排成 `to` 布局。
///
/// 两种布局必须覆盖同一个形状；`src` 至少要够长。实现是逐元素搬运，所以它既能用于
/// 行↔列主序的转换，也能用于任何由形状推出的布局——不需要为每种转换写一条路径。
pub fn remap<T: Copy>(src: &[T], from: &Layout, to: &Layout) -> Result<Vec<T>> {
    if from.shape() != to.shape() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "布局形状不一致：{:?} vs {:?}",
                from.shape().dims(),
                to.shape().dims()
            ),
        )
        .with("storage::remap"));
    }
    let count = from.shape().numel();
    if count == 0 {
        return Ok(Vec::new());
    }
    if src.len() < count {
        return Err(Error::new(
            Code::Invalid,
            format!("源缓冲 {} 个元素，形状需要 {count} 个", src.len()),
        )
        .with("storage::remap"));
    }
    let from_strides = from.strides();
    let to_strides = to.strides();
    let mut out = vec![src[0]; count];
    for linear in 0..count {
        let source = offset(&from_strides, from.shape(), linear);
        let target = offset(&to_strides, to.shape(), linear);
        out[target] = src[source];
    }
    Ok(out)
}

/// 行主序线性下标 → 该布局下的元素偏移。
fn offset(strides: &Strides, shape: &Shape, linear: usize) -> usize {
    let dims = shape.dims();
    let mut rest = linear;
    let mut offset = 0usize;
    for axis in (0..dims.len()).rev() {
        let dim = dims[axis];
        if dim == 0 {
            return 0;
        }
        let index = rest % dim;
        rest /= dim;
        offset = offset.saturating_add(index.saturating_mul(strides.get(axis).unwrap_or(0)));
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_is_the_tensor_default_and_col_major_is_its_mirror() {
        let shape = Shape::new([2, 3]);
        assert_eq!(
            Layout::new(shape.clone(), Order::RowMajor)
                .strides()
                .as_slice(),
            [3, 1]
        );
        assert_eq!(
            Layout::new(shape, Order::ColMajor).strides().as_slice(),
            [1, 2]
        );
    }

    #[test]
    fn remap_transposes_a_row_major_buffer() {
        let shape = Shape::new([2, 3]);
        let from = Layout::new(shape.clone(), Order::RowMajor);
        let to = Layout::new(shape.clone(), Order::ColMajor);
        let moved = remap(&[0u8, 1, 2, 3, 4, 5], &from, &to).expect("形状一致");
        assert_eq!(moved, [0, 3, 1, 4, 2, 5]);
        let back = remap(&moved, &to, &from).expect("回到行主序");
        assert_eq!(back, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn an_identity_layout_change_is_a_copy() {
        let shape = Shape::new([4]);
        let layout = Layout::new(shape, Order::RowMajor);
        let moved = remap(&[1.0f32, 2.0, 3.0, 4.0], &layout, &layout).expect("相同布局");
        assert_eq!(moved, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mismatched_shapes_and_short_buffers_are_rejected() {
        let from = Layout::new(Shape::new([2, 3]), Order::RowMajor);
        let to = Layout::new(Shape::new([3, 2]), Order::ColMajor);
        assert_eq!(
            remap(&[0u8; 6], &from, &to).unwrap_err().code(),
            Code::Invalid
        );
        let same = Layout::new(Shape::new([2, 3]), Order::ColMajor);
        assert_eq!(
            remap(&[0u8; 5], &from, &same).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn an_empty_shape_maps_to_nothing() {
        let shape = Shape::new([0, 3]);
        let from = Layout::new(shape.clone(), Order::RowMajor);
        let to = Layout::new(shape, Order::ColMajor);
        assert_eq!(remap::<u8>(&[], &from, &to), Ok(Vec::new()));
    }
}
