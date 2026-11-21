//! [`Layout`]：形状 + 排列顺序，以及 [`remap`] 的逐元素重排。

use diagnostic::{Code, Error, Result};
use tensor::{Order, Shape, Strides};

/// 形状 + 排列顺序；步长由 [`Strides::ordered`] 推出，累积规则只有一份。
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
        Strides::ordered(&self.shape, self.order)
    }
}

/// 把 `src` 从 `from` 布局重排成 `to` 布局。
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
#[path = "../tests/layout.rs"]
mod tests;
