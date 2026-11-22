//! 泛型只读元素视图：[`Shape`] + [`Strides`] + 调用方的 `&[E]`。

use diagnostic::{Code, Error, Result};

use crate::DType;
use crate::element::Element;
use crate::layout;
use crate::shape::Shape;
use crate::shape::Strides;

/// 泛型只读视图：形状 + 步长 + 调用方的 `&[E]`。
#[derive(Debug, Clone)]
pub struct View<'a, E: Element> {
    shape: Shape,
    strides: Strides,
    data: &'a [E],
}

impl<'a, E: Element> View<'a, E> {
    /// 由形状与连续布局的缓冲构造：元素个数必须与形状一致。
    pub fn new(shape: Shape, data: &'a [E]) -> Result<Self> {
        let required = shape.numel();
        if data.len() != required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {} 个元素，形状 {:?} 需要 {required} 个",
                    data.len(),
                    shape.dims()
                ),
            )
            .with("tensor::View::new"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// 由任意合法步长构造：转置、行/列切片这类非连续布局也是合法视图。
    pub fn strided(shape: Shape, strides: Strides, data: &'a [E]) -> Result<Self> {
        layout::check(
            shape.dims(),
            strides.as_slice(),
            data.len(),
            "tensor::View::strided",
        )?;
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    tensor_meta!(elem);

    /// 调用方的缓冲本身（同一个指针，不是副本）。
    pub fn as_slice(&self) -> &'a [E] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// 按下标取值；秩不符或越界返回 `None`（不 panic）。
    pub fn at(&self, idx: &[usize]) -> Option<E> {
        let offset = layout::offset(self.shape.dims(), self.strides.as_slice(), idx)?;
        self.data.get(offset).copied()
    }

    /// 二维视图的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
    pub fn row(&self, row: usize) -> Option<&'a [E]> {
        layout::row(self.shape.dims(), self.strides.as_slice(), self.data, row)
    }
}
