//! 逻辑张量：元数据 + 只读字节视图。

use diagnostic::Result;

use crate::shape::Shape;
use crate::strides::Strides;
use crate::validate;
use crate::DType;

/// 逻辑张量：元数据 + 只读字节视图。
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// 由元数据与连续布局的缓冲构造：`dtype` 必须在 [`crate::SUPPORTED`] 子集内、`data` 必须够长。
    pub fn new(dtype: DType, shape: Shape, data: &'a [u8]) -> Result<Self> {
        validate::check_byte_buffer(dtype, &shape, data.len(), "tensor::Tensor::new")?;
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            dtype,
            shape,
            strides,
            data,
        })
    }

    /// 元素类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 只读字节视图。
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }
}
