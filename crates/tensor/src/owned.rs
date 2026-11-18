//! 拥有缓冲的可写张量：元数据 + 持有字节向量。
//!
//! 与只读 [`crate::Tensor<'a>`] 视图互补：`Tensor` 是调用方持有的零拷贝读路径，`OwnedTensor`
//! 是分配出来、可写入的拥有形态。`dtype` / `shape` / `strides` 与设备无关；设备地址
//! 藏在分配器返回的句柄里，本层不泄漏（ADR 0025 §2）。

use diagnostic::{Code, Error, Result};

use crate::DType;
use crate::shape::Shape;
use crate::strides::Strides;
use crate::tensor::Tensor;
use crate::validate;

/// 拥有缓冲的可写张量：元数据 + 持有字节向量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensor {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    bytes: Vec<u8>,
}

impl OwnedTensor {
    /// 按形状分配并清零；`dtype` 必须在 [`crate::SUPPORTED`] 子集内。
    pub fn zeros(dtype: DType, shape: Shape) -> Result<Self> {
        let required = shape.element_bytes(dtype);
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(required).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!(
                    "{dtype:?} 的 {:?} 需要 {required} 字节：超出可用内存或地址空间",
                    shape.dims()
                ),
            )
            .with("tensor::OwnedTensor::zeros")
        })?;
        bytes.resize(required, 0);
        Self::from_bytes(dtype, shape, bytes)
    }

    /// 由已有字节向量构造：`dtype` 受支持且缓冲够长，布局取行主序连续。
    pub fn from_bytes(dtype: DType, shape: Shape, bytes: Vec<u8>) -> Result<Self> {
        validate::check_byte_buffer(
            dtype,
            &shape,
            bytes.len(),
            "tensor::OwnedTensor::from_bytes",
        )?;
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            dtype,
            shape,
            strides,
            bytes,
        })
    }

    tensor_meta!(field);

    /// 只读字节。
    pub fn data(&self) -> &[u8] {
        &self.bytes
    }

    /// 可写字节。
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// 借出只读字节视图（零拷贝读路径）；元数据构造时已校验缓冲够长，这里只做一次同口径复核。
    pub fn view(&self) -> Result<Tensor<'_>> {
        Tensor::new(self.dtype, self.shape.clone(), &self.bytes)
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }
}
