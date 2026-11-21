//! 逻辑张量：元数据 + 只读字节视图。

use diagnostic::Result;

use crate::DType;
use crate::shape::Shape;
use crate::shape::Strides;

pub use owned::OwnedTensor;

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

    tensor_meta!(field);

    /// 只读字节视图。
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }
}

mod validate {
    //! 构造期共享校验：元素子集与字节缓冲长度。

    use diagnostic::{Code, Error, Result};

    use crate::shape::Shape;
    use crate::{DType, supports};

    /// 校验 `dtype` 属于计算层支持子集；不支持时附上构造点标记。
    pub(crate) fn check_supported(dtype: DType, marker: &str) -> Result<()> {
        if supports(dtype) {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!("{dtype:?} 不在计算层支持的元素子集内"),
        )
        .with(marker))
    }

    /// 校验字节缓冲够长：先过子集校验，再按形状与类型核算所需字节数。
    pub(crate) fn check_byte_buffer(
        dtype: DType,
        shape: &Shape,
        buffer_len: usize,
        marker: &str,
    ) -> Result<()> {
        check_supported(dtype, marker)?;
        let required = shape.element_bytes(dtype);
        if buffer_len < required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {buffer_len} 字节，{dtype:?} 的 {:?} 需要 {required} 字节",
                    shape.dims()
                ),
            )
            .with(marker));
        }
        Ok(())
    }
}

mod owned {
    //! 拥有缓冲的可写张量：元数据 + 持有字节向量。

    use diagnostic::{Code, Error, Result};

    use super::validate;
    use crate::DType;
    use crate::shape::Shape;
    use crate::shape::Strides;
    use crate::tensor::Tensor;

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
}
