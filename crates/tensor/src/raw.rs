//! 运行期 `dtype` 的字节视图：[`Shape`] + [`Strides`] + 调用方的 `&[u8]`。
//!
//! [`Tensor`](crate::Tensor) 把元素类型钉在泛型上，编译器知道每个元素多宽；字节视图恰好相反——
//! 精度来自文件头 / 权重档案，构造时才知道是什么（ADR 0025 §2）。两者的契约也不同：元素张量要求
//! 逐元素对齐（`len == numel`），字节视图只要求容量够装下 `element_bytes(numel)`，因为 `U4` 这类
//! 亚字节精度本来就不是一个元素一个字节。
//!
//! 布局算式与元素张量共用 [`crate::layout`] 一份，不另起一套。

use diagnostic::Result;

use crate::invalid;

use crate::layout;
use crate::shape::element_bytes;
use crate::{DType, Shape, Strides, supports};

/// 运行期 `dtype` 的只读字节视图。
#[derive(Debug, Clone)]
pub struct Raw<'a> {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    data: &'a [u8],
}

impl<'a> Raw<'a> {
    /// 由 `dtype`、形状与连续布局的字节缓冲构造。精度须在计算层子集内，缓冲须装得下。
    pub fn new(dtype: DType, shape: Shape, data: &'a [u8]) -> Result<Self> {
        let strides = Strides::contiguous(&shape);
        Self::strided(dtype, shape, strides, data)
    }

    /// 由任意合法步长构造（缓冲按最高偏移判可达性）。
    pub fn strided(dtype: DType, shape: Shape, strides: Strides, data: &'a [u8]) -> Result<Self> {
        if !supports(dtype) {
            let allowed = crate::SUPPORTED;
            return Err(invalid(
                format!("{dtype:?} 不在计算层子集 {allowed:?} 内"),
                "tensor::Raw::new",
            ));
        }
        let numel = shape.numel();
        // 字节视图的缓冲按容量判：亚字节精度下 `element_bytes` 会小于 `numel`。
        let need = element_bytes(numel, dtype);
        if data.len() < need {
            return Err(invalid(
                format!(
                    "缓冲 {} 字节，形状 {:?} 的 {numel} 个 {:?} 需要 {need} 字节",
                    data.len(),
                    shape.dims(),
                    dtype
                ),
                "tensor::Raw::strided",
            ));
        }
        layout::check(
            shape.dims(),
            strides.as_slice(),
            data.len(),
            "tensor::Raw::strided",
        )?;
        Ok(Self {
            dtype,
            shape,
            strides,
            data,
        })
    }

    /// 元素类型：运行期字段，不是泛型常量。
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

    /// 调用方的字节缓冲本身（同一个指针，不是副本）。
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        layout::is_contiguous(self.shape.dims(), self.strides.as_slice())
    }
}
