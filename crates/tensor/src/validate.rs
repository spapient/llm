//! 构造期共享校验：元素子集与字节缓冲长度。
//!
//! [`crate::Tensor`] 与 [`crate::OwnedTensor`] 的构造口径必须一致：同一批 `dtype` 子集、
//! 同一条「缓冲够长」判定。校验集中在这里，避免两个构造点各写一遍导致文案或阈值漂移。

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
