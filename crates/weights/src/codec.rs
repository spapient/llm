//! 段级编解码闭集：码位稳定，未知码在打开期拒绝。
//!
//! - [`Codec::Store`]：原样存储，读端零拷贝；
//! - [`Codec::Rans`]：[`crate::rans`] 熵编码帧，仅在严格变小时才会被写端选用。

use diagnostic::{Code, Error, Result};

/// 一个段的存储编解码方式。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    /// 原样字节（code 0）。
    Store = 0,
    /// rANS 熵编码帧（code 1）。
    Rans = 1,
}

impl Codec {
    /// 线上码位。
    pub(crate) fn code(self) -> u8 {
        self as u8
    }

    /// 从码位解析；未知码显式报错而非静默回退。
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Store),
            1 => Ok(Self::Rans),
            other => Err(
                Error::new(Code::Unsupported, format!("未知段 codec {other}"))
                    .with("weights::codec"),
            ),
        }
    }
}
