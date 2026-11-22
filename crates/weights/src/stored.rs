//! 已解析段条目：共享描述符 + 重算出的绝对字节区间，以及单段无损口径。

use crate::bad;
use diagnostic::Result;

use crate::quant::Quant;
use crate::seg::Descriptor;
use crate::{Codec, Wire};

/// 一个已解析的段：共享描述符 + 重算出的绝对字节区间。
#[derive(Debug, Clone)]
pub struct Stored {
    pub(crate) desc: Descriptor,
    pub(crate) inner: StoredInner,
}

/// 段的存储形态：内联帧在本段名下；引用段解引用到代表段，区间/编码/CRC 在布局时复制自代表段。
#[derive(Debug, Clone)]
pub(crate) enum StoredInner {
    Inline {
        codec: Codec,
        start: usize,
        len: usize,
        crc: u32,
    },
    Ref {
        rep: usize,
        codec: Codec,
        start: usize,
        len: usize,
        crc: u32,
    },
}

impl Stored {
    /// 段名。
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    /// 存储帧的取值视图（codec、绝对起点、帧长、段级 CRC）；内联与引用两种形态共用一次解构。
    fn span(&self) -> (Codec, usize, usize, u32) {
        match &self.inner {
            StoredInner::Inline {
                codec,
                start,
                len,
                crc,
            }
            | StoredInner::Ref {
                codec,
                start,
                len,
                crc,
                ..
            } => (*codec, *start, *len, *crc),
        }
    }

    /// 存储编解码方式；引用段沿用代表段的 codec。
    pub fn codec(&self) -> Codec {
        self.span().0
    }

    /// 元素类型。
    pub fn dtype(&self) -> Wire {
        self.desc.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &[u64] {
        &self.desc.shape
    }

    /// 量化方案（若有）。
    pub fn quant(&self) -> Option<&Quant> {
        self.desc.quant.as_ref()
    }

    /// 数据起点；引用段沿用代表段的区间。
    pub fn start(&self) -> usize {
        self.span().1
    }

    /// 存储字节数；引用段沿用代表段的帧长。
    pub fn len(&self) -> usize {
        self.span().2
    }

    /// 是否零字节。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 是否为引用段（共享代表段字节，自身数据区不占帧）。
    pub fn is_ref(&self) -> bool {
        matches!(self.inner, StoredInner::Ref { .. })
    }

    /// 代表段下标；内联段返回 `None`。
    pub fn rep_idx(&self) -> Option<usize> {
        match &self.inner {
            StoredInner::Ref { rep, .. } => Some(*rep),
            StoredInner::Inline { .. } => None,
        }
    }

    /// 逻辑字节数：元素类型 × 元素个数，也是 `rans::decode` 的分配上限（来自描述符而非帧）。
    pub fn logical_len(&self) -> Result<usize> {
        usize::try_from(self.desc.dtype.bytes_for(tensor::product(&self.desc.shape)))
            .map_err(|_| bad("段的逻辑长度超 usize"))
    }

    /// 段级 CRC；引用段复制自代表段，不另存校验字段。
    pub(crate) fn crc(&self) -> u32 {
        self.span().3
    }
}

/// 单段是否无损：不带量化方案，且 codec 属于无损的 `Store`/`Rans`/`Axes`。
pub(crate) fn lossless_entry(seg: &Stored) -> bool {
    seg.quant().is_none() && matches!(seg.codec(), Codec::Store | Codec::Rans | Codec::Axes)
}
