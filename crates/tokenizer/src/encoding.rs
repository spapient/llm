//! 一次编码的结果：编号、字符区间与特殊标记三者等长。

use crate::Token;
use diagnostic::{Code, Error, Result};

/// 编码产出的对齐记录：token 编号、在原文本中的字节区间、是否特殊 token。
///
/// 三段长度在构造期闭合；暂不产出元数据的实现可用 [`Encoding::from_ids`]，
/// 区间与标记留空。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    ids: Vec<Token>,
    offsets: Vec<(usize, usize)>,
    special: Vec<bool>,
}

impl Encoding {
    /// 只带编号；对齐区间与特殊标记留空。
    #[must_use]
    pub fn from_ids(ids: Vec<Token>) -> Self {
        Self {
            ids,
            offsets: Vec::new(),
            special: Vec::new(),
        }
    }

    /// 带完整元数据；三段长度必须一致，否则报 [`Code::Invalid`]。
    ///
    /// 区间是原文本的 UTF-8 字节 `[start, end)`；`special` 标记该编号是否特殊 token。
    pub fn new(ids: Vec<Token>, offsets: Vec<(usize, usize)>, special: Vec<bool>) -> Result<Self> {
        let len = ids.len();
        if offsets.len() != len || special.len() != len {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "编码元数据长度不齐：编号 {len}、区间 {}、特殊标记 {}",
                    offsets.len(),
                    special.len()
                ),
            )
            .with("tokenizer::Encoding::new"));
        }
        Ok(Self {
            ids,
            offsets,
            special,
        })
    }

    /// token 编号序列。
    #[must_use]
    pub fn ids(&self) -> &[Token] {
        &self.ids
    }

    /// 每个编号对应的字节区间；未产出时为空。
    #[must_use]
    pub fn offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    /// 第 `index` 条是否特殊 token；越界返回 [`None`]。
    #[must_use]
    pub fn is_special(&self, index: usize) -> Option<bool> {
        self.special.get(index).copied()
    }

    /// token 个数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 是否为空序列。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 拆开编号序列，丢弃对齐元数据。
    #[must_use]
    pub fn into_ids(self) -> Vec<Token> {
        self.ids
    }
}
