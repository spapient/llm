//! 特殊 token 句柄：模型约定的控制编号，不进词片查表路径。

use crate::Token;
use diagnostic::{Code, Error, Result};

/// 七个特殊槽位的稳定 wire 序号（unk=0 … mask=6），勿重排。
const UNK: usize = 0;
const PAD: usize = 1;
const BOS: usize = 2;
const EOS: usize = 3;
const SEP: usize = 4;
const CLS: usize = 5;
const MASK: usize = 6;

/// 一组特殊 token 的编号占位；未配置的位置为 [`None`]。
///
/// 编号是否落在词表内由装配期校验；本结构只持有句柄，不做词表反查。
/// 槽位按 wire 序号存放（下标即序号），读写都走同一张下标表，无按名扇出。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    slots: [Option<u32>; 7],
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self::empty()
    }
}

impl SpecialTokens {
    /// 共享的空句柄常量，供 trait 默认实现与装配侧复用。
    pub const EMPTY: Self = Self::empty();

    /// 空句柄：没有任何特殊 token。
    pub const fn empty() -> Self {
        Self { slots: [None; 7] }
    }

    /// 登记序列起始编号。
    #[must_use]
    pub const fn with_bos(mut self, id: u32) -> Self {
        self.slots[BOS] = Some(id);
        self
    }

    /// 登记序列结束编号。
    #[must_use]
    pub const fn with_eos(mut self, id: u32) -> Self {
        self.slots[EOS] = Some(id);
        self
    }

    /// 登记词表外编号。
    #[must_use]
    pub const fn with_unk(mut self, id: u32) -> Self {
        self.slots[UNK] = Some(id);
        self
    }

    /// 登记填充编号。
    #[must_use]
    pub const fn with_pad(mut self, id: u32) -> Self {
        self.slots[PAD] = Some(id);
        self
    }

    /// 登记分隔编号。
    #[must_use]
    pub const fn with_sep(mut self, id: u32) -> Self {
        self.slots[SEP] = Some(id);
        self
    }

    /// 登记分类编号。
    #[must_use]
    pub const fn with_cls(mut self, id: u32) -> Self {
        self.slots[CLS] = Some(id);
        self
    }

    /// 登记掩码编号。
    #[must_use]
    pub const fn with_mask(mut self, id: u32) -> Self {
        self.slots[MASK] = Some(id);
        self
    }

    /// 序列起始编号。
    #[must_use]
    pub fn bos(&self) -> Option<Token> {
        self.slots[BOS].map(Token::new)
    }

    /// 序列结束编号。
    #[must_use]
    pub fn eos(&self) -> Option<Token> {
        self.slots[EOS].map(Token::new)
    }

    /// 词表外编号。
    #[must_use]
    pub fn unk(&self) -> Option<Token> {
        self.slots[UNK].map(Token::new)
    }

    /// 填充编号。
    #[must_use]
    pub fn pad(&self) -> Option<Token> {
        self.slots[PAD].map(Token::new)
    }

    /// 分隔编号。
    #[must_use]
    pub fn sep(&self) -> Option<Token> {
        self.slots[SEP].map(Token::new)
    }

    /// 分类编号。
    #[must_use]
    pub fn cls(&self) -> Option<Token> {
        self.slots[CLS].map(Token::new)
    }

    /// 掩码编号。
    #[must_use]
    pub fn mask(&self) -> Option<Token> {
        self.slots[MASK].map(Token::new)
    }

    /// 编号是否命中已登记的特殊 token。
    #[must_use]
    pub fn is_special(&self, id: u32) -> bool {
        self.slots.contains(&Some(id))
    }

    /// 已登记特殊 token 个数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// 是否一个特殊 token 都没登记。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 七个特殊槽位按稳定 wire 序展开，供中性序列化读取。
    pub(crate) const fn slots(&self) -> [(u8, Option<u32>); 7] {
        [
            (UNK as u8, self.slots[UNK]),
            (PAD as u8, self.slots[PAD]),
            (BOS as u8, self.slots[BOS]),
            (EOS as u8, self.slots[EOS]),
            (SEP as u8, self.slots[SEP]),
            (CLS as u8, self.slots[CLS]),
            (MASK as u8, self.slots[MASK]),
        ]
    }

    /// 按 wire 序号回填一个特殊槽位；未知序号报 `Invalid`。
    pub(crate) fn set_slot(&mut self, kind: u8, id: u32) -> Result<()> {
        let slot = self
            .slots
            .get_mut(kind as usize)
            .ok_or_else(|| invalid_slot(kind))?;
        *slot = Some(id);
        Ok(())
    }
}

fn invalid_slot(kind: u8) -> Error {
    Error::new(Code::Invalid, format!("未知特殊槽位序号 {kind}"))
}
