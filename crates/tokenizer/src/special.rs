//! 特殊 token 句柄：模型约定的控制编号，不进词片查表路径。

use crate::Token;

/// 一组特殊 token 的编号占位；未配置的位置为 [`None`]。
///
/// 编号是否落在词表内由装配期校验；本结构只持有句柄，不做词表反查。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
    pad: Option<u32>,
    sep: Option<u32>,
    cls: Option<u32>,
    mask: Option<u32>,
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
        Self {
            bos: None,
            eos: None,
            unk: None,
            pad: None,
            sep: None,
            cls: None,
            mask: None,
        }
    }

    /// 登记序列起始编号。
    #[must_use]
    pub const fn with_bos(mut self, id: u32) -> Self {
        self.bos = Some(id);
        self
    }

    /// 登记序列结束编号。
    #[must_use]
    pub const fn with_eos(mut self, id: u32) -> Self {
        self.eos = Some(id);
        self
    }

    /// 登记词表外编号。
    #[must_use]
    pub const fn with_unk(mut self, id: u32) -> Self {
        self.unk = Some(id);
        self
    }

    /// 登记填充编号。
    #[must_use]
    pub const fn with_pad(mut self, id: u32) -> Self {
        self.pad = Some(id);
        self
    }

    /// 登记分隔编号。
    #[must_use]
    pub const fn with_sep(mut self, id: u32) -> Self {
        self.sep = Some(id);
        self
    }

    /// 登记分类编号。
    #[must_use]
    pub const fn with_cls(mut self, id: u32) -> Self {
        self.cls = Some(id);
        self
    }

    /// 登记掩码编号。
    #[must_use]
    pub const fn with_mask(mut self, id: u32) -> Self {
        self.mask = Some(id);
        self
    }

    /// 序列起始编号。
    #[must_use]
    pub fn bos(&self) -> Option<Token> {
        self.bos.map(Token::new)
    }

    /// 序列结束编号。
    #[must_use]
    pub fn eos(&self) -> Option<Token> {
        self.eos.map(Token::new)
    }

    /// 词表外编号。
    #[must_use]
    pub fn unk(&self) -> Option<Token> {
        self.unk.map(Token::new)
    }

    /// 填充编号。
    #[must_use]
    pub fn pad(&self) -> Option<Token> {
        self.pad.map(Token::new)
    }

    /// 分隔编号。
    #[must_use]
    pub fn sep(&self) -> Option<Token> {
        self.sep.map(Token::new)
    }

    /// 分类编号。
    #[must_use]
    pub fn cls(&self) -> Option<Token> {
        self.cls.map(Token::new)
    }

    /// 掩码编号。
    #[must_use]
    pub fn mask(&self) -> Option<Token> {
        self.mask.map(Token::new)
    }

    /// 编号是否命中已登记的特殊 token。
    #[must_use]
    pub fn is_special(&self, id: u32) -> bool {
        [
            self.bos, self.eos, self.unk, self.pad, self.sep, self.cls, self.mask,
        ]
        .contains(&Some(id))
    }

    /// 已登记特殊 token 个数。
    #[must_use]
    pub fn len(&self) -> usize {
        [
            self.bos, self.eos, self.unk, self.pad, self.sep, self.cls, self.mask,
        ]
        .iter()
        .flatten()
        .count()
    }

    /// 是否一个特殊 token 都没登记。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
