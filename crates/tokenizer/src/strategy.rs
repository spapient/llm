//! 序列策略配置：插桩、截断与填充的纯数据约定。
//!
//! P0 只定义配置形状与构造期校验；策略的套用随流水线段在后续阶段落地。

use diagnostic::{Code, Error, Result};

/// 截断保留哪一侧。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// 保留尾部，截掉开头。
    Left,
    /// 保留开头，截掉尾部。
    Right,
}

/// 填充对齐方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadDirection {
    /// 填充放在序列前。
    Left,
    /// 填充放在序列后。
    Right,
}

/// 填充配置：目标长度、填充编号与方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    length: usize,
    pad_id: u32,
    direction: PadDirection,
}

impl Padding {
    /// 构造填充配置；目标长度必须非零，否则报 [`Code::Invalid`]。
    pub fn new(length: usize, pad_id: u32, direction: PadDirection) -> Result<Self> {
        if length == 0 {
            return Err(
                Error::new(Code::Invalid, "填充目标长度不能为 0").with("tokenizer::Padding::new")
            );
        }
        Ok(Self {
            length,
            pad_id,
            direction,
        })
    }

    /// 目标长度。
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }

    /// 填充编号。
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// 对齐方向。
    #[must_use]
    pub fn direction(&self) -> PadDirection {
        self.direction
    }
}

/// 一次编码的序列策略；默认不加 BOS/EOS、不截断、不填充。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strategy {
    add_bos: bool,
    add_eos: bool,
    add_sep: bool,
    max_len: Option<usize>,
    truncation: Option<Truncation>,
    padding: Option<Padding>,
}

impl Default for Strategy {
    fn default() -> Self {
        Self::no_op()
    }
}

impl Strategy {
    /// 不做任何序列加工的空策略。
    pub const fn no_op() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
            add_sep: false,
            max_len: None,
            truncation: None,
            padding: None,
        }
    }

    /// 编码前补 BOS。
    #[must_use]
    pub const fn with_bos(mut self) -> Self {
        self.add_bos = true;
        self
    }

    /// 编码后补 EOS。
    #[must_use]
    pub const fn with_eos(mut self) -> Self {
        self.add_eos = true;
        self
    }

    /// 在片段分界处补 SEP。
    #[must_use]
    pub const fn with_sep(mut self) -> Self {
        self.add_sep = true;
        self
    }

    /// 限制最大长度；截断侧由 [`Self::with_truncation`] 指定。
    #[must_use]
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    /// 指定截断侧。
    #[must_use]
    pub const fn with_truncation(mut self, side: Truncation) -> Self {
        self.truncation = Some(side);
        self
    }

    /// 指定填充。
    #[must_use]
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = Some(padding);
        self
    }

    /// 是否补 BOS。
    #[must_use]
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// 是否补 EOS。
    #[must_use]
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    /// 是否在片段分界处补 SEP。
    #[must_use]
    pub fn add_sep(&self) -> bool {
        self.add_sep
    }

    /// 最大长度上限。
    #[must_use]
    pub fn max_len(&self) -> Option<usize> {
        self.max_len
    }

    /// 截断侧。
    #[must_use]
    pub fn truncation(&self) -> Option<Truncation> {
        self.truncation
    }

    /// 填充配置。
    #[must_use]
    pub fn padding(&self) -> Option<&Padding> {
        self.padding.as_ref()
    }
}
