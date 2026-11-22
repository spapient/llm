//! 带类别与上下文的错误，以及全仓库统一的结果别名。

use crate::code::Code;
use core::fmt;
use std::path::Path;

/// 带类别与上下文的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    code: Code,
    message: String,
    ctx: Vec<String>,
    notes: Vec<String>,
}

impl Error {
    /// 由类别与消息构造。
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            ctx: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// 追加一层上下文（链式，便于 `map_err` 里就地补充位置）。
    #[must_use]
    pub fn with(mut self, ctx: impl Into<String>) -> Self {
        self.ctx.push(ctx.into());
        self
    }

    /// 追加一条**同级**诊断（同一次校验里的另一个失败）。
    #[must_use]
    pub fn also(mut self, other: &Self) -> Self {
        self.notes.push(other.to_string());
        self
    }

    /// 追加的同级诊断（已渲染为文本，含各自的类别与上下文）。
    #[must_use]
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// 类别。
    #[must_use]
    pub fn code(&self) -> Code {
        self.code
    }

    /// 原始消息（不含上下文）。
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// 上下文帧，按追加顺序。
    #[must_use]
    pub fn ctx(&self) -> &[String] {
        &self.ctx
    }

    /// 由一次读文件失败构造：路径不存在是 `missing`，其余是 `io`。
    pub fn from_io(path: &Path, error: &std::io::Error) -> Self {
        let code = if error.kind() == std::io::ErrorKind::NotFound {
            Code::Missing
        } else {
            Code::Io
        };
        Self::new(code, format!("读取 {} 失败：{error}", path.display()))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)?;
        render(&self.ctx, " → ", formatter)?;
        render(&self.notes, "\n  ", formatter)?;
        Ok(())
    }
}

/// 逐行渲染一组帧，各自带固定前缀；把循环收敛到一处以便 Display 保持低复杂度。
fn render(frames: &[String], prefix: &str, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    for frame in frames {
        write!(formatter, "{prefix}{frame}")?;
    }
    Ok(())
}

impl std::error::Error for Error {
    /// 底层源因刻意不挂进错误链：[`Error`] 是可克隆、可比较的值类型（退出码契约与诊断对拍
    /// 依赖 `Clone`/`PartialEq`/`Eq`），而 `Box<dyn Error>` 无法满足这两条。底层 `io::Error`
    /// 的类别已由 [`Self::from_io`] 归进 [`Code`]，消息也格式化进 [`Self::message`]；需要回溯
    /// 原始系统错误时从 `Code::Io` 与 message 文本取。不引入 `source` 链是刻意取舍，不是遗漏。
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// 本 crate 的结果别名：`Result<T, llm_diagnostic::Error>`。
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
#[path = "../tests/error_tests.rs"]
mod tests;
