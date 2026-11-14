//! 带类别与上下文的错误，以及全仓库统一的结果别名。
//!
//! [`Error`] 是“错误即值”的载体：仓库不打日志、不随处 panic，而是把失败的类别
//! （[`Code`]）、原始消息、从内到外的上下文帧与同级诊断都收进一个值，交给入口层
//! 映射成退出码。读文件的分类统一走 [`Error::from_io`]，避免每个调用点各猜一遍。

use crate::code::Code;
use core::fmt;
use std::path::Path;

/// 带类别与上下文的错误。
///
/// `context` 是按“从内到外”追加的帧，`Display` 时用 ` → ` 连接，所以最外层（最靠近调用点）
/// 在最后——诊断读起来是“发生了什么 → 在哪一层被发现的”。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    code: Code,
    message: String,
    context: Vec<String>,
    notes: Vec<String>,
}

impl Error {
    /// 由类别与消息构造。
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// 追加一层上下文（链式，便于 `map_err` 里就地补充位置）。
    #[must_use]
    pub fn with(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// 追加一条**同级**诊断（同一次校验里的另一个失败）。
    ///
    /// `Result` 只能携一个错误，但“装载一次报全”对改数据的人比“只报第一条”有用得多。
    /// 类别与上下文取自第一条，追加的行各自带着自己的类别文本——所以 `missing` 与
    /// `invalid` 混在一起时仍然可以分辨，而不是被压成同一个码。
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
    pub fn context(&self) -> &[String] {
        &self.context
    }

    /// 由一次读文件失败构造：路径不存在是 `missing`，其余是 `io`。
    ///
    /// 这条分类放在错误类型所在的层：它同时是退出码契约（`missing` → 4、`io` → 5），
    /// 各读文件的层各写一遍，脚本看到的分类迟早会不一致。
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
        render(&self.context, " → ", formatter)?;
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

impl std::error::Error for Error {}

/// 本 crate 的结果别名：`Result<T, llm_diagnostic::Error>`。
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_is_appended_from_inside_out() {
        let error = Error::new(Code::Missing, "模板缺配对的 `.k`")
            .with("op::load")
            .with("cutile::builtin");
        assert_eq!(error.message(), "模板缺配对的 `.k`");
        assert_eq!(error.context(), ["op::load", "cutile::builtin"]);
        assert_eq!(
            error.to_string(),
            "missing: 模板缺配对的 `.k` → op::load → cutile::builtin"
        );
    }

    #[test]
    fn a_peer_diagnostic_keeps_its_own_category() {
        let first = Error::new(Code::Missing, "缺片段 `load`").with("`cpu`");
        let second = Error::new(Code::Invalid, "`dot` 的见证不成立");
        let error = first.also(&second).with("cutile::builtin::load");
        assert_eq!(error.code(), Code::Missing, "首条的类别不被改写");
        assert_eq!(error.notes().len(), 1);
        assert_eq!(
            error.to_string(),
            "missing: 缺片段 `load` → `cpu` → cutile::builtin::load\n  invalid: `dot` 的见证不成立"
        );
    }

    #[test]
    fn a_missing_path_is_missing_and_a_broken_read_is_io() {
        let absent = std::io::Error::new(std::io::ErrorKind::NotFound, "没有这个文件");
        let error = Error::from_io(Path::new("/tmp/nope"), &absent);
        assert_eq!(error.code(), Code::Missing, "{error}");
        assert!(error.message().contains("/tmp/nope"), "{error}");

        let denied = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "拒绝访问");
        let error = Error::from_io(Path::new("/tmp/nope"), &denied);
        assert_eq!(error.code(), Code::Io, "{error}");
    }

    #[test]
    fn an_error_is_usable_as_a_std_error() {
        let error: Box<dyn std::error::Error> = Box::new(Error::new(Code::Io, "读设备档案失败"));
        assert!(error.to_string().contains("io:"));
    }
}
