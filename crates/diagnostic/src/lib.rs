//! 全仓库唯一的错误类型：类别闭集 [`Code`] + 带上下文的 [`Error`] + [`Result`]。
//!
//! **拥有**：`Error`、`Code`、`Result`。**不拥有**：任何领域的错误码字面量——各层用自己的语义
//! 填 `Code`，但类型只有这一个。**依赖**：无。**被依赖**：除 `prof` / `prim` / `testkit`
//! 以外的全部成员（见 `docs/ARCHITECTURE.md` 的所有权表）。
//!
//! 存量代码里还有旧形态：`prof` / `prim` 边界的 `Result<_, String>`，以及 `llm-config`
//! 自带的 `Error`。收敛方向只有一个：新代码用本 crate，旧签名在 crate 边界处用 [`Error::new`]
//! 显式给出类别——不做"无类别的 `From<String>` 猜测"，否则类别会失去意义。
//!
//! 类别同时是**退出码的对外契约**（[`Code::exit_code`]）：L7 入口不再自己决定失败该返回 1 还是 2，
//! 而是把类别映射成退出码，于是"缺设备"与"声明写错"在脚本里可区分。
//!
//! # 例子
//!
//! ```
//! use llm_diagnostic::{Code, Error};
//!
//! let error = Error::new(Code::Unsupported, "dp4a 不支持 f32").with("cap::negotiate");
//! assert_eq!(error.code(), Code::Unsupported);
//! assert_eq!(error.context(), ["cap::negotiate"]);
//! assert!(error.to_string().contains("unsupported"));
//! ```

#![deny(unsafe_code)]

use core::fmt;

/// 错误类别闭集。
///
/// 加一个变体就要同步 [`Code::ALL`] 与 [`Code::as_str`]——类别是日志与退出码的对外契约，
/// 不允许随手加。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Code {
    /// 所需能力不存在：缺设备、缺原语、缺证据等级。
    Unsupported,
    /// 输入不合法：档案字段越界、帧格式错误、布局不匹配。
    Invalid,
    /// 缺东西：文件不存在、模板缺配对的 `.k`、键未登记。
    Missing,
    /// 读写出错。
    Io,
    /// 内部不变量被破坏（调用方无错，是 bug）。
    Internal,
}

impl Code {
    /// 全部类别，供 `doctor`、测试与退出码映射遍历。
    pub const ALL: [Self; 5] = [
        Self::Unsupported,
        Self::Invalid,
        Self::Missing,
        Self::Io,
        Self::Internal,
    ];

    /// 稳定的短名：写进日志与用户可见诊断，不随翻译变化。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Invalid => "invalid",
            Self::Missing => "missing",
            Self::Io => "io",
            Self::Internal => "internal",
        }
    }

    /// 进程退出码：类别到 shell 的契约，脚本靠它区分"缺设备"与"写错了"。
    ///
    /// 取值刻意分散且固定（加类别才算破坏兼容）：
    ///
    /// | 类别 | 退出码 | 含义 |
    /// |---|---|---|
    /// | `Invalid` | 2 | 输入/声明/用法不合法 |
    /// | `Unsupported` | 3 | 所需能力不存在（缺设备、缺原语、缺证据等级） |
    /// | `Missing` | 4 | 缺文件、缺键、缺设备档案 |
    /// | `Io` | 5 | 读写出错 |
    /// | `Internal` | 70 | 内部不变量被破坏（调用方无错） |
    ///
    /// 成功固定是 0；`70` 借用 `EX_SOFTWARE` 的惯例，避免与 shell 的 126/127 撞车。
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Invalid => 2,
            Self::Unsupported => 3,
            Self::Missing => 4,
            Self::Io => 5,
            Self::Internal => 70,
        }
    }
}

impl fmt::Display for Code {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// 带类别与上下文的错误。
///
/// `context` 是按"从内到外"追加的帧，`Display` 时用 ` → ` 连接，所以最外层（最靠近调用点）
/// 在最后——诊断读起来是"发生了什么 → 在哪一层被发现的"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    code: Code,
    message: String,
    context: Vec<String>,
}

impl Error {
    /// 由类别与消息构造。
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: Vec::new(),
        }
    }

    /// 追加一层上下文（链式，便于 `map_err` 里就地补充位置）。
    #[must_use]
    pub fn with(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// 类别。
    pub fn code(&self) -> Code {
        self.code
    }

    /// 原始消息（不含上下文）。
    pub fn message(&self) -> &str {
        &self.message
    }

    /// 上下文帧，按追加顺序。
    pub fn context(&self) -> &[String] {
        &self.context
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)?;
        for frame in &self.context {
            write!(formatter, " → {frame}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

/// 本 crate 的结果别名：`Result<T, llm_diagnostic::Error>`。
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_is_a_closed_stable_set() {
        let names: Vec<&str> = Code::ALL.iter().map(|code| code.as_str()).collect();
        assert_eq!(
            names,
            ["unsupported", "invalid", "missing", "io", "internal"]
        );
        let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), Code::ALL.len(), "类别名必须唯一");
    }

    #[test]
    fn every_code_has_its_own_exit_code() {
        let codes: Vec<u8> = Code::ALL.iter().map(|code| code.exit_code()).collect();
        assert_eq!(codes, [3, 2, 4, 5, 70]);
        let unique: std::collections::BTreeSet<u8> = codes.iter().copied().collect();
        assert_eq!(unique.len(), Code::ALL.len(), "退出码必须两两不同");
        assert!(!codes.contains(&0), "0 只表示成功");
    }

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
    fn an_error_is_usable_as_a_std_error() {
        let error: Box<dyn std::error::Error> = Box::new(Error::new(Code::Io, "读设备档案失败"));
        assert!(error.to_string().contains("io:"));
    }
}
