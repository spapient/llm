//! 错误类别闭集：全仓库唯一的失败分类，同时是日志短名与进程退出码的对外契约。

use core::fmt;

/// 错误类别闭集。
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
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Invalid => "invalid",
            Self::Missing => "missing",
            Self::Io => "io",
            Self::Internal => "internal",
        }
    }

    /// 进程退出码：类别到 shell 的契约，脚本靠它区分“缺设备”与“写错了”。
    #[must_use]
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

#[cfg(test)]
#[path = "../tests/code_tests.rs"]
mod tests;
