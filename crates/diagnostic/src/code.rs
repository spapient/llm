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
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    testkit::test_cases! {
    fn a_code_is_a_closed_stable_set() {
        let names: Vec<&str> = Code::ALL.iter().map(|code| code.as_str()).collect();
        assert_eq!(
            names,
            ["unsupported", "invalid", "missing", "io", "internal"]
        );
        let unique = BTreeSet::from_iter(names.iter().copied());
        assert_eq!(unique.len(), Code::ALL.len(), "类别名必须唯一");
    }

    fn every_code_has_its_own_exit_code() {
        let codes: Vec<u8> = Code::ALL.iter().map(|code| code.exit_code()).collect();
        assert_eq!(codes, [3, 2, 4, 5, 70]);
        let unique = BTreeSet::from_iter(codes.iter().copied());
        assert_eq!(unique.len(), codes.len(), "退出码必须两两不同");
        assert!(!codes.contains(&0), "0 只表示成功");
    }
    }
}
