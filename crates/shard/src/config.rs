//! 运行时配置：部署参数由 `cli` 注入，本 crate 不读取配置文件，也不猜测配置来源。
//!
//! 配置只包含运行时旋钮；烘焙进生成内核的常量属于内核身份，不放进 [`RuntimeConfig`]。

use cutile::prof::Emit;
use diagnostic::{Code, Error, Result};
use std::collections::BTreeSet;
use std::sync::OnceLock;
use std::time::Duration;

/// 一门语言的语法门：解析器程序、固定前缀参数与语法检查参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageFront {
    /// 这门语言。
    pub emit: Emit,
    /// 解析器：程序名 + 固定前缀参数。
    pub tool: Vec<String>,
    /// 只做语法检查的参数；源文件路径由执行器追加在末尾。
    pub check: Vec<String>,
}

impl LanguageFront {
    /// 从任意字符串集合构造一条语法门配置。
    pub fn new<T, C>(emit: Emit, tool: T, check: C) -> Self
    where
        T: IntoIterator,
        T::Item: Into<String>,
        C: IntoIterator,
        C::Item: Into<String>,
    {
        Self {
            emit,
            tool: tool.into_iter().map(Into::into).collect(),
            check: check.into_iter().map(Into::into).collect(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.tool.is_empty() || self.tool.iter().any(|part| part.trim().is_empty()) {
            return invalid(format!("`{}` 的语法前端命令不能为空", self.emit));
        }
        if self.check.is_empty() {
            return invalid(format!("`{}` 的语法检查参数不能为空", self.emit));
        }
        Ok(())
    }
}

/// 编译与执行层的部署旋钮。
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    /// 各语言语法门。新增语言只需增加一条配置。
    pub gates: Vec<LanguageFront>,
    /// 本机自检用的 host C 编译参数。
    pub host_flags: Vec<String>,
    /// L3 达成率门槛，必须在 `0..=1` 内且为有限数。
    pub l3_threshold: f64,
    /// 缓存锁等待上限，必须大于零。
    pub lock_wait: Duration,
}

impl RuntimeConfig {
    /// 校验整份配置，拒绝会在执行期造成歧义或越界的值。
    ///
    /// # Errors
    ///
    /// 语法门命令或检查参数为空、语言重复登记、门槛不是 `0..=1` 内的有限数，或锁等待为零。
    pub fn validate(&self) -> Result<()> {
        self.check_gates()?;
        self.check_scalars()?;
        Ok(())
    }

    /// 逐门校验前端命令，并拒绝同一语言重复登记。
    fn check_gates(&self) -> Result<()> {
        let mut languages = BTreeSet::new();
        for gate in &self.gates {
            gate.validate()?;
            if !languages.insert(gate.emit.as_str()) {
                return invalid(format!("`{}` 的语法前端重复登记", gate.emit));
            }
        }
        Ok(())
    }

    /// 校验数值旋钮：门槛落在 `0..=1` 内且有限，锁等待大于零。
    fn check_scalars(&self) -> Result<()> {
        if !self.l3_threshold.is_finite() || !(0.0..=1.0).contains(&self.l3_threshold) {
            return invalid(format!(
                "L3 达成率门槛必须是 0..=1 内的有限数，得到 {}",
                self.l3_threshold
            ));
        }
        if self.lock_wait.is_zero() {
            return invalid("缓存锁等待上限必须大于零".to_string());
        }
        Ok(())
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            gates: vec![
                LanguageFront::new(
                    Emit::Cuda,
                    ["clang"],
                    [
                        "-x",
                        "cuda",
                        "--cuda-host-only",
                        "-nocudainc",
                        "-nocudalib",
                        "-fsyntax-only",
                    ],
                ),
                LanguageFront::new(Emit::Opencl, ["clang"], ["-x", "cl", "-fsyntax-only"]),
                LanguageFront::new(Emit::C, ["clang"], ["-x", "c", "-fsyntax-only"]),
                LanguageFront::new(
                    Emit::Metal,
                    ["xcrun", "-sdk", "macosx", "metal"],
                    ["-fsyntax-only"],
                ),
                LanguageFront::new(
                    Emit::Ascendc,
                    ["ccec"],
                    ["-xcce", "--cce-aicore-algorithm=dav-c220", "-fsyntax-only"],
                ),
            ],
            host_flags: ["-O2", "-std=c99"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            l3_threshold: 0.7,
            lock_wait: Duration::from_secs(5),
        }
    }
}

static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

/// 取当前配置；未注入时使用内置默认值。
#[must_use]
pub fn config() -> &'static RuntimeConfig {
    CONFIG.get_or_init(RuntimeConfig::default)
}

/// 校验并安装进程级配置；只能成功安装一次。
///
/// # Errors
///
/// 配置未通过 [`RuntimeConfig::validate`]，或此前已经读取过／安装过配置。
pub fn install(value: RuntimeConfig) -> Result<()> {
    value.validate()?;
    CONFIG.set(value).map_err(|_| {
        Error::new(
            Code::Invalid,
            "运行时配置已经安装，启动后不能替换".to_string(),
        )
    })
}

fn invalid<T>(message: String) -> Result<T> {
    Err(Error::new(Code::Invalid, message).with("shard::RuntimeConfig::validate"))
}

#[cfg(test)]
#[path = "../tests/config_tests.rs"]
mod tests;
