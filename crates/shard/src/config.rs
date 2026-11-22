//! 运行时配置：把散落在各文件的**运行时**常量收进一份纯数据，由 `cli` 从 toml 填充后注入。
//!
//! `RuntimeConfig` 是部署形态的纯数据（对齐 ADR 0048）：本 crate 不读 toml、不决定配置来源，只收这份
//! 注入进来的数据。默认值面向「本机无配置文件」的自检场景。
//!
//! 只收运行时旋钮；`POINT_THREADS` / `REDUCE_CHUNK` / `REDUCE_FUNC` 这类**烘焙进生成内核**的
//! codegen 常量不进 `RuntimeConfig`——它们不是部署策略，是内核身份的一部分。

use cutile::prof::Emit;
use std::sync::OnceLock;
use std::time::Duration;

/// 一门语言的语法门：解析器程序 + 固定语法检查参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageFront {
    /// 这门语言。
    pub emit: Emit,
    /// 解析器：程序名 + 固定前缀参数（`clang` 指本机的 clang 前端；其余按档案写法原样探测）。
    pub tool: Vec<String>,
    /// 只做语法检查的参数；源文件路径由执行器追加在末尾。
    pub check: Vec<String>,
}

/// 编译/执行层的可配置常量。字段一一对应旧散落常量。
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// 各语言语法门。加一门语言 = 加一行，不动代码。
    pub gates: Vec<LanguageFront>,
    /// 本机自检用的 host C 编译参数。
    pub host_flags: Vec<String>,
    /// L3 达成率门槛。
    pub l3_threshold: f64,
    /// 缓存锁等待上限。
    pub lock_wait: Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            gates: vec![
                LanguageFront {
                    emit: Emit::Cuda,
                    tool: sv(&["clang"]),
                    check: sv(&[
                        "-x",
                        "cuda",
                        "--cuda-host-only",
                        "-nocudainc",
                        "-nocudalib",
                        "-fsyntax-only",
                    ]),
                },
                LanguageFront {
                    emit: Emit::Opencl,
                    tool: sv(&["clang"]),
                    check: sv(&["-x", "cl", "-fsyntax-only"]),
                },
                LanguageFront {
                    emit: Emit::C,
                    tool: sv(&["clang"]),
                    check: sv(&["-x", "c", "-fsyntax-only"]),
                },
                LanguageFront {
                    emit: Emit::Metal,
                    tool: sv(&["xcrun", "-sdk", "macosx", "metal"]),
                    check: sv(&["-fsyntax-only"]),
                },
                LanguageFront {
                    emit: Emit::Ascendc,
                    tool: sv(&["ccec"]),
                    check: sv(&["-xcce", "--cce-aicore-algorithm=dav-c220", "-fsyntax-only"]),
                },
            ],
            host_flags: sv(&["-O2", "-std=c99"]),
            l3_threshold: 0.7,
            lock_wait: Duration::from_secs(5),
        }
    }
}

fn sv(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_string()).collect()
}

/// 进程级单例：与 `toolchain::C_TOOLCHAIN` 同构——配置是启动期注入一次的部署常量。
static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

/// 取当前配置；未注入时用内置默认。
#[must_use]
pub fn config() -> &'static RuntimeConfig {
    CONFIG.get_or_init(RuntimeConfig::default)
}

/// 注入配置（`cli` 启动时调用，只能成功一次）；已注入时返回 `Err` 并保留旧值。
pub fn install(value: RuntimeConfig) -> Result<(), RuntimeConfig> {
    CONFIG.set(value)
}
