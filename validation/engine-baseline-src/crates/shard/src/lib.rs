//! 分布式分片与设备执行：并行编排、集合通信、权重切分和后端契约。
//!
//! - [`Topology`] / [`Plan`] / [`ParallelConfig`]：同一套 TP/PP/DP 拓扑与部署策略；
//! - [`Collective`] / [`Local`] / [`FileRemote`]：共享泛型算法的集合通信端口与实现；
//! - [`Sharding`] / [`Tiling`] / [`Placement`]：整除切分、缓冲复用与模型层归属；
//! - [`Queue`] / [`Job`]：三档稳定 FIFO 优先级调度；
//! - [`backend`]：设备后端契约、配置表驱动实现与编译执行回读。
//!
//! rank 的唯一布局规则：`rank = dp_idx * (tp * pp) + pp_idx * tp + tp_idx`。
//! 正向（[`Topology::rank`]）、反向（[`Topology::idxs`]）与组内成员（[`Topology::group`]）
//! 都从这条规则派生，因此"某一维的第几组"只有一套算术。
//!
//! # 例子
//!
//! 一个 rank 的两件事在这里一起走完：我在三维世界的哪里（[`Topology`]）、我手里是哪一块权重
//! （[`Tiling`]）、以及归约一次的数值语义（[`Local`]）。归约在进程内就能验证，不需要真起进程。
//!
//! ```
//! use llm_shard::{Collective, Local, Parallel, Reduce, Sharding, Tiling, Topology};
//!
//! let topology = Topology::new(2, 2, 2).expect("合法");
//! assert_eq!(topology.world(), 8);
//! assert_eq!(topology.rank(1, 1, 1).expect("rank"), 7);
//! let tensor = topology.group(Parallel::Tensor, 0).expect("第一个张量并行组");
//! assert_eq!(tensor.ranks(), [0, 1]);
//!
//! // 同一层的权重 `[8, 6]` 按列切给这两个 rank：各得 `[4, 6]`，拼回去正好是原样。
//! let tiling = Tiling::new(Sharding::Column, 2, 8, 6).expect("8 能被 2 整除");
//! assert_eq!(tiling.weight_shape(1).expect("第 1 份"), [4, 6]);
//! assert_eq!(tiling.sharding().merge(), None, "按列切不需要通信");
//!
//! // 两个 rank 各持一份部分和，归约后两边都拿到完整结果。
//! let mut local = Local::new(topology, 2).expect("宽度非零");
//! local.load(0, &[1.0, 2.0]).expect("rank 0");
//! local.load(1, &[3.0, 4.0]).expect("rank 1");
//! let mut buf = [0.0; 2];
//! local.all_reduce(&tensor, Reduce::Sum, &mut buf).expect("归约");
//! assert_eq!(buf, [4.0, 6.0]);
//! assert_eq!(local.replica(1).expect("rank 1"), [4.0, 6.0]);
//! ```

#![deny(unsafe_code)]

pub(crate) use diagnostic::Code;

use diagnostic::Error;

/// 构造带错误码与 `shard::<类型>::<方法>` 站点的诊断。
pub(crate) fn refuse(code: Code, site: &'static str, message: std::fmt::Arguments<'_>) -> Error {
    Error::new(code, message.to_string()).with(site)
}

/// 构造指定错误码与站点的诊断值，供 `map_err` / `ok_or_else` 使用。
macro_rules! reject {
    ($code:ident, $site:literal, $($arg:tt)+) => {
        $crate::refuse($crate::Code::$code, $site, format_args!($($arg)+))
    };
}

/// 供 `map_err` / `ok_or_else` 使用的 `Code::Invalid` 诊断。
macro_rules! invalid {
    ($site:literal, $($arg:tt)+) => {
        $crate::refuse($crate::Code::Invalid, $site, format_args!($($arg)+))
    };
}

/// 按指定错误码与站点立即返回诊断；缺省为 `Code::Invalid`。
macro_rules! fail {
    ($code:ident, $site:literal, $($arg:tt)+) => {
        return Err($crate::refuse($crate::Code::$code, $site, format_args!($($arg)+)))
    };
    ($site:literal, $($arg:tt)+) => {
        return Err(invalid!($site, $($arg)+))
    };
}

/// [`fail!`] 的值形态，站点由调用方表达式给出。
macro_rules! refuse_at {
    ($site:expr, $($arg:tt)+) => {
        Err(crate::refuse(
            crate::Code::Invalid,
            $site,
            format_args!($($arg)+),
        ))
    };
}

/// 只读访问器表：`|s| s.字段…` 给出取值，宏生成带文档的 `const` 方法。
macro_rules! accessors {
    ($($(#[$meta:meta])* $name:ident -> $ty:ty = |$s:ident| $value:expr;)+) => {
        $(
            $(#[$meta])*
            #[must_use]
            pub const fn $name(&self) -> $ty {
                let $s = self;
                $value
            }
        )+
    };
}

// ---- 运行时配置：纯数据、校验与一次性安装 ----

use cutile::prof::Emit;
use diagnostic::Result;
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
    /// 是否显式启用逐候选实测择优；默认 `false`，启发式始终确定性执行。
    pub autotune: bool,
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
            autotune: false,
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
mod fixtures {
    pub fn catalogue() -> cutile::builtin::Catalogue {
        cutile::builtin::Catalogue::try_builtin().expect("内置档案必须合法")
    }

    pub fn profile(vendor: &str) -> cutile::prof::Profile {
        catalogue()
            .profile(vendor)
            .cloned()
            .unwrap_or_else(|| panic!("缺少 `{vendor}` 内置档案"))
    }
}

mod collective;
mod placement;
mod topology;

pub use collective::{Collective, FileRemote, Group, Local, Reduce, ReduceScalar, Remote};
pub use placement::{Assembler, LinearGeo, Placement, Sharding, Tiling};
pub use topology::{Job, Parallel, ParallelConfig, Plan, Priority, Queue, Topology};

/// 运行时配置路径；实现平铺在 crate 根部。
pub mod config {
    pub use crate::{LanguageFront, RuntimeConfig, config, install};
}

// ---- 优化面（执行半边自旧 crate 并入） ----------------------------------------
//
// `device` / `cli` 以 `shard::` 名义消费这些路径：后端契约、工具链、执行驱动。
pub mod backend;
pub mod bench;
pub mod build;
pub mod device;
pub mod doctor;
/// 执行驱动与批量任务编排的公开入口。
pub mod execution {
    pub use crate::pipeline::{Execution, execute};
}
/// 设备语言门路径；实现由 [`pipeline`] 统一提供。
pub mod gate {
    pub use crate::pipeline::{LanguageGate, language_gate};
}
/// 后端测量台账及状态报告的公开入口。
pub mod ledger {
    pub use crate::doctor::{BackendRow, LedgerCell, Status, ledger};
}
pub mod pipeline;
pub mod platform;
pub mod runner;
/// 工具链探测路径；实现由 [`runner`] 统一提供。
pub mod toolchain {
    pub(crate) use crate::runner::probe_with;
    #[cfg(test)]
    pub(crate) use crate::runner::skip_without_host_compiler;
    pub use crate::runner::{
        Toolchain, clang, detect, for_profile, parse_version, probe, probe_declared,
    };
}
pub mod verify;
pub mod witness;

/// 历史测量路径；实现已统一收进 [`verify`]。
pub mod sweep {
    pub use crate::verify::verify_one;
}

/// 设备后端安装与 Metal 执行接口，统一由 [`backend`] 提供。
pub mod backends {
    pub use crate::backend::{compile_run_metal, install, metal_driver, metal_kernel};
}
/// Metal 双源编译执行接口。
pub mod metal {
    pub use crate::backend::{compile_run_metal, metal_driver, metal_kernel};
}

pub use runner::{compile_flags, driver};
