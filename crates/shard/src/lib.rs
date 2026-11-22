//! 分布式分片：集合通信 + 三维并行拓扑 + 张量并行切分。
//!
//! 本 crate 收口三类正交事实：
//!
//! - 通信：[`Collective`]（后端端口）、进程内实现 [`Local`]、未挂载的远端占位 [`Remote`]，
//!   以及端口签名里的两个值类型 [`Group`]（升序进程组）与 [`Reduce`]（归约闭集与它的算术）；
//! - 拓扑：[`Topology`]（TP/PP/DP 世界与进程组）、[`Parallel`]（并行维度）；
//! - 切分：[`Sharding`]（按列 / 按行）与 [`Tiling`]（一层线性权重的 TP 形状与合并方式）。
//!
//! # 四个文件的分界是"互相定义的量"，不是行数
//!
//! [`Group`] 与 [`Reduce`] 是 [`Collective`] 签名里的两个参数类型，而 [`Local`] 与 [`Remote`] 各自
//! 只是同一端口的一段 `impl`：改一次签名，这五处必须一起读，所以它们在同一个文件里。拓扑与切分各自
//! 有独立的算术（rank 布局的正解与反解、整除判据），改动不会同时波及，才分开。
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

/// 判据拒绝的唯一入口：全 crate 只在这里绑定错误码与站点标签。
///
/// 各入口只负责说清"为什么"——`site` 是这个 crate 的路径式标签（`shard::<类型>::<方法>`），
/// 让诊断能指回是哪一处拒绝的。
pub(crate) fn refuse(code: Code, site: &'static str, message: std::fmt::Arguments<'_>) -> Error {
    Error::new(code, message.to_string()).with(site)
}

/// 判据拒绝的**值**形态：`invalid!("…{x}…")` 即 `refuse(Code::Invalid, SITE, format_args!(…))`。
///
/// 只在要交给 `map_err` / `ok_or_else` 的地方用；能直接返回的地方写 [`fail!`]。
macro_rules! invalid {
    ($site:literal, $($arg:tt)+) => {
        $crate::refuse($crate::Code::Invalid, $site, format_args!($($arg)+))
    };
}

/// 判据失败并立即返回：`fail!("site", "…{x}…")`，第二形态可换错误码（占位后端用 `Unsupported`）。
///
/// 三层壳（`Error::new` + `format!` + `.with`）在每个判据处都要重写一遍，按 100 列宽它必然被
/// `rustfmt` 竖排成五行以上，而判据本身只有一句中文——宏收掉壳，一句判据才留得下一行。
macro_rules! fail {
    ($code:ident, $site:literal, $($arg:tt)+) => {
        return Err($crate::refuse($crate::Code::$code, $site, format_args!($($arg)+)))
    };
    ($site:literal, $($arg:tt)+) => {
        return Err(invalid!($site, $($arg)+))
    };
}

mod collective;
mod tiling;
mod topology;

pub use collective::{Collective, Group, Local, Reduce, Remote};
pub use tiling::{Sharding, Tiling};
pub use topology::{Parallel, Topology};

pub mod config;
pub use config::{Config, Gate};

// ---- 优化面（原 `optim` crate 并入） ----------------------------------------
//
// `device` / `cli` 以 `llm_optim::` 名义消费这些路径：后端契约、工具链、执行驱动。
pub mod backend;
pub mod bench;
pub mod build;
pub mod device;
pub mod doctor;
pub mod execution;
pub mod gate;
pub mod ir;
pub mod ledger;
pub mod pipeline;
pub mod runner;
pub mod toolchain;
pub mod verify;
pub mod witness;
pub use runner::{compile_flags, driver};
