//! 服务侧状态机：[`Sequence`]、[`State`]、[`Scheduler`] 与连续批 [`Batch`]。
//!
//! **拥有**：`Sequence`、`Scheduler`、`Batch`、`Admission`、`State`，以及页账（[`kv`]）、
//! 提交与执行（[`exec`]）、设备后端 FFI（[`metal`]，macOS 越界岛，Metal `Executor` 落点，不单独成 crate）。
//! **不拥有**：采样策略（它只决定 `advance` 取哪个 token，不改变状态机）、
//! kernel 编译（`cutile`）、缓冲分配（`storage`）。**依赖**：`tensor`、`diagnostic`、`storage`。
//! **被依赖**：`serve`、`llm`。
//!
//! 页账与提交契约原是两个独立成员（`llm-kv` / `llm-exec`）：它们的生产消费者只有本 crate，
//! 合并后"谁在借页"与"谁在提交"的不变量在同一处断言，成员与依赖边各少一个。
//!
//! 连续批的关键不是"批"，而是**准入**：每个序列占多少页是显式算出来的，页不够就拒绝，
//! 而不是让 KV 缓冲在跑到一半时溢出。
//!
//! # 页账跟着序列一起长
//!
//! 准入只预留**提示长度**的页：按 `max_new` 全量预留会把池子提前锁死，而请求可能很早就结束。
//! 之后每生成一个 token，由 [`Scheduler::advance`] 预留它要落的那一页；池不够时它返回
//! [`Progress::Blocked`]，序列与页表都不动——调用方可以等一个序列 [`Scheduler::finish`] 之后
//! 原样重试，而不是让 KV 缓冲在生成到一半时越界。
//!
//! `Sequence::step` 因此不是公开 API：它只推进计数、不碰页，唯一的调用者是 [`Scheduler::advance`]。
//! "生成必须经过页账"因此是编译期就能保证的，而不是一句约定。
//!
//! # KV 数据只经页表进入
//!
//! 页里的数在 [`kv::Pager`] 里，而拿到一行只经 [`Scheduler::slot`]：先 `admit`、再按位置问页。
//! 于是"往一个还没有页的位置写 KV"写不出来（那个位置根本不在页表里），而"往已经归还的页写"
//! 会在 [`kv::Pager`] 的存活校验上失败——失效的 [`kv::Slot`] 不会被当成合法的写入地址。
//! [`Scheduler::write_kv`] 与 [`Scheduler::kv`] 只是把这条路径说完整：模型不必自己拿页表再算
//! 一次偏移，而页表与数据的口径只有一个。
//!
//! # 活性：准入保证"各自跑得完"，批内仍需调用方腾页
//!
//! 准入还会拒绝 [`Sequence::pages_required`] 超过整个页池的请求——那种序列独占页池也跑不完，
//! 放进来只会让整批卡在 [`Progress::Blocked`] 上而无人能结束。但"每条都跑得完"不等于"同时放
//! 多少都行"：几条都跑得完的序列挤在一起仍可能互相占满页而全部 [`Progress::Blocked`]。所以
//! `Blocked` 不是错误，而是调用方的动作信号——结束最接近完成的那一条（它的页会回来），再原样
//! 重试同一步。这一层不替调用方选牺牲者：选谁结束是调度策略，不属于状态机。但"现在谁在等"
//! 必须由这一层回答，于是有 [`Scheduler::blocked`]：它与单步结局同源，调用方据此挑牺牲者，
//! 不必自己重算页数。
//!
//! # 例子
//!
//! ```
//! use llm_engine::{Progress, Scheduler, Sequence};
//! use llm_engine::exec::Stream;
//! use llm_engine::kv::{Cache, Pager};
//!
//! let mut scheduler = Scheduler::new(Pager::new(Cache::new(4, 16).expect("页池"), 4).expect("行宽"));
//! let mut sequence = Sequence::new(0, &[0; 8], 4, 16, Stream::new(0)).expect("序列");
//! assert_eq!(
//!     scheduler.admit(&sequence).expect("准入"),
//!     Some(0),
//!     "页池空着，没有可接走的前缀：提示段一个位置都没算过",
//! );
//! assert_eq!(
//!     scheduler.advance(&mut sequence).expect("生成一个 token"),
//!     Progress::Running,
//! );
//! assert_eq!(sequence.produced(), 1);
//! assert_eq!(scheduler.cache().in_use(), 1, "长度 9 仍在第一页里");
//! assert!(scheduler.finish(sequence.id()).expect("释放页"));
//! ```

#![deny(unsafe_code)]

mod ascend;
pub mod backends;
pub mod batch;
pub mod cuda;
mod dialect;
pub mod exec;
pub mod ifb;
pub mod kv;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod metal_host;
pub mod metal_pipeline;
pub mod msl;
pub mod norm;
pub mod platform;
pub mod program;
pub mod radix;
pub mod scan;
pub mod scheduler;
pub mod sequence;
pub mod tiled;
mod toolchain;
pub mod wmma;

pub use batch::{Admission, Batch, SequenceOutcome};
pub use ifb::{Budget, Candidate, Phase, Verdict, chunk, pack};
pub use platform::{Abi, Arch, Host, Machine, Row, Standing, Why, audit, require, standing};
pub use scheduler::Scheduler;
pub use sequence::{Progress, Sequence, State};

#[cfg(test)]
mod tests;
