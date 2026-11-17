//! 分布式分片：集合通信 + 三维并行拓扑 + 张量并行切分。
//!
//! 本 crate 收口三类正交事实：
//!
//! - 通信：[`Group`]（升序进程组）、[`Reduce`]（归约闭集与它的算术）、[`Collective`]（后端契约）、
//!   进程内实现 [`Local`] 与未挂载的远端占位 [`Remote`]；
//! - 拓扑：[`Topology`]（TP/PP/DP 世界与进程组）、[`Parallel`]（并行维度）、[`Partition`]（恰好覆盖的切分）；
//! - 切分：[`Sharding`]（按列 / 按行）与 [`Tiling`]（一层线性权重的 TP 形状与合并方式）。
//!
//! rank 的唯一布局规则：`rank = dp_index * (tp * pp) + pp_index * tp + tp_index`。
//! 正向（[`Topology::rank`]）、反向（[`Topology::indices`]）与组内成员（[`Topology::group`]）
//! 都从这条规则派生，因此"某一维的第几组"只有一套算术。
//!
//! # 例子
//!
//! ```
//! use llm_shard::{Parallel, Topology};
//!
//! let topology = Topology::new(2, 2, 2).expect("合法");
//! assert_eq!(topology.world(), 8);
//! let tensor = topology.group(Parallel::Tensor, 0).expect("第一个张量并行组");
//! assert_eq!(tensor.ranks(), [0, 1]);
//! let data = topology.group(Parallel::Data, 0).expect("第一个数据并行组");
//! assert_eq!(data.ranks(), [0, 4]);
//! assert_eq!(topology.rank(1, 1, 1).expect("rank"), 7);
//! ```
//!
//! 归约的数值语义可以用进程内实现直接验证，不需要真的起进程：
//!
//! ```
//! use llm_shard::{Collective, Local, Parallel, Reduce, Topology};
//!
//! let topology = Topology::new(2, 1, 1).expect("合法");
//! let mut local = Local::new(topology, 2).expect("宽度非零");
//! local.load(0, &[1.0, 2.0]).expect("rank 0");
//! local.load(1, &[3.0, 4.0]).expect("rank 1");
//! let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
//! let mut buffer = [0.0; 2];
//! local.all_reduce(&group, Reduce::Sum, &mut buffer).expect("归约");
//! assert_eq!(buffer, [4.0, 6.0]);
//! assert_eq!(local.replica(1).expect("rank 1"), [4.0, 6.0]);
//! ```

#![deny(unsafe_code)]

mod collective;
mod group;
mod local;
mod partition;
mod reduce;
mod remote;
mod tiling;
mod topology;

pub use collective::Collective;
pub use group::Group;
pub use local::Local;
pub use partition::Partition;
pub use reduce::Reduce;
pub use remote::Remote;
pub use tiling::{Sharding, Tiling};
pub use topology::{Parallel, Topology};
