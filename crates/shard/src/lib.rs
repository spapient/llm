//! 分布式分片：集合通信契约 + 三维并行拓扑。
//!
//! 本 crate 收口两类正交事实：
//!
//! - 通信：[`Group`]（升序进程组）、[`Reduce`]（归约闭集）、[`Collective`]（后端契约）；
//! - 拓扑：[`Topology`]（TP/PP/DP 世界与进程组）、[`Parallel`]（并行维度）、[`Partition`]（均匀切分）。
//!
//! rank 的唯一布局规则：`rank = dp_index * (tp * pp) + pp_index * tp + tp_index`。
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
//! ```

#![deny(unsafe_code)]

mod collective;
mod group;
mod partition;
mod reduce;
mod topology;

pub use collective::Collective;
pub use group::Group;
pub use partition::Partition;
pub use reduce::Reduce;
pub use topology::{Parallel, Topology};
