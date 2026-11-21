//! 进程内集合通信：[`Local`]。
//!
//! [`Collective`] 是后端契约，但它今天一个实现都没有——于是"TP 的归约到底做对没有"只能靠读代码。
//! [`Local`] 把"每个 rank 各有一份缓冲"放进同一个进程：先 [`Local::load`] 每个 rank 的输入，
//! 再 [`Collective::all_reduce`] 拿到组内归约结果。数值语义与真实后端一致（先归约、再写回
//! 每个成员），差别只在没有网络——所以它可以由 `cargo test` 直接验证。
//!
//! 归约顺序固定按 [`Group`] 的 rank 升序（`Group` 构造期已保证升序）：浮点加法不满足结合律，
//! 顺序一变，最后几位就随启动顺序变化，对拍会变成"有时通过"。
//!
//! 一次 `all_reduce` 之后，组内每个 rank 的镜像与调用方缓冲都一样——这正是 all-reduce 与
//! reduce-scatter 的分界；只写调用方那一份会让"第二个 rank 再读一次"读到未被归约的旧值。

use crate::collective::{Collective, Group, Reduce};
use crate::topology::Topology;
use diagnostic::{Code, Error, Result};

/// 进程内的集合通信：每个 rank 一份镜像，归约在本进程内完成。
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    topology: Topology,
    width: usize,
    replicas: Vec<Vec<f32>>,
}

impl Local {
    /// 为一个拓扑建好每个 rank 的零值镜像；`width` 是每个 rank 参与归约的元素个数。
    ///
    /// # Errors
    ///
    /// `width` 为零（那样归约没有可读写的元素）。
    pub fn new(topology: Topology, width: usize) -> Result<Self> {
        if width == 0 {
            return Err(Error::new(Code::Invalid, "归约宽度不能为 0").with("shard::Local::new"));
        }
        Ok(Self {
            topology,
            width,
            replicas: vec![vec![0.0; width]; topology.world()],
        })
    }

    /// 拓扑。
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// 每个 rank 的元素个数。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 写入某个 rank 的输入；长度必须与 [`Local::width`] 一致。
    pub fn load(&mut self, rank: u32, values: &[f32]) -> Result<()> {
        if values.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "rank {rank} 的输入宽度 {} 与归约宽度 {} 不一致",
                    values.len(),
                    self.width
                ),
            )
            .with("shard::Local::load"));
        }
        self.replica_mut(rank)?.copy_from_slice(values);
        Ok(())
    }

    /// 某个 rank 当前的镜像（`load` 或 `all_reduce` 之后的值）。
    pub fn replica(&self, rank: u32) -> Result<&[f32]> {
        let index = self.index_of(rank)?;
        Ok(&self.replicas[index])
    }

    /// 组内归约结果：按 [`Group`] 的顺序逐 rank 折叠，不改变任何镜像。
    pub fn reduce(&self, group: &Group, reduce: Reduce) -> Result<Vec<f32>> {
        let mut total = vec![reduce.identity(); self.width];
        for rank in group.ranks() {
            let replica = self.replica(*rank)?;
            for (slot, value) in total.iter_mut().zip(replica) {
                *slot = reduce.apply(*slot, *value);
            }
        }
        Ok(total)
    }

    /// 把归约结果写回组内每个 rank 的镜像。
    fn distribute(&mut self, group: &Group, reduced: &[f32]) -> Result<()> {
        for rank in group.ranks() {
            self.replica_mut(*rank)?.copy_from_slice(reduced);
        }
        Ok(())
    }

    fn replica_mut(&mut self, rank: u32) -> Result<&mut [f32]> {
        let index = self.index_of(rank)?;
        Ok(&mut self.replicas[index])
    }

    /// rank 必须落在本拓扑内：越界写在别的 rank 上会静默改掉无关的镜像。
    fn index_of(&self, rank: u32) -> Result<usize> {
        let index = rank as usize;
        if index >= self.replicas.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("rank {rank} 超出 0..{}", self.replicas.len()),
            )
            .with("shard::Local::replica"));
        }
        Ok(index)
    }
}

impl Collective for Local {
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()> {
        if buffer.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲宽度 {} 与归约宽度 {} 不一致", buffer.len(), self.width),
            )
            .with("shard::Local::all_reduce"));
        }
        let reduced = self.reduce(group, reduce)?;
        buffer.copy_from_slice(&reduced);
        self.distribute(group, &reduced)
    }
}

#[cfg(test)]
#[path = "../tests/local_tests.rs"]
mod tests;
