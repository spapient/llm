//! 集合通信：[`Collective`] 端口、它的两个实现，以及端口签名里的两个值类型。
//!
//! 五件东西放在一个文件里是因为它们互相定义：改一次 `all_reduce` 的签名，[`Group`] 与
//! [`Reduce`] 的语义、[`Local`] 与 [`Remote`] 的 `impl` 必须一起读。分成四个文件时，读者要
//! 在四个 `mod` 与四条 `pub use` 之间拼回这层关系，而它们各自的算术都不足以独立成篇。

use crate::topology::Topology;
use diagnostic::{Code, Result};

/// 通信后端要实现的东西：对组内缓冲做一次集合操作。
pub trait Collective {
    /// 对 `buf` 做原地归约；返回时表示"已完成"。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()>;
}

/// 参与一次通信的进程组。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    ranks: Vec<u32>,
}

impl Group {
    /// 校验并构造。
    ///
    /// # Errors
    ///
    /// 组为空，或 rank 不是严格升序（重复或乱序）。
    pub fn new(ranks: impl Into<Vec<u32>>) -> Result<Self> {
        let ranks = ranks.into();
        if ranks.is_empty() {
            fail!("shard::Group::new", "通信组不能为空");
        }
        for pair in ranks.windows(2) {
            if pair[0] >= pair[1] {
                fail!(
                    "shard::Group::new",
                    "通信组必须升序且不重复：{} >= {}",
                    pair[0],
                    pair[1]
                );
            }
        }
        Ok(Self { ranks })
    }

    /// 组内全部 rank。
    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    /// 组大小。
    pub fn size(&self) -> usize {
        self.ranks.len()
    }
}

/// 归约种类闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reduce {
    /// 求和（梯度同步的默认选择）。
    Sum,
    /// 取最大（张量并行里的数值稳定选项）。
    Max,
    /// 取最小。
    Min,
    /// 取乘积。
    Prod,
}

/// 一种归约的全部事实：短名、单位元、二元运算。
type Row = (&'static str, f32, fn(f32, f32) -> f32);

impl Reduce {
    /// 全部种类，供日志与测试遍历。
    pub const ALL: [Self; 4] = [Self::Sum, Self::Max, Self::Min, Self::Prod];

    /// 一种归约的三个事实放在同一行：短名、单位元、二元运算。
    ///
    /// 三件事必须同时成立，所以写在一张表里。旧形态是三个各自四臂的 `match`：加一种归约要
    /// 改三处，而"忘了 `Prod` 的单位元是 `1.0`"只会在某些输入上悄悄算错。行序必须与判别式
    /// 一致（下面三个方法都走 `TABLE[self as usize]`），这条由 `tests` 里的
    /// `the_kind_table_follows_the_enum_order` 钉住。
    ///
    /// 极值的单位元是 `NEG_INFINITY` / `INFINITY` 而不是 `MIN` / `MAX`：后者会让"取最大"在
    /// 全是负值的梯度上返回 `MIN`。极值本身走 [`nan_preserving`] 而不是 `f32::max`。
    const TABLE: [Row; 4] = [
        ("sum", 0.0, |left, right| left + right),
        ("max", f32::NEG_INFINITY, |left, right| {
            nan_preserving(left, right, f32::max)
        }),
        ("min", f32::INFINITY, |left, right| {
            nan_preserving(left, right, f32::min)
        }),
        ("prod", 1.0, |left, right| left * right),
    ];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        Self::TABLE[self as usize].0
    }

    /// 单位元：与任何值 `apply` 之后都不改变它。
    pub fn identity(self) -> f32 {
        Self::TABLE[self as usize].1
    }

    /// 二元运算：满足结合律，且对 [`Reduce::identity`] 中立。
    pub fn apply(self, left: f32, right: f32) -> f32 {
        (Self::TABLE[self as usize].2)(left, right)
    }

    /// 把一个非空切片折成一个值；空切片返回 `None`。
    pub fn fold(self, values: &[f32]) -> Option<f32> {
        let (first, rest) = values.split_first()?;
        Some(
            rest.iter()
                .fold(*first, |acc, value| self.apply(acc, *value)),
        )
    }
}

/// 极值归约：任一侧是 NaN 就传播 NaN，否则交给 `pick`。
///
/// `f32::max` / `f32::min` 会**静默丢掉** NaN（它们返回另一侧），这与"归约"的语义相反：
/// 一项 NaN 混进梯度里，取最大值后应当还是 NaN，而不是悄悄变成一个有限的数。
fn nan_preserving(left: f32, right: f32, pick: fn(f32, f32) -> f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        return f32::NAN;
    }
    pick(left, right)
}

/// 远端集合通信占位后端：与 [`Local`] 同一 [`Collective`] 端口，未挂载真实通信栈。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Remote;

impl Collective for Remote {
    fn all_reduce(&mut self, _group: &Group, _reduce: Reduce, _buf: &mut [f32]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Remote::all_reduce",
            "远端集合通信未挂载：跨进程 all-reduce 在占位适配上不可用"
        );
    }
}

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
            fail!("shard::Local::new", "归约宽度不能为 0");
        }
        Ok(Self {
            topology,
            width,
            replicas: vec![vec![0.0; width]; topology.world()],
        })
    }

    /// 写入某个 rank 的输入；长度必须等于构造时给的归约宽度。
    ///
    /// # Errors
    ///
    /// 长度不符，或 `rank` 不在本拓扑内。
    pub fn load(&mut self, rank: u32, values: &[f32]) -> Result<()> {
        self.check_width("shard::Local::load", values.len())?;
        self.replica_mut("shard::Local::load", rank)?
            .copy_from_slice(values);
        Ok(())
    }

    /// 某个 rank 当前的镜像（`load` 或 `all_reduce` 之后的值）。
    ///
    /// # Errors
    ///
    /// `rank` 不在本拓扑内。
    pub fn replica(&self, rank: u32) -> Result<&[f32]> {
        Ok(&self.replicas[self.slot("shard::Local::replica", rank)?])
    }

    /// 组内归约结果：按 [`Group`] 的顺序逐 rank 折叠，不改变任何镜像。
    ///
    /// # Errors
    ///
    /// 组里出现本拓扑之外的 rank（`Group` 本身不绑定拓扑）。
    pub fn reduce(&self, group: &Group, reduce: Reduce) -> Result<Vec<f32>> {
        let mut total = vec![reduce.identity(); self.width];
        for rank in group.ranks() {
            for (slot, value) in total.iter_mut().zip(self.replica(*rank)?) {
                *slot = reduce.apply(*slot, *value);
            }
        }
        Ok(total)
    }

    /// 把归约结果写回组内每个 rank 的镜像。
    fn distribute(&mut self, group: &Group, reduced: &[f32]) -> Result<()> {
        for rank in group.ranks() {
            self.replica_mut("shard::Local::distribute", *rank)?
                .copy_from_slice(reduced);
        }
        Ok(())
    }

    fn replica_mut(&mut self, site: &'static str, rank: u32) -> Result<&mut [f32]> {
        let idx = self.slot(site, rank)?;
        Ok(&mut self.replicas[idx])
    }

    /// 缓冲必须正好是归约宽度：短了读不全，长了富余元素无人归约。
    ///
    /// `site` 与 [`Self::slot`] 同理——两个入口共用一条判据，诊断要指回真正拒绝的那个。
    fn check_width(&self, site: &'static str, len: usize) -> Result<()> {
        if len == self.width {
            return Ok(());
        }
        Err(crate::refuse(
            Code::Invalid,
            site,
            format_args!("宽度 {len} 与归约宽度 {} 不一致", self.width),
        ))
    }

    /// rank 必须落在本拓扑内：越界写在别的 rank 上会静默改掉无关的镜像。
    ///
    /// `site` 由调用方给：同一条越界判据有四个入口，诊断必须指回真正拒绝的那一个。
    fn slot(&self, site: &'static str, rank: u32) -> Result<usize> {
        let idx = rank as usize;
        if idx < self.replicas.len() {
            return Ok(idx);
        }
        Err(crate::refuse(
            Code::Invalid,
            site,
            format_args!("rank {rank} 超出 0..{}", self.replicas.len()),
        ))
    }
}

impl Collective for Local {
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()> {
        self.check_width("shard::Local::all_reduce", buf.len())?;
        let reduced = self.reduce(group, reduce)?;
        buf.copy_from_slice(&reduced);
        self.distribute(group, &reduced)
    }
}

#[cfg(test)]
#[path = "../tests/collective_tests.rs"]
mod tests;
