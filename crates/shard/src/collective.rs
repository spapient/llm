//! 集合通信：[`Collective`] 端口、它的两个实现，以及端口签名里的两个值类型。
//!
//! 五件东西放在一个文件里是因为它们互相定义：改一次 `all_reduce` 的签名，[`Group`] 与
//! [`Reduce`] 的语义、[`Local`] 与 [`Remote`] 的 `impl` 必须一起读。分成四个文件时，读者要
//! 在四个 `mod` 与四条 `pub use` 之间拼回这层关系，而它们各自的算术都不足以独立成篇。
//!
//! 元素档由 [`ReduceScalar`] 约束参数化：端口默认实例化 [`f32`]（训练梯度档），后端可换成
//! 其他满足标量算术的类型。跨进程序列化的 [`crate::FileRemote`] 字节协议固定 f32，仍实现
//! [`Collective<f32>`]。

use core::ops::{Add, Mul};

use crate::topology::Topology;
use diagnostic::Result;

/// 可参与一次归约的标量档：加法、乘法、偏序与四个归约单位元都由类型自己给出。
///
/// 这是 shard 内自包含的约束，不反向依赖 tensor 的元素标记层级——集合通信只需要算术中立元，
/// 不需要张量 dtype 分类。浮点档覆写 [`max`](Self::max) / [`min`](Self::min) 保留 NaN 传播；
/// 整数档走偏序默认实现，没有 NaN 概念。
pub trait ReduceScalar:
    Copy + Default + 'static + Send + Sync + Add<Output = Self> + Mul<Output = Self> + PartialOrd
{
    /// 求和单位元（与任何值相加不变）。
    const ZERO: Self;
    /// 乘积单位元（与任何值相乘不变）。
    const ONE: Self;
    /// 取最大值的单位元（小于任何有限值）。
    const MIN_LIMIT: Self;
    /// 取最小值的单位元（大于任何有限值）。
    const MAX_LIMIT: Self;

    /// 取大；整数档按偏序，浮点档由实现覆写以传播 NaN。
    fn max(left: Self, right: Self) -> Self {
        if left >= right { left } else { right }
    }

    /// 取小；整数档按偏序，浮点档由实现覆写以传播 NaN。
    fn min(left: Self, right: Self) -> Self {
        if left <= right { left } else { right }
    }
}

impl ReduceScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_LIMIT: Self = f32::NEG_INFINITY;
    const MAX_LIMIT: Self = f32::INFINITY;

    fn max(left: Self, right: Self) -> Self {
        if left.is_nan() || right.is_nan() {
            f32::NAN
        } else {
            f32::max(left, right)
        }
    }

    fn min(left: Self, right: Self) -> Self {
        if left.is_nan() || right.is_nan() {
            f32::NAN
        } else {
            f32::min(left, right)
        }
    }
}

impl ReduceScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_LIMIT: Self = f64::NEG_INFINITY;
    const MAX_LIMIT: Self = f64::INFINITY;

    fn max(left: Self, right: Self) -> Self {
        if left.is_nan() || right.is_nan() {
            f64::NAN
        } else {
            f64::max(left, right)
        }
    }

    fn min(left: Self, right: Self) -> Self {
        if left.is_nan() || right.is_nan() {
            f64::NAN
        } else {
            f64::min(left, right)
        }
    }
}

/// 整数标量档批量实现：单位元取类型零/一与类型上下界，极值按偏序，无 NaN。
macro_rules! impl_integer_scalar {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ReduceScalar for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                const MIN_LIMIT: Self = Self::MIN;
                const MAX_LIMIT: Self = Self::MAX;
            }
        )+
    };
}

impl_integer_scalar!(u8, u16, u32, u64, i8, i16, i32, i64, usize, isize);

/// 通信后端要实现的东西：对组内缓冲做一次集合操作。元素档 `T` 默认 [`f32`]。
pub trait Collective<T: ReduceScalar = f32> {
    /// 对 `buf` 做原地归约；返回时表示"已完成"。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [T]) -> Result<()>;

    /// 全部收集：`buf` 长 `group.size() × width`，每个成员把它的 `width` 份按组内顺序写进对应
    /// 分块；操作后每个成员都拿到组内全部贡献的拼接。缺实现返回 `Unsupported`。
    fn all_gather(&mut self, _group: &Group, _buf: &mut [T]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Collective::all_gather",
            "该集合通信后端没有 all_gather"
        );
    }

    /// 广播：`src` 号成员的值广播给组内全体。缺实现返回 `Unsupported`。
    fn broadcast(&mut self, _group: &Group, _src: u32, _buf: &mut [T]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Collective::broadcast",
            "该集合通信后端没有 broadcast"
        );
    }

    /// 归约散布：`buf` 长 `group.size() × width`，每个成员把它的 `width` 份按组内顺序用
    /// `reduce` 折进对应分块，散布出的结果是组内各分块的归约拼接。分块互不相交时退化为
    /// [`Self::all_gather`]；rank 贡献重叠（ZeRO 切分优化器态）时才是真正的归约。缺实现返回
    /// `Unsupported`。
    fn reduce_scatter(&mut self, _group: &Group, _reduce: Reduce, _buf: &mut [T]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Collective::reduce_scatter",
            "该集合通信后端没有 reduce_scatter"
        );
    }
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

impl Reduce {
    /// 全部种类，供日志与测试遍历。
    pub const ALL: [Self; 4] = [Self::Sum, Self::Max, Self::Min, Self::Prod];

    /// 一种归约的稳定短名。行序必须与判别式一致，由测试
    /// `the_kind_table_follows_the_enum_order` 钉住。单位元与二元运算按元素档在
    /// [`ReduceScalar`] 上派生，不再按 f32 存闭包表。
    const SHORT: [(&'static str, Self); 4] = [
        ("sum", Self::Sum),
        ("max", Self::Max),
        ("min", Self::Min),
        ("prod", Self::Prod),
    ];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        Self::SHORT[self as usize].0
    }

    /// 单位元：与任何值 `apply` 之后都不改变它。
    pub fn identity<T: ReduceScalar>(self) -> T {
        match self {
            Self::Sum => T::ZERO,
            Self::Prod => T::ONE,
            Self::Max => T::MIN_LIMIT,
            Self::Min => T::MAX_LIMIT,
        }
    }

    /// 二元运算：满足结合律，且对 [`Reduce::identity`] 中立。
    pub fn apply<T: ReduceScalar>(self, left: T, right: T) -> T {
        match self {
            Self::Sum => left + right,
            Self::Prod => left * right,
            Self::Max => T::max(left, right),
            Self::Min => T::min(left, right),
        }
    }

    /// 把一个非空切片折成一个值；空切片返回 `None`。
    pub fn fold<T: ReduceScalar>(self, values: &[T]) -> Option<T> {
        let (first, rest) = values.split_first()?;
        Some(
            rest.iter()
                .fold(*first, |acc, value| self.apply(acc, *value)),
        )
    }
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
pub struct Local<T: ReduceScalar = f32> {
    topology: Topology,
    width: usize,
    replicas: Vec<Vec<T>>,
}

impl<T: ReduceScalar> Local<T> {
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
            replicas: vec![vec![T::ZERO; width]; topology.world()],
        })
    }

    /// 写入某个 rank 的输入；长度必须等于构造时给的归约宽度。
    ///
    /// # Errors
    ///
    /// 长度不符，或 `rank` 不在本拓扑内。
    pub fn load(&mut self, rank: u32, values: &[T]) -> Result<()> {
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
    pub fn replica(&self, rank: u32) -> Result<&[T]> {
        Ok(&self.replicas[self.slot("shard::Local::replica", rank)?])
    }

    /// 组内归约结果：按 [`Group`] 的顺序逐 rank 折叠，不改变任何镜像。
    ///
    /// # Errors
    ///
    /// 组里出现本拓扑之外的 rank（`Group` 本身不绑定拓扑）。
    pub fn reduce(&self, group: &Group, reduce: Reduce) -> Result<Vec<T>> {
        let mut total = vec![T::ZERO; self.width];
        self.reduce_into(group, reduce, &mut total)?;
        Ok(total)
    }

    /// 把组内归约写进调用方给的缓冲；长度必须等于归约宽度。
    ///
    /// [`Collective::all_reduce`] 直接写进它自己的 `buf`，所以一次归约**不分配**堆：
    /// 旧形态的 `all_reduce` 先 `reduce` 出一个 `Vec` 再拷进 `buf`，每次调用多一次分配。
    ///
    /// 内层按元素折叠是同元素跨 rank 的**串行依赖**（后一个值依赖前一个累加结果），故保留
    /// 逐元素循环；跨 rank 之间无可并行的元素独立度。
    ///
    /// # Errors
    ///
    /// 缓冲长度与归约宽度不一致，或组里出现本拓扑之外的 rank。
    pub fn reduce_into(&self, group: &Group, reduce: Reduce, out: &mut [T]) -> Result<()> {
        self.check_width("shard::Local::reduce_into", out.len())?;
        out.fill(reduce.identity());
        for rank in group.ranks() {
            let replica = self.replica(*rank)?;
            for (slot, value) in out.iter_mut().zip(replica) {
                *slot = reduce.apply(*slot, *value);
            }
        }
        Ok(())
    }

    /// 把归约结果写回组内每个 rank 的镜像。
    fn distribute(&mut self, group: &Group, reduced: &[T]) -> Result<()> {
        for rank in group.ranks() {
            self.replica_mut("shard::Local::distribute", *rank)?
                .copy_from_slice(reduced);
        }
        Ok(())
    }

    fn replica_mut(&mut self, site: &'static str, rank: u32) -> Result<&mut [T]> {
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
        refuse_at!(site, "宽度 {len} 与归约宽度 {} 不一致", self.width)
    }

    /// rank 必须落在本拓扑内：越界写在别的 rank 上会静默改掉无关的镜像。
    ///
    /// `site` 由调用方给：同一条越界判据有四个入口，诊断必须指回真正拒绝的那一个。
    fn slot(&self, site: &'static str, rank: u32) -> Result<usize> {
        let idx = rank as usize;
        if idx < self.replicas.len() {
            return Ok(idx);
        }
        refuse_at!(site, "rank {rank} 超出 0..{}", self.replicas.len())
    }
}

impl<T: ReduceScalar> Collective<T> for Local<T> {
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [T]) -> Result<()> {
        self.reduce_into(group, reduce, buf)?;
        self.distribute(group, buf)
    }

    fn all_gather(&mut self, group: &Group, buf: &mut [T]) -> Result<()> {
        let width = self.width;
        if buf.len() != group.size() * width {
            return refuse_at!(
                "shard::Local::all_gather",
                "缓冲长度 {} 与 all_gather 期望的 {} 不一致",
                buf.len(),
                group.size() * width
            );
        }
        for (idx, rank) in group.ranks().iter().enumerate() {
            buf[idx * width..(idx + 1) * width].copy_from_slice(self.replica(*rank)?);
        }
        Ok(())
    }

    fn broadcast(&mut self, group: &Group, src: u32, buf: &mut [T]) -> Result<()> {
        self.check_width("shard::Local::broadcast", buf.len())?;
        if !group.ranks().contains(&src) {
            return refuse_at!("shard::Local::broadcast", "源 rank {src} 不在组内");
        }
        buf.copy_from_slice(self.replica(src)?);
        Ok(())
    }

    fn reduce_scatter(&mut self, group: &Group, reduce: Reduce, buf: &mut [T]) -> Result<()> {
        let width = self.width;
        if buf.len() != group.size() * width {
            return refuse_at!(
                "shard::Local::reduce_scatter",
                "缓冲长度 {} 与 reduce_scatter 期望的 {} 不一致",
                buf.len(),
                group.size() * width
            );
        }
        buf.fill(reduce.identity());
        for (idx, rank) in group.ranks().iter().enumerate() {
            let replica = self.replica(*rank)?;
            for (out, value) in buf[idx * width..(idx + 1) * width].iter_mut().zip(replica) {
                *out = reduce.apply(*out, *value);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) use crate::remote::FileRemote;

#[cfg(test)]
#[path = "../tests/collective_tests.rs"]
mod tests;
