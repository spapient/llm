//! 集合通信：统一端口、泛型标量、进程内镜像与文件传输实现。
//!
//! [`Local`] 与 [`FileRemote`] 只负责镜像读写；归约、收集、广播和散布共享同一组泛型算法，
//! 因此两种传输不会出现语义分叉。跨进程序列化协议固定为小端 `f32`。

use core::ops::{Add, Mul};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::PathBuf;

use crate::Code;
use crate::topology::Topology;
use diagnostic::{Error, Result};

/// 可参与归约的标量：算术、偏序与四个单位元由元素类型提供。
pub trait ReduceScalar:
    Copy + 'static + Send + Sync + Add<Output = Self> + Mul<Output = Self> + PartialOrd
{
    /// 求和单位元。
    const ZERO: Self;
    /// 乘积单位元。
    const ONE: Self;
    /// 取最大值的单位元。
    const MIN_LIMIT: Self;
    /// 取最小值的单位元。
    const MAX_LIMIT: Self;

    /// 取大；浮点实现必须传播 NaN。
    fn max(left: Self, right: Self) -> Self {
        if left >= right { left } else { right }
    }

    /// 取小；浮点实现必须传播 NaN。
    fn min(left: Self, right: Self) -> Self {
        if left <= right { left } else { right }
    }
}

macro_rules! impl_float_scalar {
    ($($ty:ty),+ $(,)?) => {$(
        impl ReduceScalar for $ty {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const MIN_LIMIT: Self = <$ty>::NEG_INFINITY;
            const MAX_LIMIT: Self = <$ty>::INFINITY;

            fn max(left: Self, right: Self) -> Self {
                if left.is_nan() || right.is_nan() { <$ty>::NAN } else { <$ty>::max(left, right) }
            }

            fn min(left: Self, right: Self) -> Self {
                if left.is_nan() || right.is_nan() { <$ty>::NAN } else { <$ty>::min(left, right) }
            }
        }
    )+};
}

impl_float_scalar!(f32, f64);

macro_rules! impl_integer_scalar {
    ($($ty:ty),+ $(,)?) => {$(
        impl ReduceScalar for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MIN_LIMIT: Self = Self::MIN;
            const MAX_LIMIT: Self = Self::MAX;
        }
    )+};
}

impl_integer_scalar!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize
);

/// 集合通信后端端口；元素档默认为 `f32`。
pub trait Collective<T: ReduceScalar = f32> {
    /// 对 `buf` 做原地归约。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [T]) -> Result<()>;

    /// 按组内顺序拼接各成员的 `width` 个元素。
    fn all_gather(&mut self, _group: &Group, _buf: &mut [T]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Collective::all_gather",
            "该集合通信后端没有 all_gather"
        );
    }

    /// 将 `src` 成员的值复制到 `buf`。
    fn broadcast(&mut self, _group: &Group, _src: u32, _buf: &mut [T]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Collective::broadcast",
            "该集合通信后端没有 broadcast"
        );
    }

    /// 将各成员对应分块归约到输出分块。
    fn reduce_scatter(&mut self, _group: &Group, _reduce: Reduce, _buf: &mut [T]) -> Result<()> {
        fail!(
            Unsupported,
            "shard::Collective::reduce_scatter",
            "该集合通信后端没有 reduce_scatter"
        );
    }
}

/// 严格升序且不重复的进程组。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    ranks: Vec<u32>,
}

impl Group {
    /// 校验并构造进程组。
    ///
    /// # Errors
    ///
    /// 组为空，或 rank 重复／乱序。
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
    #[must_use]
    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    /// 组大小。
    #[must_use]
    pub fn size(&self) -> usize {
        self.ranks.len()
    }
}

/// 归约种类闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Reduce {
    /// 求和。
    Sum,
    /// 取最大。
    Max,
    /// 取最小。
    Min,
    /// 取乘积。
    Prod,
}

impl Reduce {
    /// 全部归约种类。
    pub const ALL: [Self; 4] = [Self::Sum, Self::Max, Self::Min, Self::Prod];
    const NAMES: [&'static str; 4] = ["sum", "max", "min", "prod"];

    /// 稳定短名。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    /// 当前元素档下的单位元。
    #[must_use]
    pub fn identity<T: ReduceScalar>(self) -> T {
        match self {
            Self::Sum => T::ZERO,
            Self::Prod => T::ONE,
            Self::Max => T::MIN_LIMIT,
            Self::Min => T::MAX_LIMIT,
        }
    }

    /// 应用当前二元归约运算。
    #[must_use]
    pub fn apply<T: ReduceScalar>(self, left: T, right: T) -> T {
        match self {
            Self::Sum => left + right,
            Self::Prod => left * right,
            Self::Max => T::max(left, right),
            Self::Min => T::min(left, right),
        }
    }

    /// 折叠非空切片；空切片返回 `None`。
    #[must_use]
    pub fn fold<T: ReduceScalar>(self, values: &[T]) -> Option<T> {
        let (first, rest) = values.split_first()?;
        Some(
            rest.iter()
                .fold(*first, |acc, value| self.apply(acc, *value)),
        )
    }
}

/// 未挂载真实通信栈的远端占位后端。
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

/// 镜像读写端口；集合算法只依赖这一组原语。
trait ReplicaStore<T: ReduceScalar> {
    fn width(&self) -> usize;
    fn read_into(&self, rank: u32, out: &mut [T]) -> Result<()>;
    fn write(&mut self, rank: u32, values: &[T]) -> Result<()>;

    fn check_width(&self, site: &'static str, len: usize) -> Result<()> {
        if len == self.width() {
            return Ok(());
        }
        refuse_at!(site, "宽度 {len} 与归约宽度 {} 不一致", self.width())
    }

    fn check_group_width(&self, site: &'static str, group: &Group, len: usize) -> Result<usize> {
        let width = self.width();
        let Some(expected) = group.size().checked_mul(width) else {
            return refuse_at!(site, "组大小 {} 与宽度 {width} 的乘积溢出", group.size());
        };
        if len != expected {
            return refuse_at!(site, "缓冲长度 {len} 与集合操作期望的 {expected} 不一致");
        }
        Ok(width)
    }
}

fn reduce_store<S, T>(
    store: &S,
    group: &Group,
    reduce: Reduce,
    out: &mut [T],
    site: &'static str,
) -> Result<()>
where
    S: ReplicaStore<T>,
    T: ReduceScalar,
{
    store.check_width(site, out.len())?;
    out.fill(reduce.identity());
    let mut slot = vec![T::ZERO; store.width()];
    for rank in group.ranks() {
        store.read_into(*rank, &mut slot)?;
        reduce_values(reduce, out, &slot);
    }
    Ok(())
}

/// 将一个镜像槽归约进输出；归约顺序与组内 rank 顺序一致。
fn reduce_values<T: ReduceScalar>(reduce: Reduce, out: &mut [T], slot: &[T]) {
    for (acc, value) in out.iter_mut().zip(slot) {
        *acc = reduce.apply(*acc, *value);
    }
}

fn all_reduce_store<S, T>(
    store: &mut S,
    group: &Group,
    reduce: Reduce,
    buf: &mut [T],
    site: &'static str,
) -> Result<()>
where
    S: ReplicaStore<T>,
    T: ReduceScalar,
{
    reduce_store(store, group, reduce, buf, site)?;
    for rank in group.ranks() {
        store.write(*rank, buf)?;
    }
    Ok(())
}

fn gather_store<S, T>(store: &S, group: &Group, buf: &mut [T], site: &'static str) -> Result<()>
where
    S: ReplicaStore<T>,
    T: ReduceScalar,
{
    let width = store.check_group_width(site, group, buf.len())?;
    for (chunk, rank) in buf.chunks_exact_mut(width).zip(group.ranks()) {
        store.read_into(*rank, chunk)?;
    }
    Ok(())
}

fn broadcast_store<S, T>(
    store: &S,
    group: &Group,
    src: u32,
    buf: &mut [T],
    site: &'static str,
) -> Result<()>
where
    S: ReplicaStore<T>,
    T: ReduceScalar,
{
    store.check_width(site, buf.len())?;
    if !group.ranks().contains(&src) {
        return refuse_at!(site, "源 rank {src} 不在组内");
    }
    store.read_into(src, buf)
}

fn scatter_store<S, T>(
    store: &S,
    group: &Group,
    reduce: Reduce,
    buf: &mut [T],
    site: &'static str,
) -> Result<()>
where
    S: ReplicaStore<T>,
    T: ReduceScalar,
{
    let width = store.check_group_width(site, group, buf.len())?;
    buf.fill(reduce.identity());
    let mut slot = vec![T::ZERO; width];
    for (chunk, rank) in buf.chunks_exact_mut(width).zip(group.ranks()) {
        store.read_into(*rank, &mut slot)?;
        reduce_values(reduce, chunk, &slot);
    }
    Ok(())
}

/// 进程内集合通信：每个 rank 一份镜像。
#[derive(Debug, Clone, PartialEq)]
pub struct Local<T: ReduceScalar = f32> {
    topology: Topology,
    width: usize,
    replicas: Vec<Vec<T>>,
}

impl<T: ReduceScalar> Local<T> {
    /// 为拓扑中的每个 rank 建立零值镜像。
    ///
    /// # Errors
    ///
    /// `width` 为零。
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

    /// 写入某个 rank 的输入。
    ///
    /// # Errors
    ///
    /// 长度不符或 rank 越界。
    pub fn load(&mut self, rank: u32, values: &[T]) -> Result<()> {
        self.check_width("shard::Local::load", values.len())?;
        self.replica_mut("shard::Local::load", rank)?
            .copy_from_slice(values);
        Ok(())
    }

    /// 读取某个 rank 的当前镜像。
    ///
    /// # Errors
    ///
    /// rank 越界。
    pub fn replica(&self, rank: u32) -> Result<&[T]> {
        Ok(&self.replicas[self.slot("shard::Local::replica", rank)?])
    }

    /// 分配并返回组内归约结果。
    ///
    /// # Errors
    ///
    /// 组里出现拓扑之外的 rank。
    pub fn reduce(&self, group: &Group, reduce: Reduce) -> Result<Vec<T>> {
        let mut total = vec![T::ZERO; self.width];
        self.reduce_into(group, reduce, &mut total)?;
        Ok(total)
    }

    /// 将组内归约写入调用方缓冲，全程不分配结果数组。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符或组里出现拓扑之外的 rank。
    pub fn reduce_into(&self, group: &Group, reduce: Reduce, out: &mut [T]) -> Result<()> {
        reduce_store(self, group, reduce, out, "shard::Local::reduce_into")
    }

    fn replica_mut(&mut self, site: &'static str, rank: u32) -> Result<&mut [T]> {
        let idx = self.slot(site, rank)?;
        Ok(&mut self.replicas[idx])
    }

    fn slot(&self, site: &'static str, rank: u32) -> Result<usize> {
        let idx = rank as usize;
        if idx < self.replicas.len() {
            return Ok(idx);
        }
        refuse_at!(site, "rank {rank} 超出 0..{}", self.replicas.len())
    }
}

impl<T: ReduceScalar> ReplicaStore<T> for Local<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn read_into(&self, rank: u32, out: &mut [T]) -> Result<()> {
        out.copy_from_slice(self.replica(rank)?);
        Ok(())
    }

    fn write(&mut self, rank: u32, values: &[T]) -> Result<()> {
        self.replica_mut("shard::Local::write", rank)?
            .copy_from_slice(values);
        Ok(())
    }
}

impl<T: ReduceScalar> Collective<T> for Local<T> {
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [T]) -> Result<()> {
        all_reduce_store(self, group, reduce, buf, "shard::Local::all_reduce")
    }

    fn all_gather(&mut self, group: &Group, buf: &mut [T]) -> Result<()> {
        gather_store(self, group, buf, "shard::Local::all_gather")
    }

    fn broadcast(&mut self, group: &Group, src: u32, buf: &mut [T]) -> Result<()> {
        broadcast_store(self, group, src, buf, "shard::Local::broadcast")
    }

    fn reduce_scatter(&mut self, group: &Group, reduce: Reduce, buf: &mut [T]) -> Result<()> {
        scatter_store(self, group, reduce, buf, "shard::Local::reduce_scatter")
    }
}

/// 文件传输集合通信：共享文件承载 `world × width` 个小端 `f32` 槽位。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRemote {
    rank: u32,
    world: u32,
    width: usize,
    bytes: u64,
    backing: PathBuf,
}

impl FileRemote {
    /// 打开一个 rank 的文件句柄，并将共享文件预留到完整长度。
    ///
    /// # Errors
    ///
    /// `world` 或 `width` 为零、rank 越界、文件长度溢出或文件不可创建。
    pub fn open(rank: u32, world: u32, width: usize, backing: impl Into<PathBuf>) -> Result<Self> {
        Self::check_open_config(rank, world, width)?;
        let bytes = Self::file_bytes(world, width)?;
        let remote = Self {
            rank,
            world,
            width,
            bytes,
            backing: backing.into(),
        };
        remote.resize()?;
        Ok(remote)
    }

    /// 打开配置只负责进程拓扑与非零宽度，和文件字节换算分离。
    fn check_open_config(rank: u32, world: u32, width: usize) -> Result<()> {
        if world == 0 {
            fail!("shard::FileRemote::open", "进程数 world 不能为 0");
        }
        if rank >= world {
            fail!("shard::FileRemote::open", "rank {rank} 超出 0..{world}");
        }
        if width == 0 {
            fail!("shard::FileRemote::open", "归约宽度不能为 0");
        }
        Ok(())
    }

    /// 共享文件包含 `world × width` 个 `f32`，所有换算必须受溢出保护。
    fn file_bytes(world: u32, width: usize) -> Result<u64> {
        let slot_bytes = width
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid!("shard::FileRemote::open", "槽位字节数溢出"))?;
        u64::try_from(slot_bytes)
            .ok()
            .and_then(|slot| slot.checked_mul(u64::from(world)))
            .ok_or_else(|| invalid!("shard::FileRemote::open", "共享文件字节数溢出"))
    }

    /// 本进程的 rank。
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// 将某个 rank 的输入写入对应槽位。
    ///
    /// # Errors
    ///
    /// 长度不符、rank 越界或文件不可写。
    pub fn load(&self, rank: u32, values: &[f32]) -> Result<()> {
        self.check_width("shard::FileRemote::load", values.len())?;
        self.check_rank("shard::FileRemote::load", rank)?;
        self.write_slot(rank, values)
    }

    /// 读取某个 rank 的当前槽位。
    ///
    /// # Errors
    ///
    /// rank 越界或文件不可读。
    pub fn replica(&self, rank: u32) -> Result<Vec<f32>> {
        self.check_rank("shard::FileRemote::replica", rank)?;
        let mut slot = vec![0.0; self.width];
        self.read_slot(rank, &mut slot)?;
        Ok(slot)
    }

    /// 将组内归约写入调用方缓冲。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符、组里出现 world 之外的 rank 或文件不可读。
    pub fn reduce_into(&self, group: &Group, reduce: Reduce, out: &mut [f32]) -> Result<()> {
        reduce_store(self, group, reduce, out, "shard::FileRemote::reduce_into")
    }

    fn check_rank(&self, site: &'static str, rank: u32) -> Result<()> {
        if rank < self.world {
            return Ok(());
        }
        refuse_at!(site, "rank {rank} 超出 0..{}", self.world)
    }

    fn offset(&self, rank: u32) -> u64 {
        u64::from(rank) * (self.width as u64) * size_of::<f32>() as u64
    }

    fn io(&self, action: &str, error: std::io::Error) -> Error {
        Error::new(
            Code::Io,
            format!("{action} `{}` 失败：{error}", self.backing.display()),
        )
    }

    fn seek_slot(&self, file: &mut File, rank: u32) -> Result<()> {
        file.seek(SeekFrom::Start(self.offset(rank)))
            .map_err(|error| self.io("定位", error))
            .map(|_| ())
    }

    fn resize(&self) -> Result<()> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.backing)
            .map_err(|error| self.io("创建", error))?
            .set_len(self.bytes)
            .map_err(|error| self.io("预留", error))
    }

    fn read_slot(&self, rank: u32, out: &mut [f32]) -> Result<()> {
        let bytes = self.read_bytes(rank)?;
        for (slot, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *slot = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    /// 文件定位与完整读取共用一条路径，调用方只负责协议解码。
    fn read_bytes(&self, rank: u32) -> Result<Vec<u8>> {
        self.check_rank("shard::FileRemote::read", rank)?;
        let mut file = File::open(&self.backing).map_err(|error| self.io("打开", error))?;
        self.seek_slot(&mut file, rank)?;
        let mut bytes = vec![0u8; self.width * size_of::<f32>()];
        file.read_exact(&mut bytes)
            .map_err(|error| self.io("读", error))?;
        Ok(bytes)
    }

    fn write_slot(&self, rank: u32, values: &[f32]) -> Result<()> {
        self.check_rank("shard::FileRemote::write", rank)?;
        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.backing)
            .map_err(|error| self.io("打开", error))?;
        self.seek_slot(&mut file, rank)?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        file.write_all(&bytes).map_err(|error| self.io("写", error))
    }
}

impl ReplicaStore<f32> for FileRemote {
    fn width(&self) -> usize {
        self.width
    }

    fn read_into(&self, rank: u32, out: &mut [f32]) -> Result<()> {
        self.read_slot(rank, out)
    }

    fn write(&mut self, rank: u32, values: &[f32]) -> Result<()> {
        self.write_slot(rank, values)
    }
}

impl Collective for FileRemote {
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()> {
        all_reduce_store(self, group, reduce, buf, "shard::FileRemote::all_reduce")
    }

    fn all_gather(&mut self, group: &Group, buf: &mut [f32]) -> Result<()> {
        gather_store(self, group, buf, "shard::FileRemote::all_gather")
    }

    fn broadcast(&mut self, group: &Group, src: u32, buf: &mut [f32]) -> Result<()> {
        broadcast_store(self, group, src, buf, "shard::FileRemote::broadcast")
    }

    fn reduce_scatter(&mut self, group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()> {
        scatter_store(
            self,
            group,
            reduce,
            buf,
            "shard::FileRemote::reduce_scatter",
        )
    }
}

#[cfg(test)]
#[path = "../tests/collective_tests.rs"]
mod tests;
