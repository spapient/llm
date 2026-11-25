//! 模型层并行归属：把 [`nn::Stack`] 映射到本 rank 的 PP 层段与 TP 分片。
//!
//! PP 按层名块号切段；TP 按投影后缀选择列切或行切，并由 [`Tiling`] 计算形状与范围；
//! DP 不改变层和权重归属。段外层 [`held`](Placement::held) 为 `false`，非分层参数由各 PP 副本持有。
//! 本模块只生成几何与切片计划，不装载权重或发起通信。

use nn::{Contract, Linear, Stack};

use crate::topology::{ParallelConfig, Plan};
use diagnostic::Result;

/// 段内投影的 TP 分片计划三元组：分片布局、权重形状、行/列区间；非投影层三槽均 `None`。
type ShardPlan = (Option<Tiling>, Option<[usize; 2]>, Option<(usize, usize)>);

/// 投影几何的只读副本：脱离 [`nn::Linear`] 引用，供装配侧长期持有。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearGeo {
    /// 输入宽度。
    pub in_features: usize,
    /// 输出宽度。
    pub out_features: usize,
}

/// 一条层在本 rank 的并行归属；载荷 `T` 由装配侧自带（层句柄 / 键），计划层不解引用它。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement<T> {
    name: String,
    layer: Option<usize>,
    held: bool,
    linear: Option<LinearGeo>,
    tiling: Option<Tiling>,
    shard_shape: Option<[usize; 2]>,
    shard_range: Option<(usize, usize)>,
    /// 本 rank 在 TP 组内的份号；段外或非投影层为 0（切片时段外回 [`None`]，不读它）。
    part: usize,
    payload: T,
}

impl<T> Placement<T> {
    accessors! {
        /// 层名（与权重条目前缀一致）。
        name -> &str = |s| s.name.as_str();
        /// 解析出的块号；非分层参数为 `None`。
        layer -> Option<usize> = |s| s.layer;
        /// 本 rank 是否持有这一层（PP 段内，或非分层参数）。
        held -> bool = |s| s.held;
        /// 投影几何；非投影层为 `None`。
        linear -> Option<LinearGeo> = |s| s.linear;
        /// 整份 TP 切分计划；段外或非投影层为 `None`。
        tiling -> Option<&Tiling> = |s| s.tiling.as_ref();
        /// 本 rank 分片形状 `[out, in]`；段外或非投影层为 `None`。
        shard_shape -> Option<[usize; 2]> = |s| s.shard_shape;
        /// 本 rank 分片在完整权重里的 `(起点, 长度)`。
        shard_range -> Option<(usize, usize)> = |s| s.shard_range;
    }

    /// 按本 rank 归属几何把完整权重取成本 rank 本地子块；段外层或非投影层返回 [`None`]。
    ///
    /// 切片只产出本地子块，不装载进图、不发起通信；行切子块合并仍需 [`Sharding::merge`]
    /// 标注的求和归约，运行时接线未闭合。
    ///
    /// # Errors
    ///
    /// 完整权重长度与形状积不符（委托 [`Tiling::split`]）。
    pub fn split<E: Copy>(&self, full: &[E]) -> Option<Result<Vec<E>>> {
        self.tiling.map(|tiling| tiling.split(self.part, full))
    }

    /// 装配侧自带的载荷原样带回。
    pub fn into_payload(self) -> T {
        self.payload
    }

    accessors! {
        /// 载荷只读视图。
        payload -> &T = |s| &s.payload;
    }
}

/// 投影后缀 → TP 切法：一行一事实，加一种角色只加一行。
///
/// Q/K/V 沿输出维列切（分片直接进下一层，无需通信）；输出投影沿输入维行切（各 rank 出
/// 部分和，合并需求和归约，与 [`Sharding::Row`] 的 `merge` 一致）。
const TP_SHARDING: [(&str, Sharding); 4] = [
    (".q", Sharding::Column),
    (".k", Sharding::Column),
    (".v", Sharding::Column),
    (".o", Sharding::Row),
];

/// 按投影名后缀定切法；未登记角色回退列切（输出维），不猜行切——行切必须配求和归约。
fn sharding_of(name: &str) -> Sharding {
    TP_SHARDING
        .iter()
        .find(|(suffix, _)| name.ends_with(suffix))
        .map_or(Sharding::Column, |(_, sharding)| *sharding)
}

/// 层名里的块号：取第一个可解析为 `usize` 的点分段；没有则为非分层参数。
///
/// `blk.0.q` → `0`；`tok_embeddings` → `None`。段序按名字书写顺序，不依赖固定前缀。
fn layer_idx(name: &str) -> Option<usize> {
    name.split('.')
        .filter_map(|seg| seg.parse::<usize>().ok())
        .next()
}

/// 装配归属计划：由本 rank 的 [`Plan`] 驱动，把层清单归到本 rank 名下。
pub struct Assembler {
    plan: Plan,
}

impl Assembler {
    /// 由本 rank 的并行计划构造。
    pub fn new(plan: Plan) -> Self {
        Self { plan }
    }

    /// 由并行策略与本 rank 构造；拓扑由策略度数派生，与训练侧 `Distributed::from_config`
    /// 走同一条策略→拓扑判据。
    ///
    /// # Errors
    ///
    /// 策略度数非法，或 `rank` 超出派生拓扑的世界范围。
    pub fn from_config(rank: u32, policy: &ParallelConfig) -> Result<Self> {
        let plan = Plan::new(policy.topology()?, rank)?;
        Ok(Self::new(plan))
    }

    /// 便捷入口：归属一份 nn 层栈，载荷为单元。
    ///
    /// `blocks` 是 Transformer 块总数（PP 按它切段）；必须能被 PP 度数整除，判据复用
    /// [`Plan::layer_range`]。
    pub fn place_stack(&self, stack: &Stack, blocks: usize) -> Result<Vec<Placement<()>>> {
        let layers = stack.layers().iter().map(|layer| ((), layer.as_ref()));
        self.place(blocks, layers)
    }

    /// 泛型归属入口：逐条给 `(载荷, 层契约)`，载荷原样写回 [`Placement`]。
    ///
    /// # Errors
    ///
    /// `blocks` 为 0 或不能被 PP 整除；段内某投影的被切维不能被 TP 份数整除（[`Tiling`] 判据）。
    pub fn place<'a, T, I>(&self, blocks: usize, layers: I) -> Result<Vec<Placement<T>>>
    where
        I: IntoIterator<Item = (T, &'a dyn Contract)>,
    {
        let (start, end) = self.plan.layer_range(blocks)?;
        layers
            .into_iter()
            .map(|(payload, layer)| self.assign(start, end, layer, payload))
            .collect()
    }

    /// 单条归属：PP 段判定 + 段内投影 TP 分片。
    fn assign<T>(
        &self,
        start: usize,
        end: usize,
        layer: &dyn Contract,
        payload: T,
    ) -> Result<Placement<T>> {
        let name = layer.name().to_string();
        let block = layer_idx(&name);
        let held = block.is_none_or(|idx| idx >= start && idx < end);
        let linear = layer.linear();
        let geo = linear.map(Self::linear_geo);
        let rank = self.plan.tp_idx();
        let (tiling, shard_shape, shard_range) = self.sharded(held, linear, &name, rank)?;

        Ok(Placement {
            name,
            layer: block,
            held,
            linear: geo,
            tiling,
            shard_shape,
            shard_range,
            part: rank,
            payload,
        })
    }

    /// 线性投影几何只读副本：脱离 [`nn::Linear`] 引用，供装配侧长期持有。
    fn linear_geo(linear: &Linear) -> LinearGeo {
        LinearGeo {
            in_features: linear.in_features(),
            out_features: linear.out_features(),
        }
    }

    /// 段内投影的 TP 分片计划；段外层或非投影层三个槽位均为 `None`。
    fn sharded(
        &self,
        held: bool,
        linear: Option<&Linear>,
        name: &str,
        rank: usize,
    ) -> Result<ShardPlan> {
        match (held, linear) {
            (true, Some(linear)) => {
                let parts = self.plan.topology().tp();
                let sharding = sharding_of(name);
                let tiling =
                    Tiling::new(sharding, parts, linear.out_features(), linear.in_features())?;
                let shape = tiling.weight_shape(rank)?;
                let range = tiling.range(rank)?;
                Ok((Some(tiling), Some(shape), Some(range)))
            }
            _ => Ok((None, None, None)),
        }
    }
}

// ---- 权重切分几何与可复用切片 ----

use crate::collective::Reduce;

/// 张量并行里线性层的两种切法。
///
/// 判别式就是它在 `TABLE` 里的行号（`#[repr(usize)]`），于是短名、被切轴、所需归约三件事
/// 写在同一行里：加一种切法只加一行。旧形态是三个各自两臂的 `match`——"`Row` 忘了要
/// `Sum`"这类错只会在反向传播时表现为静默的梯度翻倍。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Sharding {
    /// 按列切：切输出维，分片直接进下一层，不需要通信。
    Column = 0,
    /// 按行切：切输入维，每个 rank 产出一份部分和，必须求和归约。
    Row = 1,
}

/// 一种切法的全部事实：短名、被切的是权重 `[out, in]` 的哪一维、合并所需的归约。
type Row = (&'static str, usize, Option<Reduce>);

impl Sharding {
    /// 全部切法，供日志与测试遍历。
    pub const ALL: [Self; 2] = [Self::Column, Self::Row];

    /// 每种切法一行。行序必须与判别式一致（三个访问器都走 `TABLE[self as usize]`），
    /// 这条由 `tests` 里的 `every_sharding_row_matches_its_discriminant` 钉住。
    const TABLE: [Row; 2] = [("column", 0, None), ("row", 1, Some(Reduce::Sum))];

    fn row(self) -> Row {
        Self::TABLE[self as usize]
    }

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        self.row().0
    }

    /// 被切的是权重 `[out, in]` 的哪一维。
    ///
    /// 全 crate 只有这一处把切法翻译成轴：分片形状、随份数变化的长度都从它派生。
    pub fn axis(self) -> usize {
        self.row().1
    }

    /// 合并各 rank 的输出需要哪种归约；`Column` 不需要（`None`）。
    pub fn merge(self) -> Option<Reduce> {
        self.row().2
    }
}

/// 一层线性权重的 TP 切分计划：切法、份数与完整的 `[out, in]` 形状。
///
/// 分片是**整除**的等分：`Tiling::new` 拒绝任何摊余数的形状，因此第 `rank` 份永远是
/// `(rank · 被切维长度 / 份数, 被切维长度 / 份数)`。TP 要求各 rank 的权重形状一致，
/// 有余数就没有合法的分片，故这里不保留"前几份多拿一个"的均匀切分——那条分支在任何
/// 可达路径上都取不到。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    sharding: Sharding,
    parts: usize,
    shape: [usize; 2],
}

impl Tiling {
    /// 由切法、份数与完整权重形状构造；被切的那一维必须能被份数整除。
    ///
    /// # Errors
    ///
    /// 份数为零、形状里出现零宽，或被切维不能被份数整除。
    pub fn new(sharding: Sharding, parts: usize, out: usize, inp: usize) -> Result<Self> {
        if parts == 0 {
            fail!("shard::Tiling::new", "切分份数不能为 0");
        }
        if out == 0 || inp == 0 {
            fail!(
                "shard::Tiling::new",
                "权重形状 [{out}, {inp}] 含零宽：切出来是空分片"
            );
        }
        let shape = [out, inp];
        let split = shape[sharding.axis()];
        if !split.is_multiple_of(parts) {
            fail!(
                "shard::Tiling::new",
                "{} 的第 {} 维长度 {split} 不能被 {parts} 整除：各 rank 的权重形状会不一致",
                sharding.as_str(),
                sharding.axis()
            );
        }
        Ok(Self {
            sharding,
            parts,
            shape,
        })
    }

    accessors! {
        /// 切法。
        sharding -> Sharding = |s| s.sharding;
        /// 份数（通常等于 `Topology::tp()`）。
        parts -> usize = |s| s.parts;
    }

    /// 第 `rank` 份的权重形状 `[out, in]`：完整形状把被切维换成该份的长度。
    pub fn weight_shape(&self, rank: usize) -> Result<[usize; 2]> {
        let (_, width) = self.range(rank)?;
        let mut shape = self.shape;
        shape[self.sharding.axis()] = width;
        Ok(shape)
    }

    /// 第 `rank` 份在完整权重里的 `(起点, 长度)`，沿 [`Sharding::axis`] 那一维。
    pub fn range(&self, rank: usize) -> Result<(usize, usize)> {
        if rank >= self.parts {
            fail!(
                "shard::Tiling::range",
                "份下标 {rank} 超出 0..{}",
                self.parts
            );
        }
        let width = self.width();
        Ok((rank * width, width))
    }

    /// 取第 `rank` 份的本地权重子块（行主序 `[out, in]`）。
    ///
    /// 列切的行段在缓冲里连续，一次切出整段；行切的列段跨行分散，只能按行拷列段——行内仍是
    /// 整块拷贝，外层按行循环是数据布局决定的分散拷贝，不是逐元素遍历。切片只产出本 rank 本地
    /// 子块，不装载进图、不发起通信。
    ///
    /// # Errors
    ///
    /// `rank` 越出份数（复用 [`Self::range`]）；`full` 长度不等于完整形状积，或形状积溢出。
    pub fn split<T: Copy>(&self, rank: usize, full: &[T]) -> Result<Vec<T>> {
        let mut local = Vec::new();
        self.split_into(rank, full, &mut local)?;
        Ok(local)
    }

    /// 取第 `rank` 份并写入调用方缓冲；先清空旧值，成功后保留容量供下一层复用。
    ///
    /// # Errors
    ///
    /// 判据与 [`Self::split`] 相同；失败时缓冲保持为空，不暴露半份分片。
    pub fn split_into<T: Copy>(&self, rank: usize, full: &[T], local: &mut Vec<T>) -> Result<()> {
        local.clear();
        let (start, width) = self.range(rank)?;
        let [out, inp] = self.shape;
        self.check_len(full.len(), out, inp)?;
        local.reserve(out * inp / self.parts);
        match self.sharding {
            Sharding::Column => {
                let lo = start * inp;
                local.extend_from_slice(&full[lo..lo + width * inp]);
            }
            Sharding::Row => {
                for row in full.chunks_exact(inp) {
                    local.extend_from_slice(&row[start..start + width]);
                }
            }
        }
        Ok(())
    }

    /// 校验完整权重长度等于形状积；形状积溢出或不符即拒绝。
    fn check_len(&self, full_len: usize, out: usize, inp: usize) -> Result<()> {
        let expected = out
            .checked_mul(inp)
            .ok_or_else(|| invalid!("shard::Tiling::split", "权重形状积溢出"))?;
        if full_len != expected {
            fail!(
                "shard::Tiling::split",
                "权重长度 {full_len} 与完整形状 [{out}, {inp}] 的积 {expected} 不符",
            );
        }
        Ok(())
    }

    /// 被切维在每份里的长度：构造期已保证整除。
    fn width(&self) -> usize {
        self.shape[self.sharding.axis()] / self.parts
    }
}

#[cfg(test)]
#[path = "../tests/placement_tests.rs"]
mod tests;
