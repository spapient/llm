//! 模型层的并行归属：把一份 [`nn::Stack`] 层清单映射到本 rank 的 PP 段与 TP 分片。
//!
//! [`Plan`] 只描述"进程在三维世界的哪里"；本模块回答"这份模型里哪些层、哪片权重归本
//! rank"。两件事正交：
//!
//! - **流水并行（PP）**：按层名里的块号把 Transformer 块切到本 rank 的层段；段外层标记
//!   [`held`](Placement::held) 为 `false`，非分层参数（嵌入、归一化、块外项）没有块号，所有
//!   PP 副本各自持有一份；
//! - **张量并行（TP）**：段内投影按后缀角色定切法（Q/K/V 列切、输出投影行切），用
//!   [`Tiling`] 算出本 rank 分片形状与权重行/列范围；
//! - **数据并行（DP）**：副本不切层、不切权重，只影响梯度归约（由 `Collective` 端口驱动），
//!   归属视图不做 DP 裁剪。
//!
//! 本模块只产出**归属几何与权重范围计划**：不装载真实权重、不发起通信。真实切片装载接
//! 参数表、跨进程集合通信仍受安全通道边界约束（见分账 §5.3）。

use nn::{Contract, Linear, Stack};

use crate::plan::Plan;
use crate::policy::ParallelConfig;
use crate::tiling::{Sharding, Tiling};
use diagnostic::Result;

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
    /// 层名（与权重条目前缀一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 解析出的块号；非分层参数为 `None`。
    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    /// 本 rank 是否持有这一层（PP 段内，或非分层参数）。
    pub fn held(&self) -> bool {
        self.held
    }

    /// 投影几何；非投影层为 `None`。
    pub fn linear(&self) -> Option<LinearGeo> {
        self.linear
    }

    /// 整份 TP 切分计划（切法 / 份数 / 完整形状）；段外或非投影层为 `None`。
    pub fn tiling(&self) -> Option<&Tiling> {
        self.tiling.as_ref()
    }

    /// 本 rank 分片形状 `[out, in]`；段外或非投影层为 `None`。
    pub fn shard_shape(&self) -> Option<[usize; 2]> {
        self.shard_shape
    }

    /// 本 rank 分片在完整权重里的 `(起点, 长度)`，沿 [`Sharding::axis`] 那一维；段外或非投影层为 `None`。
    pub fn shard_range(&self) -> Option<(usize, usize)> {
        self.shard_range
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

    /// 载荷只读视图。
    pub fn payload(&self) -> &T {
        &self.payload
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
    ) -> Result<(Option<Tiling>, Option<[usize; 2]>, Option<(usize, usize)>)> {
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

#[cfg(test)]
#[path = "../tests/placement_tests.rs"]
mod tests;
