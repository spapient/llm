//! 三维并行拓扑：[`Parallel`] 与 [`Topology`]。

use crate::collective::Group;
use diagnostic::Result;

/// 维数：rank 布局的进制表有多长。
const DIM: usize = 3;

/// 三种并行维度。
///
/// 判别式**就是**它在 `[tp, pp, dp]` 里的位置（最次要维在前）：`#[repr(usize)]` 把这条约定
/// 变成类型的一部分，于是短名、并行度、组内下标全都走同一张行表，不再各自写一个三臂 `match`
/// ——旧形态下加一维要改四处，漏掉一处只会在某些输入上悄悄错位。行序由 `tests` 里的
/// `every_axis_row_matches_its_discriminant` 钉住。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Parallel {
    /// 张量并行：同一层的权重按列切。
    Tensor = 0,
    /// 流水并行：层按段切。
    Pipeline = 1,
    /// 数据并行：同一份权重、不同的批。
    Data = 2,
}

impl Parallel {
    /// 全部维度，供日志与 `doctor` 遍历。
    pub const ALL: [Self; DIM] = [Self::Tensor, Self::Pipeline, Self::Data];

    /// 每个维度一行：稳定的短名。行序 = 判别式。
    const NAMES: [&'static str; DIM] = ["tp", "pp", "dp"];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

/// 三维并行拓扑。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    tp: usize,
    pp: usize,
    dp: usize,
    world: usize,
}

impl Topology {
    /// 三个维度都必须非零，且乘积（总进程数）必须能用 `u32` 表示。
    ///
    /// # Errors
    ///
    /// 任一维度为零，或 `tp · pp · dp` 溢出 `usize` / 超过 `u32::MAX` 时返回说明。
    pub fn new(tp: usize, pp: usize, dp: usize) -> Result<Self> {
        if [tp, pp, dp].contains(&0) {
            fail!(
                "shard::Topology::new",
                "并行度必须非零（tp = {tp}，pp = {pp}，dp = {dp}）"
            );
        }
        let world = tp
            .checked_mul(pp)
            .and_then(|plan| plan.checked_mul(dp))
            .filter(|world| *world <= u32::MAX as usize)
            .ok_or_else(|| {
                invalid!(
                    "shard::Topology::new",
                    "tp · pp · dp = {tp} · {pp} · {dp} 超出 u32：rank 布局无解，启动再多进程也对不上"
                )
            })?;
        Ok(Self { tp, pp, dp, world })
    }

    /// 单进程恒合法拓扑（1×1×1）：维度非零、乘积 1 在 `u32` 内，是 [`Self::new`] 的合法特例，
    /// 无需再包一层 `Result`。供进程内恒等编排直接取常量，避免在已知合法的输入上 `expect`。
    pub const fn single() -> Self {
        Self {
            tp: 1,
            pp: 1,
            dp: 1,
            world: 1,
        }
    }

    /// 张量并行度。
    pub fn tp(&self) -> usize {
        self.tp
    }

    /// 流水并行度。
    pub fn pp(&self) -> usize {
        self.pp
    }

    /// 数据并行度。
    pub fn dp(&self) -> usize {
        self.dp
    }

    /// 总进程数。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 校验启动的进程数与拓扑一致（不一致时继续跑只会得到静默的错误结果）。
    pub fn check_world(&self, world: usize) -> Result<()> {
        if world != self.world {
            fail!(
                "shard::Topology::check",
                "启动 {world} 个进程，但拓扑要求 {}",
                self.world
            );
        }
        Ok(())
    }

    /// 由三个下标算出全局 rank：`dp_idx · (tp · pp) + pp_idx · tp + tp_idx`，即把
    /// `[tp_idx, pp_idx, dp_idx]` 按 `[tp, pp, dp]` 这组进制编码成一个混合进制数。
    ///
    /// # Errors
    ///
    /// 任一下标超出它那一维的并行度。
    pub fn rank(&self, tp_idx: usize, pp_idx: usize, dp_idx: usize) -> Result<u32> {
        self.check_axis(Parallel::Tensor, tp_idx)?;
        self.check_axis(Parallel::Pipeline, pp_idx)?;
        self.check_axis(Parallel::Data, dp_idx)?;
        let flat = self.weigh([tp_idx, pp_idx, dp_idx]);
        // `new` 已把 world 限在 u32 以内，而 flat < world，故这次收窄不会失败。
        u32::try_from(flat).map_err(|_| invalid!("shard::Topology::rank", "rank {flat} 超出 u32"))
    }

    /// [`Topology::rank`] 的反解：`(tp_idx, pp_idx, dp_idx)`。
    ///
    /// # Errors
    ///
    /// `rank` 不在 `0..world`。
    pub fn idxs(&self, rank: u32) -> Result<(usize, usize, usize)> {
        let position = rank as usize;
        if position >= self.world {
            fail!(
                "shard::Topology::idxs",
                "rank {rank} 超出 0..{}",
                self.world
            );
        }
        let [tp, pp, dp] = self.digits(position, None);
        Ok((tp, pp, dp))
    }

    /// 第 `idx` 个指定维度的进程组：组内只沿该维移动，另外两维固定。
    ///
    /// 组是一条**等差序列**：沿本维加一位就是加这一维的步长，所以整组由「首成员」与步长两个数
    /// 决定，成员不必再逐个走 [`Topology::rank`] 的三次下标校验。组号则只在另外两维里解码。
    ///
    /// # Errors
    ///
    /// `idx` 超出该维的组数（即 `world / 该维并行度`）。
    pub fn group(&self, parallel: Parallel, idx: usize) -> Result<Group> {
        let own = parallel as usize;
        let degree = self.degree(parallel);
        let count = self.world / degree;
        if idx >= count {
            fail!(
                "shard::Topology::group",
                "{} 组下标 {idx} 超出 0..{count}",
                parallel.as_str()
            );
        }
        let stride = self.strides()[own];
        let head = self.weigh(self.digits(idx, Some(own)));
        let ranks = (0..degree)
            .map(|step| u32::try_from(head + step * stride).ok())
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(|| invalid!("shard::Topology::group", "组内 rank 超出 u32"))?;
        Group::new(ranks)
    }

    /// 三个维度的进制，从最次要维排到最主维：全 crate 唯一的一处 rank 布局配置。
    fn radices(&self) -> [usize; DIM] {
        [self.tp, self.pp, self.dp]
    }

    /// 某一维的并行度，也是它每个组的大小。
    fn degree(&self, parallel: Parallel) -> usize {
        self.radices()[parallel as usize]
    }

    /// 每一维的 rank 步长：沿该维加一位，rank 就加这么多。`strides[k] = ∏_{j<k} radices[j]`。
    ///
    /// 这是 rank 布局的第二种等价说法：`rank = Σ idxs[k] · strides[k]`。它把 [`Self::radices`]
    /// 从"进制"翻译成"权重"，此后编码（[`Self::weigh`]）与解码（[`Self::digits`]）都只做乘除，
    /// 不再各自重写一遍进制循环。
    fn strides(&self) -> [usize; DIM] {
        let radices = self.radices();
        let mut strides = [1; DIM];
        for k in 1..DIM {
            strides[k] = strides[k - 1] * radices[k - 1];
        }
        strides
    }

    /// 编码：三个下标按各自的步长加权求和。
    fn weigh(&self, idxs: [usize; DIM]) -> usize {
        idxs.iter().zip(self.strides()).map(|(n, s)| n * s).sum()
    }

    /// 解码：按 `[tp, pp, dp]` 的进制逐位取余，`skip` 那一维跳过并留 0。
    ///
    /// `skip = None` 是完整反解（[`Self::idxs`]）；`skip = Some(own)` 是"本维不动、只在另外
    /// 两维里数"，即 [`Self::group`] 的组号——此时商只在留下的两维上推进，所以组号是**压缩**
    /// 过的进制数，而不是完整 rank 的高位。两者共用这一处循环，`spread` 就是它 `skip = None`
    /// 的形态。
    fn digits(&self, mut flat: usize, skip: Option<usize>) -> [usize; DIM] {
        let radices = self.radices();
        let mut digits = [0; DIM];
        for axis in 0..DIM {
            if Some(axis) == skip {
                continue;
            }
            digits[axis] = flat % radices[axis];
            flat /= radices[axis];
        }
        digits
    }

    /// 某一维的下标必须落在它自己的并行度以内。
    fn check_axis(&self, parallel: Parallel, idx: usize) -> Result<()> {
        let limit = self.degree(parallel);
        if idx >= limit {
            fail!(
                "shard::Topology::rank",
                "{} 下标 {idx} 超出 0..{limit}",
                parallel.as_str()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/topology_tests.rs"]
mod tests;
