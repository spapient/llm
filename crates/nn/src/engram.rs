//! Engram 条件记忆（§5.8.2）：哈希 N-gram、`O(1)` 槽、ρ 稀疏分配。

use diagnostic::{Code, Error, Result};
use crate::conv::scan::Scalar;

use crate::{Contract, ContractSpec, ParamRole, Shape};

/// 乘-XOR 终混（64 位全雪崩，splitmix64 finalizer）。
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// `φ_{n,k}`：确定性乘-XOR 哈希，把 N-gram（含表种子）映射成槽哈希。
pub fn phi(ngram: &[u32], seed: u64) -> u64 {
    let mut h = seed;
    for &t in ngram {
        h = mix(h ^ u64::from(t));
    }
    h
}

/// 槽号 `z = φ(ngram, seed) mod buckets`：槽公式只有这一份。
fn bucket(ngram: &[u32], seed: u64, buckets: usize) -> usize {
    (phi(ngram, seed) % buckets as u64) as usize
}

/// 尾部 `order` 元 N-gram 的槽号；历史不足 `order`（或几何退化）时 `None`。
pub fn slot_of(tokens: &[u32], order: usize, seed: u64, buckets: usize) -> Option<usize> {
    if buckets == 0 || order < 2 {
        return None;
    }
    let tail = tokens.len().checked_sub(order)?;
    Some(bucket(&tokens[tail..], seed, buckets))
}

/// 每 token 槽数 `K(N−1)`：`n` 为阶（≥2），`k` 为表数；与表长无关（`O(1)` 查找）。
pub const fn slots(n: usize, k: usize) -> usize {
    k * n.saturating_sub(1)
}

/// 稀疏参数分配：`P_MoE = ρ·P_sparse`，`P_Engram = (1−ρ)·P_sparse`（`ρ∈[0,1]`）。
pub fn allocate<S: Scalar>(rho: S, p_sparse: S) -> (S, S) {
    debug_assert!(rho >= S::ZERO && rho <= S::ONE);
    let om = S::ONE.sub(rho);
    (rho.mul(p_sparse), om.mul(p_sparse))
}

/// 下一个 ≥ `lower` 的素数（表长取素数，避免合数槽聚集）。
pub fn next_prime(mut lower: usize) -> usize {
    if lower < 2 {
        return 2;
    }
    while !is_prime(lower) {
        lower += 1;
    }
    lower
}

/// 试除判素（`n` 为表长量级，小正整数）。
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3usize;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// 一张 N-gram 嵌入表：素数长、`d` 维嵌入，哈希槽 `O(1)` 读写。
#[derive(Debug, Clone)]
pub struct Table<S: Scalar = f32> {
    len: usize,
    dim: usize,
    data: Vec<S>,
}

impl<S: Scalar> Table<S> {
    /// 构造：`len` 自动提升到素数，`dim` 为嵌入维。
    pub fn new(len: usize, dim: usize) -> Self {
        assert!(dim > 0, "嵌入维必须非零");
        let len = next_prime(len);
        Self {
            len,
            dim,
            data: vec![S::ZERO; len * dim],
        }
    }

    /// 表长（素数）。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 恒为假（表长至少为 2），仅供容器式 API 完整性。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 嵌入维。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 槽下标：`z = φ(ngram, seed) mod len`。
    pub fn slot(&self, ngram: &[u32], seed: u64) -> usize {
        bucket(ngram, seed, self.len)
    }

    /// `O(1)` 查表：返回该槽的嵌入切片（长度 = [`Table::dim`]）。
    pub fn get(&self, ngram: &[u32], seed: u64) -> &[S] {
        let z = self.slot(ngram, seed);
        &self.data[z * self.dim..(z + 1) * self.dim]
    }

    /// `O(1)` 写表：把 `value` 写入该槽（覆盖）。
    pub fn set(&mut self, ngram: &[u32], seed: u64, value: &[S]) {
        assert_eq!(value.len(), self.dim, "嵌入长度必须等于 dim");
        let z = self.slot(ngram, seed);
        self.data[z * self.dim..(z + 1) * self.dim].copy_from_slice(value);
    }
}

/// `LmHead` 前的条件记忆旁路：尾部 N-gram → 表槽 → 门控残差。
#[derive(Debug, Clone)]
pub struct Bypass<S: Scalar = f32> {
    table: Table<S>,
    seed: u64,
    order: usize,
    gate: S,
}

/// 旁路报错的出处。
const SITE: &str = "nn::engram::Bypass";

impl<S: Scalar> Bypass<S> {
    /// 构造：`order ≥ 2`（1-gram 没有上下文，只是词表拷贝），`gate` 是残差强度。
    pub fn new(table: Table<S>, seed: u64, order: usize, gate: S) -> Result<Self> {
        if order < 2 {
            return Err(Error::new(
                Code::Invalid,
                format!("N-gram 阶必须 ≥ 2（order = {order}）"),
            )
            .with(SITE));
        }
        Ok(Self {
            table,
            seed,
            order,
            gate,
        })
    }

    crate::accessors! {
        /// N-gram 阶 `N`。
        order -> usize = order,
        /// 残差强度。
        gate -> S = gate,
    }

    /// 只读的表。
    pub fn table(&self) -> &Table<S> {
        &self.table
    }

    /// 尾部 N-gram 的槽嵌入；历史不足 `order` 时 `None`（不猜记忆）。
    pub fn recall(&self, tokens: &[u32]) -> Option<&[S]> {
        let tail = tokens.len().checked_sub(self.order)?;
        Some(self.table.get(&tokens[tail..], self.seed))
    }

    /// 把记忆加进隐藏向量：`hidden += gate · recall`。
    pub fn apply(&self, hidden: &mut [S], tokens: &[u32]) -> Result<bool> {
        if hidden.len() != self.table.dim() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "隐藏向量长度 {} 与嵌入维 {} 不符",
                    hidden.len(),
                    self.table.dim()
                ),
            )
            .with(SITE));
        }
        let Some(memory) = self.recall(tokens) else {
            return Ok(false);
        };
        for (slot, &value) in hidden.iter_mut().zip(memory) {
            *slot = slot.add(value.mul(self.gate));
        }
        Ok(true)
    }
}

/// 条件记忆层的装配视图（§5.8.2 的第二个执行形态）：`[width]` 隐藏向量 + 尾部 N-gram → 按门加回。
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryBlock {
    spec: ContractSpec,
    width: usize,
    order: usize,
    seed: u64,
    buckets: usize,
    gate: f32,
}

/// 记忆层报错的出处。
const MEMORY_SITE: &str = "nn::engram::MemoryBlock";

/// 行宽与槽数：两者都是查表节点的建表几何，为 0 会建出一张空表或零宽行。
fn check_shape(width: usize, buckets: usize) -> Result<()> {
    if width == 0 {
        return Err(
            Error::new(Code::Invalid, "条件记忆的行宽必须为正".to_string()).with(MEMORY_SITE),
        );
    }
    if buckets == 0 {
        return Err(
            Error::new(Code::Invalid, "条件记忆的表槽数必须为正".to_string()).with(MEMORY_SITE),
        );
    }
    if buckets > MAX_BUCKETS {
        return Err(Error::new(
            Code::Invalid,
            format!("表槽数 {buckets} 超出下标通道的精确上界 {MAX_BUCKETS}"),
        )
        .with(MEMORY_SITE));
    }
    Ok(())
}

/// 阶与门控强度：前者决定「有没有上下文可查」，后者决定「加回去还看得见主链」。
fn check_params(order: usize, gate: f32) -> Result<()> {
    if order < 2 {
        return Err(Error::new(
            Code::Invalid,
            format!("N-gram 阶必须 ≥ 2（order = {order}）"),
        )
        .with(MEMORY_SITE));
    }
    if !gate.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("门控强度必须是有限值，得到 {gate}"))
                .with(MEMORY_SITE),
        );
    }
    Ok(())
}

/// 表槽数上界：槽号经查表算子的下标通道传输，`2^24` 以上的整数会被静默合并到相邻槽——表
/// 看起来建成了，实际两个槽共用同一行，读回来的记忆属于别的 N-gram。
const MAX_BUCKETS: usize = 1 << 24;

impl MemoryBlock {
    /// 查表节点算子名：`[slots, table] → [width]`。
    pub const EMBED: &'static str = "embed_f32";
    /// 门控缩放节点算子名：`[width] × [width]`（逐元素）。
    pub const SCALE: &'static str = "mul_f32";
    /// 加回主链节点算子名：`[width] + [width]`（逐元素）。
    pub const RESIDUAL: &'static str = "add_f32";

    /// 构造：行宽与槽数为正、`order ≥ 2`、门控有限。
    ///
    /// `order < 2` 时没有上下文可查（1-gram 只是词表拷贝，见 [`Bypass::new`]）；`gate` 非有限会让
    /// 整行隐藏向量变成 `NaN` 而看不出是哪一层坏的。
    ///
    /// # Errors
    ///
    /// `width == 0`、`buckets == 0`、`order < 2`、`gate` 非有限——返回 `Code::Invalid`。
    pub fn new(
        name: &str,
        width: usize,
        order: usize,
        seed: u64,
        buckets: usize,
        gate: f32,
    ) -> Result<Self> {
        check_shape(width, buckets)?;
        check_params(order, gate)?;
        let spec = ContractSpec::new_with_role(
            name,
            Shape::new([width]),
            [Self::EMBED, Self::SCALE, Self::RESIDUAL],
            ParamRole::Embedding,
        )?;
        Ok(Self {
            spec,
            width,
            order,
            seed,
            buckets,
            gate,
        })
    }

    crate::accessors! {
        /// 隐藏宽度（本层的输入与输出宽度）。
        width -> usize = width,
        /// N-gram 阶 `N`。
        order -> usize = order,
        /// 槽哈希种子（与写表时同一把）。
        seed -> u64 = seed,
        /// 表槽数 `M`：驻留权重条目为 `[buckets, width]`。
        buckets -> usize = buckets,
        /// 门控强度 `γ`。
        gate -> f32 = gate,
    }
}

impl Contract for MemoryBlock {
    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    fn memory(&self) -> Option<&MemoryBlock> {
        Some(self)
    }
}

#[cfg(test)]
#[path = "../tests/engram_tests.rs"]
mod engram_tests;
