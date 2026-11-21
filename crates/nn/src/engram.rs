//! Engram 条件记忆（§5.8.2）：哈希 N-gram、`O(1)` 槽、ρ 稀疏分配。
//!
//! 静态 N-gram 模式查表，不是第四前向、不替代块表（E12 KV）、不替代 CSA/HCA、不是有证书上界
//! 压缩（§5.8.2 禁止越界）。每 token 取槽 `K(N−1)`，与表长 `M` 无关 ⇒ 查找 `O(1)`（数学验证 40）。
//!
//! [`Bypass`] 是这套公式的执行侧：它是 `LmHead` 之前的那条旁路——取尾部 N-gram、查一个槽、
//! 按门控强度加进隐藏向量。表本身只读，写侧（训练/蒸馏）不在本模块。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

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
///
/// 表内查表（[`Table::slot`]）与装配侧每步回填的槽表都走它——同一份 N-gram 只有一种落槽方式，
/// 因此「旁路算出的槽」与「训练写表时的槽」不可能是两个。
fn bucket(ngram: &[u32], seed: u64, buckets: usize) -> usize {
    (phi(ngram, seed) % buckets as u64) as usize
}

/// 尾部 `order` 元 N-gram 的槽号；历史不足 `order`（或几何退化）时 `None`。
///
/// `None` 与 [`Bypass::apply`] 返回 `false` 是同一句判据：**不猜记忆**。装配侧据此把这一行的门控
/// 压成零，而不是拿第 0 槽顶替——顶替出来的向量在残差里就是一个真实的偏移。
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
///
/// `ρ=1` 纯 MoE；公开最优 `ρ*≈75%–80%`，Engram-27B 取 `ρ=74.3%`（§5.8.2）。
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
///
/// 值类型泛型化到 [`Scalar`]（默认 `f32` 前向、`f64` 验证对拍），与 `csa` / `mix` 同一精度闭集，
/// 不因精度复制两份表实现。
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
///
/// 表只读（写侧归训练/蒸馏），旁路只回答一个问题：这一步该给主干的末尾隐藏向量加多少记忆。
/// 历史不足 `order` 时**不加**，而不是拿零向量补位——补位会把"没有证据"伪装成"证据是零"，
/// 而零向量在残差里就是一个真实的偏移。
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

    /// N-gram 阶 `N`。
    pub fn order(&self) -> usize {
        self.order
    }

    /// 残差强度。
    pub fn gate(&self) -> S {
        self.gate
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
    ///
    /// 返回是否**有记忆可加**（历史足够）；`gate = 0` 时命中也算加过，只是加的是零。
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
///
/// 与 [`Bypass`] 是**同一套公式的两个位置**，不是两份实现：槽函数同为 [`slot_of`]，门控同样是
/// `hidden + gate · table[slot]`，历史不足 `order` 时同样不加。区别只在"谁算"——`Bypass` 在 `nn`
/// 内一次算完，本结构只声明装配几何（槽数、行宽、阶、种子、门控强度），算术交给装配侧三个算子
/// 节点（查表 → 门控缩放 → 加回主链），因为服务侧的执行单元是算子图而不是逐层函数。
///
/// 表本身是这一层的驻留权重条目 `[buckets, width]`，与词表嵌入同为查表角色（表行按槽随机访问，
/// 豁免权重衰减比按行归一更贴切）。
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
            vec![
                Self::EMBED.to_string(),
                Self::SCALE.to_string(),
                Self::RESIDUAL.to_string(),
            ],
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

    /// 隐藏宽度（本层的输入与输出宽度）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// N-gram 阶 `N`。
    pub fn order(&self) -> usize {
        self.order
    }

    /// 槽哈希种子（与写表时同一把）。
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// 表槽数 `M`：驻留权重条目为 `[buckets, width]`。
    pub fn buckets(&self) -> usize {
        self.buckets
    }

    /// 门控强度 `γ`。
    pub fn gate(&self) -> f32 {
        self.gate
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
mod tests;
