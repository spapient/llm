//! Engram 条件记忆（§5.8.2）：哈希 N-gram、`O(1)` 槽、ρ 稀疏分配。
//!
//! 静态 N-gram 模式查表，不是第四前向、不替代块表（E12 KV）、不替代 CSA/HCA、不是有证书上界
//! 压缩（§5.8.2 禁止越界）。每 token 取槽 `K(N−1)`，与表长 `M` 无关 ⇒ 查找 `O(1)`（数学验证 40）。
//!
//! [`Bypass`] 是这套公式的执行侧：它是 `LmHead` 之前的那条旁路——取尾部 N-gram、查一个槽、
//! 按门控强度加进隐藏向量。表本身只读，写侧（训练/蒸馏）不在本模块。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

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
/// 值类型泛型化到 [`Scalar`]（默认 `f32` 前向、`f64` 验证对拍），与 [`crate::csa`] / [`crate::mix`]
/// 同一精度闭集，不因精度复制两份表实现。
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
        (phi(ngram, seed) % self.len as u64) as usize
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 数学验证 40：N=3、K=8 ⇒ 16 槽/token。
    #[test]
    fn slots_match_n3_k8() {
        assert_eq!(slots(3, 8), 16);
        assert_eq!(slots(4, 8), 24);
        assert_eq!(slots(2, 1), 1);
    }

    /// φ 是确定性的：同输入必同输出。
    #[test]
    fn phi_is_deterministic() {
        let g = [7u32, 11, 13];
        assert_eq!(phi(&g, 0), phi(&g, 0));
        assert_ne!(phi(&g, 0), phi(&g, 1), "不同种子应散开");
    }

    /// ρ 分配构成单位剖分：ρ·p + (1−ρ)·p = p。
    #[test]
    fn allocate_partitions_sparse() {
        for rho in [0.0f32, 0.25, 0.743, 1.0] {
            let (moe, engram) = allocate(rho, 100.0);
            assert!((moe + engram - 100.0).abs() < 1e-6);
            assert!((moe - rho * 100.0).abs() < 1e-6);
            assert!((engram - (1.0 - rho) * 100.0).abs() < 1e-6);
        }
    }

    /// 表长提升到素数。
    #[test]
    fn next_prime_is_prime_and_rounds_up() {
        assert_eq!(next_prime(1), 2);
        assert_eq!(next_prime(10), 11);
        assert_eq!(next_prime(11), 11);
        assert!(is_prime(next_prime(9973)));
    }

    /// 同 N-gram 落到同槽，读写往返一致。
    #[test]
    fn table_get_set_roundtrip() {
        let mut t = Table::<f32>::new(10, 4);
        assert!(t.len() >= 10 && is_prime(t.len()));
        let g = [3u32, 1, 4];
        t.set(&g, 7, &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(t.get(&g, 7), &[0.1, 0.2, 0.3, 0.4]);
    }

    /// f64 表复用同一份实现（泛型不复制）。
    #[test]
    fn table_f64_reuses_impl() {
        let mut t = Table::<f64>::new(10, 2);
        let g = [1u32, 1, 2];
        t.set(&g, 3, &[0.5, 1.5]);
        assert_eq!(t.get(&g, 3), &[0.5, 1.5]);
    }

    /// 槽数与表长无关（O(1) 查找，§5.8.2）。
    #[test]
    fn slot_is_independent_of_table_length() {
        // `slots` 只依赖 N、K，不接收表长 M；不同表长对同 N-gram 的槽下标由 φ mod M 决定，
        // 单次查找复杂度 O(N)（哈希 N-gram），与 M 无关。
        let small = Table::<f32>::new(16, 1);
        let large = Table::<f32>::new(16_384, 1);
        let g = [5u32, 9];
        assert!(small.slot(&g, 0) < small.len());
        assert!(large.slot(&g, 0) < large.len());
    }

    /// 历史不足 `order` 时不加记忆（不是加零向量）。
    #[test]
    fn a_short_history_adds_nothing() {
        let bypass = Bypass::new(Table::<f32>::new(11, 2), 7, 3, 1.0).expect("合法旁路");
        let mut hidden = [1.0f32, 2.0];
        assert!(!bypass.apply(&mut hidden, &[1, 2]).expect("合法宽度"));
        assert_eq!(hidden, [1.0, 2.0]);
        assert!(bypass.recall(&[1, 2]).is_none());
        assert!(bypass.recall(&[]).is_none());
    }

    /// 记忆按 `gate` 缩放进残差；查的是**尾部** N-gram，历史里更早的 token 不算。
    #[test]
    fn gate_scales_the_memory_residual() {
        let mut table = Table::<f32>::new(11, 2);
        table.set(&[1, 2, 3], 7, &[2.0, -4.0]);
        let half = Bypass::new(table.clone(), 7, 3, 0.5).expect("合法旁路");
        let mut hidden = [1.0f32, 2.0];
        assert!(half.apply(&mut hidden, &[9, 1, 2, 3]).expect("合法宽度"));
        assert_eq!(hidden, [2.0, 0.0], "9 不在尾部三元组里");
        let off = Bypass::new(table, 7, 3, 0.0).expect("合法旁路");
        let mut untouched = [1.0f32, 2.0];
        assert!(off.apply(&mut untouched, &[1, 2, 3]).expect("合法宽度"));
        assert_eq!(untouched, [1.0, 2.0], "gate = 0 时残差为零");
    }

    /// 阶 < 2 与嵌入宽度不符都在入口失败。
    #[test]
    fn a_bypass_rejects_bad_order_and_width() {
        let bad = Bypass::new(Table::<f32>::new(11, 2), 7, 1, 1.0);
        assert_eq!(bad.unwrap_err().code(), Code::Invalid, "1-gram 没有上下文");
        let bypass = Bypass::new(Table::<f32>::new(11, 2), 7, 2, 0.25).expect("合法旁路");
        assert_eq!(bypass.order(), 2);
        assert_eq!(bypass.gate(), 0.25);
        assert_eq!(bypass.table().dim(), 2);
        assert_eq!(
            bypass.apply(&mut [0.0f32; 3], &[1, 2]).unwrap_err().code(),
            Code::Invalid
        );
    }
}
