//! Engram 条件记忆（§5.8.2）：哈希 N-gram、`O(1)` 槽、ρ 稀疏分配。
//!
//! 静态 N-gram 模式查表，不是第四前向、不替代块表（E12 KV）、不替代 CSA/HCA、不是有证书上界
//! 压缩（§5.8.2 禁止越界）。每 token 取槽 `K(N−1)`，与表长 `M` 无关 ⇒ 查找 `O(1)`（数学验证 40）。

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
pub fn allocate(rho: f32, p_sparse: f32) -> (f32, f32) {
    debug_assert!((0.0..=1.0).contains(&rho));
    (rho * p_sparse, (1.0 - rho) * p_sparse)
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
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3usize;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// 一张 N-gram 嵌入表：素数长、`d` 维嵌入，哈希槽 `O(1)` 读写。
#[derive(Debug, Clone)]
pub struct Table {
    len: usize,
    dim: usize,
    data: Vec<f32>,
}

impl Table {
    /// 构造：`len` 自动提升到素数，`dim` 为嵌入维。
    pub fn new(len: usize, dim: usize) -> Self {
        assert!(dim > 0, "嵌入维必须非零");
        let len = next_prime(len);
        Self {
            len,
            dim,
            data: vec![0.0; len * dim],
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
    pub fn get(&self, ngram: &[u32], seed: u64) -> &[f32] {
        let z = self.slot(ngram, seed);
        &self.data[z * self.dim..(z + 1) * self.dim]
    }

    /// `O(1)` 写表：把 `value` 写入该槽（覆盖）。
    pub fn set(&mut self, ngram: &[u32], seed: u64, value: &[f32]) {
        assert_eq!(value.len(), self.dim, "嵌入长度必须等于 dim");
        let z = self.slot(ngram, seed);
        self.data[z * self.dim..(z + 1) * self.dim].copy_from_slice(value);
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
        let mut t = Table::new(10, 4);
        assert!(t.len() >= 10 && is_prime(t.len()));
        let g = [3u32, 1, 4];
        t.set(&g, 7, &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(t.get(&g, 7), &[0.1, 0.2, 0.3, 0.4]);
    }

    /// 槽数与表长无关（O(1) 查找，§5.8.2）。
    #[test]
    fn slot_is_independent_of_table_length() {
        // `slots` 只依赖 N、K，不接收表长 M；不同表长对同 N-gram 的槽下标由 φ mod M 决定，
        // 单次查找复杂度 O(N)（哈希 N-gram），与 M 无关。
        let small = Table::new(16, 1);
        let large = Table::new(16_384, 1);
        let g = [5u32, 9];
        assert!(small.slot(&g, 0) < small.len());
        assert!(large.slot(&g, 0) < large.len());
    }
}
