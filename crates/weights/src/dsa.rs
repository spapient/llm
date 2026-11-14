//! ML-DSA 编码与采样（FIPS 204 §7.1、§7.3），零依赖、纯安全。
//!
//! 多项式统一为 `[i32; 256]`、系数 `0..q-1`；位打包只有一个 `bit_pack`（带范围 a,b），简单打包
//! 是 a=0 的特例，XOF 由 `Xof` 按需消费。

use crate::keccak;
use crate::ntt::{self, N};

/// 多项式：256 个系数。
type Poly = [i32; N];
/// 多项式向量。
type PolyVec = Vec<Poly>;

/// 拒绝采样：3 字节生成一个 `0..q-1` 系数，超出范围（含 b2 高位）返回 None。
fn coeff_from_three_bytes(b: [u8; 3]) -> Option<i32> {
    let z = (i32::from(b[2] & 0x7f) << 16) | (i32::from(b[1]) << 8) | i32::from(b[0]);
    (z < ntt::Q as i32).then_some(z)
}

/// 拒绝采样：半字节配合 η∈{2,4} 生成 `−η..η` 系数，非法半字节返回 None。
fn coeff_from_half_byte(eta: i32, b: u8) -> Option<i32> {
    match eta {
        2 if b < 15 => Some(2 - i32::from(b % 5)),
        4 if b < 9 => Some(4 - i32::from(b)),
        _ => None,
    }
}

/// 位宽：表示 `0..value` 所需位数。
fn bit_len(value: i32) -> usize {
    32 - value.leading_zeros() as usize
}

/// 通用位打包：系数在 `[−a, b]`，编码 `b - coeff`（a=0 时即 coeff 本身），每个值占 bit_len(a+b) 位。
pub fn bit_pack(w: &Poly, a: i32, b: i32) -> Vec<u8> {
    let width = bit_len(a + b);
    let mut out = vec![0u8; N * width / 8];
    let (mut temp, mut bit_index, mut byte_index) = (0u32, 0usize, 0usize);
    for &coeff in &w[..] {
        let value = if a == 0 { coeff } else { b - coeff };
        temp |= (value as u32) << bit_index;
        bit_index += width;
        while bit_index > 7 {
            out[byte_index] = temp as u8;
            temp >>= 8;
            byte_index += 1;
            bit_index -= 8;
        }
    }
    out
}

/// 位解包：`bit_pack` 的逆，返回系数；a=0 时直接取值，否则还原 `b - value`。
pub fn bit_unpack(v: &[u8], a: i32, b: i32) -> Poly {
    let width = bit_len(a + b);
    let mask = (1u32 << width) - 1;
    let mut w = [0i32; N];
    let (mut temp, mut bit_index, mut index) = (0i32, 0usize, 0usize);
    for &byte in v {
        temp |= i32::from(byte) << bit_index;
        bit_index += 8;
        while bit_index >= width {
            let value = (temp as u32 & mask) as i32;
            w[index] = if a == 0 { value } else { b - value };
            bit_index -= width;
            temp >>= width;
            index += 1;
        }
    }
    w
}

/// 可游标消费的 XOF：一次性按 `rate` 生成足够输出，逐块读取。
struct Xof {
    bytes: Vec<u8>,
    at: usize,
}

impl Xof {
    fn shake(use_shake256: bool, seed: &[u8], bound: usize) -> Self {
        // 拒绝采样有界：每系数最多消耗少量字节，取宽松上界，越界即说明参数异常。
        let bytes = if use_shake256 {
            keccak::shake256(seed, bound)
        } else {
            keccak::shake128(seed, bound)
        };
        Self { bytes, at: 0 }
    }

    fn take(&mut self, n: usize) -> &[u8] {
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        slice
    }
}

/// 均匀拒绝采样一个 NTT 域多项式（算法 30），seed 已含位置后缀。
pub fn rej_ntt_poly(seed: &[u8]) -> Poly {
    let bytes = keccak::shake128(seed, N * 8);
    let (chunks, _) = bytes.as_chunks::<3>();
    let coeffs = chunks.iter().filter_map(|c| coeff_from_three_bytes(*c));
    let mut w = [0i32; N];
    for (slot, coeff) in w.iter_mut().zip(coeffs) {
        *slot = coeff;
    }
    w
}

/// ExpandA（算法 32）：从 32 字节种子生成 k×l 个 NTT 域多项式，种子后缀绑定 (s,r)。
pub fn expand_a<const K: usize, const L: usize>(rho: &[u8; 32]) -> [[Poly; L]; K] {
    core::array::from_fn(|r| {
        core::array::from_fn(|s| rej_ntt_poly(&[&rho[..], &[s as u8], &[r as u8]].concat()))
    })
}

/// ExpandS（算法 33）：生成 s1（l 个）与 s2（k 个），系数在 `[−η, η]`，后缀为 16 位索引。
pub fn expand_s<const K: usize, const L: usize>(
    eta: i32,
    rho: &[u8; 64],
) -> ([Poly; L], [Poly; K]) {
    let poly = |i: usize| rej_bounded_poly(eta, &[&rho[..], &(i as u16).to_le_bytes()].concat());
    let s1: [Poly; L] = core::array::from_fn(poly);
    let s2 = core::array::from_fn(|r| poly(r + L));
    (s1, s2)
}

/// 中心取模：返回 `−alpha/2 < r ≤ alpha/2` 内的代表（FIPS 204 §2.3）。
fn mod_pm(m: i32, alpha: i32) -> i32 {
    let r = m.rem_euclid(alpha);
    if r > alpha / 2 { r - alpha } else { r }
}

/// Power2Round（算法 35）：`r ≡ r1·2^d + r0`，r0 中心取模 2^d。
pub fn power2round(r: i32, d: u32) -> (i32, i32) {
    let r0 = mod_pm(r, 1 << d);
    ((r - r0) >> d, r0)
}

/// Decompose（算法 36）：`r ≡ r1·(2γ2) + r0`，处理 r1 顶格的边界特例。
pub fn decompose(gamma2: i32, r: i32) -> (i32, i32) {
    let rp = r.rem_euclid(ntt::Q as i32);
    let r0 = mod_pm(rp, 2 * gamma2);
    if rp - r0 == ntt::Q as i32 - 1 {
        (0, r0 - 1)
    } else {
        ((rp - r0) / (2 * gamma2), r0)
    }
}

/// MakeHint（算法 39）：r 加上 z 是否改变高位。
pub fn make_hint(gamma2: i32, z: i32, r: i32) -> bool {
    decompose(gamma2, r).0 != decompose(gamma2, r + z).0
}

/// UseHint（算法 40）：按提示位还原调整后的高位，m 为高位箱数。
pub fn use_hint(gamma2: i32, m: i32, h: i32, r: i32) -> i32 {
    let (r1, r0) = decompose(gamma2, r);
    if h == 0 {
        return r1;
    }
    (if r0 > 0 { r1 + 1 } else { r1 - 1 }).rem_euclid(m)
}

/// ExpandMask（算法 34）：生成 l 个系数在 `[−γ1+1, γ1]` 的掩码多项式，种子后缀绑定 (mu+r)。
pub fn expand_mask<const L: usize>(gamma1: i32, rho: &[u8; 64], mu: u16) -> [Poly; L] {
    // 宽度与 bit_unpack 的 BitLen(a+b) 一致（a=γ1−1,b=γ1）。
    let width = bit_len(2 * gamma1 - 1);
    core::array::from_fn(|r| {
        let seed = [&rho[..], &(mu + r as u16).to_le_bytes()].concat();
        bit_unpack(&keccak::shake256(&seed, 32 * width), gamma1 - 1, gamma1)
    })
}

/// HintBitPack（算法 20）：记录每个多项式中 1 的位置，末尾 k 个字节为各多项式的累计索引。
pub fn hint_pack<const K: usize>(h: &[Poly; K]) -> Vec<u8> {
    hint_pack_dyn(&h[..])
}

/// HintBitUnpack（算法 21）：`hint_pack` 的逆，校验累计索引单调、位置严格递增，非法输入返回 None。
pub fn hint_unpack<const K: usize>(omega: i32, y: &[u8]) -> Option<[Poly; K]> {
    let v = hint_unpack_dyn(K, omega, y)?;
    Some(core::array::from_fn(|i| v[i]))
}

/// SampleInBall（算法 29）：生成系数 ∈{−1,0,1}、Hamming 重量为 τ 的挑战多项式。
pub fn sample_in_ball(tau: usize, rho: &[u8]) -> Poly {
    let mut c = [0i32; N];
    let mut xof = Xof::shake(true, rho, N * 4);
    let sign = xof.take(8).to_vec();
    for i in (N - tau)..N {
        let mut j = xof.take(1)[0] as usize;
        while j > i {
            j = xof.take(1)[0] as usize;
        }
        c[i] = c[j];
        let bit = (sign[(i + tau - N) / 8] >> ((i + tau - N) & 7)) & 1;
        c[j] = 1 - 2 * i32::from(bit);
    }
    c
}

/// ML-DSA 参数集（FIPS 204 Table 1），三流程只依赖这一组闭集，D5 复用时换实例即可。
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// 向量维度 k。
    pub k: usize,
    /// 向量维度 l。
    pub l: usize,
    /// 小系数界 η。
    pub eta: i32,
    /// 挑战稀疏度 τ（=β）。
    pub tau: usize,
    /// 掩码界 γ1。
    pub gamma1: i32,
    /// 低位界 γ2。
    pub gamma2: i32,
    /// 提示位上限 ω。
    pub omega: i32,
    /// 安全强度 λ。
    pub lambda: usize,
}

/// 私钥运行时表示：签名所需的全部组件（FIPS 204 算法 7）。
#[derive(Debug, Clone)]
pub struct SecretKey {
    /// 公开种子 rho。
    pub rho: [u8; 32],
    /// 签名随机种子 K。
    pub seed: [u8; 32],
    /// 公钥指纹 tr。
    pub tr: [u8; 64],
    /// 小系数向量 s1（l 个）。
    pub s1: PolyVec,
    /// 小系数向量 s2（k 个）。
    pub s2: PolyVec,
    /// 舍入余项 t0（k 个）。
    pub t0: PolyVec,
}

/// 签名尝试上限（FIPS 204 附录 C），正确密钥触达概率约 2^−256。
const MAX_ATTEMPTS: u16 = 814;

/// 一个参数集：`(k, l)` 与 FIPS 204 Table 1 的 `(η, τ, γ1 的位数, γ2 的除数, ω, λ)`。
const fn params(shape: (usize, usize), rest: (i32, usize, u32, i32, i32, usize)) -> Params {
    Params {
        k: shape.0,
        l: shape.1,
        eta: rest.0,
        tau: rest.1,
        gamma1: 1 << rest.2,
        gamma2: (ntt::Q as i32 - 1) / rest.3,
        omega: rest.4,
        lambda: rest.5,
    }
}

/// ML-DSA-44 参数（NIST 安全类别 2）：k=4,l=4,η=2,τ=39,γ1=2^17,γ2=(q-1)/88,ω=80,λ=128。
pub const ML_DSA_44: Params = params((4, 4), (2, 39, 17, 88, 80, 128));

/// ML-DSA-65 参数（NIST 安全类别 3）：k=6,l=5,η=4,τ=49,γ1=2^19,γ2=(q-1)/32,ω=55,λ=192。
pub const ML_DSA_65: Params = params((6, 5), (4, 49, 19, 32, 55, 192));

/// ML-DSA-87 参数（NIST 安全类别 5）：k=8,l=7,η=2,τ=60,γ1=2^19,γ2=(q-1)/32,ω=75,λ=256。
pub const ML_DSA_87: Params = params((8, 7), (2, 60, 19, 32, 75, 256));

/// 系数取模到 `0..q-1`（处理加减后的负值）。
fn reduce(x: i32) -> i32 {
    x.rem_euclid(ntt::Q as i32)
}

/// 把 `0..q-1` 非负表示转为中心表示：常规域统一用中心表示，转换只发生在 NTT 桥梁处。
fn center(poly: Poly) -> Poly {
    let half = ntt::Q as i32 / 2;
    poly.map(|v| if v > half { v - ntt::Q as i32 } else { v })
}

/// inv_ntt 后立即转中心表示，保证常规域表示一致。
fn inv_center(poly: &Poly) -> Poly {
    center(ntt::inv_ntt(poly))
}

/// 两个常规域多项式逐系数相加，输出中心表示。
pub fn add(a: &Poly, b: &Poly) -> Poly {
    core::array::from_fn(|i| mod_pm(a[i] + b[i], ntt::Q as i32))
}

/// 两个常规域多项式逐系数相减，输出中心表示。
pub fn sub(a: &Poly, b: &Poly) -> Poly {
    core::array::from_fn(|i| mod_pm(a[i] - b[i], ntt::Q as i32))
}

/// NTT 域矩阵乘向量：每个结果多项式是 l 个逐元素乘积之和（NTT 域线性，求和后仍在该域）。
pub fn mat_vec<const K: usize, const L: usize>(a: &[[Poly; L]; K], v: &[Poly; L]) -> [Poly; K] {
    core::array::from_fn(|r| {
        let mut acc = [0i32; N];
        for s in 0..L {
            let prod = ntt::multiply(&a[r][s], &v[s]);
            // NTT 域保持非负表示，逐系数求和后 reduce（不调用常规域的 add）。
            for i in 0..N {
                acc[i] = reduce(acc[i] + prod[i]);
            }
        }
        acc
    })
}

/// 解出的四件套：`ρ`、`t1`、`c` 与提示向量——`open` 与验签共用这一份形状。
type Opened = (Vec<u8>, PolyVec, PolyVec, [u8; 32], PolyVec);

impl Params {
    /// t1 系数上界：Power2Round 后高位占 `bit_len(q−1)−d` 位（d=13），即 10 位上界 1023。
    fn t1_bound() -> i32 {
        (1 << (bit_len(ntt::Q as i32 - 1) - 13)) - 1
    }

    /// 公钥字节数：`rho` 加 `k` 个 `t1` 的定长打包；长度由参数集唯一确定，因此是**校验**。
    #[must_use]
    pub fn pk_len(&self) -> usize {
        32 + self.k * (N * bit_len(Self::t1_bound()) / 8)
    }

    /// 签名的最小字节数：提示段长度可变（`total + k`，`total ≤ omega`），所以这里是下界。
    #[must_use]
    pub fn sig_len_min(&self) -> usize {
        self.lambda / 4 + self.l * (N * (1 + bit_len(self.gamma1 - 1)) / 8) + self.k
    }

    /// 公钥编码（算法 22）：rho 后接 k 个 t1 的简单打包（10 位/系数）。
    pub fn pk_encode(&self, rho: &[u8; 32], t1: &[Poly]) -> Vec<u8> {
        let mut out = rho.to_vec();
        let bound = Self::t1_bound();
        out.extend(t1[..self.k].iter().flat_map(|p| bit_pack(p, 0, bound)));
        out
    }

    /// 公钥解码（算法 23）：返回 rho 与 t1；截断的公钥是文件里能出现的东西，不是编程错误。
    pub fn pk_decode(&self, pk: &[u8]) -> Option<([u8; 32], PolyVec)> {
        if pk.len() != self.pk_len() {
            return None;
        }
        let bound = Self::t1_bound();
        let step = N * bit_len(bound) / 8;
        let rho = copy_of(&pk[..32]);
        let t1 = (0..self.k)
            .map(|i| bit_unpack(&pk[32 + i * step..32 + (i + 1) * step], 0, bound))
            .collect();
        Some((rho, t1))
    }

    /// 公钥与签名一起解码：这是「截断的侧车」唯一的判定点，公钥长度由 `pk_decode` 校验。
    fn open(&self, pk: &[u8], sig: &[u8]) -> Option<Opened> {
        let (c_tilde, z, h) = self.sig_decode(sig)?;
        let (rho, t1) = self.pk_decode(pk)?;
        Some((c_tilde, z, h, rho, t1))
    }

    /// 签名编码（算法 26）：c_tilde、l 个 z 的位打包、提示位打包。
    pub fn sig_encode(&self, c_tilde: &[u8], z: &[Poly], h: &[Poly]) -> Vec<u8> {
        let mut out = c_tilde.to_vec();
        let pack = |p: &Poly| bit_pack(p, self.gamma1 - 1, self.gamma1);
        out.extend(z[..self.l].iter().flat_map(pack));
        out.extend(hint_pack_dyn(&h[..self.k]));
        out
    }

    /// 签名解码（算法 27）：返回 c_tilde、z 与提示向量；长度不足或提示非法时为 `None`。
    pub fn sig_decode(&self, sig: &[u8]) -> Option<(Vec<u8>, PolyVec, PolyVec)> {
        if sig.len() < self.sig_len_min() {
            return None;
        }
        let c_len = self.lambda / 4;
        let z_step = N * (1 + bit_len(self.gamma1 - 1)) / 8;
        let z_end = c_len + self.l * z_step;
        let z: PolyVec = (0..self.l)
            .map(|i| {
                bit_unpack(
                    &sig[c_len + i * z_step..c_len + (i + 1) * z_step],
                    self.gamma1 - 1,
                    self.gamma1,
                )
            })
            .collect();
        let h = hint_unpack_dyn(self.k, self.omega, &sig[z_end..])?;
        Some((sig[..c_len].to_vec(), z, h))
    }

    /// 密钥生成（算法 6）：从 32 字节种子确定性生成公钥字节与私钥；K,L 为编译期维度。
    pub fn key_gen_internal<const K: usize, const L: usize>(
        &self,
        xi: &[u8; 32],
    ) -> (Vec<u8>, SecretKey) {
        let h = keccak::shake256(&[&xi[..], &[K as u8], &[L as u8]].concat(), 128);
        let rho = copy_of(&h[..32]);
        let rho_prime = copy_of(&h[32..96]);
        let cap_k = copy_of(&h[96..128]);

        let (s1, s2) = expand_s::<K, L>(self.eta, &rho_prime);
        let a = expand_a::<K, L>(&rho);
        let s1_hat: [Poly; L] = core::array::from_fn(|i| ntt::ntt(&s1[i]));
        let t_hat = mat_vec(&a, &s1_hat);
        // t 保持非负表示（Power2Round 要求输入 0..q-1）：inv_ntt 非负，加 s2 用 reduce 归一。
        let t_conv: PolyVec = (0..K).map(|i| ntt::inv_ntt(&t_hat[i])).collect();
        let t: PolyVec = t_conv
            .iter()
            .zip(&s2)
            .map(|(a, b)| core::array::from_fn(|i| reduce(a[i] + b[i])))
            .collect();

        let (t1, t0) = power2round_vec(&t);
        let pk = self.pk_encode(&rho, &t1);
        let tr = copy_of(&keccak::shake256(&pk, 64));
        let sk = SecretKey {
            rho,
            seed: cap_k,
            tr,
            s1: s1.to_vec(),
            s2: s2.to_vec(),
            t0: t0.to_vec(),
        };
        (pk, sk)
    }

    /// 签名（算法 7）：rnd 全 0 为确定性签名，否则混入每消息随机；达到上限（≈2^−256）返回 None。
    pub fn sign_internal<const K: usize, const L: usize>(
        &self,
        sk: &SecretKey,
        mu: &[u8; 64],
        rnd: &[u8; 32],
    ) -> Option<Vec<u8>> {
        let rho_prime = copy_of(&keccak::shake256(&[&sk.seed[..], rnd, mu].concat(), 64));
        let a = expand_a::<K, L>(&sk.rho);
        let hat = NttSecret::of(sk);
        (0..MAX_ATTEMPTS).find_map(|i| {
            let attempt = self.sign_attempt::<K, L>(&a, mu, &rho_prime, i * L as u16, &hat);
            attempt.map(|(c_tilde, z, h)| self.sig_encode(&c_tilde, &z, &h))
        })
    }

    /// 验证（算法 8）：解码签名、重算承诺哈希并比对，全部通过才为真。
    pub fn verify_internal<const K: usize, const L: usize>(
        &self,
        pk: &[u8],
        mu: &[u8; 64],
        sig: &[u8],
    ) -> bool {
        let Some((c_tilde, z, h, rho, t1)) = self.open(pk, sig) else {
            return false;
        };
        if infinity_norm(&z) >= self.gamma1 - self.tau as i32 {
            return false;
        }
        let a = expand_a::<K, L>(&rho);
        let z_hat: PolyVec = z.iter().map(ntt::ntt).collect();
        let mut w_approx = mat_vec(&a, &core::array::from_fn(|i| z_hat[i]));
        let c = sample_in_ball(self.tau, &c_tilde);
        let c_hat = ntt::ntt(&c);
        let t1_hat: PolyVec = t1.iter().map(ntt::ntt).collect();
        for i in 0..K {
            let shift = ntt::multiply(&c_hat, &t1_hat[i]);
            // t1·2^d：在 NTT 域乘 2^d，用 i64 避免 i32 左移溢出，保持 NTT 域非负。
            let scaled = shift.map(|v| (i64::from(v) << 13).rem_euclid(ntt::Q) as i32);
            w_approx[i] = core::array::from_fn(|j| reduce(w_approx[i][j] - scaled[j]));
        }
        // NTT 域算完后必须 inv_ntt 回常规中心表示，UseHint 作用于常规域系数。
        let w_regular: PolyVec = w_approx.iter().map(inv_center).collect();
        let w1 = use_hint_vec(&h, &w_regular, self.gamma2);
        let mut c_tilde_p = mu.to_vec();
        c_tilde_p.extend(w1_encode(self.gamma2, &w1));
        let c_tilde_p = keccak::shake256(&c_tilde_p, self.lambda / 4);
        c_tilde == c_tilde_p
    }
}

/// 把切片拷进定长数组（长度由类型给出）。
fn copy_of<const M: usize>(bytes: &[u8]) -> [u8; M] {
    let mut out = [0u8; M];
    out.copy_from_slice(bytes);
    out
}

/// 按提示向量还原 w1：逐系数调用 use_hint。
fn use_hint_vec(h: &[Poly], w: &[Poly], gamma2: i32) -> PolyVec {
    let m = (ntt::Q as i32 - 1) / (2 * gamma2);
    h.iter()
        .zip(w)
        .map(|(hp, wp)| core::array::from_fn(|i| use_hint(gamma2, m, hp[i], wp[i])))
        .collect()
}

/// 向量逐系数相加（Vec 版）。
fn add_vec(a: &[Poly], b: &[Poly]) -> PolyVec {
    a.iter().zip(b).map(|(x, y)| add(x, y)).collect()
}

/// 对向量中每个多项式做 Power2Round，返回 (t1, t0)。
fn power2round_vec(r: &[Poly]) -> (PolyVec, PolyVec) {
    r.iter()
        .map(|poly| {
            let pairs: [(i32, i32); N] = core::array::from_fn(|i| power2round(poly[i], 13));
            (pairs.map(|pair| pair.0), pairs.map(|pair| pair.1))
        })
        .unzip()
}

/// 无限范数：全部系数绝对值的最大值。
fn infinity_norm(polys: &[Poly]) -> i32 {
    polys.iter().flatten().map(|c| c.abs()).max().unwrap_or(0)
}

/// w1 编码（算法 28）：每个系数 0..15 占 4 位，简单打包后拼接。
fn w1_encode(gamma2: i32, w1: &[Poly]) -> Vec<u8> {
    // 上位箱数 m=(q-1)/(2γ2)：44 集需 6 位，65/87 集需 4 位。
    let m = (ntt::Q as i32 - 1) / (2 * gamma2);
    w1.iter().flat_map(|p| bit_pack(p, 0, m - 1)).collect()
}

/// 私钥的 NTT 域预算，循环外算一次，避免每次尝试重复变换。
struct NttSecret {
    s1: PolyVec,
    s2: PolyVec,
    t0: PolyVec,
}

impl NttSecret {
    /// 把 s1/s2/t0 一次性变换到 NTT 域。
    fn of(key: &SecretKey) -> Self {
        let hat = |v: &PolyVec| v.iter().map(ntt::ntt).collect();
        Self {
            s1: hat(&key.s1),
            s2: hat(&key.s2),
            t0: hat(&key.t0),
        }
    }
}

/// 对向量逐系数取 Decompose 的高位 r1（w1）。
fn high_bits(polys: &[Poly], gamma2: i32) -> PolyVec {
    polys
        .iter()
        .map(|p| p.map(|v| decompose(gamma2, v).0))
        .collect()
}

/// 对向量逐系数取 Decompose 的低位 r0。
fn low_bits(polys: &[Poly], gamma2: i32) -> PolyVec {
    polys
        .iter()
        .map(|p| p.map(|v| decompose(gamma2, v).1))
        .collect()
}

impl Params {
    /// 单次签名尝试：通过全部范数与提示计数检查才返回 (c_tilde, z, h)，否则 None。
    fn sign_attempt<const K: usize, const L: usize>(
        &self,
        a: &[[Poly; L]; K],
        mu: &[u8; 64],
        rho_prime: &[u8; 64],
        kappa: u16,
        hat: &NttSecret,
    ) -> Option<(Vec<u8>, PolyVec, PolyVec)> {
        let y = expand_mask::<L>(self.gamma1, rho_prime, kappa);
        let y_hat: [Poly; L] = core::array::from_fn(|i| ntt::ntt(&y[i]));
        let w: PolyVec = (0..K).map(|i| inv_center(&mat_vec(a, &y_hat)[i])).collect();
        let w1 = high_bits(&w, self.gamma2);
        let mut c_tilde = mu.to_vec();
        c_tilde.extend(w1_encode(self.gamma2, &w1));
        let c_tilde = keccak::shake256(&c_tilde, self.lambda / 4);
        let c_hat = ntt::ntt(&sample_in_ball(self.tau, &c_tilde));

        let cs1 = inv_vec(&hat.s1, &c_hat);
        let cs2 = inv_vec(&hat.s2, &c_hat);
        let z = add_vec(&y, &cs1);
        let w_cs2 = sub_vec(&w, &cs2);
        let r0 = low_bits(&w_cs2, self.gamma2);
        let bound = self.gamma1 - self.tau as i32;
        if infinity_norm(&z) >= bound || infinity_norm(&r0) >= self.gamma2 - self.tau as i32 {
            return None;
        }
        let ct0 = inv_vec(&hat.t0, &c_hat);
        let inside = add_vec(&w_cs2, &ct0);
        let h = make_hint_vec(&ct0, &inside, self.gamma2);
        if infinity_norm(&ct0) >= self.gamma2 || hint_count(&h) > self.omega {
            return None;
        }
        Some((c_tilde, z, h))
    }
}

/// NTT 域逐元素乘 c_hat 后回到常规域。
fn inv_vec(hat: &[Poly], c_hat: &Poly) -> PolyVec {
    hat.iter()
        .map(|p| inv_center(&ntt::multiply(c_hat, p)))
        .collect()
}

/// 向量逐系数相减。
fn sub_vec(a: &[Poly], b: &[Poly]) -> PolyVec {
    a.iter().zip(b).map(|(x, y)| sub(x, y)).collect()
}

/// 逐系数构造提示向量（MakeHint），输入为 -ct0 与 w-cs2+ct0。
fn make_hint_vec(ct0: &[Poly], inside: &[Poly], gamma2: i32) -> PolyVec {
    ct0.iter()
        .zip(inside)
        .map(|(c, r)| core::array::from_fn(|i| i32::from(make_hint(gamma2, -c[i], r[i]))))
        .collect()
}

/// 提示向量中 1 的总数。
fn hint_count(h: &[Poly]) -> i32 {
    h.iter().flat_map(|p| p.iter()).filter(|&&c| c == 1).sum()
}

/// 动态维度的提示打包（hint_pack 的 Vec 版），供 Params 方法使用。
fn hint_pack_dyn(h: &[Poly]) -> Vec<u8> {
    let total = hint_count(h) as usize;
    let mut out = vec![0u8; total + h.len()];
    let mut index = 0usize;
    for (poly, stop) in h.iter().zip(total..) {
        for (j, coeff) in poly.iter().enumerate() {
            if *coeff == 1 {
                out[index] = j as u8;
                index += 1;
            }
        }
        out[stop] = index as u8;
    }
    out
}

/// 动态维度的提示解包（hint_unpack 的 Vec 版），供 Params 方法使用。
fn hint_unpack_dyn(k: usize, omega: i32, y: &[u8]) -> Option<PolyVec> {
    let total = y.len().checked_sub(k)?;
    let mut h = vec![[0i32; N]; k];
    let mut index = 0usize;
    for (i, &stop) in y[total..].iter().enumerate() {
        index = mark_positions(y, &mut h[i], (index, stop as usize), omega as usize)?;
    }
    let rest = y.get(index..total)?;
    rest.iter().all(|&b| b == 0).then_some(h)
}

/// 写出一段严格递增且不越 ω 的提示位置，返回新的已消耗下标。
fn mark_positions(y: &[u8], poly: &mut Poly, span: (usize, usize), omega: usize) -> Option<usize> {
    let (from, stop) = span;
    if !(from..=omega).contains(&stop) {
        return None;
    }
    for at in from..stop {
        if at > from && y[at - 1] >= y[at] {
            return None;
        }
        poly[y[at] as usize] = 1;
    }
    Some(stop)
}

/// 小系数拒绝采样（算法 31），系数在 `[−η, η]`。
pub fn rej_bounded_poly(eta: i32, seed: &[u8]) -> Poly {
    let mut xof = Xof::shake(true, seed, N * 8);
    let mut w = [0i32; N];
    let mut j = 0usize;
    while j < N {
        let z = xof.take(1)[0];
        for half in [z & 0x0f, z >> 4] {
            if j < N
                && let Some(coeff) = coeff_from_half_byte(eta, half)
            {
                w[j] = coeff;
                j += 1;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M'（算法 2）→ μ：域分隔、空上下文长度、消息体，再对 tr‖M' 取 SHAKE256。
    fn mu_of(sk: &SecretKey, msg: &[u8]) -> [u8; 64] {
        let mut input = sk.tr.to_vec();
        input.extend([0x00, 0x00]);
        input.extend(msg);
        copy_of(&keccak::shake256(&input, 64))
    }

    #[test]
    fn a_hint_count_beyond_the_segment_is_rejected() {
        // 提示段是 `total + k`，`total` 由签名总长反推；这里 `total = 0`，k 个 stop 计数必然越界。
        let k = ML_DSA_44.k;
        let mut y = vec![0u8; k];
        for (i, slot) in y.iter_mut().enumerate() {
            *slot = (i + 1) as u8;
        }
        assert_eq!(
            hint_unpack_dyn(k, ML_DSA_44.omega, &y),
            None,
            "提示计数越界必须判无效"
        );
    }

    #[test]
    fn coeff_decoders_follow_the_rejection_rules() {
        assert_eq!(
            coeff_from_three_bytes([0x12, 0x34, 0x56]),
            Some(0x0056_3412)
        );
        assert_eq!(coeff_from_three_bytes([0x01, 0xe0, 0x7f]), None);
        assert_eq!(coeff_from_half_byte(2, 3), Some(-1));
        assert_eq!(coeff_from_half_byte(4, 8), Some(-4));
        assert_eq!(coeff_from_half_byte(4, 10), None);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let simple: Poly = core::array::from_fn(|j| (j as i32 * 11) % 44);
        assert_eq!(bit_unpack(&bit_pack(&simple, 0, 44), 0, 44), simple);
        let signed: Poly = core::array::from_fn(|j| (j as i32 % 9) - 4);
        assert_eq!(bit_unpack(&bit_pack(&signed, 4, 4), 4, 4), signed);
    }

    #[test]
    fn sampled_polys_stay_in_range() {
        let bounded = rej_bounded_poly(4, &[7u8; 40]);
        assert!(bounded.iter().all(|&c| (-4..=4).contains(&c)));
        let uniform = rej_ntt_poly(&[3u8; 40]);
        assert!(uniform.iter().all(|&c| (0..ntt::Q as i32).contains(&c)));
    }

    #[test]
    fn decompose_reconstructs_and_round_trips_hint() {
        let q = ntt::Q as i32;
        let gamma2 = (q - 1) / 32;
        for r in (0..q).step_by(997) {
            let (r1, r0) = decompose(gamma2, r);
            assert_eq!((r1 * 2 * gamma2 + r0).rem_euclid(q), r);
            for h in 0..=1 {
                let got = use_hint(gamma2, 16, h, r);
                assert!((0..16).contains(&got));
            }
        }
        // 边界特例：r1 顶格时归零、r0 减一。
        assert_eq!(decompose(gamma2, 8_118_529), (0, -261_888));
    }

    #[test]
    fn power2round_reconstructs() {
        for r in (0..ntt::Q as i32).step_by(1009) {
            let (r1, r0) = power2round(r, 13);
            assert_eq!((r1 << 13) + r0, r);
            assert!(r0.abs() <= 1 << 12);
        }
    }

    #[test]
    fn hint_pack_roundtrip_and_rejects_bad_input() {
        let mut h = [[0i32; N]; 6];
        h[0][5] = 1;
        h[0][200] = 1;
        h[3][10] = 1;
        let packed = hint_pack(&h);
        assert_eq!(hint_unpack::<6>(55, &packed), Some(h));
        assert_eq!(hint_unpack::<6>(1, &packed), None);
    }

    #[test]
    fn mask_is_in_gamma1_range() {
        let rho64: [u8; 64] = core::array::from_fn(|j| j as u8);
        let y = expand_mask::<5>(1 << 19, &rho64, 0);
        assert!(
            y.iter()
                .all(|p| p.iter().all(|&c| (-(1 << 19) + 1..=1 << 19).contains(&c)))
        );
    }

    #[test]
    fn matches_python_reference() {
        // 参考由 /tmp/dsa_ref.py 生成；文件缺失时跳过，使该测试不依赖外部环境。
        let Ok(text) = std::fs::read_to_string("/tmp/dsa_ref.json") else {
            return;
        };
        let ref_value = crate::json::parse(&text).expect("参考 JSON");
        let get = |key: &str| -> Vec<i32> {
            let field = ref_value.get(key).expect("字段");
            let values = field.array().expect("数组");
            values
                .iter()
                .map(|v| v.integer().expect("整数") as i32)
                .collect()
        };

        let rho32: [u8; 32] = core::array::from_fn(|j| j as u8);
        let a = expand_a::<6, 5>(&rho32);
        assert_eq!(&a[0][0], get("a00").as_slice());
        assert_eq!(&a[5][4], get("a54").as_slice());

        let rho64: [u8; 64] = core::array::from_fn(|j| j as u8);
        let (s1, s2) = expand_s::<6, 5>(4, &rho64);
        assert_eq!(&s1[0], get("s10").as_slice());
        assert_eq!(&s2[5], get("s25").as_slice());

        let c = sample_in_ball(49, &rho64);
        assert_eq!(&c, get("c").as_slice());
    }

    #[test]
    fn sign_then_verify_roundtrip_and_rejects_tampering() {
        let xi: [u8; 32] = core::array::from_fn(|j| (j * 7 + 1) as u8);
        let (pk, sk) = ML_DSA_65.key_gen_internal::<6, 5>(&xi);
        let mu = mu_of(&sk, b"weight-container-manifest");
        let sig = ML_DSA_65
            .sign_internal::<6, 5>(&sk, &mu, &[0u8; 32])
            .expect("签名成功");
        assert!(ML_DSA_65.verify_internal::<6, 5>(&pk, &mu, &sig));

        // 篡改消息代表后验证必须失败。
        let mut bad_mu = mu;
        bad_mu[0] ^= 1;
        assert!(!ML_DSA_65.verify_internal::<6, 5>(&pk, &bad_mu, &sig));
    }

    /// 单参数集与权威纯 Python 参考逐字节对拍；参考文件缺失时跳过。
    fn parity<const K: usize, const L: usize>(params: &Params, name: &str) {
        let (Ok(py_pk), Ok(py_sig), Ok(py_mu)) = (
            std::fs::read(format!("/tmp/py_pk_{name}.bin")),
            std::fs::read(format!("/tmp/py_sig_{name}.bin")),
            std::fs::read(format!("/tmp/py_mu_{name}.bin")),
        ) else {
            return;
        };
        let xi: [u8; 32] = core::array::from_fn(|j| (j * 7 + 1) as u8);
        let (pk, sk) = params.key_gen_internal::<K, L>(&xi);
        let mu = mu_of(&sk, b"weight-container-manifest");
        let sig = params
            .sign_internal::<K, L>(&sk, &mu, &[0u8; 32])
            .expect("签名成功");
        assert_eq!(pk, py_pk);
        assert_eq!(mu.to_vec(), py_mu);
        assert_eq!(sig, py_sig);
        assert!(params.verify_internal::<K, L>(&py_pk, &mu, &py_sig));
    }

    #[test]
    fn matches_python_full_reference() {
        parity::<4, 4>(&ML_DSA_44, "44");
        parity::<6, 5>(&ML_DSA_65, "65");
        parity::<8, 7>(&ML_DSA_87, "87");
    }

    /// 单参数集往返：正确签名通过、篡改拦截。
    fn roundtrip<const K: usize, const L: usize>(params: &Params, seed_byte: u8) {
        let xi: [u8; 32] = core::array::from_fn(|j| (j as u8).wrapping_add(seed_byte));
        let (pk, sk) = params.key_gen_internal::<K, L>(&xi);
        let mu = mu_of(&sk, b"all-parameter-sets");
        let sig = params
            .sign_internal::<K, L>(&sk, &mu, &[0u8; 32])
            .expect("签名成功");
        assert!(params.verify_internal::<K, L>(&pk, &mu, &sig));
        let mut bad_mu = mu;
        bad_mu[0] ^= 1;
        assert!(!params.verify_internal::<K, L>(&pk, &bad_mu, &sig));
    }

    #[test]
    fn all_parameter_sets_roundtrip() {
        roundtrip::<4, 4>(&ML_DSA_44, 1);
        roundtrip::<6, 5>(&ML_DSA_65, 2);
        roundtrip::<8, 7>(&ML_DSA_87, 3);
    }
}
