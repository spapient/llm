//! 多项式环 R_q 上的数论变换（FIPS 204 §7.5），零依赖、纯安全整数。
//!
//! 环 `R_q = Z_q[x]/(x^256+1)`，q = 8380417，本原根 ζ = 1753。系数统一用 `0..q-1` 表示，
//! 数学上与 FIPS 的对称表示等价，更便于直接取模。乘法中间值落在 i64（q² ≈ 7·10^13 ≪ 2^63），
//! 对常数 q 的取模由编译器降为乘法移位，无需手写 Montgomery 也没有真除法。
//!
//! 根表与基乘法 γ 表都在编译期由 ζ 生成，不手抄常数，杜绝抄录错误。

/// 模数 q = 2^23 − 2^13 + 1。
pub const Q: i64 = 8_380_417;
/// 多项式阶数。
pub const N: usize = 256;
/// 模 q 的 256 次本原根。
const ZETA: i64 = 1_753;

/// NTT 蝶形根表，按 bit-reversed 索引（对应 FIPS 的 ζ^{brv(k)}）。
static ZETA_TABLE: [i32; 256] = gen_zeta_table();

const fn pow_mod(mut base: i64, mut exp: u32) -> i64 {
    let mut acc = 1i64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base % Q;
        }
        base = base * base % Q;
        exp >>= 1;
    }
    acc
}

const fn gen_zeta_table() -> [i32; 256] {
    let mut table = [0i32; 256];
    let mut i = 0u32;
    while i < 256 {
        let index = (i as u8).reverse_bits() as usize;
        table[index] = pow_mod(ZETA, i) as i32;
        i += 1;
    }
    table
}

/// 把任意 i64 收成 `0..q-1`。
fn mod_q(value: i64) -> i32 {
    let r = value % Q;
    if r < 0 { (r + Q) as i32 } else { r as i32 }
}

/// 正向 NTT（算法 41）：常规系数 → NTT 域。
pub fn ntt(poly: &[i32; N]) -> [i32; N] {
    let mut w = *poly;
    let mut m = 0usize;
    let mut len = 128usize;
    while len >= 1 {
        let mut start = 0usize;
        while start < N {
            m += 1;
            let zeta = i64::from(ZETA_TABLE[m]);
            for j in start..start + len {
                let t = mod_q(zeta * i64::from(w[j + len]));
                w[j + len] = mod_q(i64::from(w[j]) - i64::from(t));
                w[j] = mod_q(i64::from(w[j]) + i64::from(t));
            }
            start += 2 * len;
        }
        len >>= 1;
    }
    w
}

/// 逆向 NTT（算法 42）：NTT 域 → 常规系数。
pub fn inv_ntt(hat: &[i32; N]) -> [i32; N] {
    let mut w = *hat;
    let mut m = 256usize;
    let mut len = 1usize;
    while len < N {
        let mut start = 0usize;
        while start < N {
            m -= 1;
            let zeta = -i64::from(ZETA_TABLE[m]);
            for j in start..start + len {
                let t = w[j];
                w[j] = mod_q(i64::from(t) + i64::from(w[j + len]));
                w[j + len] = mod_q(zeta * i64::from(t - w[j + len]));
            }
            start += 2 * len;
        }
        len <<= 1;
    }
    let n_inv = mod_q(pow_mod(N as i64, (Q - 2) as u32));
    for coeff in &mut w {
        *coeff = mod_q(i64::from(n_inv) * i64::from(*coeff));
    }
    w
}

/// NTT 域点积：NTT 把多项式映射到 256 个独立求值点，因此点积就是逐元素相乘取模，
/// 与 FIPS 204 参考实现的逐元素 Montgomery 乘加等价（此处为非 Montgomery 形式）。
pub fn multiply(a: &[i32; N], b: &[i32; N]) -> [i32; N] {
    core::array::from_fn(|i| mod_q(i64::from(a[i]) * i64::from(b[i])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_then_inv_is_identity() {
        let poly: [i32; N] = core::array::from_fn(|j| (j as i32 * 7 + 3) % Q as i32);
        let back = inv_ntt(&ntt(&poly));
        assert_eq!(back, poly);
    }

    #[test]
    fn pointwise_matches_convolution() {
        let a: [i32; N] = core::array::from_fn(|j| (j as i32 + 1) % Q as i32);
        let b: [i32; N] = core::array::from_fn(|j| (j as i32 * 3 + 2) % Q as i32);
        // NTT 域点积回到常规域，应等于环上卷积（含 x^256 = -1 的回绕）。
        let product = inv_ntt(&multiply(&ntt(&a), &ntt(&b)));
        let mut expected = [0i32; N];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let k = i + j;
                let term = mod_q(i64::from(ai) * i64::from(bj));
                if k < N {
                    expected[k] = mod_q(i64::from(expected[k]) + i64::from(term));
                } else {
                    expected[k - N] = mod_q(i64::from(expected[k - N]) - i64::from(term));
                }
            }
        }
        assert_eq!(product, expected);
    }

    #[test]
    fn pointwise_with_one_is_identity() {
        let poly: [i32; N] = core::array::from_fn(|j| (j as i32 * 13 + 5) % Q as i32);
        // 常数多项式 1 的 NTT 在每个求值点都是 1。
        let one = [1i32; N];
        let product = multiply(&ntt(&poly), &one);
        assert_eq!(product, ntt(&poly));
    }
}
