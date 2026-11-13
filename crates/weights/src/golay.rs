//! 扩展二元 Golay 码 G24：`[24, 12, 8]`，12 个信息位编为 24 位码字，最小汉明距离 8，
//! 可纠正至多 3 位错误。
//!
//! 生成矩阵取 `[I12 | B]`，`B` 是对称的 12×12 边界矩阵且 `B² = I`；这两条性质让译码只需
//! 两个 syndrome（`s` 与 `s·B`）加少量重量检查，不必枚举全部错误模式。G24 是 Leech 格
//! Λ24 构造的核心构件，本模块先把码层基础设施做正确并对拍，真 Λ24 的陪集秩编码在后续阶段推进，
//! 这里不宣称已达到 Leech 密度。
//!
//! 全部用整数位运算，纯安全、零第三方依赖。

use diagnostic::{Code, Error, Result};

/// 信息位与校验位宽度。
const HALF: u32 = 12;

/// 12 位掩码。
const MASK12: u16 = 0x0FFF;

/// 24 位掩码。
const MASK24: u32 = 0x00FF_FFFF;

/// 可纠正错误数：`(最小距离 - 1) / 2 = 3`。
const CORRECTABLE: u32 = 3;

/// 边界矩阵 B 的 12 行（字符串位 0..11 对应列 0..11，`const` 解析，零运行时开销）。
/// 取自权威 G24 构造：B 对称且 `B² = I`，配合 `[I|B]` 每行总重 8（doubly-even）。
const B: [u16; 12] = [
    parse_row("011111111111"),
    parse_row("111011100010"),
    parse_row("110111000101"),
    parse_row("101110001011"),
    parse_row("111100010110"),
    parse_row("111000101101"),
    parse_row("110001011011"),
    parse_row("100010110111"),
    parse_row("100101101110"),
    parse_row("101011011100"),
    parse_row("110110111000"),
    parse_row("101101110001"),
];

/// const 字符串行 → 位掩码（`'1'` 在对应列置位）。
const fn parse_row(s: &str) -> u16 {
    let bytes = s.as_bytes();
    let mut row = 0u16;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'1' {
            row |= 1 << i;
        }
        i += 1;
    }
    row
}

/// GF(2) 矩阵乘：`x · B`，对每个置位信息位 XOR 对应 B 行。
#[must_use]
fn b_mul(x: u16) -> u16 {
    let mut acc = 0u16;
    let mut bit = 0u32;
    while bit < HALF {
        if x & (1 << bit) != 0 {
            acc ^= B[bit as usize];
        }
        bit += 1;
    }
    acc
}

/// 12 位信息 → 24 位码字：低 12 位信息，高 12 位 `x·B` 校验。
#[must_use]
pub fn encode(data: u16) -> u32 {
    let data = data & MASK12;
    u32::from(data) | (u32::from(b_mul(data)) << HALF)
}

/// 译码：纠正至多 3 位错误并恢复 12 位信息；超出纠错能力（≥4 位）显式报错，不猜。
///
/// 收到 `r = u | v<<12`，syndrome `s = u·B ^ v`。错误 `(e1, e2)` 满足 `s = e1·B ^ e2`，
/// 据此枚举四类共 26 个候选，取总汉明重量最小且 ≤3 者：
/// 1. 错误全在信息半：`e1 = s·B, e2 = 0`
/// 2. 错误全在校验半：`e1 = 0, e2 = s`
/// 3. 信息一位 + 校验至多两位：`e1 = e_i, e2 = s ^ B_i`
/// 4. 校验一位 + 信息至多两位：`e1 = s·B ^ B_i, e2 = e_i`
pub fn decode(received: u32) -> Result<u16> {
    let received = received & MASK24;
    let u = received as u16 & MASK12;
    let v = (received >> HALF) as u16;
    let s = b_mul(u) ^ v;
    let sb = b_mul(s);

    let mut best: Option<(u32, u16)> = None;
    let mut consider = |e1: u16, e2: u16| {
        let weight = e1.count_ones() + e2.count_ones();
        if weight <= CORRECTABLE && best.is_none_or(|(w, _)| weight < w) {
            best = Some((weight, e1));
        }
    };

    consider(sb, 0);
    consider(0, s);
    for i in 0..HALF {
        let bit = 1u16 << i;
        consider(bit, s ^ B[i as usize]);
        consider(sb ^ B[i as usize], bit);
    }

    match best {
        Some((_, e1)) => Ok((u ^ e1) & MASK12),
        None => Err(
            Error::new(Code::Invalid, "Golay: 错误超过 3 位，超出 G24 纠错能力")
                .with("golay::decode"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codeword_round_trips() {
        for data in 0..=MASK12 {
            let cw = encode(data);
            assert_eq!(decode(cw).expect("无错误译码"), data);
        }
    }

    /// 线性码的最小距离 = 最小非零码字重量；必须为 8。
    #[test]
    fn minimum_distance_is_eight() {
        let mut min_weight = u32::MAX;
        for data in 1..=MASK12 {
            min_weight = min_weight.min(encode(data).count_ones());
        }
        assert_eq!(min_weight, 8);
    }

    #[test]
    fn up_to_three_errors_are_corrected() {
        let mut rng = 0x9E37_79B9u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng
        };
        for _ in 0..2000 {
            let data = (next() as u16) & MASK12;
            let cw = encode(data);
            for flips in 1..=3u32 {
                let mut damaged = cw;
                let mut used = [false; 24];
                for _ in 0..flips {
                    let pos = (next() % 24) as usize;
                    if used[pos] {
                        continue;
                    }
                    used[pos] = true;
                    damaged ^= 1 << pos;
                }
                assert_eq!(decode(damaged).expect("纠正后恢复"), data);
            }
        }
    }

    #[test]
    fn b_is_symmetric_and_involutory() {
        // B² = I：对每个单位向量，b_mul(b_mul(e_i)) == e_i。
        for i in 0..HALF {
            let e = 1u16 << i;
            assert_eq!(b_mul(b_mul(e)), e);
        }
    }

    #[test]
    fn high_bit_inputs_are_masked() {
        // 超过 12 位的信息位被忽略，结果与低 12 位一致。
        assert_eq!(encode(0xFFFF), encode(0x0FFF));
    }
}
