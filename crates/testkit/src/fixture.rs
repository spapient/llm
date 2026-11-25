//! 确定性输入与格式夹具：样本、伪随机、字节打包和位级 CRC 参考。

/// 按 safetensors 布局拼一份完整文件：8 字节小端头长 + 头部 JSON + 依序拼接的数据段。
#[must_use]
pub fn safetensors_bytes(json: &str, data: &[(&str, Vec<u8>)]) -> Vec<u8> {
    for (name, _) in data {
        debug_assert!(json.contains(name), "头部 JSON 必须声明数据段 `{name}`");
    }
    let payload = data.iter().map(|(_, chunk)| chunk.len()).sum::<usize>();
    let mut bytes = Vec::with_capacity(8 + json.len() + payload);
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    for (_, chunk) in data {
        bytes.extend_from_slice(chunk);
    }
    bytes
}

/// 把 `bits` 位有符号码按低 `bits` 位打包：`zero = 2^(bits-1)` 平移，低 `bits` 位落在字节内。
///
/// # Panics
///
/// `bits` 不是 1、2、4、8 时 panic；这些是唯一整字节划分且能容纳有符号平移的位宽。
#[must_use]
pub fn pack_codes(bits: usize, values: &[i8]) -> Vec<u8> {
    assert!(matches!(bits, 1 | 2 | 4 | 8), "码位宽必须是 1/2/4/8");
    let per_byte = 8 / bits;
    let zero = 1i16 << (bits - 1);
    let mask = u8::MAX >> (8 - bits);
    values
        .chunks(per_byte)
        .map(|group| {
            (0..per_byte).fold(0u8, |byte, slot| {
                let code = (i16::from(group.get(slot).copied().unwrap_or(0)) + zero) as u8 & mask;
                byte | (code << (slot * bits))
            })
        })
        .collect()
}

/// 长度不匹配必须拒绝：长输出、短输出两个用例各驱动一次 `run`，`run` 返回该次是否被拒绝。
pub fn assert_rejects_length_mismatch<F>(mut run: F)
where
    F: FnMut(&[f32], &mut [f32]) -> bool,
{
    let mut long = vec![0.0; 3];
    assert!(run(&[1.0, 2.0], &mut long), "输出比输入长必须拒绝");
    let mut short = vec![0.0; 1];
    assert!(run(&[1.0, 2.0, 3.0], &mut short), "输出比输入短必须拒绝");
}

/// CRC-32 反射多项式（IEEE 802.3 的 `0x04C1_1DB7` 反射形式）。
const CRC_POLY: u32 = 0xEDB8_8320;

/// CRC-32 的逐字节参考定义。
#[must_use]
pub fn crc32_bitwise(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        // 每一位都消费上一位的反馈，存在串行依赖，不能用独立批处理替代。
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ CRC_POLY
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// 把某个 crate 的查表 CRC-32 实现与位级参考对拍：递增序列、空串与标准 check 值。
pub fn assert_table_matches_bitwise(crc32: impl Fn(&[u8]) -> u32) {
    let data: Vec<u8> = (0u8..=200).collect();
    crate::cases!(|name, (input, want): (&[u8], u32)| {
        assert_eq!(crc32(input), want, "{name}");
    };
    "递增序列" => (&data, crc32_bitwise(&data)),
    "空串" => (&[], crc32_bitwise(&[])),
    "标准 check" => (b"123456789", 0xCBF4_3926),
    );
}

/// 确定性 PRNG（xorshift32）：测试共享，不给依赖表加 `rand`，同时让失败可复现。
#[derive(Clone, Debug)]
pub struct Rng {
    state: u32,
}

impl Rng {
    /// 固定种子。
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// 取 `[0, bound)` 内的确定性值；零界按一界处理。
    pub fn below(&mut self, bound: u32) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state % bound.max(1)
    }

    /// `±[0.5, 2]` 的确定性值：远离 0，避免平方类算子的差分为 0 造成无谓抵消。
    pub fn value(&mut self) -> f64 {
        let magnitude = 0.5 + 1.5 * f64::from(self.below(1_000)) / 1_000.0;
        match self.below(2) {
            0 => magnitude,
            _ => -magnitude,
        }
    }

    /// `[0, 1)` 内的确定性 f32。
    pub fn next_f32(&mut self) -> f32 {
        self.below(1_000_000) as f32 / 1_000_000.0
    }

    /// `[-1, 1)` 内的确定性 f32：千分网格上取值，有符号夹具共用一份。
    pub fn signed_unit(&mut self) -> f32 {
        self.below(2_001) as f32 / 1_000.0 - 1.0
    }
}

/// 基础确定性序列：`i·2654435761 + salt·40503`，映射由调用方配置。
fn sample_word(idx: usize, salt: u32) -> u32 {
    (idx as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(salt.wrapping_mul(40_503))
}

/// 确定性样本序列；同一 `(len, salt)` 永远同一批值，`T` 由映射决定。
#[must_use]
pub fn sample_with<T>(len: usize, salt: u32, map: impl Fn(u32) -> T) -> Vec<T> {
    let mut values = Vec::with_capacity(len);
    values.extend((0..len).map(|idx| map(sample_word(idx, salt))));
    values
}

/// 确定性测试样本：`sample(len, salt)` 生成 `[-1, 1)` 的 f32。
#[must_use]
pub fn sample(len: usize, salt: u32) -> Vec<f32> {
    sample_with(len, salt, |word| ((word >> 8) % 2048) as f32 / 1024.0 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let mut left = Rng::new(1234);
        let mut right = Rng::new(1234);
        for _ in 0..16 {
            let value = left.below(10);
            assert_eq!(value, right.below(10), "同种子序列一致");
            assert!(value < 10, "结果落在界内");
        }
        assert_eq!(Rng::new(7).below(1), 0, "bound=1 恒为 0");
        assert_eq!(Rng::new(7).below(0), 0, "bound=0 不产生除零");
    }

    #[test]
    fn rng_value_stays_away_from_zero_and_is_signable() {
        let mut rng = Rng::new(42);
        for _ in 0..16 {
            let value = rng.value();
            assert!(value.abs() >= 0.5 && value.abs() <= 2.0);
        }
    }

    #[test]
    fn rng_next_f32_is_in_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..16 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn rng_signed_unit_covers_both_signs_inside_the_band() {
        let mut rng = Rng::new(9);
        let draws: Vec<f32> = (0..64).map(|_| rng.signed_unit()).collect();
        assert!(draws.iter().all(|value| (-1.0..1.0).contains(value)));
        assert!(draws.iter().any(|value| *value < 0.0));
        assert!(draws.iter().any(|value| *value >= 0.0));
    }

    #[test]
    fn sample_is_reproducible_and_inside_the_band() {
        assert_eq!(sample(9, 3), sample(9, 3), "同 (len, salt) 必须同一批值");
        assert_ne!(sample(9, 3), sample(9, 4), "salt 必须真的参与计算");
        assert!(
            sample(64, 1)
                .iter()
                .all(|value| (-1.0..1.0).contains(value))
        );
        assert_eq!(sample(0, 7), Vec::<f32>::new(), "零长批是空切片");
    }

    #[test]
    fn sample_with_shares_the_base_prng_and_supports_custom_mappings() {
        assert_eq!(
            sample(16, 5),
            sample_with(16, 5, |word| ((word >> 8) % 2048) as f32 / 1024.0 - 1.0)
        );
        let scaled = sample_with(8, 1, |word| ((word >> 8) as f32) / 256.0 - 32.0);
        assert_eq!(
            scaled,
            sample_with(8, 1, |word| ((word >> 8) as f32) / 256.0 - 32.0)
        );
        let as_i8 = sample_with(8, 1, |word| ((word % 251) as i32 - 125) as i8);
        assert!(as_i8.iter().all(|value| (-125..=125).contains(value)));
        assert_eq!(sample_with::<f32>(0, 1, |_| 0.0), Vec::<f32>::new());
    }
}
