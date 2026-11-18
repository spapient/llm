//! 2/4/8-bit 对称训练后量化（PTQ）：把 f32 权重按组量化并打包，以及它的反解。
//!
//! 与 [`super::quant`] 的分工：`quant` 是**磁盘方案描述符与分组推断**（读已有量化文件的形状与
//! 标度，不做数值）；本模块是**把 f32 真正算成打包码**的正向算法与其逆变换，供离线 PTQ 与
//! round-trip 验证使用。换算算法按 ADR 0018 留在 `weights`，不进入设备内核注册表。
//!
//! 三个格式的运行时事实（位宽、对称零点、每字节码数、正码上限）与 `op::quant::Row` 的
//! `I2`/`I4`/`I8` **逐项同源**（低码在前的打包位序、对称零点一致）。本模块不自建到 `op` 的依赖边：
//! 离线换算按 ADR 0018 留在 `weights`，不需要内核层；两处常量是同一事实的离线视角与内核视角，
//! 若未来要硬性单一事实源，可加 `weights → op` 的 dev 依赖用交叉测试锁死（需同步更新架构边表）。
//!
//! 对称口径：`scale = max_abs / max_code`，`signed = round(x/scale)` 夹取到 `[-max_code, max_code]`，
//! 码的线格式与反量化公式 `y = scale * (code - zero)` 完全一致。i2/i4 是**偏移码**（`code = signed + zero`，
//! 无符号 nibble），i8 是**二补码字节**（`zero = 0`，`code = signed` 直接按 `i8` 解释）。

use diagnostic::{Code, Error, Result};

/// 一档对称量化格式的运行时事实。
///
/// 四个字段都能回答“这一档怎么量化/怎么打包”，与 `op::quant::Row`（`I2`/`I4`/`I8`）逐项一致：
/// `bits` 对应 `Row::width`，`zero` 对应 `Row::zero`，`cpb` 对应 `Row::cpb`，`max_code` 是
/// 对称码域的正码上限（`2^(bits-1) - 1`，i8 取 127 避开 -128 的非对称）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scheme {
    /// 元素位宽：2 / 4 / 8。
    pub bits: u8,
    /// 对称零点：偏移码（i2/i4）的 `signed + zero` 平移量；i8 为 0。
    pub zero: i32,
    /// 每字节码数：`8 / bits`；逐元素（i8）为 1。
    pub cpb: usize,
    /// 正码上限：`scale = max_abs / max_code` 的分母。
    pub max_code: f32,
}

impl Scheme {
    /// 码域大小：`2^bits`。
    pub const fn domain(self) -> i32 {
        1 << self.bits
    }

    /// 每个码占的位数：`8 / cpb`。
    pub const fn width(self) -> usize {
        8 / self.cpb
    }
}

/// 2 位打包：4 码/字节，零点 2，码表 `{-2,-1,0,1}`，正码上限 1。
pub const I2: Scheme = Scheme {
    bits: 2,
    zero: 2,
    cpb: 4,
    max_code: 1.0,
};

/// 4 位打包：2 码/字节，零点 8，码表 `{-8..=7}`，正码上限 7。
pub const I4: Scheme = Scheme {
    bits: 4,
    zero: 8,
    cpb: 2,
    max_code: 7.0,
};

/// 8 位逐元素：1 码/字节，零点 0（二补码 `i8`），正码上限 127。
pub const I8: Scheme = Scheme {
    bits: 8,
    zero: 0,
    cpb: 1,
    max_code: 127.0,
};

/// 按组对称量化：把 `x` 按 `group` 个元素一组，每组独立 `scale = max_abs / max_code`，
/// 逐元素 `signed = round(x/scale)` 夹取到 `[-max_code, max_code]`，再按低码在前打包。
///
/// 返回 `(packed, scales)`：`packed` 长度 `ceil(len / cpb)`，`scales` 长度 `ceil(len / group)`。
/// 全零组取 `scale = 1.0`（码全为对称零点 `zero`），反解仍精确归零，同时避开零 scale 被反解侧拒绝。
///
/// # Errors
///
/// `group` 为 0、输入含非有限值，或 `max_code` 非法（0）时返回说明。
pub fn quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
    validate_quantize(scheme, group, x)?;

    let groups = x.len().div_ceil(group);
    let mut packed = vec![0u8; x.len().div_ceil(scheme.cpb)];
    let mut scales = Vec::with_capacity(groups);

    for g in 0..groups {
        let lo = g * group;
        let hi = (lo + group).min(x.len());
        let slice = &x[lo..hi];
        let scale = group_scale(scheme, slice);
        scales.push(scale);

        for (i, &v) in slice.iter().enumerate() {
            let signed = (v / scale).round() as i32;
            let signed = signed.clamp(-(scheme.max_code as i32), scheme.max_code as i32);
            pack_into(scheme, &mut packed, lo + i, (signed + scheme.zero) as u8);
        }
    }

    Ok((packed, scales))
}

/// 量化参数校验：分组、正码上限、输入有限性。
fn validate_quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<()> {
    if group == 0 {
        return Err(invalid("量化分组必须大于 0"));
    }
    if scheme.max_code <= 0.0 {
        return Err(invalid("正码上限必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(invalid("输入必须是有限 f32"));
    }
    Ok(())
}

/// 一组元素的对称标度：全零组取 1（码全为对称零点，反解仍精确归零），否则 `max_abs / max_code`。
fn group_scale(scheme: Scheme, slice: &[f32]) -> f32 {
    let max_abs = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / scheme.max_code
    }
}

/// 按组反解：`y = scale * (code - zero)`，与反量化公式一致，供 round-trip 对拍。
///
/// `packed` 长度必须为 `ceil(len / cpb)`，`scales` 长度必须为 `ceil(len / group)`，`out` 长度
/// 为元素数。i2/i4 的码是无符号 nibble（减 `zero` 得符号值），i8 的码是二补码字节（`zero = 0`）。
///
/// # Errors
///
/// 输出长度不为 0 但分组/字节数不匹配时返回说明。
pub fn dequantize(
    scheme: Scheme,
    packed: &[u8],
    scales: &[f32],
    group: usize,
    out: &mut [f32],
) -> Result<()> {
    validate_dequantize(scheme, packed, scales, group, out)?;

    // width == 8 时 `1u8 << 8` 会溢出：用 u16 求掩码再截断，`255` 对 i8 是整字节。
    let mask = ((1u16 << scheme.width()) - 1) as u8;
    for (i, slot) in out.iter_mut().enumerate() {
        let byte = packed[i / scheme.cpb];
        let shift = (i % scheme.cpb) * scheme.width();
        let field = (byte >> shift) & mask;
        *slot = scales[i / group] * decode_code(scheme, field);
    }
    Ok(())
}

/// 反解参数校验：分组、打包字节数与输出长度、标度条数与分组数。
fn validate_dequantize(
    scheme: Scheme,
    packed: &[u8],
    scales: &[f32],
    group: usize,
    out: &[f32],
) -> Result<()> {
    if group == 0 {
        return Err(invalid("反解分组必须大于 0"));
    }
    if packed.len() != out.len().div_ceil(scheme.cpb) {
        return Err(invalid("打包字节数与输出长度不一致"));
    }
    if scales.len() != out.len().div_ceil(group) {
        return Err(invalid("标度条数与分组数不一致"));
    }
    Ok(())
}

/// 码 → 符号值：i8（`zero == 0`）按二补码解释，i2/i4 是无符号码减零点。
fn decode_code(scheme: Scheme, field: u8) -> f32 {
    if scheme.zero == 0 {
        field as i8 as f32
    } else {
        field as f32 - scheme.zero as f32
    }
}

/// 把第 `index` 个码写入对应字节的对应位段（低码在前）；字节已按 `vec![0]` 初始化，按位或累加。
fn pack_into(scheme: Scheme, packed: &mut [u8], index: usize, code: u8) {
    let shift = (index % scheme.cpb) * scheme.width();
    packed[index / scheme.cpb] |= code << shift;
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("weights::ptq")
}

#[cfg(test)]
mod tests {
    use super::{I2, I4, I8, dequantize, quantize};

    /// 打包位序：i4 低 nibble 在前，`[7, -7]` → 码 `[15, 1]` → 字节 `0x1F`。
    #[test]
    fn i4_packs_low_nibble_first() {
        let (packed, scales) = quantize(I4, 32, &[7.0, -7.0]).expect("合法输入");
        assert_eq!(packed, vec![0x1F]);
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 2];
        dequantize(I4, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [7.0, -7.0]);
    }

    /// 打包位序：i2 每字节 4 码，`[1, -1, 0, 0]` → 码 `[3, 1, 2, 2]` → 字节 `0xA7`。
    #[test]
    fn i2_packs_four_codes_per_byte() {
        let (packed, scales) = quantize(I2, 32, &[1.0, -1.0, 0.0, 0.0]).expect("合法输入");
        assert_eq!(packed, vec![0xA7]);
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 4];
        dequantize(I2, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [1.0, -1.0, 0.0, 0.0]);
    }

    /// i8 逐元素二补码：`[127, -127]` → 字节 `[127, 129]`，反解归位。
    #[test]
    fn i8_stores_twos_complement() {
        let (packed, scales) = quantize(I8, 32, &[127.0, -127.0]).expect("合法输入");
        assert_eq!(packed, vec![127, 129]);
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 2];
        dequantize(I8, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [127.0, -127.0]);
    }

    /// 量化误差有界：每元素误差不超过本组半码距（`scale / 2`），且余弦相似度随位宽提升。
    #[test]
    fn round_trip_error_is_bounded() {
        let x: Vec<f32> = (0..256)
            .map(|i| ((i as f32).sin() * 3.7) + (i as f32 % 7.0) - 3.0)
            .collect();
        // i2 只有 {-1,0,1} 三档码，余弦阈值放低；i4/i8 随位宽逐级收紧。
        for (scheme, min_cosine) in [(I2, 0.80), (I4, 0.98), (I8, 0.999)] {
            let (packed, scales) = quantize(scheme, 32, &x).expect("合法输入");
            let mut out = vec![0.0; x.len()];
            dequantize(scheme, &packed, &scales, 32, &mut out).expect("合法反解");

            let mut cosine_num = 0.0f64;
            let mut x2 = 0.0f64;
            let mut o2 = 0.0f64;
            for (a, b) in x.iter().zip(&out) {
                cosine_num += (*a as f64) * (*b as f64);
                x2 += (*a as f64) * (*a as f64);
                o2 += (*b as f64) * (*b as f64);
            }
            let cosine = cosine_num / (x2.sqrt() * o2.sqrt());
            assert!(
                cosine > min_cosine,
                "scheme {} 余弦 {cosine} 低于阈值 {min_cosine}",
                scheme.bits
            );

            // 误差界用本组 scale：round 到最近码，|x - scale*signed| <= scale/2。
            for (i, (a, b)) in x.iter().zip(&out).enumerate() {
                let group_scale = scales[i / 32];
                let err = (a - b).abs();
                assert!(
                    err <= group_scale / 2.0 + 1e-4,
                    "scheme {} 第 {i} 项误差 {err} 超过半码距 {}",
                    scheme.bits,
                    group_scale / 2.0
                );
            }
        }
    }

    /// 全零组不产生零 scale：scale 取 1，码全为对称零点（i4 为 8），反解精确归零。
    #[test]
    fn all_zero_group_is_invertible() {
        let x = [0.0f32; 64];
        let (packed, scales) = quantize(I4, 32, &x).expect("合法输入");
        assert_eq!(scales, vec![1.0, 1.0]);
        assert!(packed.iter().all(|&b| b == 0x88), "码应为零点 8");
        let mut out = [0.0; 64];
        dequantize(I4, &packed, &scales, 32, &mut out).expect("合法反解");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// 非法输入：零分组、非有限值、长度不匹配都显式报错，不静默。
    #[test]
    fn rejects_bad_inputs() {
        assert!(quantize(I4, 0, &[1.0]).is_err());
        assert!(quantize(I4, 32, &[f32::NAN]).is_err());
        assert!(quantize(I4, 32, &[f32::INFINITY]).is_err());

        let (packed, scales) = quantize(I4, 32, &[1.0, -1.0]).expect("合法输入");
        let mut out = [0.0; 3];
        assert!(dequantize(I4, &packed, &scales, 32, &mut out).is_err());
    }
}
