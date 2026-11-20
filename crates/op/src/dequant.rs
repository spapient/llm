//! 打包反量化骨架：每字节 `CPB` 个码，`y = scale · (code − zero)`。
//!
//! `dequant_i2`（4 码/字节）与 `dequant_i4`（2 码/字节）共用形状投影、确定性样本、标量参考与
//! 打包内核；位宽、掩码、零点作为参数传入，解包与长度校验只写一份。`dequant_i8` 不打包
//! （f32 承载量化值），不走本骨架。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = 0.25;

/// 打包格式的前置约束：输出长度必须是输入字节数的 `cpb` 倍。
fn check_len(q: &[u8], out: &[f32], cpb: usize) -> Result<(), Error> {
    if out.len() != q.len() * cpb {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不符：q {} 字节、out {} 元素", q.len(), out.len()),
        ));
    }
    Ok(())
}

/// 仿射参数校验：标度不能为零（否则整组塌成常数、失去量化口径），零点必须落在码域内。
///
/// 码域是 `0..2^width`：零点是码点到有符号值的平移量，越出码域意味着存在无法表示的码点，
/// 属于配置错误而不是数值极端。打包格式传自身位宽（i2=2、i4=4），逐元素格式传 16 档。
///
/// # Errors
///
/// 标度为零、零点不是有限值，或零点不在 `0..2^width` 内时返回 [`Code::Invalid`]。
pub fn check_affine(scale: f32, zero: f32, width: usize) -> Result<(), Error> {
    if scale == 0.0 || !scale.is_finite() {
        return Err(Error::new(Code::Invalid, "反量化标度必须是非零有限值"));
    }
    if !zero.is_finite() {
        return Err(Error::new(Code::Invalid, "反量化零点必须是有限值"));
    }
    let domain = 1u32 << width;
    if !(0.0..domain as f32).contains(&zero) {
        return Err(Error::new(
            Code::Invalid,
            format!("零点 {zero} 越出码域 0..{domain}（{width} 位码）"),
        ));
    }
    Ok(())
}

/// 标量代价：每元素一次减法一次乘法；读打包字节（`n / cpb` 个），写 `n` 个 f32。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost_packed(shape: &ShapeContext, cpb: usize) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "dequant 形状缺轴 `n`"))?;
    Ok(Cost {
        flops: (n as f64) * 2.0,
        read: (n / cpb) as u64,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`q`（打包字节，`len / cpb` 个）、`y`（`len` 个 f32）。
pub fn sample_packed(len: usize, cpb: usize) -> Sample {
    let bytes = len / cpb;
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "q",
                values: Values::U8((0..bytes).map(|i| (i * 37 % 251) as u8).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 解包字节内第 `slot` 个码，按对称零点回退成有符号值。
pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: f32) -> f32 {
    let width = crate::quant::width_cpb(cpb);
    let mask = (1u8 << width) - 1;
    ((byte >> (slot * width)) & mask) as f32 - zero
}

/// 标量参考：逐字节解 `cpb` 个码，与内核同一公式。
pub fn reference_packed(q: &[u8], cpb: usize, zero: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.len() * cpb);
    for byte in q {
        for slot in 0..cpb {
            out.push(SCALE * unpack(*byte, slot, cpb, zero));
        }
    }
    out
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 `cpb` 倍时返回说明。
pub fn forward_packed(
    scale: f32,
    zero: f32,
    q: &[u8],
    out: &mut [f32],
    cpb: usize,
) -> diagnostic::Result<()> {
    check_len(q, out, cpb)?;
    check_affine(scale, zero, crate::quant::width_cpb(cpb))?;
    for (byte, chunk) in q.iter().zip(out.chunks_mut(cpb)) {
        for (slot, place) in chunk.iter_mut().enumerate() {
            *place = scale * unpack(*byte, slot, cpb, zero);
        }
    }
    Ok(())
}

/// 生成一份「打包反量化」精度变体的完整公开面：`SCALE`/`ZERO` 常量、代价/样本/参考/
/// `forward` 入口、[`annotation::model`] 别名与测试挂载。`dequant_i2`/`dequant_i4` 只差一行
/// [`crate::quant::Row`]；新增一档打包反量化加一次调用即可。
#[macro_export]
macro_rules! dequant_packed {
    ($name:ident, $alias:ident, $op:tt, $neg:expr, $nonfinite:expr, $row:ident) => {
        /// 反量化标度（2 的负幂网格，f32 精确）。
        pub const SCALE: f32 = $crate::dequant::SCALE;

        /// 打包对称零点（取自配置表 `$row`）。
        pub const ZERO: f32 = $row.zero as f32;

        // 单轴定义与形状投影复用打包骨架。
        pub use $crate::dequant::{AXES, project_shape};

        /// 标量代价：每元素一次减法一次乘法；读打包字节，写 `n` 个 f32。
        pub fn cost(
            shape: &$crate::model::ShapeContext,
        ) -> diagnostic::Result<$crate::model::Cost> {
            $crate::dequant::cost_packed(shape, $row.cpb)
        }

        /// 确定性样本：`q`（打包字节）、`y`（`n` 个 f32）。
        pub fn sample(len: usize, _: usize) -> $crate::model::Sample {
            $crate::dequant::sample_packed(len, $row.cpb)
        }

        /// 标量参考：逐字节解包，与内核同一公式。
        pub fn reference(sample: &$crate::model::Sample) -> diagnostic::Result<Vec<f32>> {
            let q = sample.u8("q")?;
            Ok($crate::dequant::reference_packed(
                q,
                $row.cpb,
                $row.zero as f32,
            ))
        }

        /// 内核本体：结果写进调用方给的缓冲。
        pub fn forward(scale: f32, zero: f32, q: &[u8], out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::dequant::forward_packed(scale, zero, q, out, $row.cpb)
        }

        #[doc = concat!("`", $op, "` 的标记类型：位宽与零点见配置表，骨架见 `crate::dequant`。")]
        pub struct $name;

        // 基座契约（ADR 0050）：转发本模块的骨架入口；粒度直接取配置表的每字节码数。
        $crate::operator_via!(
            @impl $name,
            $op,
            1,
            (4096, 4096),
            false,
            $crate::quant::$row.cpb as u32,
            self::AXES,
            self::project_shape,
            self::cost,
            self::sample,
            self::reference
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = $op)]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::dequant::test_support::dequant_packed_tests!(
                sample,
                reference,
                forward,
                SCALE,
                ZERO,
                $row.cpb,
                (1u32 << $crate::quant::width_cpb($row.cpb)) as f32,
                $neg,
                $nonfinite
            );
        }
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    /// 生成 dequant_i2/i4 共有的三则测试：参考对拍、退化边界、仿射参数负例。
    macro_rules! dequant_packed_tests {
        ($sample:ident, $reference:ident, $forward:ident, $scale:ident, $zero:ident,
         $cpb:expr, $domain:expr, $neg:expr, $nonfinite:expr) => {
            #[test]
            fn dequant_matches_the_scalar_reference() {
                let sample = $sample(128, 1);
                let expected = $reference(&sample).expect("参考可计算");
                let mut out = vec![0.0; 128];
                $forward($scale, $zero, sample.u8("q").expect("q"), &mut out).expect("等长");
                assert_eq!(out, expected);
            }

            #[test]
            fn handles_empty_mismatched_and_saturated_bytes() {
                let mut empty = vec![];
                $forward($scale, $zero, &[], &mut empty).expect("空行放行");

                let mut wrong = vec![0.0; 3];
                assert!(
                    $forward($scale, $zero, &[0x12, 0x34], &mut wrong).is_err(),
                    "输出长度必须是字节数的整数倍"
                );

                let saturated = [0x00u8, 0xFF];
                let mut out = vec![0.0; saturated.len() * $cpb];
                $forward($scale, $zero, &saturated, &mut out).expect("极值字节不越界");
                assert!(out.iter().all(|v| v.is_finite()), "结果有限");
            }

            #[test]
            fn rejects_zero_scale_and_out_of_domain_zero() {
                let q = [0x52u8];
                let mut out = [0.0; $cpb];

                assert!($forward(0.0, $zero, &q, &mut out).is_err(), "零标度拒绝");
                assert!(
                    $forward($scale, $nonfinite, &q, &mut out).is_err(),
                    "非有限零点拒绝"
                );
                assert!(
                    $forward($scale, $domain, &q, &mut out).is_err(),
                    "零点越出码域"
                );
                assert!(
                    $forward($scale, $neg, &q, &mut out).is_err(),
                    "负零点越出码域"
                );
            }
        };
    }
    pub(crate) use dequant_packed_tests;
}

pub mod dequant_i2 {
    //! 2 位反量化：打包字节 `q`（每字节四个 2 位码），`y = scale * (code - 2)`。
    //!
    //! 低 2 位在前、高位在后；码是无符号 0..3，按对称零点 2 转成 -2..=1。打包骨架在
    //! [`crate::dequant`]，本模块只写位宽与零点，`#[annotation::model]` 装配。

    use crate::quant::I2;

    crate::dequant_packed!(DequantI2, DequantI2F32, "dequant_i2", -0.5, f32::NAN, I2);
}

pub mod dequant_i4 {
    //! 4 位反量化：打包字节 `q`（每字节两个 nibble），`y = scale * (nibble - 8)`。
    //!
    //! 低 4 位在前、高 4 位在后；nibble 是无符号 0..15，按对称零点 8 转有符号。打包骨架在
    //! [`crate::dequant`]，本模块只写位宽与零点，`#[annotation::model]` 装配。

    use crate::quant::I4;

    crate::dequant_packed!(
        DequantI4,
        DequantI4F32,
        "dequant_i4",
        -1.0,
        f32::INFINITY,
        I4
    );
}

pub mod dequant_i8 {
    //! 8 位反量化：`y[n] = scale * (q[n] - zero)`。
    //!
    //! 量化值以 f32 承载（当前 `Values` 只有 f32 变体），`scale` 与 `zero` 由样本标量给出。
    //! 本模块自带形状、代价、样本、参考与遍历，`#[annotation::model]` 装配。

    use crate::dequant;
    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n};
    use diagnostic::{Code, Error};

    // 单轴定义复用 unary 骨架。
    pub use crate::unary::AXES;

    /// 样本固定的反量化参数。
    pub const SCALE: f32 = 0.25;
    /// 样本固定的零点。
    pub const ZERO: f32 = 8.0;

    // 形状投影复用 unary 骨架。
    pub use crate::unary::project_shape;

    /// 标量代价：每元素一次减法一次乘法，读一次写一次。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        cost_n(shape, "dequant", 2.0, 1)
    }

    /// 确定性样本：`q`（f32 承载的量化整数）、`y`，标量 `scale`/`zero`。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            buffers: vec![
                Buffer {
                    name: "q",
                    values: Values::F32((0..len).map(|i| (i % 16) as f32).collect()),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 标量参考：与内核同一公式。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let q = sample.f32("q")?;
        Ok(q.iter().map(|value| SCALE * (value - ZERO)).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出等长、标度非零有限、零点在 `0..=15` 码域内，否则返回说明。
    pub fn forward(scale: f32, zero: f32, q: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        if q.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：q {}、out {}", q.len(), out.len()),
            ));
        }
        dequant::check_affine(scale, zero, 4)?;
        for (out, q) in out.iter_mut().zip(q) {
            *out = scale * (q - zero);
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct DequantI8;

    // 基座契约（ADR 0050）：本模块自带具体入口，直接转发。
    crate::operator_here!(
        DequantI8,
        "dequant_i8",
        1,
        (4096, 4096),
        false,
        self::reference
    );

    /// 反量化的具体语义别名。
    #[annotation::model(name = "dequant_i8", shape = [4096, 4096], dtype = "f32", skeleton = "dequant")]
    pub type DequantI8F32 = DequantI8;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn dequant_matches_the_scalar_reference() {
            let sample = sample(128, 1);
            let expected = reference(&sample).expect("参考可计算");
            let mut out = vec![0.0; 128];
            forward(SCALE, ZERO, sample.f32("q").expect("q"), &mut out).expect("等长");
            assert_eq!(out, expected);
        }

        /// 退化与边界：空行放行；长度不齐拒绝；饱和量化值与零档不产生 NaN。
        #[test]
        fn dequant_i8_handles_empty_mismatched_and_saturated_values() {
            let mut empty = vec![];
            forward(SCALE, ZERO, &[], &mut empty).expect("空行放行");

            let mut out = vec![0.0; 2];
            assert!(
                forward(SCALE, ZERO, &[1.0, 2.0, 3.0], &mut out).is_err(),
                "q 与 out 不等长拒绝"
            );

            let saturated = [0.0f32, 15.0, f32::MIN, f32::MAX];
            let mut out = vec![0.0; saturated.len()];
            forward(SCALE, ZERO, &saturated, &mut out).expect("饱和值不越界");
            assert!(out.iter().all(|v| v.is_finite()), "结果有限");
        }

        /// 仿射参数负例：零标度、非有限标度、越出 i4 码域（0..15）的零点都在解包前拦下。
        #[test]
        fn dequant_i8_rejects_zero_scale_and_out_of_domain_zero() {
            let q = [8.0f32, 12.0];
            let mut out = [0.0; 2];

            assert!(forward(0.0, ZERO, &q, &mut out).is_err(), "零标度拒绝");
            assert!(
                forward(f32::INFINITY, ZERO, &q, &mut out).is_err(),
                "非有限标度拒绝"
            );
            assert!(
                forward(SCALE, 16.0, &q, &mut out).is_err(),
                "零点越出 0..15 码域"
            );
            assert!(
                forward(SCALE, -1.0, &q, &mut out).is_err(),
                "负零点越出码域"
            );
        }
    }
}
