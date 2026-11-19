use super::{Format, cost};
use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::Error;

/// 三缓冲打包样本：`x`、打包权重 `w` 与每组分标度 `scales`。
pub fn with(m: usize, n: usize, x: Vec<f32>, w: Vec<u8>, scales: Vec<f32>) -> Sample {
    Sample {
        m,
        n,
        scalars: Vec::new(),
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32(x),
            },
            Buffer {
                name: "w",
                values: Values::U8(w),
            },
            Buffer {
                name: "scales",
                values: Values::F32(scales),
            },
        ],
    }
}

/// 只关心 `(m, n)` 的代价：宽度填进算子声明的具名轴。
pub fn at<T: Format>(m: usize, n: usize) -> Result<Cost, Error> {
    cost::<T>(&ShapeContext::new([('m', m), ('n', n)])?)
}

/// 生成 i2/i4 打包 GEMV 共有的四则测试：负例拒绝、生产形状合法、非整组报错、样本确定性。
macro_rules! packed_gemv_tests {
    ($ty:ty, $model:ident, $group:ident, $forward:ident, $w_bytes:expr, $sample_w:expr) => {
        #[test]
        fn forward_rejects_a_buffer_that_does_not_fit_the_shapes() {
            let error = $forward(&[1.0; 4], &[], &[1.0], &mut [0.0; 1]).expect_err("权重太短");
            assert_eq!(error.code(), ::diagnostic::Code::Invalid);
            assert!(
                $forward(&[1.0; 4], &[0u8; $w_bytes], &[1.0], &mut []).is_err(),
                "空输出"
            );
        }

        #[test]
        fn the_production_shape_is_a_legal_input() {
            let (m, n) = $model.shape;
            assert!(
                m.is_multiple_of($group),
                "生产形状的 M 必须是组大小的整数倍"
            );
            assert!(at::<$ty>(m, n).is_ok(), "生产形状必须能算出代价");
            assert_eq!(
                $model.dtype,
                ::prof::atom::DType::F32,
                "样本的输出缓冲是 f32"
            );
            assert_eq!($model.group as usize, $group, "注册的粒度就是注解的粒度");
        }

        #[test]
        fn a_shape_that_does_not_fit_the_grouping_is_an_error() {
            let error = at::<$ty>(33, 4).expect_err("33 不是 32 的整数倍");
            assert!(error.to_string().contains("整数倍"), "{error}");
            assert_eq!(error.code(), ::diagnostic::Code::Invalid);
        }

        #[test]
        fn sample_is_deterministic_and_sized_to_the_group() {
            let first = crate::gemv::sample::<$ty>(64, 4);
            assert_eq!(first, crate::gemv::sample::<$ty>(64, 4));
            assert_eq!(first.u8("w").expect("w").len(), 4 * $sample_w);
            assert_eq!(
                first.f32("scales").expect("scales").len(),
                4 * (64 / $group)
            );
            assert_eq!(first.f32("y").expect("y").len(), 4);
        }
    };
}
pub(crate) use packed_gemv_tests;

/// 8×2 多组样本：打包 `bits` 位权重、逐行递增输入 `x`，返回 `(w, x, scales, m, n)`。
pub fn multi_group(
    bits: usize,
    w: &[i8],
    scales: Vec<f32>,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, usize, usize) {
    let (m, n) = (8, 2);
    let packed = pack(bits, w);
    let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
    (packed, x, scales, m, n)
}

/// 把 `bits` 位有符号码按低 `bits` 位落在字节内：`zero = 2^(bits-1)` 平移、低 `bits` 位落字节。
pub fn pack(bits: usize, values: &[i8]) -> Vec<u8> {
    let cpb = 8 / bits;
    let zero = 1 << (bits - 1);
    let mask = (1u8 << bits) - 1;
    values
        .chunks(cpb)
        .map(|group| {
            (0..cpb).fold(0u8, |byte, slot| {
                let code = (group.get(slot).copied().unwrap_or(0) + zero) as u8 & mask;
                byte | (code << (slot * bits))
            })
        })
        .collect()
}

/// 生成 i2/i4 打包 GEMV 共有、数值可由位宽推导的五则测试：
/// 解包点积、非有限标度、大/小形状代价、模板签名。
macro_rules! packed_gemv_config_tests {
    ($ty:ty, $op:literal, $bits:expr, $w_note:literal, $dequant_vals:expr, $dequant_expect:expr) => {
        #[test]
        fn reference_dequantizes_and_dots() {
            let w = pack($bits, &$dequant_vals);
            let sample = with(4, 1, vec![1.0; 4], w, vec![0.5]);
            let y = crate::gemv::reference::<$ty>(&sample).expect("参考可算");
            assert_eq!(y.len(), 1);
            assert!((y[0] - $dequant_expect).abs() < 1e-6, "解包点积 = {}", y[0]);
        }

        #[test]
        fn gemv_rejects_non_finite_scales() {
            let m = 4;
            let w = vec![0u8; m / (8 / $bits)];
            let x = vec![1.0; m];
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let error = forward(&x, &w, &[bad], &mut [0.0; 1]).expect_err("非有限标度必须拒绝");
                assert_eq!(error.code(), diagnostic::Code::Invalid);
            }
        }

        #[test]
        fn cost_comes_from_the_declared_expression_without_hand_written_constants() {
            let cost = at::<$ty>(4096, 4096).expect("可估");
            assert_eq!(cost.flops, 2.0 * 4096.0 * 4096.0);
            assert_eq!(
                cost.read,
                4096 * 4096 / (8 / $bits) + 4096 * 4 + 4096 * 128 * 4,
                $w_note
            );
            assert_eq!(cost.write, 4096 * 4, "写方向只有输出那一份");
            assert_eq!(
                cost.bytes(),
                4096 * 4096 / (8 / $bits) + 4096 * 4 + 4096 * 128 * 4 + 4096 * 4,
                "合计与改造前逐字节相同"
            );
        }

        #[test]
        fn a_small_shape_pins_every_term_so_a_factor_cannot_hide() {
            let cost = at::<$ty>(64, 4).expect("可估");
            assert_eq!(cost.flops, 2.0 * 64.0 * 4.0);
            assert_eq!(cost.read, 256 / (8 / $bits) + 256 + 32);
            assert_eq!(cost.write, 16);
            assert_eq!(cost.bytes(), 256 / (8 / $bits) + 256 + 32 + 16);
        }

        #[test]
        fn call_arguments_follow_the_template_signature() {
            let signature = crate::sig::Signature::parse($op, include_str!("../ops/gemv_packed.k"))
                .expect("签名可解析");
            assert_eq!(
                signature
                    .call_args(&crate::gemv::sample::<$ty>(64, 4))
                    .expect("实参"),
                "x, w, scales, y, 64, 4, 32"
            );
            assert_eq!(signature.output(), Some("y"));
        }
    };
}
pub(crate) use packed_gemv_config_tests;
