use super::{Format, cost};
use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::Error;

/// 三缓冲打包样本：`x`、打包权重 `w` 与每组分标度 `scales`。
pub fn with(m: usize, n: usize, x: Vec<f32>, w: Vec<u8>, scales: Vec<f32>) -> Sample {
    Sample {
        m,
        n,
        scalars: Vec::new(),
        bufs: vec![
            Buf {
                name: "x",
                values: Values::F32(x),
            },
            Buf {
                name: "w",
                values: Values::U8(w),
            },
            Buf {
                name: "scales",
                values: Values::F32(scales),
            },
        ],
    }
}

/// 只关心 `(m, n)` 的代价：宽度填进算子声明的具名轴。
pub fn at<T: Format>(m: usize, n: usize) -> Result<Cost, Error> {
    cost::<T>(&ShapeCtx::new([('m', m), ('n', n)])?)
}

/// 生成三档量化 GEMV（i2 / i4 / i8）共有的行批测试：批结果逐位等于逐行调用、行批几何不符被拒。
///
/// 两条用例都只用"同一份权重与输入"这一个判据：解包、分组与累加次序若在批路径上被改动，
/// 逐位比较立刻失败——它不需要 f32 容差，因此不会把真实差额藏进舍入里。打包与否只由 `$cpb`
/// 体现（`i8` 取 1），守卫与遍历都是同一份骨架。
macro_rules! packed_gemv_batch_tests {
    ($forward:ident, $forward_batch:ident, $cpb:expr, $group:expr) => {
        #[test]
        fn a_row_batch_equals_row_by_row_calls() {
            let (m, n, rows) = (4 * $group, 3usize, 4usize);
            let weights = $crate::gemv::packed::weights(m, n, $cpb);
            let scales = vec![0.25f32; n * (m / $group)];
            let x: Vec<f32> = (0..rows * m).map(|i| (i as f32 * 0.125).sin()).collect();

            let mut batched = vec![0.0f32; rows * n];
            $forward_batch(
                &x,
                &weights,
                &scales,
                $crate::gemv::Geometry::new(rows, m, n),
                &mut batched,
            )
            .expect("行批可算");
            let mut stepwise = vec![0.0f32; rows * n];
            for row in 0..rows {
                $forward(
                    tensor::View::one(&x[row * m..(row + 1) * m]),
                    &weights,
                    tensor::View::one(&scales),
                    &mut stepwise[row * n..(row + 1) * n],
                )
                .expect("单行可算");
            }
            assert_eq!(batched, stepwise, "行批必须与逐行调用逐位相同");
        }

        #[test]
        fn a_row_batch_with_mismatched_bufs_is_an_error() {
            let weights = $crate::gemv::packed::weights(4 * $group, 2, $cpb);
            let scales = vec![1.0f32; 2 * 4];
            let mut out = vec![0.0f32; 4];
            let error = $forward_batch(
                &[1.0f32; 3 * 4 * $group],
                &weights,
                &scales,
                $crate::gemv::Geometry::new(4, 4 * $group, 2),
                &mut out,
            )
            .expect_err("行数与缓冲不符必须拒绝");
            assert_eq!(error.code(), ::diagnostic::Code::Invalid);
        }
    };
}

/// 生成 i2/i4 打包 GEMV 共有的四则测试：负例拒绝、生产形状合法、非整组报错、样本确定性。
macro_rules! packed_gemv_tests {
    ($ty:ty, $model:ident, $group:ident, $forward:ident, $w_bytes:expr, $sample_w:expr) => {
        #[test]
        fn forward_rejects_a_buf_that_does_not_fit_the_shapes() {
            let error = $forward(tensor::View::one(&[1.0; 4]), &[], tensor::View::one(&[1.0]), &mut [0.0; 1]).expect_err("权重太短");
            assert_eq!(error.code(), ::diagnostic::Code::Invalid);
            assert!(
                $forward(tensor::View::one(&[1.0; 4]), &[0u8; $w_bytes], tensor::View::one(&[1.0]), &mut []).is_err(),
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
    let x: Vec<f32> = (0..m).map(|idx| idx as f32 + 1.0).collect();
    (packed, x, scales, m, n)
}

/// 把 `bits` 位有符号码按低 `bits` 位落在字节内（实现与跨 crate 夹具共用 [`testkit::pack_codes`]）。
pub use testkit::pack_codes as pack;

/// 生成「非有限标度必须在分组守卫后、解包前拦下」的一则测试。
///
/// `i2` / `i4` / `i8` 三档的这条守卫都在骨架里（[`crate::gemv::finite_scales`]），判据与错误码
/// 完全相同：非有限标度会把整组乘积带进 NaN/Inf，任何一档都不能放过去。`$bits` 只决定一行 `m`
/// 个元素占多少字节。
macro_rules! non_finite_scale_test {
    ($bits:expr) => {
        #[test]
        fn gemv_rejects_non_finite_scales() {
            let m = 4;
            let w = vec![0u8; m / (8 / $bits)];
            let x = vec![1.0; m];
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let error = forward(tensor::View::one(&x), &w, tensor::View::one(&[bad]), &mut [0.0; 1]).expect_err("非有限标度必须拒绝");
                assert_eq!(error.code(), ::diagnostic::Code::Invalid);
            }
        }
    };
}
pub(crate) use non_finite_scale_test;

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

        $crate::gemv::packed_test_support::non_finite_scale_test!($bits);

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
            let signature = crate::sig::Signature::parse(
                $op,
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../prof/src/ops/gemv_packed.k"
                )),
            )
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

/// 生成 i2/i4 打包 GEMV 共有、数值随位宽变化的四则测试：
/// 多组参考、覆盖写对拍、手写求和、形状负例。位宽只决定查表与粒度倍数，
/// 这里的取样、点积、守卫共用同一套骨架。
macro_rules! packed_gemv_value_tests {
    (
        $ty:ty, $bits:expr,
        $w_group:expr, $scales_group:expr, $ref_assert:expr,
        $w_sum:expr, $row0:expr, $row1:expr
    ) => {
        #[test]
        fn reference_handles_multiple_groups_and_rows() {
            let (w, x, scales, m, n) = multi_group($bits, &$w_group, $scales_group.to_vec());
            let sample = with(m, n, x, w, scales);
            let y = crate::gemv::reference::<$ty>(&sample).expect("参考可算");
            assert_eq!(y.len(), n);
            $ref_assert(&y);
        }

        #[test]
        fn forward_writes_into_the_caller_buf_and_matches_the_reference() {
            let (w, x, scales, m, n) = multi_group($bits, &$w_group, $scales_group.to_vec());
            let expected =
                crate::gemv::reference::<$ty>(&with(m, n, x.clone(), w.clone(), scales.clone()))
                    .expect("参考可算");
            // 复用一块脏缓冲：结果必须是覆盖写，读到的不能是上一次的残留。
            let mut out = vec![f32::NAN; n];
            forward(tensor::View::one(&x), &w, tensor::View::one(&scales), &mut out).expect("内核可算");
            assert_eq!(out, expected, "内核与参考必须是同一段算术");
        }

        /// 手写一遍求和，不经过本文件的任何循环：分组边界、`scale` 落在哪、行偏移都由测试自己算。
        ///
        /// 这是替换内层循环（查表 + 按组切片 + 每组一次乘法）之后仍然钉住算术的判据——
        /// `reference` 与 `forward` 是同一段代码，只对它俩对拍等于自己跟自己对。
        #[test]
        fn the_kernel_agrees_with_a_hand_written_sum() {
            let m = 8;
            let n = 2;
            let w = pack($bits, &$w_sum);
            let x: Vec<f32> = (0..m).map(|idx| idx as f32 * 0.5 - 1.0).collect();
            // 行 0：组 0 = 0.25、组 1 = 0.5；行 1：组 0 = 1.0、组 1 = -1.0。
            let scales = vec![0.25, 0.5, 1.0, -1.0];

            let mut out = vec![f32::NAN; n];
            forward(tensor::View::one(&x), &w, tensor::View::one(&scales), &mut out).expect("内核可算");

            let row = |weights: [f32; 8], scale: [f32; 2]| {
                scale[0] * (0..4).map(|k| weights[k] * x[k]).sum::<f32>()
                    + scale[1] * (0..4).map(|k| weights[4 + k] * x[4 + k]).sum::<f32>()
            };
            let expected = [row($row0, [0.25, 0.5]), row($row1, [1.0, -1.0])];
            assert!((out[0] - expected[0]).abs() < 1e-5, "{:?}", out);
            assert!((out[1] - expected[1]).abs() < 1e-5, "{:?}", out);
        }

        #[test]
        fn shape_mismatch_is_an_error() {
            let cpb = 8 / $bits;
            // 输入太短。
            assert!(
                crate::gemv::reference::<$ty>(&with(
                    4,
                    1,
                    vec![1.0],
                    vec![0u8; 4 / cpb],
                    vec![1.0]
                ))
                .is_err()
            );
            // 权重太短。
            assert!(
                crate::gemv::reference::<$ty>(&with(4, 1, vec![1.0; 4], vec![0u8; 0], vec![1.0]))
                    .is_err()
            );
            // 没有分组。
            assert!(
                crate::gemv::reference::<$ty>(&with(
                    4,
                    1,
                    vec![1.0; 4],
                    vec![0u8; 4 / cpb],
                    vec![]
                ))
                .is_err()
            );
            // 分组不是 cpb 的倍数：m=5、1 组 → 组大小 5，对 2 位与 4 位都不是整组。
            assert!(
                crate::gemv::reference::<$ty>(&with(
                    5,
                    1,
                    vec![1.0; 5],
                    vec![0u8; 5usize.div_ceil(cpb)],
                    vec![1.0]
                ))
                .is_err()
            );
            // m 不是 cpb 的倍数、分组为 2 组但不整除：先被"不能整除"拦下，解包循环从不碰到残缺尾码。
            assert!(
                crate::gemv::reference::<$ty>(&with(
                    5,
                    1,
                    vec![1.0; 5],
                    vec![0u8; 5usize.div_ceil(cpb)],
                    vec![1.0, 1.0]
                ))
                .is_err()
            );
            // 非整组宽度无论分组奇偶都进不到解包：非整组被倍数校验拦，其余被整除校验拦。
            assert!(
                crate::gemv::reference::<$ty>(&with(
                    7,
                    1,
                    vec![1.0; 7],
                    vec![0u8; 7usize.div_ceil(cpb)],
                    vec![1.0]
                ))
                .is_err()
            );
            assert!(
                crate::gemv::reference::<$ty>(&with(
                    7,
                    1,
                    vec![1.0; 7],
                    vec![0u8; 7usize.div_ceil(cpb)],
                    vec![1.0, 1.0]
                ))
                .is_err()
            );
        }
    };
}
pub(crate) use packed_gemv_batch_tests;
pub(crate) use packed_gemv_value_tests;
