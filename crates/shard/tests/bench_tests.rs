//! `cutile::bench` 的用例：计时采样与统计。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;
use nn::model::Buf;

#[test]
fn an_attainment_above_one_is_an_invalid_measurement_not_a_pass() {
    // 数据驻留在缓存里时 `bytes/ns` 会超过档案的主存峰值：这不是"超过 100% 的实现质量"。
    assert_eq!(classify(Some(1.42), false), Verdict::OverPeak);
    // 装得进缓存时，分母（主存峰值）本来就不是这次测量的口径。
    assert_eq!(classify(Some(0.19), true), Verdict::CacheResident);
    assert_eq!(classify(Some(1.0), false), Verdict::Reached);
    assert_eq!(classify(Some(0.7), false), Verdict::Reached);
    assert_eq!(classify(Some(0.69), false), Verdict::Below);
    assert_eq!(classify(None, false), Verdict::Unknown);
}

#[test]
fn unregistered_device_profiles_are_not_reported_as_gpu_benchmarks() {
    let catalogue = Catalogue::try_builtin().expect("内置档案可解析");
    let cpu = catalogue.resolve("cpu").expect("CPU 档案存在");
    assert!(require_bench_backend(cpu).is_ok());

    for device in ["metal", "nvidia", "musa"] {
        let profile = catalogue.resolve(device).expect("设备档案存在");
        let error = require_bench_backend(profile).expect_err("设备计时后端尚未接入");
        assert_eq!(error.code(), Code::Unsupported, "{device}: {error}");
        assert!(
            error.to_string().contains("不能报告设备吞吐"),
            "{device}: {error}"
        );
    }
}

#[test]
fn bandwidth_attainment_is_a_byte_ratio_not_a_gigabyte_one() {
    // 0.7 MB / 1.467 ms ≈ 0.477 GB/s，占 68 GB/s 的 0.7%；除以 GB/s 会算出 0.0%。
    let measured = throughput(700_000, 2.1e6, 1.467e6, Some(68.0e9));
    assert!((measured.gb_s - 0.477).abs() < 0.01, "{measured:?}");
    assert!((measured.gflops - 1.43).abs() < 0.01, "{measured:?}");
    assert!((measured.bandwidth_ratio.expect("有峰值") - 0.0070).abs() < 1e-4);
    assert_eq!(throughput(1, 1.0, 0.0, Some(1.0)).bandwidth_ratio, None);
}

#[test]
fn gemv_i4_selects_the_cooperative_device_lowering() {
    let catalogue = Catalogue::try_builtin().expect("内置档案可解析");
    let op = catalogue.op("gemv_i4").expect("gemv_i4 已登记");
    assert_eq!(
        production_lower(op, Emit::Cuda).expect("CUDA 有生产态 lowering"),
        Lower::QuantizedGemvI4 { threads: 32 }
    );
    assert!(matches!(
        production_lower(op, Emit::Metal).expect("Metal 有生产态 lowering"),
        Lower::Loop(_)
    ));
}

#[test]
fn the_measured_p50_backfills_effective_cost_coefficients() {
    // 0.7 MB / 1.467 ms ≈ 0.477 GB/s → 4.77e8 B/s；2.1 MFLOP / 1.467 ms ≈ 1.43 GFLOP/s。
    let coefficients = cost_coefficients(700_000, 2.1e6, 1.467e6).expect("有效计时");
    assert!((coefficients.bw - 0.477e9).abs() < 1e7, "{coefficients:?}");
    assert!(
        (coefficients.flops - 1.43e9).abs() < 1e7,
        "{coefficients:?}"
    );
    assert!(
        cost_coefficients(1, 1.0, 0.0).is_none(),
        "耗时为 0 无法回填"
    );
}

#[test]
fn working_set_is_every_input_buf_but_not_the_output() {
    let sample = Sample {
        m: 1,
        n: 2,
        scalars: Vec::new(),
        bufs: vec![
            Buf {
                name: "w",
                values: Values::U8(vec![0; 8]),
            },
            Buf {
                name: "x",
                values: Values::F32(vec![1.0; 4]),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; 64]),
            },
        ],
    };
    assert_eq!(working_set(&sample, "y"), 8 + 16);
}
