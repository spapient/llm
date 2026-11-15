use super::*;
use op::model::Buffer;

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
fn bandwidth_attainment_is_a_byte_ratio_not_a_gigabyte_one() {
    // 0.7 MB / 1.467 ms ≈ 0.477 GB/s，占 68 GB/s 的 0.7%；除以 GB/s 会算出 0.0%。
    let measured = throughput(700_000, 2.1e6, 1.467e6, Some(68.0e9));
    assert!((measured.gb_s - 0.477).abs() < 0.01, "{measured:?}");
    assert!((measured.gflops - 1.43).abs() < 0.01, "{measured:?}");
    assert!((measured.bandwidth_ratio.expect("有峰值") - 0.0070).abs() < 1e-4);
    assert_eq!(throughput(1, 1.0, 0.0, Some(1.0)).bandwidth_ratio, None);
}

#[test]
fn working_set_is_every_input_buffer_but_not_the_output() {
    let sample = Sample {
        m: 1,
        n: 2,
        scalars: Vec::new(),
        buffers: vec![
            Buffer {
                name: "w",
                values: Values::U8(vec![0; 8]),
            },
            Buffer {
                name: "x",
                values: Values::F32(vec![1.0; 4]),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; 64]),
            },
        ],
    };
    assert_eq!(working_set(&sample, "y"), 8 + 16);
}
