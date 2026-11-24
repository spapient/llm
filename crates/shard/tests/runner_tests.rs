use super::*;

#[test]
fn every_timed_run_is_its_own_sample() {
    let samples = parse_samples("noise\nns 120\nns 100\nns 140\n");
    assert_eq!(samples, vec![120.0, 100.0, 140.0]);
    assert!(parse_samples("").is_empty());
}

#[test]
fn min_and_median_are_computed_from_all_samples() {
    let timing = Timing {
        samples_ns: vec![30.0, 10.0, 20.0, 40.0],
    };
    assert_eq!(timing.min_ns(), Some(10.0));
    assert_eq!(timing.p50_ns(), Some(25.0));
    assert_eq!(timing.p95_ns(), Some(40.0));
    assert_eq!(Timing::default().min_ns(), None);
    assert_eq!(Timing::default().p50_ns(), None);
    assert_eq!(Timing::default().p95_ns(), None);
}

#[test]
fn bit_patterns_decode_back_to_floats() {
    let bits = f32::to_bits(1.5);
    assert_eq!(parse_bits(&format!("{bits}\n")), vec![1.5]);
}

#[test]
fn bit_patterns_round_trip_for_random_bits() {
    let mut rng = testkit::Rng::new(0xB17);
    for _ in 0..64 {
        let bits = rng.below(u32::MAX);
        let decoded = parse_bits(&format!("{bits}\n"));
        // 按位比较：NaN 的位模式也要能往返，而 `f32 == f32` 对 NaN 不成立。
        assert_eq!(decoded[0].to_bits(), bits);
    }
}

#[test]
fn quantiles_are_ordered_for_random_samples() {
    let mut rng = testkit::Rng::new(0xA17E);
    for _ in 0..32 {
        let count = rng.below(64) + 1;
        let samples: Vec<f64> = (0..count)
            .map(|_| f64::from(rng.below(1_000_000)))
            .collect();
        let timing = Timing {
            samples_ns: samples,
        };
        let min = timing.min_ns().expect("非空");
        let p50 = timing.p50_ns().expect("非空");
        let p95 = timing.p95_ns().expect("非空");
        assert!(min <= p50 && p50 <= p95, "分位必须有序：{min} {p50} {p95}");
    }
}

#[test]
fn compiler_gets_the_flags_it_was_given() {
    // 真机用例：默认跳过，`LLM_TEST_DEVICE=cpu` 才真的编译一次。
    if testkit::device_or_skip(
        "cpu",
        "cutile: 未显式启用 CPU 真机编译用例（LLM_TEST_DEVICE=cpu）",
    ) {
        return;
    }
    let workdir = std::env::temp_dir().join(format!("optim-runner-{}", std::process::id()));
    let source = "#include <stdio.h>\n\
int main(void) { printf(\"%u\\n\", (unsigned)FLAG_MARK); return 0; }\n";
    let exe =
        compile("cc", &["-DFLAG_MARK=7".to_string()], source, &workdir).expect("带档案参数编译");
    let outcome = run(&exe).expect("运行");
    assert_eq!(outcome.bits, vec![f32::from_bits(7)]);
    let _ = std::fs::remove_dir_all(&workdir);
}
