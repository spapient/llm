use super::*;

#[test]
fn defaults_are_valid_and_cover_every_required_front() {
    let config = RuntimeConfig::default();
    config.validate().expect("默认配置必须合法");
    let languages: Vec<_> = config.gates.iter().map(|gate| gate.emit).collect();
    assert_eq!(
        languages,
        [
            Emit::Cuda,
            Emit::Opencl,
            Emit::C,
            Emit::Metal,
            Emit::Ascendc
        ]
    );
}

#[test]
fn a_front_constructor_accepts_owned_and_borrowed_strings() {
    let borrowed = LanguageFront::new(Emit::C, ["clang"], ["-fsyntax-only"]);
    let owned = LanguageFront::new(
        Emit::C,
        [String::from("clang")],
        [String::from("-fsyntax-only")],
    );
    assert_eq!(borrowed, owned);
}

#[test]
fn malformed_fronts_are_rejected_before_they_can_idx_an_empty_command() {
    for gate in [
        LanguageFront::new(Emit::C, [] as [&str; 0], ["-fsyntax-only"]),
        LanguageFront::new(Emit::C, ["  "], ["-fsyntax-only"]),
        LanguageFront::new(Emit::C, ["clang"], [] as [&str; 0]),
    ] {
        let config = RuntimeConfig {
            gates: vec![gate],
            ..RuntimeConfig::default()
        };
        assert_eq!(config.validate().unwrap_err().code(), Code::Invalid);
    }
}

#[test]
fn duplicate_languages_are_rejected_instead_of_using_the_first_row_silently() {
    let mut config = RuntimeConfig::default();
    let duplicate = config.gates[0].clone();
    config.gates.push(duplicate);
    assert!(config.validate().unwrap_err().to_string().contains("重复"));
}

#[test]
fn every_invalid_threshold_and_zero_lock_wait_is_rejected() {
    for threshold in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
        let config = RuntimeConfig {
            l3_threshold: threshold,
            ..RuntimeConfig::default()
        };
        assert_eq!(config.validate().unwrap_err().code(), Code::Invalid);
    }
    let config = RuntimeConfig {
        lock_wait: Duration::ZERO,
        ..RuntimeConfig::default()
    };
    assert_eq!(config.validate().unwrap_err().code(), Code::Invalid);
}
