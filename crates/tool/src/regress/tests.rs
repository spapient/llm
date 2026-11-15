use super::*;

fn baseline() -> Baseline {
    Baseline::new(vec![
        Case::new("tput", 100.0, 5.0, Direction::Higher).expect("场景"),
        Case::new("latency", 50.0, 10.0, Direction::Lower).expect("场景"),
    ])
    .expect("基线")
}

#[test]
fn a_higher_is_better_case_regresses_only_below_its_limit() {
    let baseline = baseline();
    assert_eq!(baseline.judge("tput", 98.0), Ok(Judgement::Pass));
    assert_eq!(baseline.judge("tput", 100.0), Ok(Judgement::Pass));
    assert!(matches!(
        baseline.judge("tput", 94.0),
        Ok(Judgement::Regressed { .. })
    ));
    assert!(matches!(
        baseline.judge("tput", 120.0),
        Ok(Judgement::Improved { .. })
    ));
}

#[test]
fn a_lower_is_better_case_mirrors_the_comparison() {
    let baseline = baseline();
    assert_eq!(baseline.judge("latency", 52.0), Ok(Judgement::Pass));
    assert!(matches!(
        baseline.judge("latency", 56.0),
        Ok(Judgement::Regressed { .. })
    ));
    assert!(matches!(
        baseline.judge("latency", 40.0),
        Ok(Judgement::Improved { .. })
    ));
}

#[test]
fn a_non_finite_measurement_never_passes() {
    assert!(matches!(
        baseline().judge("tput", f64::NAN),
        Ok(Judgement::Regressed { .. })
    ));
}

#[test]
fn an_unknown_scenario_is_a_configuration_error() {
    let error = baseline().judge("missing", 1.0).expect_err("未知场景");
    assert_eq!(error.code(), Code::Missing);
    assert!(error.to_string().contains("tput"), "{error}");
}

#[test]
fn impossible_cases_are_rejected() {
    assert_eq!(
        Case::new("x", 0.0, 5.0, Direction::Higher)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(
        Case::new("x", 1.0, 101.0, Direction::Higher)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(Baseline::new(Vec::new()).unwrap_err().code(), Code::Invalid);
    let duplicated = Case::new("x", 1.0, 1.0, Direction::Higher).expect("场景");
    assert_eq!(
        Baseline::new(vec![duplicated.clone(), duplicated])
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn judgements_and_directions_have_stable_names() {
    assert_eq!(Direction::Higher.as_str(), "higher-is-better");
    assert_eq!(Direction::Lower.as_str(), "lower-is-better");
    assert_eq!(Judgement::Pass.as_str(), "pass");
    assert!(
        Judgement::Improved {
            observed: 2.0,
            baseline: 1.0
        }
        .is_pass()
    );
}
