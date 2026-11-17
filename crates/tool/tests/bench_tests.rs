use super::*;
use crate::bench_json::{from_json, to_json};
use diagnostic::Code;

fn harness() -> Harness {
    Harness::new(
        "host-a",
        "cpu-x",
        "macos",
        "aarch64",
        "rustc 1.82",
        "cpu-reference",
    )
}

fn geometry(shape: Vec<usize>, batch: u32, dtype: &str) -> Result<Geometry> {
    Geometry::new(shape, batch, dtype)
}

fn record(name: &str, backend: &str, geom: Geometry) -> Result<BenchRecord> {
    BenchRecord::new(
        EntryKind::Op,
        name,
        backend,
        geom,
        harness(),
        Roofline::default(),
    )
}

#[test]
fn a_minimal_record_keeps_observed_and_roofline_optional() {
    let geom = geometry(vec![4096, 4096], 1, "f32").expect("几何合法");
    let record = record("gemv_f32", "cpu-reference", geom).expect("记录合法");
    assert_eq!(record.kind(), EntryKind::Op);
    assert_eq!(record.name(), "gemv_f32");
    assert_eq!(record.backend(), "cpu-reference");
    assert_eq!(record.geometry().shape(), &[4096, 4096]);
    assert_eq!(record.geometry().batch(), 1);
    assert_eq!(record.geometry().dtype(), "f32");
    assert!(record.observed().is_none(), "未测量不预填实测");
    assert!(record.roofline().peak_bw_gb_s().is_none());
    assert!(record.roofline().peak_gflops().is_none());
}

#[test]
fn observed_is_filled_only_when_measured() {
    let geom = geometry(vec![4096], 1, "f32").expect("几何合法");
    let filled = record("rope_f32", "cpu-reference", geom)
        .expect("记录合法")
        .with_observed(12.5)
        .expect("实测有限");
    assert_eq!(filled.observed(), Some(12.5));

    let geom = geometry(vec![4096], 1, "f32").expect("几何合法");
    let nan = record("rope_f32", "cpu-reference", geom)
        .expect("记录合法")
        .with_observed(f64::NAN);
    assert!(nan.is_err(), "非有限实测不能回填");
    assert_eq!(nan.unwrap_err().code(), Code::Invalid);
}

#[test]
fn empty_identity_and_invalid_geometry_are_rejected() {
    let cases: [(&str, &str, Result<Geometry>); 5] = [
        ("", "cpu-reference", geometry(vec![4], 1, "f32")),
        ("gemv", "", geometry(vec![4], 1, "f32")),
        ("gemv", "cpu-reference", geometry(vec![], 1, "f32")),
        ("gemv", "cpu-reference", geometry(vec![4], 0, "f32")),
        ("gemv", "cpu-reference", geometry(vec![4], 1, "")),
    ];
    for (name, backend, geom) in cases {
        let geom = match geom {
            Ok(geom) => geom,
            Err(_) => continue,
        };
        let err = record(name, backend, geom).expect_err("必须拒绝");
        assert_eq!(err.code(), Code::Invalid, "输入：{name}/{backend}");
    }
}

#[test]
fn invalid_geometry_itself_is_an_error() {
    assert!(geometry(vec![], 1, "f32").is_err());
    assert!(geometry(vec![4], 0, "f32").is_err());
    assert!(geometry(vec![4], 1, "").is_err());
}

#[test]
fn roofline_rejects_negative_or_nonfinite_peaks() {
    assert!(Roofline::new(Some(-1.0), None).is_err());
    assert!(Roofline::new(None, Some(f64::INFINITY)).is_err());
    let ok = Roofline::new(Some(1024.0), Some(512.0)).expect("上界合法");
    assert_eq!(ok.peak_bw_gb_s(), Some(1024.0));
    assert_eq!(ok.peak_gflops(), Some(512.0));
}

#[test]
fn entry_kind_short_names_are_stable() {
    assert_eq!(EntryKind::Op.as_str(), "op");
    assert_eq!(EntryKind::Layer.as_str(), "layer");
    assert_eq!(EntryKind::Model.as_str(), "model");
}

fn filled_record() -> BenchRecord {
    let geom = geometry(vec![4096, 4096], 4, "f32").expect("几何合法");
    let roofline = Roofline::new(Some(1024.0), Some(512.0)).expect("上界合法");
    BenchRecord::new(
        EntryKind::Model,
        "batch4_step",
        "cpu-reference",
        geom,
        harness(),
        roofline,
    )
    .expect("记录合法")
    .with_observed(1_500_000.0)
    .expect("实测有限")
}

fn assert_round_trip(record: &BenchRecord) {
    let text = to_json(record);
    let parsed = from_json(&text).expect("回读合法");
    assert_eq!(&parsed, record);
}

#[test]
fn json_round_trips_record_without_observed() {
    let geom = geometry(vec![4096], 1, "f32").expect("几何合法");
    let record = record("gemv \"v1\" / \\x", "cpu-reference", geom).expect("记录合法");
    assert_round_trip(&record);

    let parsed = from_json(&to_json(&record)).expect("回读合法");
    assert!(parsed.observed().is_none());
    assert!(parsed.roofline().peak_bw_gb_s().is_none());
    assert!(parsed.roofline().peak_gflops().is_none());
}

#[test]
fn json_round_trips_record_with_observed_and_roofline() {
    let record = filled_record();
    assert_round_trip(&record);

    let parsed = from_json(&to_json(&record)).expect("回读合法");
    assert_eq!(parsed.kind(), EntryKind::Model);
    assert_eq!(parsed.observed(), Some(1_500_000.0));
    assert_eq!(parsed.roofline().peak_bw_gb_s(), Some(1024.0));
    assert_eq!(parsed.roofline().peak_gflops(), Some(512.0));
}

#[test]
fn json_rejects_malformed_and_truncated_input() {
    let cases = [
        "{",
        "{\"kind\": \"op\"",
        "{\"geometry\": {\"shape\": [4",
        "{\"name\": \"x\"",
        "[1, 2]",
        "{} trailing",
        "not json",
    ];
    for text in cases {
        let err = from_json(text).expect_err("截断 / 非法输入必须拒绝");
        assert_eq!(err.code(), Code::Invalid, "输入：{text}");
    }
}

#[test]
fn json_rejects_duplicate_keys() {
    let top_level = r#"
        {"kind": "op", "kind": "layer", "name": "x", "backend": "b",
         "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
         "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                     "toolchain": "t", "driver": "d"},
         "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
         "observed": null}
    "#;
    let err = from_json(top_level).expect_err("顶层重复键必须拒绝");
    assert_eq!(err.code(), Code::Invalid);

    let nested = r#"
        {"kind": "op", "name": "x", "backend": "b",
         "geometry": {"shape": [1], "shape": [2], "batch": 1, "dtype": "f32"},
         "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                     "toolchain": "t", "driver": "d"},
         "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
         "observed": null}
    "#;
    let err = from_json(nested).expect_err("几何对象内重复键必须拒绝");
    assert_eq!(err.code(), Code::Invalid);
}

#[test]
fn json_rejects_non_finite_and_wrong_typed_values() {
    let base = |observed: &str| -> String {
        format!(
            r#"
            {{"kind": "op", "name": "x", "backend": "b",
              "geometry": {{"shape": [1], "batch": 1, "dtype": "f32"}},
              "harness": {{"host": "h", "cpu": "c", "os": "o", "arch": "a",
                          "toolchain": "t", "driver": "d"}},
              "roofline": {{"peak_bw_gb_s": null, "peak_gflops": null}},
              "observed": {observed}}}
            "#
        )
    };
    let cases = [
        base("NaN"),
        base("Infinity"),
        base("-Infinity"),
        base("\"12\""),
        r#"
            {"kind": "dense", "name": "x", "backend": "b",
             "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
             "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                         "toolchain": "t", "driver": "d"},
             "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
             "observed": null}
        "#.to_string(),
        r#"
            {"kind": "op", "name": "x", "backend": "b",
             "geometry": {"shape": [1], "batch": 1.5, "dtype": "f32"},
             "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                         "toolchain": "t", "driver": "d"},
             "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
             "observed": null}
        "#.to_string(),
        r#"
            {"kind": "op", "name": "x", "backend": "b",
             "geometry": {"shape": ["x"], "batch": 1, "dtype": "f32"},
             "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                         "toolchain": "t", "driver": "d"},
             "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
             "observed": null}
        "#.to_string(),
    ];
    for text in cases {
        let err = from_json(&text).expect_err("坏数值 / 类型不符必须拒绝");
        assert_eq!(err.code(), Code::Invalid, "输入：{text}");
    }
}

#[test]
fn json_rejects_missing_required_fields() {
    let cases = [
        r#"{"name": "x", "backend": "b",
            "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
            "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                        "toolchain": "t", "driver": "d"},
            "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
            "observed": null}"#,
        r#"{"kind": "op", "backend": "b",
            "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
            "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                        "toolchain": "t", "driver": "d"},
            "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
            "observed": null}"#,
        r#"{"kind": "op", "name": "x", "backend": "b",
            "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                        "toolchain": "t", "driver": "d"},
            "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
            "observed": null}"#,
        r#"{"kind": "op", "name": "x", "backend": "b",
            "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
            "roofline": {"peak_bw_gb_s": null, "peak_gflops": null},
            "observed": null}"#,
        r#"{"kind": "op", "name": "x", "backend": "b",
            "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
            "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                        "toolchain": "t", "driver": "d"},
            "observed": null}"#,
        r#"{"kind": "op", "name": "x", "backend": "b",
            "geometry": {"shape": [1], "batch": 1, "dtype": "f32"},
            "harness": {"host": "h", "cpu": "c", "os": "o", "arch": "a",
                        "toolchain": "t", "driver": "d"},
            "roofline": {"peak_bw_gb_s": null, "peak_gflops": null}}"#,
    ];
    for text in cases {
        let err = from_json(text).expect_err("缺必填字段必须拒绝");
        assert_eq!(err.code(), Code::Invalid, "输入：{text}");
    }
}
