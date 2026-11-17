use super::*;
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
