use super::*;

#[test]
fn ctx_is_appended_from_inside_out() {
    let error = Error::new(Code::Missing, "模板缺配对的 `.k`")
        .with("prof::load")
        .with("cutile::builtin");
    assert_eq!(error.message(), "模板缺配对的 `.k`");
    assert_eq!(error.ctx(), ["prof::load", "cutile::builtin"]);
    assert_eq!(
        error.to_string(),
        "missing: 模板缺配对的 `.k` → prof::load → cutile::builtin"
    );
}

#[test]
fn a_peer_diagnostic_keeps_its_own_category() {
    let first = Error::new(Code::Missing, "缺片段 `load`").with("`cpu`");
    let second = Error::new(Code::Invalid, "`dot` 的见证不成立");
    let error = first.also(&second).with("cutile::builtin::load");
    assert_eq!(error.code(), Code::Missing, "首条的类别不被改写");
    assert_eq!(error.notes().len(), 1);
    assert_eq!(
        error.to_string(),
        "missing: 缺片段 `load` → `cpu` → cutile::builtin::load\n  invalid: `dot` 的见证不成立"
    );
}

#[test]
fn a_missing_path_is_missing_and_a_broken_read_is_io() {
    let absent = std::io::Error::new(std::io::ErrorKind::NotFound, "没有这个文件");
    let error = Error::from_io(Path::new("/tmp/nope"), &absent);
    assert_eq!(error.code(), Code::Missing, "{error}");
    assert!(error.message().contains("/tmp/nope"), "{error}");

    let denied = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "拒绝访问");
    let error = Error::from_io(Path::new("/tmp/nope"), &denied);
    assert_eq!(error.code(), Code::Io, "{error}");
}

#[test]
fn an_error_is_usable_as_a_std_error() {
    let error: Box<dyn std::error::Error> = Box::new(Error::new(Code::Io, "读设备档案失败"));
    assert!(error.to_string().contains("io:"));
}
