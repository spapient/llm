use super::*;
use diagnostic::Code;

#[test]
fn closed_set_round_trip() {
    for text in ["row", "col", "swz<3,3,3>", "pad(4)"] {
        let spec = LayoutSpec::parse(text).unwrap();
        assert_eq!(spec.as_str(), text);
    }
    let error = LayoutSpec::parse("interleaved").unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("不在布局闭集内"), "{error}");
}
