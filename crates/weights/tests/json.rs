//! `json` 模块的单元测试。

use super::*;



#[test]

fn an_object_keeps_its_written_order() {

    let value = parse(r#"{"b": 1, "a": 2}"#).expect("合法");

    let fields = value.object().expect("对象");

    let names: Vec<&str> = fields.iter().map(|(name, _)| name.as_str()).collect();

    assert_eq!(names, ["b", "a"]);

    assert_eq!(value.get("a").and_then(|it| it.integer().ok()), Some(2));

}



#[test]

fn every_shape_the_header_uses_round_trips() {

    let text = r#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;

    let value = parse(text).expect("合法");

    assert_eq!(value.object().expect("对象").len(), 2);

    let entry = value.get("w").expect("有条目");

    assert_eq!(

        entry.get("dtype").and_then(|it| it.string().ok()),

        Some("F32")

    );

    let shape = entry.get("shape").expect("有形状").array().expect("数组");

    assert_eq!(shape.len(), 2);

    assert_eq!(shape[1].integer().expect("整数"), 3);

}



#[test]

fn escapes_including_surrogate_pairs_are_decoded() {

    let value = parse(r#""a\"b\\c\/d\ne\u0041\ud83d\ude00""#).expect("合法");

    assert_eq!(value.string().expect("字符串"), "a\"b\\c/d\neA😀");

}



#[test]

fn non_integers_are_rejected_with_their_position() {

    for text in [r#"{"a": 1.5}"#, r#"{"a": 1e3}"#] {

        let error = parse(text).expect_err("小数或指数");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("只接受整数"), "{error}");

    }

    let leading_zero = parse(r#"{"a": 01}"#).expect_err("前导零");

    assert!(

        leading_zero.to_string().contains("前导零"),

        "{leading_zero}"

    );

}



#[test]

fn duplicate_keys_are_rejected_rather_than_silently_merged() {

    let error = parse(r#"{"a": 1, "a": 2}"#).expect_err("重复键");

    assert!(error.to_string().contains("重复键"), "{error}");

    assert!(error.to_string().contains("位置 "), "{error}");

}



#[test]

fn truncated_and_trailing_input_is_rejected() {

    for text in [r#"{"a": 1"#, r#"{"a": "#, r#"{"a": 1} x"#, ""] {

        assert!(parse(text).is_err(), "{text:?}");

    }

    // 结尾空白是合法的：头部由长度前缀定界，最后一个字节常带 `\n`。

    assert!(parse("{} \n").is_ok());

}



#[test]

fn deep_nesting_is_refused_instead_of_overflowing_the_stack() {

    let text = format!("{}{}", "[".repeat(MAX_DEPTH + 2), "]".repeat(MAX_DEPTH + 2));

    let error = parse(&text).expect_err("过深");

    assert!(error.to_string().contains("嵌套"), "{error}");

}



#[test]

fn accessors_report_the_actual_kind() {

    let value = parse(r#""x""#).expect("合法");

    let error = value.object().expect_err("不是对象");

    assert!(error.to_string().contains("字符串"), "{error}");

    assert_eq!(value.kind(), "字符串");

    assert!(value.get("missing").is_none());

}



#[test]

fn the_ledger_reader_keeps_a_decimal_as_written() {

    let value = parse_ledger(r#"{"volume_ratio": 0.500000, "n": 3}"#).expect("合法");

    let ratio = value.get("volume_ratio").expect("有比值");

    assert_eq!(ratio.number_text().expect("数字"), "0.500000");

    assert_eq!(ratio.kind(), "小数");

    // 整数在账本模式下仍是整数：`3` 与 `3.0` 读回来不是同一个值。

    let count = value.get("n").expect("有计数");

    assert_eq!(count.integer().expect("整数"), 3);

    assert_eq!(count.kind(), "整数");

}



#[test]

fn the_ledger_reader_still_rejects_malformed_numbers() {

    for text in [

        r#"{"a": 00.5}"#,

        r#"{"a": 1.}"#,

        r#"{"a": 1e}"#,

        r#"{"a": 1e+}"#,

    ] {

        let error = parse_ledger(text).expect_err(text);

        assert_eq!(error.code(), Code::Invalid, "{text}");

    }

    // 指数写法保留原文，不折算成 `1000`。

    let value = parse_ledger(r#"{"a": 1e3}"#).expect("合法");

    assert_eq!(

        value.get("a").expect("有值").number_text().expect("数字"),

        "1e3"

    );

    // 头部入口一个数字都不放宽。

    assert!(parse(r#"{"a": 0.5}"#).is_err());

    assert!(parse(r#"{"a": 1e3}"#).is_err());

}
