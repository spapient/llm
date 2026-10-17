//! Black-box tests for the `lib::json` value, parser, and serializer.
//!
//! These tests exercise the public `lib` API only, per the repository test
//! organization rules: parsing, value access, serialization, canonical output,
//! and the error positions reported by [`jsonerror`].

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use lib::{jsonerror, jsonnumber, jsonvalue, tostring};

/// A value round-trips through parse and serialize, comparing numbers by value
/// (a real that serializes to an integer literal re-parses as an integer).
fn roundtrip(document: &str) {
    let value = match parse_str(document) {
        Ok(value) => value,
        Err(error) => panic!("failed to parse {document}: {error}"),
    };
    let reencoded = match parse_str(&tostring(&value)) {
        Ok(value) => value,
        Err(error) => panic!("failed to re-parse {document}: {error}"),
    };
    assert!(
        values_equivalent(&value, &reencoded),
        "round-trip mismatch for {document}: {value:?} != {reencoded:?}"
    );
}

fn values_equivalent(left: &jsonvalue, right: &jsonvalue) -> bool {
    match (left, right) {
        (jsonvalue::number(a), jsonvalue::number(b)) => number_to_f64(a) == number_to_f64(b),
        (jsonvalue::array(a), jsonvalue::array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_equivalent(x, y))
        }
        (jsonvalue::object(a), jsonvalue::object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| values_equivalent(v, w)))
        }
        _ => left == right,
    }
}

fn number_to_f64(number: &jsonnumber) -> f64 {
    match number {
        jsonnumber::integer(value) => *value as f64,
        jsonnumber::unsigned(value) => *value as f64,
        jsonnumber::real(value) => *value,
    }
}

fn parse_str(document: &str) -> Result<jsonvalue, jsonerror> {
    lib::parse_str(document)
}

#[test]
fn scalars_parse_and_access() {
    assert!(parse_str("null").unwrap().isnull());
    assert_eq!(parse_str("true").unwrap().getbool(), Some(true));
    assert_eq!(parse_str("false").unwrap().getbool(), Some(false));
    assert_eq!(parse_str("42").unwrap().getinteger(), Some(42));
    assert_eq!(parse_str("-17").unwrap().getinteger(), Some(-17));
    assert_eq!(parse_str("3.5").unwrap().getreal(), Some(3.5));
    assert_eq!(parse_str("\"hello\"").unwrap().getstr(), Some("hello"));
}

#[test]
fn arrays_and_objects_parse() {
    let array = parse_str("[1, 2, 3]").unwrap();
    assert_eq!(array.getarray().map(|v| v.len()), Some(3));
    assert_eq!(array.getarray().unwrap()[0].getinteger(), Some(1));

    let object = parse_str("{\"a\": 1, \"b\": [true, null]}").unwrap();
    assert_eq!(object.get("a").and_then(jsonvalue::getinteger), Some(1));
    let b = object.get("b").and_then(jsonvalue::getarray).unwrap();
    assert_eq!(b[0].getbool(), Some(true));
    assert!(b[1].isnull());
}

#[test]
fn nested_and_empty_containers() {
    assert_eq!(
        parse_str("[]").unwrap().getarray().map(|v| v.len()),
        Some(0)
    );
    assert_eq!(
        parse_str("{}").unwrap().getobject().map(BTreeMap::len),
        Some(0)
    );
    let nested = parse_str("{\"x\": {\"y\": [1, {\"z\": \"v\"}]}}").unwrap();
    assert_eq!(
        nested
            .get("x")
            .and_then(|x| x.get("y"))
            .and_then(jsonvalue::getarray)
            .and_then(|a| a.get(1))
            .and_then(|o| o.get("z"))
            .and_then(jsonvalue::getstr),
        Some("v")
    );
}

#[test]
fn strings_decode_escapes_and_unicode() {
    assert_eq!(
        parse_str(r#""a\"b\\c\/d\be\ff\ng\rh\t""#).unwrap().getstr(),
        Some("a\"b\\c/d\u{08}e\u{0c}f\ng\rh\t")
    );
    assert_eq!(
        parse_str(r#""\u0041\u00e9\u4e2d""#).unwrap().getstr(),
        Some("Aé中")
    );
    // Surrogate pair: U+1F600.
    assert_eq!(
        parse_str(r#""\ud83d\ude00""#).unwrap().getstr(),
        Some("\u{1f600}")
    );
    // Multibyte raw UTF-8.
    assert_eq!(parse_str("\"中\"").unwrap().getstr(), Some("中"));
}

#[test]
fn numbers_keep_integer_precision() {
    assert_eq!(
        parse_str("9223372036854775807").unwrap().getinteger(),
        Some(i64::MAX)
    );
    assert_eq!(
        parse_str("-9223372036854775808").unwrap().getinteger(),
        Some(i64::MIN)
    );
    assert_eq!(
        parse_str("18446744073709551615").unwrap().getunsigned(),
        Some(u64::MAX)
    );
    let real = parse_str("1e10").unwrap();
    assert!(matches!(real, jsonvalue::number(jsonnumber::real(_))));
}

#[test]
fn malformed_input_is_rejected() {
    assert!(parse_str("").is_err());
    assert!(parse_str("nul").is_err());
    assert!(parse_str("[1,]").is_err());
    assert!(parse_str("{\"a\" 1}").is_err());
    assert!(parse_str("{\"a\":}").is_err());
    assert!(parse_str("\"abc").is_err());
    assert!(parse_str("\"\\q\"").is_err());
    assert!(parse_str("01").is_err());
    assert!(parse_str("1.").is_err());
    assert!(parse_str("1e").is_err());
    assert!(parse_str("18446744073709551616").is_err()); // u64::MAX + 1
    assert!(parse_str("1 2").is_err());
    assert!(parse_str(r#""\ud83d""#).is_err()); // lone high surrogate
    assert!(parse_str(r#""\ude00""#).is_err()); // lone low surrogate
}

#[test]
fn error_reports_line_and_column() {
    let error = parse_str("{\n  \"a\": 1,\n  \"b\": tru\n}").unwrap_err();
    assert_eq!(error.getline(), 3);
    assert_eq!(error.getcolumn(), 8);
    assert!(error.to_string().contains("invalid literal"));
}

#[test]
fn serialization_is_canonical_and_compact() {
    assert_eq!(tostring(&parse_str("null").unwrap()), "null");
    assert_eq!(tostring(&parse_str("true").unwrap()), "true");
    assert_eq!(tostring(&parse_str("42").unwrap()), "42");
    assert_eq!(tostring(&parse_str("3.5").unwrap()), "3.5");
    assert_eq!(tostring(&parse_str("\"hi\"").unwrap()), "\"hi\"");
    assert_eq!(tostring(&parse_str("[1,2,3]").unwrap()), "[1,2,3]");
    assert_eq!(
        tostring(&parse_str("{\"b\":2,\"a\":1}").unwrap()),
        "{\"a\":1,\"b\":2}"
    );
    assert_eq!(
        tostring(&parse_str("\"a\\nb\\tc\"").unwrap()),
        "\"a\\nb\\tc\""
    );
}

#[test]
fn random_documents_roundtrip() {
    let documents = [
        "null",
        "true",
        "false",
        "0",
        "-0.5",
        "1.25e3",
        "\"\"",
        "\"text\"",
        "[]",
        "[[]]",
        "{}",
        "{\"\":1}",
        "{\"a\":[1,{\"b\":null}],\"c\":\"d\"}",
        "[[1,2],[3,[4]]]",
        "{\"n\":1.5,\"s\":\"\\u4e2d\\u6587\",\"b\":false}",
    ];
    for document in documents {
        roundtrip(document);
    }
}

#[test]
fn constructed_values_serialize() {
    let mut object = std::collections::BTreeMap::new();
    object.insert("k".to_string(), jsonvalue::string("v".to_string()));
    object.insert("n".to_string(), jsonvalue::number(jsonnumber::integer(7)));
    let value = jsonvalue::object(object);
    assert_eq!(tostring(&value), "{\"k\":\"v\",\"n\":7}");
}
