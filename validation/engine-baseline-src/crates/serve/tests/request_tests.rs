use super::*;

#[test]
fn a_request_parses_key_value_lines() {
    let request = Request::parse(b"id=7\nprompt=hello world\nmax_new=16\n").expect("合法");
    assert_eq!(request.id(), 7);
    assert_eq!(request.prompt(), "hello world");
    assert_eq!(request.max_new(), 16);
}

/// 值取第一个 `=` 之后的全部：提示里本来就可能有等号（代码片段、`a=b` 这类输入）。
#[test]
fn a_value_may_contain_the_separator() {
    let request = Request::parse(b"id=1\nprompt=x=1+2\nmax_new=1").expect("合法");
    assert_eq!(request.prompt(), "x=1+2");
}

/// 空行与 CRLF 都不该改变文法：传输层不做换行规范化时这里仍然收得下。
#[test]
fn blank_lines_and_carriage_returns_are_tolerated() {
    let request = Request::parse(b"\r\nid=2\r\n\r\nprompt=hi\r\nmax_new=3\r\n").expect("合法");
    assert_eq!(request.id(), 2);
    assert_eq!(request.prompt(), "hi");
}

#[test]
fn a_line_without_a_separator_is_refused() {
    let error = Request::parse(b"id=1\nprompt\nmax_new=2").unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("第 2 行"), "{error}");
}

#[test]
fn an_unknown_field_is_refused_by_name() {
    let error = Request::parse(b"id=1\nprompt=hi\nmax_new=2\ntemperature=0.7").unwrap_err();
    assert!(error.to_string().contains("temperature"), "{error}");
}

#[test]
fn a_duplicate_field_is_refused() {
    let error = Request::parse(b"id=1\nid=2\nprompt=hi\nmax_new=2").unwrap_err();
    assert!(error.to_string().contains("出现了两次"), "{error}");
}

#[test]
fn every_missing_field_is_named() {
    for body in [
        &b"prompt=hi\nmax_new=2"[..],
        &b"id=1\nmax_new=2"[..],
        &b"id=1\nprompt=hi"[..],
    ] {
        let error = Request::parse(body).unwrap_err();
        assert!(error.to_string().contains("缺少字段"), "{error}");
    }
}

#[test]
fn a_number_that_does_not_parse_is_refused_with_its_field() {
    let error = Request::parse(b"id=seven\nprompt=hi\nmax_new=2").unwrap_err();
    assert!(error.to_string().contains("`id`"), "{error}");
    let error = Request::parse(b"id=1\nprompt=hi\nmax_new=-2").unwrap_err();
    assert!(error.to_string().contains("`max_new`"), "{error}");
}

#[test]
fn a_body_that_is_not_utf8_is_refused() {
    let error = Request::parse(&[0xff, 0xfe]).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("UTF-8"), "{error}");
}

/// 空提示是合法的（引擎允许空上下文），但字段本身仍要写出：缺 `prompt` 是另一回事。
#[test]
fn an_empty_prompt_is_legal_but_the_field_is_still_required() {
    let request = Request::parse(b"id=1\nprompt=\nmax_new=1").expect("合法");
    assert_eq!(request.prompt(), "");
    assert_eq!(
        Request::parse(b"id=1\nmax_new=1").unwrap_err().code(),
        Code::Invalid
    );
}
