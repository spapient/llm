//! 身份契约：四个格式闭集共用一套读法——判别值即下标、规范名唯一且能双向解析、
//! `Display` 与 `FromStr` 读同一张行表。
//!
//! 这里只断言**跨闭集共有**的身份契约；每个闭集表里的字面事实（位宽、线上 code、块字节、
//! scale、编码族）由 `src/*.rs` 的内联行表逐条锁定。分层的目的是让每个事实只有一个断言点：
//! 改一张表只会让一处红灯，而不是两个文件各写一遍同一个期望值。

use core::fmt::Display;
use core::str::FromStr;

use llm_dtype::{DType, Element, Format, MxType, QType, UnknownCode, UnknownName};

/// 闭集身份：下标、规范名、`Display`、`FromStr` 与按名解析必须指向同一行。
fn assert_identity<F>(unknown: &str)
where
    F: Format + Display + FromStr,
{
    for (index, format) in F::ALL.iter().enumerate() {
        assert_eq!(format.index(), index, "{format:?} 的判别值必须等于行表下标");
        assert_eq!(
            format.name().parse::<F>().ok(),
            Some(*format),
            "{format:?} 的 `FromStr` 与规范名分家"
        );
        assert_eq!(
            Format::parse_name(format.name()),
            Some(*format),
            "{format:?} 的规范名不能往返"
        );
        assert_eq!(
            format!("{format}"),
            format.name(),
            "{format:?} 的 `Display` 与规范名分家"
        );
    }
    assert!(unknown.parse::<F>().is_err(), "`{unknown}` 不在闭集里");
    assert_eq!(F::parse_name(unknown), None, "`{unknown}` 不能被解析");

    let mut names: Vec<&str> = F::ALL.iter().map(|format| format.name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        F::ALL.len(),
        "规范名必须唯一（解析不能有歧义）"
    );
}

#[test]
fn every_closed_set_obeys_the_identity_contract() {
    // 四个闭集跑同一套断言：新增一个元素、一个 MX 格式或一个标量类型不会只补到其中一处。
    assert_identity::<DType>("i7");
    assert_identity::<QType>("q4_9");
    assert_identity::<MxType>("mxfp3");
    assert_identity::<Element>("e1m1");
}

#[test]
fn the_trait_view_matches_the_const_accessors() {
    // trait 是泛型视角、inherent `const fn` 是常量视角；同一张行表读出同一个答案。
    for qtype in QType::ALL {
        assert_eq!(Format::index(*qtype), *qtype as usize);
        assert_eq!(Format::name(*qtype), qtype.name());
        assert_eq!(Format::bits(*qtype), qtype.bits());
    }
    for mx in MxType::ALL {
        assert_eq!(Format::index(*mx), *mx as usize);
        assert_eq!(Format::name(*mx), mx.name());
        assert_eq!(Format::bits(*mx), mx.bits());
    }
    for dtype in DType::ALL {
        assert_eq!(Format::index(*dtype), *dtype as usize);
        assert_eq!(Format::name(*dtype), dtype.name());
        assert_eq!(Format::bits(*dtype), dtype.bits());
    }
    for element in Element::ALL {
        assert_eq!(Format::index(*element), *element as usize);
        assert_eq!(Format::name(*element), element.name());
        assert_eq!(Format::bits(*element), element.bits());
    }
}

#[test]
fn inherent_parse_name_forwards_to_the_trait() {
    // `DType::parse_name(..)` 这类路径调用不导入 `Format`，是 `annotation` / `prof` / `weights`
    // 实际用的入口；inherent 只转发，不能读另一张表。
    for dtype in DType::ALL {
        assert_eq!(DType::parse_name(dtype.name()), Some(*dtype));
    }
    for qtype in QType::ALL {
        assert_eq!(QType::parse_name(qtype.name()), Some(*qtype));
    }
    for mx in MxType::ALL {
        assert_eq!(MxType::parse_name(mx.name()), Some(*mx));
    }
    for element in Element::ALL {
        assert_eq!(Element::parse_name(element.name()), Some(*element));
    }
    assert_eq!(DType::parse_name("i7"), None);
}

#[test]
fn parse_errors_say_what_went_wrong_without_allocating() {
    // 错误类型也是公开 API：描述固定，不因为输入是外部字符串就分配、也不夹带原文。
    assert_eq!(UnknownName.to_string(), "unknown format name");
    assert_eq!(UnknownCode { code: 0 }.to_string(), "unknown dtype code 0");
    assert_eq!(
        UnknownCode { code: u16::MAX }.to_string(),
        "unknown dtype code 65535"
    );
}
