//! 身份契约：四个格式闭集共用一套读法——判别值即下标、规范名唯一且能双向解析、
//! `Display` 与 `FromStr` 读同一张行表，并且 trait 视角与 `const fn` 视角指向同一行。
//!
//! 这里只断言**跨闭集共有**的身份契约；每个闭集表里的字面事实（位宽、线上 code、块字节、
//! scale、编码族）由 `src/*.rs` 的内联行表逐条锁定。分层的目的是让每个事实只有一个断言点：
//! 改一张表只会让一处红灯，而不是两个文件各写一遍同一个期望值。

use core::str::FromStr;

use llm_dtype::{BlockFormat, DType, Element, Format, MxType, QType, UnknownCode, UnknownName};

/// 闭集身份：下标、规范名、`Display`、`FromStr` 与按名解析必须指向同一行。
///
/// 不需要 `F: Display` 这个额外 bound：[`Format`] 已经把 `Display` 算作闭集契约的一部分。
fn assert_identity<F>(unknown: &str)
where
    F: Format + FromStr,
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

/// 泛型 trait 视角与 inherent `const fn` 视角必须读出同一份事实：按名解析只能走 trait，
/// 常量求值只能走 inherent，两条路读的必须是同一张行表。
///
/// 两个视角没法写成泛型断言：inherent 方法与 trait 方法同名，泛型上下文里只会解析到 trait。
/// 所以按具体类型展开——新增一个闭集在这里加一行，多一个块格式加一个 `block`。
macro_rules! assert_one_table {
    ($format:ty) => {
        assert_one_table!($format, @identity);
    };
    ($format:ty, block) => {
        assert_one_table!($format, @identity);
        for format in <$format>::ALL {
            assert_eq!(
                BlockFormat::block_len(*format),
                <$format>::block_len(*format),
                "{format:?} 的块长两个视角分家"
            );
            assert_eq!(
                BlockFormat::block_bytes(*format),
                <$format>::block_bytes(*format),
                "{format:?} 的块字节两个视角分家"
            );
            assert_eq!(
                BlockFormat::block_count(*format, 300),
                <$format>::block_count(*format, 300),
                "{format:?} 的块数两个视角分家"
            );
            assert_eq!(
                BlockFormat::bytes_for(*format, 300),
                <$format>::bytes_for(*format, 300),
                "{format:?} 的字节数两个视角分家"
            );
        }
    };
    ($format:ty, @identity) => {
        for format in <$format>::ALL {
            assert_eq!(
                Format::index(*format),
                *format as usize,
                "{format:?} 的判别值必须等于行表下标"
            );
            assert_eq!(
                Format::name(*format),
                <$format>::name(*format),
                "{format:?} 的规范名两个视角分家"
            );
            assert_eq!(
                Format::bits(*format),
                <$format>::bits(*format),
                "{format:?} 的位宽两个视角分家"
            );
            // `DType::parse_name(..)` 这类路径调用不导入 `Format`，是 `annotation` / `prof` /
            // `weights` 实际用的入口；inherent 只转发，不能读另一张表。
            assert_eq!(
                <$format>::parse_name(<$format>::name(*format)),
                Some(*format),
                "{format:?} 的按名解析两个视角分家"
            );
        }
    };
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
fn the_trait_view_forwards_to_the_const_accessors() {
    // 身份与块几何是同一份事实的两个视角：trait 只转发，不能另写一份公式。
    assert_one_table!(DType);
    assert_one_table!(QType, block);
    assert_one_table!(MxType, block);
    assert_one_table!(Element);
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
