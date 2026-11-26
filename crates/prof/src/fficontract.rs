//! C 头原型 ↔ Rust extern 的领域词法解析层。
//!
//! 通用符号名集与宽类序列对齐内核在 [`diagnostic::scan`]；本文件只把本仓 FFI 的领域词法
//! （`cutile_` 前缀、`const float *` / `*const f32` / `usize` 等）映射成
//! [`FfiSignature`] 后交给通用内核。参数名不参与对齐：C 侧 `src`/`dst` 与 Rust 侧局部名
//! 差异合法，只核指针 const 性与整数宽度这一 ABI 形状。

use diagnostic::Result;
use diagnostic::scan::{FfiParam, FfiSignature, ffi_align};
use std::collections::BTreeSet;

/// 核对 `expected` 符号集在头与 Rust extern 中名集相等、宽类序列一致；左为头、右为 extern。
pub fn align(expected: &BTreeSet<&str>, header: &str, rust: &str) -> Result<()> {
    ffi_align(expected, &parse_header(header), &parse_rust_externs(rust))
}

/// 解析头里的 `void cutile_*(…);` 原型；跨折行按 `;` 分段。
fn parse_header(text: &str) -> Vec<FfiSignature> {
    text.split(';')
        .filter_map(|chunk| {
            let at = chunk.find("void cutile_")?;
            let rest = &chunk[at + "void ".len()..];
            let (name, args) = split_prototype(rest)?;
            let params = split_params(args)
                .map(header_param)
                .collect::<Option<Vec<_>>>()?;
            Some(FfiSignature {
                name: name.to_owned(),
                params,
            })
        })
        .collect()
}

/// 解析 Rust 桥里的 `fn cutile_*(…);` extern 声明；签名可跨多行，按 `;` 分段收集。
fn parse_rust_externs(text: &str) -> Vec<FfiSignature> {
    text.split(';')
        .filter_map(|chunk| {
            let at = chunk.find("fn cutile_")?;
            let rest = &chunk[at + "fn ".len()..];
            let (name, args) = split_prototype(rest)?;
            let params = split_params(args)
                .map(rust_param)
                .collect::<Option<Vec<_>>>()?;
            Some(FfiSignature {
                name: name.to_owned(),
                params,
            })
        })
        .collect()
}

/// 从 `name(…)` 正文取出符号名与括号内参数字符串。
fn split_prototype(rest: &str) -> Option<(&str, &str)> {
    let paren = rest.find('(')?;
    let close = rest[paren + 1..].find(')')?;
    Some((rest[..paren].trim(), &rest[paren + 1..paren + 1 + close]))
}

fn split_params(args: &str) -> impl Iterator<Item = &str> {
    args.split(',').filter(|param| !param.trim().is_empty())
}

fn header_param(raw: &str) -> Option<FfiParam> {
    let token = raw.trim();
    if token.contains('*') {
        Some(if token.starts_with("const") {
            FfiParam::PtrConst
        } else {
            FfiParam::PtrMut
        })
    } else if token.starts_with("size_t") {
        Some(FfiParam::Size)
    } else {
        None
    }
}

fn rust_param(raw: &str) -> Option<FfiParam> {
    let ty = raw.split(':').nth(1)?.trim();
    match ty {
        "*const f32" => Some(FfiParam::PtrConst),
        "*mut f32" => Some(FfiParam::PtrMut),
        "usize" => Some(FfiParam::Size),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str =
        "void cutile_a(const float *x, float *y, size_t n);\nvoid cutile_b(size_t rounds);";
    const RUST: &str =
        "fn cutile_a(x: *const f32, y: *mut f32, n: usize);\nfn cutile_b(rounds: usize);";

    #[cfg(test)]
    testkit::test_cases! {
    fn header_parses_pointer_and_size_classes() {
        let sigs = parse_header(HEADER);
        let a = sigs
            .iter()
            .find(|sig| sig.name == "cutile_a")
            .expect("cutile_a");
        assert_eq!(
            a.params,
            vec![FfiParam::PtrConst, FfiParam::PtrMut, FfiParam::Size]
        );
    }

    fn rust_externs_parse_matching_classes() {
        let sigs = parse_rust_externs(RUST);
        let b = sigs
            .iter()
            .find(|sig| sig.name == "cutile_b")
            .expect("cutile_b");
        assert_eq!(b.params, vec![FfiParam::Size]);
    }

    fn multiline_rust_externs_parse() {
        let rust = "fn cutile_a(\n    x: *const f32,\n    y: *mut f32,\n    n: usize\n);";
        let sigs = parse_rust_externs(rust);
        let a = sigs
            .iter()
            .find(|sig| sig.name == "cutile_a")
            .expect("cutile_a");
        assert_eq!(
            a.params,
            vec![FfiParam::PtrConst, FfiParam::PtrMut, FfiParam::Size]
        );
    }

    fn aligned_signatures_pass() {
        let expected: BTreeSet<&str> = ["cutile_a", "cutile_b"].into_iter().collect();
        assert!(align(&expected, HEADER, RUST).is_ok());
    }

    fn shape_mismatch_is_rejected() {
        let rust = "fn cutile_a(x: *mut f32, y: *mut f32, n: usize);";
        let expected: BTreeSet<&str> = ["cutile_a"].into_iter().collect();
        let err = align(&expected, HEADER, rust).unwrap_err();
        assert!(err.to_string().contains("形状不一致"));
    }

    fn missing_and_unknown_symbols_are_rejected() {
        let expected: BTreeSet<&str> = ["cutile_a"].into_iter().collect();
        let rust = "fn cutile_z(x: usize);";
        let err = align(&expected, HEADER, rust).unwrap_err();
        assert!(err.to_string().contains("缺签名") || err.to_string().contains("未登记"));
    }
    }
}
