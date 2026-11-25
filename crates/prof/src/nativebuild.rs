//! 主机 GEMM 微内核的构建期生成：三族配置校验 → `gemm_micro.k` 注解归属校验 →
//! 槽位替换与具名展开 → 静态 C 库编译，并产出契约 Rust 薄包装。
//!
//! 本模块只在构建脚本内经 `build.rs` 的 `#[path]` 编入，不进入运行期 crate。

use std::fmt::Write as _;
use std::path::Path;

use crate::{buildgen, fail};

/// Rust 常量与 C 模板槽位的唯一映射：槽位、微内核族、配置键。
const NATIVE_FIELDS: &[(&str, &str, &str)] = &[
    ("FMA_MR", "fma", "mr"),
    ("FMA_NR", "fma", "nr"),
    ("FMA_KB", "fma", "kb"),
    ("SDOT_MR", "sdot", "mr"),
    ("SDOT_NR", "sdot", "nr"),
    ("SDOT_CHAINS", "sdot", "chains"),
    ("SME_MR", "sme", "mr"),
    ("SME_NR", "sme", "nr"),
    ("SME_MIN_ELEMS", "sme", "min_elems"),
];

/// 构建 `gemm_micro.k`：同一正文供静态生产桥与既有验证 harness 使用。
pub fn build_native(manifest: &Path, out: &Path) {
    let ops = manifest.join("src").join("ops");
    let config = ops.join("kernels.toml");
    let template = ops.join("gemm_micro.k");
    let header = ops.join("gemm_micro.h");
    let bridge = manifest.join("src").join("runtime.rs");
    println!("cargo:rerun-if-changed={}", config.display());
    println!("cargo:rerun-if-changed={}", template.display());
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", bridge.display());
    let table = buildgen::read(&config).unwrap_or_else(|error| fail(format!("{error}")));
    let rendered = render_native(&super::read(&template), &table);
    let c = out.join("gemm_micro.c");
    std::fs::write(&c, rendered)
        .unwrap_or_else(|error| fail(format!("写 {} 失败：{error}", c.display())));
    align_contract(&super::read(&header), &super::read(&bridge));
    let native = native_table(&table);
    let mut contract = String::new();
    for (slot, family, key) in NATIVE_FIELDS {
        let value = native_positive(native, family, key);
        let _ = writeln!(
            contract,
            "#[doc = \"gemm_micro 配置常量。\"] pub const {slot}: usize = {value};"
        );
    }
    let contract_path = out.join("gemm_micro_contract.rs");
    std::fs::write(&contract_path, contract)
        .unwrap_or_else(|error| fail(format!("写 {} 失败：{error}", contract_path.display())));
    compile_native(&c, &ops);
}

/// 头原型 ↔ Rust extern 按 [`NATIVE_SYMBOLS`] 符号集对齐；漂移直接终止构建。
fn align_contract(header: &str, rust: &str) {
    let expected: std::collections::BTreeSet<&str> =
        NATIVE_SYMBOLS.iter().map(|(_, symbol)| *symbol).collect();
    if let Err(error) = crate::fficontract::align(&expected, header, rust) {
        fail(error);
    }
}

/// SME 段无平台裁剪；其余 f32/i8 瓦片已在 Rust intrinsic 中实现，统一单次编译。
fn compile_native(c: &Path, ops: &Path) {
    cc::Build::new()
        .file(c)
        .include(ops)
        .opt_level(3)
        .flag_if_supported("-std=c99")
        .compile("cutile_gemm_micro");
}

/// 生成三族配置常量并校验／渲染所有具名注解。
fn render_native(text: &str, table: &toml::Table) -> String {
    let native = native_table(table);
    validate_native_config(native);
    validate_native_annotations(text, fail);
    let out = replace_native_slots(native, text);
    if out.contains("{{") {
        fail("`gemm_micro.k` 仍有未登记的 `{{…}}` 槽位");
    }
    apply_native_unrolls(native, out)
}

fn native_table(table: &toml::Table) -> &toml::Table {
    table
        .get("gemm_micro")
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| fail("`kernels.toml.gemm_micro` 必须是表"))
}

fn validate_native_config(native: &toml::Table) {
    for (family, allowed) in [
        ("fma", &["mr", "nr", "kb"][..]),
        ("sdot", &["mr", "nr", "chains"][..]),
        ("sme", &["mr", "nr", "min_elems", "unroll"][..]),
    ] {
        validate_native_family(native, family, allowed);
    }
    for family in native.keys() {
        if !["fma", "sdot", "sme"].contains(&family.as_str()) {
            fail(format!("`gemm_micro.{family}` 不是已登记微内核族"));
        }
    }
}

fn validate_native_family(native: &toml::Table, family: &str, allowed: &[&str]) {
    let section = native
        .get(family)
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| fail("`gemm_micro.{family}` 必须是表"));
    for key in section.keys() {
        if !allowed.contains(&key.as_str()) {
            fail(format!("`gemm_micro.{family}.{key}` 不是已登记字段"));
        }
    }
    for key in allowed {
        let _ = native_positive(native, family, key);
    }
}

fn replace_native_slots(native: &toml::Table, text: &str) -> String {
    let mut out = text.to_owned();
    for (slot, family, key) in NATIVE_FIELDS {
        let value = native_positive(native, family, key);
        out = out.replace(&format!("{{{{{slot}}}}}"), &value.to_string());
    }
    out
}

fn apply_native_unrolls(native: &toml::Table, out: String) -> String {
    let marker = "/* cutile-unroll: sme */";
    let factor = native_positive(native, "sme", "unroll");
    let count = out.matches(marker).count();
    if factor > 1 && count != 1 {
        fail(format!(
            "`gemm_micro.k` 的 sme 展开点应恰好一次，实得 {count}"
        ));
    }
    out.replace(marker, &format!("#pragma unroll {factor}"))
}

/// 一个配置字段：必须是非零、可由 usize 承载的整数。
fn native_positive(native: &toml::Table, family: &str, key: &str) -> usize {
    let value = native
        .get(family)
        .and_then(toml::Value::as_table)
        .and_then(|section| section.get(key))
        .and_then(toml::Value::as_integer)
        .unwrap_or_else(|| fail(format!("`gemm_micro.{family}.{key}` 必须是正整数")));
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| fail(format!("`gemm_micro.{family}.{key}` 超出正 usize 范围")))
}

const NATIVE_EXPORTS: [&str; 3] = ["sme", "sme_pack", "sme_stage"];
const NATIVE_PROBES: [(&str, &str); 2] = [("sme_pure", "sme_tile"), ("sme_mix", "sme_tile")];

/// 注解短名 → 链接符号名；对齐门按符号名核对头原型与 Rust extern。
const NATIVE_SYMBOLS: [(&str, &str); 5] = [
    ("sme", "cutile_sme_micro"),
    ("sme_pack", "cutile_sme_pack"),
    ("sme_stage", "cutile_sme_stage"),
    ("sme_pure", "cutile_sme_pure"),
    ("sme_mix", "cutile_sme_mix"),
];

/// 校验 `gemm_micro.k` 的导出、探针归属和具名展开；调用方提供构建期失败出口。
fn validate_native_annotations(text: &str, fail: impl Fn(String) -> !) {
    let mut exports = std::collections::BTreeSet::new();
    let mut probes = std::collections::BTreeSet::new();
    let mut unrolls = std::collections::BTreeSet::new();
    for raw in text.lines().map(str::trim) {
        validate_export(raw, &mut exports, &fail);
        validate_probe(raw, &mut probes, &fail);
        validate_unroll(raw, &mut unrolls, &fail);
    }
    require_names(&exports, NATIVE_EXPORTS, "生产导出", &fail);
    require_names(&probes, NATIVE_PROBES.map(|(name, _)| name), "探针", &fail);
}

fn require_names<'a>(
    names: &std::collections::BTreeSet<String>,
    required: impl IntoIterator<Item = &'a str>,
    kind: &str,
    fail: &impl Fn(String) -> !,
) {
    for name in required {
        if !names.contains(name) {
            fail(format!("`gemm_micro.k` 缺{kind} `{name}`"));
        }
    }
}

fn annotation<'a>(line: &'a str, kind: &str) -> Option<&'a str> {
    line.strip_prefix(&format!("/* {kind}"))?
        .strip_suffix("*/")
        .map(str::trim)
}

fn validate_export(
    raw: &str,
    exports: &mut std::collections::BTreeSet<String>,
    fail: &impl Fn(String) -> !,
) {
    if let Some(name) = annotation(raw, "cutile-export:")
        && (!NATIVE_EXPORTS.contains(&name) || !exports.insert(name.to_string()))
    {
        fail(format!("未知或重复的 `cutile-export` 名字 `{name}`"));
    }
}

fn validate_probe(
    raw: &str,
    probes: &mut std::collections::BTreeSet<String>,
    fail: &impl Fn(String) -> !,
) {
    if let Some(body) = annotation(raw, "cutile-probe:") {
        let (name, core) = probe_fields(body, fail);
        let expected = probe_core(name, fail);
        if core != expected || !probes.insert(name.to_string()) {
            fail(format!("重复或核心不符的 `cutile-probe` `{body}`"));
        }
    }
}

fn probe_fields<'a>(body: &'a str, fail: &impl Fn(String) -> !) -> (&'a str, &'a str) {
    let Some((name, core)) = body.split_once(',') else {
        fail(format!("探针 `{body}` 缺 `core=<名字>`"));
    };
    let Some(core) = core.trim().strip_prefix("core=") else {
        fail(format!("探针 `{body}` 缺 `core=<名字>`"));
    };
    if core.contains(',') {
        fail(format!("探针 `{body}` 含未知字段"));
    }
    (name.trim(), core)
}

fn probe_core<'a>(name: &str, fail: &impl Fn(String) -> !) -> &'a str {
    let Some((_, core)) = NATIVE_PROBES.iter().find(|(probe, _)| *probe == name) else {
        fail(format!("未知 `cutile-probe` 名字 `{name}`"));
    };
    core
}

fn validate_unroll(
    raw: &str,
    unrolls: &mut std::collections::BTreeSet<String>,
    fail: &impl Fn(String) -> !,
) {
    if let Some(name) = annotation(raw, "cutile-unroll:")
        && (!["sme"].contains(&name) || !unrolls.insert(name.to_string()))
    {
        fail(format!("未知或重复的具名展开 `{name}`"));
    }
}
