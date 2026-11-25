//! 片段签名声明的纯字符串规则：`/* cutile-sign: <prim>/<impl> */`（ADR 0027 §5），
//! 与片段语料的三个保留词（标记、见证文件名、尾块文件名）。
//!
//! 同一条规则有两个调用点，而它们不能共用编译产物：本 crate 的 `build.rs` 在**构建期**拿它比对
//! 档案里的 `provides`（扫描语料目录要区分"片段 / 见证 / 尾块"），`Fragment::parse` 在**装载期**
//! 解析同一行。文件住在 `src/` 里的无依赖模块，构建脚本用 `#[path]` 把它编进来——于是构建期与
//! 装载期读的是同一份文本，而不是两份互相漂移的抄写。
//!
//! 本模块不认识原语闭集，也不构造诊断：闭集比对、`impl` 非空判定与错误格式化留给调用方。
//!
//! # 例子
//!
//! ```
//! use llm_prof::sign;
//!
//! let declared = sign::strip_marker("/* cutile-sign: dot/fma */").expect("这是声明");
//! assert_eq!(sign::split_slash(declared), Some(("dot", "fma")));
//! assert_eq!(sign::strip_marker("/* 别的注释 */"), None);
//! ```

/// 签名声明的标记：片段第一行 `/* cutile-sign: <prim>/<impl> */` 里的 `cutile-sign:`。
pub const MARKER: &str = "cutile-sign:";

/// 原生算子模板标记：只有显式声明的完整翻译单元可以绕过 DSL 子集解析。
#[allow(dead_code)] // 运行库使用；构建脚本共享本模块但不走原生整程序路径。
pub const NATIVE_MARKER: &str = "cutile-native:";

/// 见证源码的文件名（不含扩展名）。
pub const WITNESS_STEM: &str = "witness";

/// 可选前置声明的文件名（不含扩展名）：`preamble.<ext>`。
pub const PREAMBLE_STEM: &str = "preamble";

/// 写回尾块片段的文件名（不含扩展名）：一份文本装六种形式。
///
/// 它的签名声明不带 `/`（`/* cutile-sign: epilogue */`）：原语片段的声明回答"这一段实现哪条
/// 签名"，而一份尾块文本实现六种形式，逐个形式写声明会让同一份文件出现六条。
pub const EPILOGUE_STEM: &str = "epilogue";

/// 剥掉 `/* */` 外壳与 [`MARKER`]，得到声明正文 `<prim>/<impl>`；形态不符返回 `None`。
pub fn strip_marker(first: &str) -> Option<&str> {
    strip_annotation(first, MARKER)
}

/// 原生模板名必须同时匹配首行标记与算子名；普通模板不凭解析失败切换路径。
#[allow(dead_code)] // 同上：这是运行库与 shard 的共享分类入口。
pub fn is_native_template(source: &str, op: &str) -> bool {
    source
        .lines()
        .next()
        .and_then(|line| strip_annotation(line, NATIVE_MARKER))
        == Some(op)
}

fn strip_annotation<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    line.trim()
        .strip_prefix("/*")?
        .strip_suffix("*/")
        .map(str::trim)?
        .strip_prefix(marker)
        .map(str::trim)
}

/// 把 `<prim>/<impl>` 按 `/` 切成两半；缺 `/` 返回 `None`。
pub fn split_slash(decl: &str) -> Option<(&str, &str)> {
    decl.split_once('/')
}

const NATIVE_EXPORTS: [&str; 6] = ["fma", "fma_scalar", "sdot", "sme", "sme_pack", "sme_stage"];
const NATIVE_PROBES: [(&str, &str); 6] = [
    ("fma_pure", "fma_tile"),
    ("fma_block", "fma_tile"),
    ("sdot_pure", "sdot_tile"),
    ("sdot_block", "sdot_tile"),
    ("sme_pure", "sme_tile"),
    ("sme_mix", "sme_tile"),
];

/// 校验 `gemm_micro.k` 的导出、探针归属和具名展开；调用方提供构建期失败出口。
#[allow(dead_code)] // 构建脚本共享；运行库只装载已有签名。
pub(crate) fn validate_native_annotations(text: &str, fail: impl Fn(String) -> !) {
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
        && (!["fma_neon", "fma_avx2", "fma_scalar", "sdot", "sme"].contains(&name)
            || !unrolls.insert(name.to_string()))
    {
        fail(format!("未知或重复的具名展开 `{name}`"));
    }
}

/// 由 `cutile-probe` 语料生成的 Rust 薄包装；保持既有公开名字，生产默认不包含。
#[allow(dead_code)]
pub(crate) const NATIVE_PROBE_RS: &str = r#"
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[doc = "FMA 纯端口探针：环内无内存访问。"]
pub fn port_fma(rounds: usize) -> bool {
    if rounds == 0 { return false; }
    unsafe { cutile_fma_pure(rounds); }
    true
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[doc = "sdot 纯端口探针：无 dotprod 时拒绝执行。"]
pub fn port_sdot(rounds: usize) -> bool {
    if rounds == 0 || !sdot_available() { return false; }
    unsafe { cutile_sdot_pure(rounds); }
    true
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[doc = "FMA 取数探针：调用与生产相同的 fma_tile。"]
pub fn port_fma_block(rounds: usize, a: &[f32], b: &[f32], k: usize) -> bool {
    let Ok(need) = area(FMA_NR, k, "FMA probe") else { return false; };
    let Ok(width) = c_int(k, "k") else { return false; };
    if rounds == 0 || k == 0 || a.len() < need || b.len() < need { return false; }
    unsafe { cutile_fma_block(rounds, a.as_ptr(), b.as_ptr(), width); }
    true
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[doc = "sdot 取数探针：调用与生产相同的 sdot_tile。"]
pub fn port_sdot_block(rounds: usize, a: &[i8], b: &[i8], k: usize) -> bool {
    let Ok(need) = area(SDOT_MR, k, "sdot probe") else { return false; };
    let Ok(width) = c_int(k, "k") else { return false; };
    if rounds == 0 || k == 0 || !k.is_multiple_of(16) || !sdot_available()
        || a.len() < need || b.len() < need { return false; }
    unsafe { cutile_sdot_block(rounds, a.as_ptr(), b.as_ptr(), width); }
    true
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[doc = "SME 纯外积探针：与生产共用 fmopa 核心，环内无内存访问。"]
pub fn port_run(rounds: usize) -> bool {
    if rounds == 0 || !sme_available() { return false; }
    unsafe { cutile_sme_pure(rounds); }
    true
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[doc = "SME 固定缓存行取数探针：与生产共用 fmopa 核心。"]
pub fn port_mix(rounds: usize, a: &[f32; SME_MR], b: &[f32; SME_NR]) -> bool {
    if rounds == 0 || !sme_available() { return false; }
    unsafe { cutile_sme_mix(rounds, a.as_ptr(), b.as_ptr()); }
    true
}
"#;
