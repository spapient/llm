//! 静态质量门集成：在真实仓库源文件上跑 ccn/call/dup/docs，`cargo test` 不可绕过。
//!
//! 防假绿：扫描文件数有下限。生产代码先经 `strip_tests` 剔除测试条目；超标项走棘轮——
//! 新增与恶化失败，存量只准降，已达标的基线必须删除。
//!
//! 这个文件只**测量**：限值一律问 `Gate::limit`，本文件不持有阈值、也不决定「哪个度量用
//! 哪个数」。判定与结清都在 `llm_tool::gate` 里，因此换阈值不需要动这里。

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use llm_tool::{
    call, ccn, docs, dup,
    gate::{Gate, Metric, Over},
    lex, strip_tests,
};

const MIN_SOURCES: usize = 100;
/// 文档下限：根 README + 各成员 README + `docs/`（含 ADR）。低于它说明扫描路径写错了。
const MIN_DOCUMENTS: usize = 30;

#[test]
fn every_source_passes_the_static_gate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let files = rust_files(&root.join("crates"));
    assert!(
        files.len() >= MIN_SOURCES,
        "扫描文件数 {} 低于下限 {MIN_SOURCES}",
        files.len()
    );

    let gate = Gate::load().expect("阈值配置加载");
    let mut bad = Vec::new();
    // 身份 → 本次仍超限的度量；棘轮按（身份, 度量）结清，不按身份。
    let mut live = BTreeMap::new();

    for path in &files {
        let name = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
        let raw = std::fs::read_to_string(path).expect("read source");
        let text = strip_tests(&raw).unwrap_or_else(|err| panic!("{name}: {err:?}"));

        if let Some(lines) = dup::file_lines(&text, &gate) {
            judge(&gate, &name, Metric::Lines, lines, &mut bad, &mut live);
        }
        for finding in ccn::analyze(&text).expect("ccn 分析") {
            let key = format!("{name}:{}#{}", finding.name, finding.nth);
            let (ccn, nloc) = (finding.ccn as usize, finding.nloc as usize);
            judge(&gate, &key, Metric::Ccn, ccn, &mut bad, &mut live);
            judge(&gate, &key, Metric::Nloc, nloc, &mut bad, &mut live);
        }
        let calls = call::scan(&text).expect("call 扫描").len();
        judge(&gate, &name, Metric::Call, calls, &mut bad, &mut live);
        if let Err(err) = lex::parse(&raw) {
            bad.push(format!("{name}: 词法错误 {err:?}"));
        }
    }

    bad.extend(gate.stale(&live));

    let documents = document_files(root);
    assert!(
        documents.len() >= MIN_DOCUMENTS,
        "扫描文档数 {} 低于下限 {MIN_DOCUMENTS}",
        documents.len()
    );

    for doc in documents {
        let text = std::fs::read_to_string(&doc).expect("read doc");
        for link in docs::links(&text) {
            let target = doc.parent().unwrap_or(Path::new("")).join(&link.target);
            if !target.exists() {
                bad.push(format!(
                    "{}: 失效链接 {}（行 {}）",
                    doc.display(),
                    link.target,
                    link.line
                ));
            }
        }
    }

    assert!(bad.is_empty(), "静态质量门失败：\n  {}", bad.join("\n  "));
}

/// 记录一次测量：先问门禁是否超限，超限才进入判定与结清名单。
///
/// 「是否超限」也是门禁的回答（[`Gate::limit`]），所以这里既没有阈值也没有比较方向。
fn judge(
    gate: &Gate,
    key: &str,
    metric: Metric,
    value: usize,
    bad: &mut Vec<String>,
    live: &mut BTreeMap<String, BTreeSet<Metric>>,
) {
    if value <= gate.limit(metric) {
        return;
    }
    live.entry(key.to_string()).or_default().insert(metric);
    if let Some(reason) = gate.check(&Over {
        key: key.to_string(),
        metric,
        value,
    }) {
        bad.push(reason);
    }
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(dir, &mut out);
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name() != Some("target".as_ref()) {
                walk(&path, out);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") && is_production(&path) {
            out.push(path);
        }
    }
}

fn is_production(path: &Path) -> bool {
    path.components().any(|part| part.as_os_str() == "src")
        && path.file_name() != Some("tests.rs".as_ref())
}

/// 纳入链接检查的文档：根 README、`docs/**/*.md`（含 ADR）、各成员 README。
///
/// crate 的 README 曾经不在门里，于是目录树、模块清单与 ADR 链接可以长期指向不存在的东西：
/// 设计文档写的是"现在有什么"，它过期就是错误信息，不是文风问题。
fn document_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_markdown(&root.join("docs"), &mut out);
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "md") {
                out.push(path);
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(root.join("crates")) {
        for entry in entries.flatten() {
            let readme = entry.path().join("README.md");
            if readme.is_file() {
                out.push(readme);
            }
        }
    }
    out
}

fn walk_markdown(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_markdown(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
}
