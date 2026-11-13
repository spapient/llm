//! 静态质量门集成：在真实仓库源文件上跑 ccn/call/dup/docs，`cargo test` 不可绕过。
//!
//! 防假绿：扫描文件数有下限。生产代码先经 `strip_tests` 剔除测试条目；超标项走棘轮——
//! 新增与恶化失败，存量只准降，已达标的基线必须删除。

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use llm_tool::{
    call, ccn, docs, dup,
    gate::{Gate, Metric, Over},
    lex, strip_tests,
};

const MIN_SOURCES: usize = 100;

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
    let mut live = BTreeSet::new();

    for path in &files {
        let name = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
        let raw = std::fs::read_to_string(path).expect("read source");
        let text = strip_tests(&raw).unwrap_or_else(|err| panic!("{name}: {err:?}"));

        if let Some(lines) = dup::file_lines(&text, &gate) {
            judge(
                &gate,
                Over {
                    key: name.to_string(),
                    metric: Metric::Lines,
                    value: lines,
                    limit: gate.max_file_lines,
                },
                &mut bad,
                &mut live,
            );
        }
        for finding in ccn::analyze(&text).expect("ccn 分析") {
            let key = format!("{name}:{}#{}", finding.name, finding.nth);
            if finding.ccn as usize > gate.max_ccn {
                judge(
                    &gate,
                    Over {
                        key: key.clone(),
                        metric: Metric::Ccn,
                        value: finding.ccn as usize,
                        limit: gate.max_ccn,
                    },
                    &mut bad,
                    &mut live,
                );
            }
            if finding.nloc as usize > gate.max_nloc {
                judge(
                    &gate,
                    Over {
                        key,
                        metric: Metric::Nloc,
                        value: finding.nloc as usize,
                        limit: gate.max_nloc,
                    },
                    &mut bad,
                    &mut live,
                );
            }
        }
        let calls = call::scan(&text).expect("call 扫描");
        if !calls.is_empty() {
            judge(
                &gate,
                Over {
                    key: name.to_string(),
                    metric: Metric::Call,
                    value: calls.len(),
                    limit: 0,
                },
                &mut bad,
                &mut live,
            );
        }
        if let Err(err) = lex::parse(&raw) {
            bad.push(format!("{name}: 词法错误 {err:?}"));
        }
    }

    bad.extend(gate.stale(&live));

    for doc in markdown(&root.join("docs")) {
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

fn judge(gate: &Gate, over: Over, bad: &mut Vec<String>, live: &mut BTreeSet<String>) {
    live.insert(over.key.clone());
    if let Some(reason) = gate.check(&over) {
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

fn markdown(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
    out
}
