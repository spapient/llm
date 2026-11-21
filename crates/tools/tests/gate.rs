//! 静态质量门集成：在真实仓库源文件上跑 ccn/call/dup/docs，`cargo test` 不可绕过。
//!
//! 防假绿：扫描文件数有下限。生产代码先经 `strip_tests` 剔除测试条目；超标项走棘轮——
//! 新增与恶化失败，存量只准降，已达标的基线必须删除。
//!
//! 这个文件只**测量**：限值一律问 `Gate::limit`，本文件不持有阈值、也不决定「哪个度量用
//! 哪个数」。判定与结清都在 `llm_tools::gate` 里，因此换阈值不需要动这里。
//!
//! 扫描分四类，适用的度量不同：
//!
//! - **生产代码**（`crates/*/src` 非测试夹具；入口二进制 `cli` 的 `src/main.rs` 与 `src/bin/cutile.rs` 也在此列）：全套门——
//!   裸 unwrap/expect/panic、ccn、nloc、单文件行数、词法。
//! - **构建脚本**（`crates/*/build.rs`）：守门词法与单文件行数；脚本对 cargo 注入的环境变量
//!   用 `expect`、用 `fail()` panic 是构建期惯例，不套运行时零 panic 门，复杂度指标随脚本
//!   清理再纳入。
//! - **外部集成测试**（`crates/*/tests/`）：测试代码惯例可用 `expect`/`unwrap`，只守门词法
//!   能解析，不套生产限值。
//!
//! 本仓不设 `examples/`：可演示的用法由测试夹具与集成测试承担，不再单开一类示例程序。
//!
//! 另有 CCN 跨实现对拍：以旧 `tools gate ccn`（syn 实现）为基线，验证零依赖新分析器在真实
//! 源文件上逐函数零漂移。旧仓库/二进制不在本机时静默跳过，保证 CI 可移植。

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use llm_tools::{
    analyze, call, ccn, docs, dup,
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
    let crates = root.join("crates");

    let production = rust_files(&crates);
    let build_scripts = build_script_files(&crates);
    let integration_tests = integration_test_files(&crates);

    let scanned = production.len() + build_scripts.len();
    assert!(
        scanned >= MIN_SOURCES,
        "扫描文件数 {scanned} 低于下限 {MIN_SOURCES}",
    );

    let gate = Gate::load().expect("阈值配置加载");
    let mut bad = Vec::new();
    // 身份 → 本次仍超限的度量；棘轮按（身份, 度量）结清，不按身份。
    let mut live = BTreeMap::new();

    for path in &production {
        scan_production(&gate, root, path, &mut bad, &mut live);
    }
    for path in &build_scripts {
        scan_build_script(&gate, root, path, &mut bad, &mut live);
    }
    for path in &integration_tests {
        scan_integration_test(root, path, &mut bad);
    }

    bad.extend(gate.stale(&live));
    bad.extend(forbidden_example_dirs(root, &crates));

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

/// 生产代码与根入口：全套度量。内联 `#[cfg(test)]` 条目先剔除再量复杂度/行数/调用形态。
fn scan_production(
    gate: &Gate,
    root: &Path,
    path: &Path,
    bad: &mut Vec<String>,
    live: &mut BTreeMap<String, BTreeSet<Metric>>,
) {
    let name = label(root, path);
    let raw = std::fs::read_to_string(path).expect("read source");
    let text = strip_tests(&raw).unwrap_or_else(|err| panic!("{name}: {err:?}"));

    if let Some(lines) = dup::file_lines(&text, gate) {
        judge(gate, &name, Metric::Lines, lines, bad, live);
    }
    for finding in ccn::analyze(&text).expect("ccn 分析") {
        let key = format!("{name}:{}#{}", finding.name, finding.nth);
        let (ccn, nloc) = (finding.ccn as usize, finding.nloc as usize);
        judge(gate, &key, Metric::Ccn, ccn, bad, live);
        judge(gate, &key, Metric::Nloc, nloc, bad, live);
    }
    let calls = call::scan(&text).expect("call 扫描").len();
    judge(gate, &name, Metric::Call, calls, bad, live);
    if let Err(err) = lex::parse(&raw) {
        bad.push(format!("{name}: 词法错误 {err:?}"));
    }
}

/// 构建脚本：词法 + 单文件行数。构建期 `expect(OUT_DIR)` 与 `fail()` panic 不纳入零 panic 门。
fn scan_build_script(
    gate: &Gate,
    root: &Path,
    path: &Path,
    bad: &mut Vec<String>,
    live: &mut BTreeMap<String, BTreeSet<Metric>>,
) {
    let name = label(root, path);
    let raw = std::fs::read_to_string(path).expect("read build script");
    if let Some(lines) = dup::file_lines(&raw, gate) {
        judge(gate, &name, Metric::Lines, lines, bad, live);
    }
    if let Err(err) = lex::parse(&raw) {
        bad.push(format!("{name}: 词法错误 {err:?}"));
    }
}

/// 外部集成测试：只守门词法可解析；生产限值不套到测试代码上。
fn scan_integration_test(root: &Path, path: &Path, bad: &mut Vec<String>) {
    let name = label(root, path);
    let raw = std::fs::read_to_string(path).expect("read integration test");
    if let Err(err) = lex::parse(&raw) {
        bad.push(format!("{name}: 词法错误 {err:?}"));
    }
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
    walk(dir, &mut out, "rs", true, &is_production);
    out
}

/// 各 crate 根目录下的 `build.rs`（不在 `src` 里，原扫描集漏掉了这一类）。
fn build_script_files(crates: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(crates) else {
        return out;
    };
    for entry in entries.flatten() {
        let build = entry.path().join("build.rs");
        if build.is_file() {
            out.push(build);
        }
    }
    out
}

/// 本仓不设示例程序：根与各 crate 下出现 `examples/` 目录即判失败，可演示用法由测试夹具
/// 与集成测试承担。
fn forbidden_example_dirs(root: &Path, crates: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut check = |dir: &Path| {
        let example = dir.join("examples");
        if example.is_dir() {
            found.push(example.display().to_string());
        }
    };
    check(root);
    if let Ok(entries) = std::fs::read_dir(crates) {
        for entry in entries.flatten() {
            check(&entry.path());
        }
    }
    found
}

/// 各 crate `tests/` 目录下的外部集成测试（与内联 `#[cfg(test)]` 模块分开）。
fn integration_test_files(crates: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(crates) else {
        return out;
    };
    for entry in entries.flatten() {
        let tests = entry.path().join("tests");
        if tests.is_dir() {
            walk(&tests, &mut out, "rs", true, &|_| true);
        }
    }
    out
}

/// 递归收集匹配扩展名的文件；`skip_target` 为真时跳过 `target` 目录，`keep` 是文件谓词。
fn walk(
    dir: &Path,
    out: &mut Vec<PathBuf>,
    ext: &str,
    skip_target: bool,
    keep: &dyn Fn(&Path) -> bool,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let is_target = path.file_name() == Some("target".as_ref());
            if !(skip_target && is_target) {
                walk(&path, out, ext, skip_target, keep);
            }
        } else if path.extension().is_some_and(|e| e == ext) && keep(&path) {
            out.push(path);
        }
    }
}

fn is_production(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return true;
    };
    let test_fixture = name == "tests.rs" || name.ends_with("_tests.rs");
    path.components().any(|part| part.as_os_str() == "src") && !test_fixture
}

/// 仓库内相对路径，失败时退回原样，保证坏消息里能定位文件。
/// 分隔符统一成 `/`：台账基线按正斜杠登记，Windows 的反斜杠不能参与匹配。
fn label(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// 纳入链接检查的文档：根 README、`docs/**/*.md`（含 ADR）、各成员 README。
///
/// crate 的 README 曾经不在门里，于是目录树、模块清单与 ADR 链接可以长期指向不存在的东西：
/// 设计文档写的是"现在有什么"，它过期就是错误信息，不是文风问题。
fn document_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(&root.join("docs"), &mut out, "md", false, &|_| true);
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

// ---------------------------------------------------------------------------
// CCN 跨实现对拍
// ---------------------------------------------------------------------------
//
// 以 llm.u 旧 `tools gate ccn`（syn 实现）为基线，验证零依赖新分析器在真实源文件上逐函数
// 零漂移。旧仓库/二进制不在本机时静默跳过，保证 CI 可移植。

/// 旧仓库根与旧工具二进制（本机开发环境路径）。
const OLD_ROOT: &str = "/Users/fei/code/llm.u";
const OLD_BIN: &str = "/Users/fei/code/llm.u/target/debug/tools";

/// 对拍文件数下限：低于此数说明遍历坏了，而不是"仓库没有源文件"。
const MIN_FILES: usize = 40;

#[test]
fn ccn_matches_the_syn_baseline_on_every_old_source() {
    if testkit::skip(format!(
        "tools: 旧 tools 基线二进制不存在（{OLD_BIN}），CCN 对拍用例未运行"
    )) {
        return;
    }
    let files = sources(Path::new(OLD_ROOT));
    assert!(
        files.len() >= MIN_FILES,
        "对拍文件数 {} 低于下限 {MIN_FILES}",
        files.len()
    );

    let baseline = baseline(&files);
    let mut drift = 0;
    for file in &files {
        let src = std::fs::read_to_string(file).expect("read source");
        let got: BTreeMap<(String, usize), u32> = analyze(&src)
            .unwrap_or_else(|err| panic!("analyze {file:?}: {err:?}"))
            .into_iter()
            .map(|f| ((f.name, f.nth), f.ccn))
            .collect();
        let want = match baseline.get(&file.to_string_lossy().to_string()) {
            Some(rows) => rows,
            None => continue,
        };
        let want_keys: BTreeSet<_> = want.keys().collect();
        let got_keys: BTreeSet<_> = got.keys().collect();
        for key in want_keys.symmetric_difference(&got_keys) {
            eprintln!("DRIFT key {file:?} {key:?}");
            drift += 1;
        }
        for key in want_keys.intersection(&got_keys) {
            if want[key] != got[key] {
                eprintln!(
                    "DRIFT ccn {file:?} {key:?} baseline={} new={}",
                    want[key], got[key]
                );
                drift += 1;
            }
        }
    }
    assert_eq!(drift, 0, "CCN 对拍出现 {drift} 处漂移");
}

/// 旧二进制输出：文件 → (函数名, 序号) → ccn。
fn baseline(files: &[PathBuf]) -> BTreeMap<String, BTreeMap<(String, usize), u32>> {
    let output = Command::new(OLD_BIN)
        .arg("gate")
        .arg("ccn")
        .args(files)
        .output()
        .expect("run old tools");
    assert!(
        output.status.success(),
        "旧 gate ccn 失败：{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut map = BTreeMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 5 {
            continue;
        }
        let (path, identity) = (cols[0].to_string(), cols[1]);
        let Some(ccn) = cols
            .iter()
            .find(|c| c.starts_with("ccn="))
            .and_then(|c| c[4..].parse().ok())
        else {
            continue;
        };
        let Some((name, nth)) = identity.split_once('#') else {
            continue;
        };
        let nth: usize = nth.parse().expect("nth");
        map.entry(path)
            .or_insert_with(BTreeMap::new)
            .insert((name.to_string(), nth), ccn);
    }
    map
}

/// 递归收集仓库内全部 `.rs` 文件（排除 target）。
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_sources(dir, &mut out);
    out
}

fn walk_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read dir").flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name() == Some("target".as_ref())
                || path.file_name() == Some(".git".as_ref())
            {
                continue;
            }
            walk_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}
