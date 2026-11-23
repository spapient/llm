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
//! - **挂载的测试模块**（`src` 里 `#[cfg(test)] mod 名字;` 指向的文件）：与内联 cfg(test)
//!   条目同理——运行时构建看不到它们，所以只守门词法，不套生产限值。
//!
//! 本仓不设 `examples/`：可演示的用法由测试夹具与集成测试承担，不再单开一类示例程序。
//!
//! 另有 CCN 跨实现对拍：以旧 `tools gate ccn`（syn 实现）为基线，验证零依赖新分析器在真实
//! 源文件上逐函数零漂移。旧仓库/二进制不在本机时静默跳过，保证 CI 可移植。

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use diagnostic::data::{Table, Value};
use llm_tools::{
    analyze, call, ccn, docs, dup,
    gate::{Gate, Metric, Over},
    lex,
    strip::Mount,
    strip_tests, submodules, test_modules, words,
};

const MIN_SOURCES: usize = 100;
/// 文档下限：根 README + 各成员 README + `docs/`（含 ADR）。低于它说明扫描路径写错了。
///
/// 它探测的是"扫描路径还在扫到东西"，不是文档数量目标——合并文档是本仓的既定方向，所以下限要
/// 明显低于当前篇数，否则每合并一次就得改门禁。
const MIN_DOCUMENTS: usize = 12;

#[test]
fn every_source_passes_the_static_gate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let crates = root.join("crates");

    let production = rust_files(&crates);
    let mounted_tests = mounted_test_modules(&production);
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
        if mounted_tests.contains(path) {
            scan_integration_test(&gate, root, path, &mut bad);
        } else {
            scan_production(&gate, root, path, &mut bad, &mut live);
        }
    }
    for path in &build_scripts {
        scan_build_script(&gate, root, path, &mut bad, &mut live);
    }
    for path in &integration_tests {
        scan_integration_test(&gate, root, path, &mut bad);
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
    vocabulary(&name, &raw, bad);
}

/// 一概念一词：标识符级词表。
///
/// `scripts/name-lint.sh` 只看文件名，看不到 `buffer` 与 `buf` 同时存在于一个概念上；这里
/// 按标识符判定，外来 API 名（Metal 选择器、`char_indices`、`Index` 等）在词表里豁免。
/// 用原文而非 `strip_tests` 后的文本：测试与被测代码是同一个词表。
fn vocabulary(name: &str, src: &str, bad: &mut Vec<String>) {
    for hit in words::scan(src).unwrap_or_else(|err| panic!("{name}: {err:?}")) {
        bad.push(format!("{name}: {}", hit.describe()));
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
    vocabulary(&name, &raw, bad);
}

/// 外部集成测试与 `#[path]` 挂进 `src` 的测试模块。
///
/// 函数长度与圈复杂度是生产面的度量，不套到测试代码上；但**文件行数受
/// [`Gate::test_limit`] 管**——测试巨石和口袋文件是同一个病：一个文件里躺着几十条互不
/// 相干的用例，改一条得先读懂全部。
fn scan_integration_test(gate: &Gate, root: &Path, path: &Path, bad: &mut Vec<String>) {
    let name = label(root, path);
    let raw = std::fs::read_to_string(path).expect("read integration test");
    if let Err(err) = lex::parse(&raw) {
        bad.push(format!("{name}: 词法错误 {err:?}"));
    }
    vocabulary(&name, &raw, bad);
    let lines = raw.lines().count();
    if lines > gate.test_limit() {
        bad.push(format!(
            "{name}: 测试文件 {lines} 行超过上限 {}——按主题拆成 `tests/<主题>/` 下的多个模块",
            gate.test_limit()
        ));
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
    if let Some(reason) = gate.reject_reason(&Over {
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

/// `#[cfg(test)] mod 名字;` 挂载的文件集合；递归展开——被挂载的文件自己也可能挂载子模块，
/// 那些子模块随父文件一起只在 `cfg(test)` 下编译。
///
/// 声明解析在 [`llm_tools::strip`]，本函数只负责把模块名映射回文件系统。
fn mounted_test_modules(production: &[PathBuf]) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    let mut queue: Vec<(PathBuf, bool)> = production
        .iter()
        .map(|path| (path.clone(), false))
        .collect();
    while let Some((file, inherited)) = queue.pop() {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let names = if inherited {
            submodules(&text)
        } else {
            test_modules(&text)
        };
        let Ok(names) = names else {
            continue;
        };
        for mount in names {
            let target = mount_candidates(&file, &mount)
                .into_iter()
                .find(|path| path.is_file());
            if let Some(target) = target
                && out.insert(target.clone())
            {
                queue.push((target, true));
            }
        }
    }
    out
}

/// `mod 名字;` 的落点：显式 `#[path]` 相对当前文件目录优先，否则按模块名推断
/// `目录/名字.rs` 与 `目录/名字/mod.rs`。
fn mount_candidates(file: &Path, mount: &Mount) -> Vec<PathBuf> {
    let parent = file.parent().unwrap_or_else(|| Path::new(""));
    let mut out = Vec::new();
    if let Some(path) = &mount.path {
        out.push(parent.join(path));
    }
    let dir = module_dir(file);
    out.push(dir.join(format!("{}.rs", mount.name)));
    out.push(dir.join(mount.name.clone()).join("mod.rs"));
    out
}

/// 声明文件的子模块目录：`lib.rs`/`main.rs`/`mod.rs` 的子模块与自身同目录，其余以文件名命名
/// （`a/b.rs` 的子模块在 `a/b/` 下）。
fn module_dir(file: &Path) -> PathBuf {
    let parent = file.parent().unwrap_or_else(|| Path::new(""));
    match file.file_stem().and_then(std::ffi::OsStr::to_str) {
        Some("lib" | "main" | "mod") | None => parent.to_path_buf(),
        Some(stem) => parent.join(stem),
    }
}

#[test]
fn cfg_test_mount_sets_test_scope_only_under_test_chain() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let production = rust_files(&root.join("crates"));
    let mounted = mounted_test_modules(&production);

    // cfg(test) 父模块链下、经 `#[path]` 平级挂载的种子夹具归测试口径。
    let seeds = root.join("crates/fuzz/src/seeds.rs");
    assert!(
        mounted.contains(&seeds),
        "cfg(test) 链下 #[path] 挂载的 seeds 应纳入测试口径",
    );

    // 生产入口 `#[path]` 拆出去、`pub use` 复用的子模块不得被测试口径豁免。
    for prod in ["tower.rs", "pulse.rs", "cell.rs"] {
        let path = root.join(format!("crates/algorithm/src/{prod}"));
        assert!(
            !mounted.contains(&path),
            "生产拆分子模块 {prod} 不得按测试口径豁免",
        );
    }
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

// ---------------------------------------------------------------------------
// Gate 结构本身：阈值解析、闭集键校验、棘轮结清
// ---------------------------------------------------------------------------

/// 本仓库三个阈值的键与值；改动它们就改 `lints/vocabulary.toml`。
const GATE_KEYS: [(&str, i64); 4] = [
    ("ccn_max", 5),
    ("nloc_max", 60),
    ("file_max_lines", 600),
    ("test_max_lines", 800),
];

/// 一份与仓库同阈值、没有存量债的 `[gate]` 表。
fn sound_gate() -> Table {
    let gate = GATE_KEYS
        .iter()
        .map(|(key, value)| (*key, Value::int(*value)))
        .collect();
    Table::from_pairs(vec![("gate", Value::table(gate))])
}

/// 造一张只含一条 ccn 债务的 gate 与一份实测超限：登记值 `ccn`、实测 `measured`。
fn debt_gate(ccn: i64, measured: usize) -> (Gate, Over) {
    let mut config = sound_gate();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(ccn)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let over = Over {
        key: "crates/x/src/y.rs:run#0".into(),
        metric: Metric::Ccn,
        value: measured,
    };
    (gate, over)
}

/// 项目配置：三个阈值来自构建期生成的 `lints/vocabulary.toml`；存量债务只允许挂在入口编排层
///（`crates/cli/src/` 下的编排与二进制入口），运行时生产代码 `crates/*/src` 一条都不能有。
#[test]
fn the_project_config_lists_only_root_entry_legacy_debt() {
    let gate = Gate::load().expect("项目配置合法");
    let limits =
        [Metric::Ccn, Metric::Nloc, Metric::Lines, Metric::Call].map(|metric| gate.limit(metric));
    assert_eq!(
        limits,
        [5, 60, 600, 0],
        "阈值来自构建期生成的 `lints/vocabulary.toml`；`Call` 没有预算"
    );
    for msg in gate.stale(&BTreeMap::new()) {
        assert!(
            msg.starts_with("crates/cli/src/"),
            "运行时生产代码没有债务条目，存量只能是入口编排既有逻辑：{msg}"
        );
    }
}

#[test]
fn a_test_boulder_over_the_limit_must_fail() {
    let gate = Gate::load().expect("阈值配置加载");
    let limit = gate.test_limit();
    let dir = std::env::temp_dir().join(format!("llm-gate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("造临时目录");
    let path = dir.join("boulder.rs");
    // 每行都是一个合法空行：本用例量的是行数棘轮，不是词法。
    let body = "\n".repeat(limit + 1);
    std::fs::write(&path, &body).expect("写临时文件");

    let mut bad = Vec::new();
    scan_integration_test(&gate, &dir, &path, &mut bad);
    assert_eq!(bad.len(), 1, "超过 {limit} 行必须恰好报一条：{bad:?}");
    assert!(bad[0].contains("超过上限"), "{bad:?}");

    std::fs::write(&path, "\n".repeat(limit)).expect("重写临时文件");
    let mut ok = Vec::new();
    scan_integration_test(&gate, &dir, &path, &mut ok);
    assert!(ok.is_empty(), "等于上限不该报：{ok:?}");
    std::fs::remove_dir_all(&dir).expect("清理临时目录");
}

#[test]
fn every_threshold_must_be_positive() {
    for (key, _) in GATE_KEYS {
        let mut config = sound_gate();
        config.set(&format!("gate.{key}"), Value::int(0));
        assert_eq!(
            Gate::from_table(&config).unwrap_err().message(),
            "[gate] 阈值必须为正",
            "`{key} = 0` 必须被拒"
        );
    }
}

#[test]
fn an_unknown_gate_key_is_an_error_not_noise() {
    let mut config = sound_gate();
    config.set("gate.simd_nloc", Value::int(96));
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate] 未知键 simd_nloc",
        "从未被读到的档位必须报错，而不是静默失效"
    );
}

#[test]
fn a_threshold_must_be_a_positive_integer() {
    let mut config = sound_gate();
    config.set("gate.ccn_max", Value::text("5"));
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate] 缺正整数 ccn_max"
    );
}

#[test]
fn a_vocabulary_without_a_gate_section_is_rejected() {
    assert_eq!(
        Gate::from_table(&Table::new()).unwrap_err().message(),
        "vocabulary.toml 缺 [gate]"
    );
}

#[test]
fn a_debt_row_without_a_metric_is_rejected() {
    let mut config = sound_gate();
    config.push(
        "gate.debt",
        Value::table(vec![("key", Value::text("crates/x/src/y.rs:run#0"))]),
    );
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate.debt] crates/x/src/y.rs:run#0 无度量"
    );
}

#[test]
fn a_debt_row_registers_only_the_metrics_it_names() {
    let (gate, over) = debt_gate(9, 9);
    assert!(gate.reject_reason(&over).is_none(), "登记范围内应通过");
    assert!(
        gate.reject_reason(&Over { value: 10, ..over }).is_some(),
        "超过登记值应失败"
    );
    assert_eq!(
        gate.stale(&BTreeMap::new()).len(),
        1,
        "未观测到的条目应提示删除"
    );
}

/// 登记值高于实测即失败：基线是快照，不是预算，改善必须在同一提交里记账。
#[test]
fn a_register_above_the_measurement_must_be_lowered() {
    let (gate, over) = debt_gate(47, 6);
    assert_eq!(
        gate.reject_reason(&over).unwrap(),
        "crates/x/src/y.rs:run#0: Ccn 实测 6 < 登记 47：把基线条目改成 6",
        "顶着 47 的登记值活着的 ccn=6 必须被要求记账"
    );
}

/// 同一身份登记 ccn 与 nloc：只观测到 ccn 仍超限时，nloc 那条必须被要求删除。
#[test]
fn a_satisfied_metric_must_drop_even_when_the_identity_is_still_live() {
    let mut config = sound_gate();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(9)),
            ("nloc", Value::int(90)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let live = BTreeMap::from([(
        "crates/x/src/y.rs:run#0".to_string(),
        BTreeSet::from([Metric::Ccn]),
    )]);
    assert_eq!(
        gate.stale(&live),
        vec!["crates/x/src/y.rs:run#0: Nloc 已达标，删除基线条目".to_string()],
        "身份仍在册不能让已达标的那条度量继续豁免"
    );
    assert_eq!(
        gate.stale(&BTreeMap::new()).len(),
        2,
        "一条都没观测到就该全删"
    );
}

/// 三种判定：恶化失败、登记范围内通过、全新身份失败。
#[test]
fn ratchet_blocks_new_and_worse_allows_within_budget() {
    let (gate, over) = debt_gate(9, 10);
    assert!(gate.reject_reason(&over).is_some(), "恶化必须失败");
    assert!(
        gate.reject_reason(&Over { value: 9, ..over }).is_none(),
        "登记范围内应通过（此时仍超阈值）"
    );
    assert!(
        gate.reject_reason(&Over {
            key: "crates/x/src/y.rs:new#0".into(),
            ..over
        })
        .is_some(),
        "全新身份直接失败"
    );
}

/// 架构变体对拍用例总数的下限：17 份逐族对拍、245 条用例（`algorithm` 恢复后的实测值）。
///
/// 这个数不是质量目标，是**存在证明**。`crates/algorithm` 曾被整目录删除而全仓没有一条用例
/// 变红——因为"这些族是什么"只写在注释与治理表里，没有任何东西要求它存在；注释和表格都能
/// 无声消失，用例数不能。族少一个、对拍被掏空一半，这里就红。
const ARCHITECTURE_FAMILY_CASES: usize = 248;

#[test]
fn every_architecture_family_keeps_its_conformance_suite() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let dir = root.join("crates/algorithm");
    let mut suites = Vec::new();
    walk(&dir.join("tests"), &mut suites, "rs", false, &|_| true);

    let mut cases = 0;
    let mut orphans = Vec::new();
    for suite in &suites {
        let text = std::fs::read_to_string(suite).expect("对拍文件可读");
        cases += text.matches("#[test]").count();
        if !mounted(&dir, suite) {
            orphans.push(suite.display().to_string());
        }
    }

    assert!(
        orphans.is_empty(),
        "对拍存在但没被挂载，等于没跑（孤儿文件不算覆盖）：{orphans:?}"
    );
    assert!(
        cases >= ARCHITECTURE_FAMILY_CASES,
        "架构变体对拍从 {ARCHITECTURE_FAMILY_CASES} 条降到 {cases} 条"
    );
}

/// 这份对拍是否被 `src` 里某条 `#[path]` 真挂载：目标按所在目录解析后比较，路径写法不影响判定。
fn mounted(dir: &Path, suite: &Path) -> bool {
    let mut sources = Vec::new();
    walk(&dir.join("src"), &mut sources, "rs", false, &|_| true);
    let Ok(target) = std::fs::canonicalize(suite) else {
        return false;
    };
    sources.iter().any(|source| {
        let text = std::fs::read_to_string(source).unwrap_or_default();
        let here = Path::new(source).parent();
        text.match_indices("path = \"")
            .filter_map(|(at, mark)| {
                let rest = &text[at + mark.len()..];
                let end = rest.find('"')?;
                Some(here?.join(&rest[..end]))
            })
            .any(|path| std::fs::canonicalize(path).is_ok_and(|real| real == target))
    })
}
