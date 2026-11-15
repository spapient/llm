//! CCN 跨实现对拍：以 llm.u 旧 `tool gate ccn`（syn 实现）为基线，验证零依赖新分析器
//! 在真实源文件上逐函数零漂移。旧仓库/二进制不在本机时静默跳过，保证 CI 可移植。

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use llm_tool::analyze;

/// 旧仓库根与旧工具二进制（本机开发环境路径）。
const OLD_ROOT: &str = "/Users/fei/code/llm.u";
const OLD_BIN: &str = "/Users/fei/code/llm.u/target/debug/tool";

/// 对拍文件数下限：低于此数说明遍历坏了，而不是"仓库没有源文件"。
const MIN_FILES: usize = 40;

#[test]
fn ccn_matches_the_syn_baseline_on_every_old_source() {
    if testkit::skip(format!(
        "tool: 旧 tool 基线二进制不存在（{OLD_BIN}），CCN 对拍用例未运行"
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
        .expect("run old tool");
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
    walk(dir, &mut out);
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read dir").flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name() == Some("target".as_ref())
                || path.file_name() == Some(".git".as_ref())
            {
                continue;
            }
            walk(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}
