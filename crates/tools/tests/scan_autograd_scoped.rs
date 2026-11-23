//! 临时：只扫 `llm-autograd` 的静态门度量，核对归并后 ccn/nloc/行数/禁止形态/词表。
//! 跑完即删，不并入门禁基线。

use std::path::Path;

use llm_tools::{
    call, ccn,
    gate::{Gate, Metric, Over},
    lex, strip_tests, words,
};

#[test]
fn autograd_passes_scoped_gate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let ag = root.join("crates/autograd");
    let gate = Gate::load().expect("阈值配置加载");
    let mut bad = Vec::new();

    let mut prod: Vec<_> = std::fs::read_dir(ag.join("src"))
        .expect("src dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|s| s == "rs"))
        .collect();
    prod.sort();

    for path in &prod {
        let name = format!("autograd/src/{}", path.file_name().unwrap().to_string_lossy());
        let raw = std::fs::read_to_string(path).expect("read");
        let text = strip_tests(&raw).expect("strip tests");
        let lines = raw.lines().count();
        if Some(gate.reject_reason(&Over { key: name.clone(), metric: Metric::Lines, value: lines })).is_some() {
            bad.push(format!("{name}: 文件 {lines} 行超线"));
        }
        for f in ccn::analyze(&text).expect("ccn") {
            let key = format!("{name}:{}#{}", f.name, f.nth);
            for (metric, value) in [(Metric::Ccn, f.ccn as usize), (Metric::Nloc, f.nloc as usize)] {
                if gate.reject_reason(&Over { key: key.clone(), metric, value }).is_some() {
                    bad.push(format!("{key}: {metric:?}={value} 超限"));
                }
            }
        }
        if call::scan(&text).expect("call").len() > gate.limit(Metric::Call) {
            bad.push(format!("{name}: 存在裸 unwrap/expect/panic"));
        }
        if lex::parse(&raw).is_err() {
            bad.push(format!("{name}: 词法错误"));
        }
        for hit in words::scan(&raw).expect("words") {
            bad.push(format!("{name}: {}", hit.describe()));
        }
    }

    let mut tests: Vec<_> = std::fs::read_dir(ag.join("tests"))
        .expect("tests dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|s| s == "rs"))
        .collect();
    tests.sort();
    for path in &tests {
        let name = format!("autograd/tests/{}", path.file_name().unwrap().to_string_lossy());
        let raw = std::fs::read_to_string(path).expect("read");
        let lines = raw.lines().count();
        if lines > gate.test_limit() {
            bad.push(format!("{name}: 测试 {lines} 行超线"));
        }
        if lex::parse(&raw).is_err() {
            bad.push(format!("{name}: 词法错误"));
        }
        for hit in words::scan(&raw).expect("words") {
            bad.push(format!("{name}: {}", hit.describe()));
        }
    }

    assert!(bad.is_empty(), "autograd 静态门失败：\n  {}", bad.join("\n  "));
}
