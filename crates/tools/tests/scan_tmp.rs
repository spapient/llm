use llm_tools::{ccn, call};
use std::path::Path;

#[test]
fn scan_train() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("train/src");
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let mut rows = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        for x in ccn::analyze(&src).unwrap() {
            rows.push((x.ccn, x.nloc, x.name.clone(), f.clone()));
        }
        let c = call::scan(&src).unwrap().len();
        if c > 0 { rows.push((1000 + c as u32, 0, format!("FORBIDDEN_CALLS={c}"), f.clone())); }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    println!("=== top CCN / NLOC ({}/{}) ===", files.len(), rows.len());
    for (ccn, nloc, name, f) in rows.iter().take(28) {
        println!("{:>4} {:>4}  {:<28} {}", ccn, nloc, name,
            f.strip_prefix(root.parent().unwrap()).unwrap().display());
    }
    println!("=== NLOC top ===");
    let mut byn: Vec<_> = rows.iter().filter(|r| r.0 < 1000).collect();
    byn.sort_by(|a,b| b.1.cmp(&a.1));
    for (ccn, nloc, name, f) in byn.iter().take(12) {
        println!("{:>4} {:>4}  {:<28} {}", ccn, nloc, name,
            f.strip_prefix(root.parent().unwrap()).unwrap().display());
    }
}

fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() { collect(&p, out); } else if p.extension().is_some_and(|x| x == "rs") { out.push(p); }
    }
}
