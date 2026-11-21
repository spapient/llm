//! 台账与树的一致性：表和 `src/` 对不上就红。

use std::path::{Path, PathBuf};

use crate::wire::{self, Blocked, Wiring};
use crate::Mhc;

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries.flatten().map(|e| e.path()) {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_family_in_the_ledger_still_exists_in_the_tree_and_is_named_once() {
    let mut files = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    let ledger = wire::ledger();
    assert_eq!(ledger.len(), 17, "台账必须覆盖 17 族");
    let mut names: Vec<&str> = ledger.iter().map(|f| f.name).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "台账出现重复族名：{names:?}");
    for name in names {
        let want = format!("{name}.rs");
        let hit = files
            .iter()
            .any(|p| p.file_name().is_some_and(|n| n == want.as_str()));
        assert!(hit, "台账里的族 {name} 在 src/ 下已无对应文件");
    }
}

#[test]
fn no_family_is_blocked_without_saying_what_is_missing() {
    for family in wire::ledger() {
        let Wiring::Blocked(why) = family.wiring else {
            continue;
        };
        let text = match why {
            Blocked::Operator(op) => op,
            Blocked::Layer(layer) => layer,
            Blocked::Geometry(geo) => geo,
        };
        assert!(!text.trim().is_empty(), "族 {} 没写缺什么", family.name);
    }
}

#[test]
fn mhc_is_the_only_wired_family_and_it_lands_on_a_stream_square_matrix() {
    let wired: Vec<&str> = wire::ledger()
        .iter()
        .filter(|f| matches!(f.wiring, Wiring::Wired(_)))
        .map(|f| f.name)
        .collect();
    assert_eq!(wired, ["mhc"], "已通族清单变了，台账口径要同步");
    let layer = wire::mhc_layer(&Mhc::identity(2).expect("两条流的恒等重组")).expect("接线");
    assert_eq!((layer.rows(), layer.cols()), (2, 2));
    let wide = wire::mhc_layer(&Mhc::identity(4).expect("四条流的恒等重组")).expect("接线");
    assert_eq!((wide.rows(), wide.cols()), (4, 4), "流数变了层声明必须跟着变");
    assert!(Mhc::<f32>::identity(0).is_err(), "零条流不该构造出重组");
}
