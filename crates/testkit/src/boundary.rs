//! 层边界规则：错误边界、厂商名归属、unsafe 岛与设备后端装配。

use std::collections::{BTreeMap, BTreeSet};

use crate::architecture::{
    DOC, backticks, contract, rust_files, source_text, src_root, workspace, workspace_root,
};
use crate::ownership::{is_public_definition, public_item_name};

/// 允许用 `String` 做**消息片段**的文件（不是"允许用 `String` 报错"）。
pub(crate) const MESSAGE_FRAGMENTS: &[&str] = &[];

/// 错误边界不得退回字符串（`docs/ARCHITECTURE.md` 第三节）：凡是被 L7 直接调用的层——数据层
/// （`prof`）、内核层（`cutile`，含 `report`）——失败类型必须是
/// `diagnostic::Error`，否则类别与退出码就失去了来源。
#[test]
pub(crate) fn the_error_boundary_does_not_fall_back_to_strings() {
    let contract = contract();
    let mut paths = Vec::new();
    for member in contract.layers.keys() {
        let root = src_root(member);
        let files = rust_files(&root);
        assert!(
            !files.is_empty(),
            "层表成员 `{member}` 在 {} 下一个源文件都没有：扫描集由层表派生，这里必须扫得到东西，\
         否则合并/删除成员会让本测试静默通过",
            root.display()
        );
        paths.extend(files);
    }
    let mut offences = Vec::new();
    for path in paths {
        if MESSAGE_FRAGMENTS
            .iter()
            .any(|exempt| path == std::path::Path::new(exempt))
        {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("源文件可读");
        for (idx, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            // 只认错误位是 `String` 的形态；`Result<Vec<String>, Error>` 这类是合法的。
            if line.contains("Result<") && line.contains(", String>") {
                offences.push(format!("{}:{}", path.display(), idx + 1));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "这些位置仍以 `String` 报错，应改为 `diagnostic::Error`：{offences:?}"
    );
}

/// 厂商名基线：机制层已无按厂商命名的公开入口（`cutile` 的 17 个 `*_program` 收成了
/// `Backend` + `Mode` 一个泛型入口），因此这里保持为空——再有人加按厂商起名的公开项，门禁
/// 直接报越界而不是记进基线。
pub(crate) const VENDOR_PUBLIC_BASELINE: &[(&str, &str)] = &[];

/// 允许公开项带厂商名的成员：后端实现层（L6），`cutile` 的 `Backend` 缝由它填。
pub(crate) const VENDOR_OWNERS: [&str; 1] = ["device"];

/// 厂商名清单；判据是"公开项以 `厂商名_` 起头"。
pub(crate) const VENDORS: [&str; 8] = [
    "ascend", "cuda", "hip", "metal", "musa", "rocm", "sycl", "vulkan",
];

/// 越界岛白名单：`docs/adr/0035-unsafe-ffi-boundary.md` §6 的表格，路径 → `allow(unsafe_code)` 条数。
pub(crate) const UNSAFE_ISLANDS: &[(&str, usize)] = &[
    ("crates/device/src/executor.rs", 1),
    ("crates/nn/src/kernel_fast_i8.rs", 4),
    ("crates/nn/src/kernel_fast_pool.rs", 3),
    ("crates/prof/src/native/gemm8x8.rs", 11),
    ("crates/prof/src/native/i8.rs", 1),
    ("crates/prof/src/native/sme.rs", 8),
    ("crates/storage/src/mmap.rs", 1),
];

/// 文档正文点名的 `crate::名字` 必须在源码里真的存在。
#[test]
pub(crate) fn the_document_names_only_existing_items() {
    let (members, _) = workspace();
    let mut missing = BTreeSet::new();
    for span in backticks(DOC) {
        let Some((crate_name, name)) = reference(&span) else {
            continue;
        };
        if members.contains(crate_name) && !declares_any(crate_name, name) {
            missing.insert(format!("{crate_name}::{name}"));
        }
    }
    assert!(
        missing.is_empty(),
        "文档点名的这些路径在源码里不存在（改名或删模块后没有同步）：{missing:?}"
    );
}

/// 厂商名不得出现在机制层的公开面上：`cutile` 只持 `Backend` 缝，厂商实现归 L6（`device`）。
#[test]
pub(crate) fn the_mechanism_layer_exposes_no_vendor_names() {
    let (members, _) = workspace();
    let mut found = BTreeSet::new();
    for member in &members {
        if VENDOR_OWNERS.contains(&member.as_str()) {
            continue;
        }
        let root = src_root(member);
        for path in rust_files(&root) {
            let Some(text) = source_text(&path) else {
                continue;
            };
            for name in text.lines().filter_map(public_item_name) {
                if VENDORS.iter().any(|vendor| vendor_prefix(name, vendor)) {
                    found.insert((member.clone(), name.to_string()));
                }
            }
        }
    }
    let baseline = VENDOR_PUBLIC_BASELINE
        .iter()
        .map(|(crate_name, name)| ((*crate_name).to_string(), (*name).to_string()))
        .collect::<BTreeSet<_>>();
    let added = found.difference(&baseline).collect::<Vec<_>>();
    let stale = baseline.difference(&found).collect::<Vec<_>>();
    assert!(
        added.is_empty(),
        "机制层的公开项出现厂商名（厂商名只该出现在 `{VENDOR_OWNERS:?}`）：{added:?}"
    );
    assert!(
        stale.is_empty(),
        "厂商名基线过期（这些项已经不在了，从 `VENDOR_PUBLIC_BASELINE` 删掉）：{stale:?}"
    );
}

/// `allow(unsafe_code)` 只允许出现在 ADR 0035 §6 列举的越界岛里。
#[test]
pub(crate) fn unsafe_islands_are_the_ones_the_adr_lists() {
    let mut found = BTreeMap::new();
    for member in workspace().0 {
        let root = src_root(&member);
        for path in rust_files(&root) {
            let Some(text) = source_text(&path) else {
                continue;
            };
            let count = text
                .lines()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    // 模块级 `#![allow]` 与条目级 `#[allow]` 都算；文档注释里的提法不算。
                    trimmed.starts_with('#') && trimmed.contains("allow(unsafe_code)")
                })
                .count();
            if count > 0 {
                let relative = path
                    .strip_prefix(workspace_root())
                    .expect("扫描路径在仓内")
                    .display()
                    .to_string();
                found.insert(relative, count);
            }
        }
    }
    let expected = UNSAFE_ISLANDS
        .iter()
        .map(|(path, count)| ((*path).to_string(), *count))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        found, expected,
        "越界岛与 ADR 0035 §6 的白名单不一致：多一处是新岛（先改 ADR 说清不变量），少一处是基线过期"
    );
}

/// 设备后端必须在**产品路径**上装配，不能只活在测试里。
#[test]
pub(crate) fn the_device_backends_are_installed_on_the_product_path() {
    let mut entries = 0;
    let mut missing = Vec::new();
    for path in rust_files(&workspace_root().join("crates/cli/src")) {
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        // 库入口：`pub fn entry(`。二进制入口：`fn main(` 且不转调库入口（转调的由库那处装着）。
        let library = body.contains("pub fn entry(");
        let standalone = body.contains("fn main(") && !body.contains("llm_cli::entry");
        if !library && !standalone {
            continue;
        }
        entries += 1;
        if !body.contains("backends::install()") {
            missing.push(path.display().to_string());
        }
    }
    assert!(
        entries >= 2,
        "只扫到 {entries} 个产品入口：`llm` 与 `cutile` 两个二进制各自要有一条产品路径"
    );
    assert!(
        missing.is_empty(),
        "这些产品入口没有装配设备后端（`device::backends::install()`）：{missing:?}——\
     装配点只留在 `tests/` 时，用户拿到的是“本机没有该后端”这条错误信息"
    );
}

fn reference(span: &str) -> Option<(&str, &str)> {
    let (head, tail) = span.split_once("::")?;
    let is_crate = !head.is_empty()
        && head
            .chars()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == '_');
    if !is_crate {
        return None;
    }
    let end = tail
        .find(|value: char| !(value.is_alphanumeric() || value == '_'))
        .unwrap_or(tail.len());
    let name = &tail[..end];
    (!name.is_empty()).then_some((head, name))
}

fn declares_any(crate_name: &str, name: &str) -> bool {
    const KINDS: [&str; 8] = [
        "struct", "enum", "trait", "type", "fn", "const", "union", "mod",
    ];
    let root = src_root(crate_name);
    rust_files(&root).into_iter().any(|path| {
        source_text(&path).is_some_and(|text| {
            text.lines().any(|line| {
                is_public_definition(line, name, &KINDS) || is_module_or_macro(line, name)
            })
        })
    })
}

fn is_module_or_macro(line: &str, name: &str) -> bool {
    let trimmed = line.trim_start();
    let rest = trimmed
        .strip_prefix("pub(crate) ")
        .or_else(|| trimmed.strip_prefix("pub "))
        .unwrap_or(trimmed);
    ["mod ", "macro_rules! "].iter().any(|prefix| {
        rest.strip_prefix(prefix)
            .is_some_and(|tail| ident_is(tail.trim_start(), name))
    })
}

fn ident_is(text: &str, name: &str) -> bool {
    text.strip_prefix(name).is_some_and(|after| {
        after
            .chars()
            .next()
            .is_none_or(|next| !(next.is_alphanumeric() || next == '_'))
    })
}

fn vendor_prefix(name: &str, vendor: &str) -> bool {
    name.strip_prefix(vendor)
        .is_some_and(|after| after.starts_with('_'))
}
