//! 层边界规则：错误边界、厂商名归属、unsafe 岛与设备后端装配。
//!
//! 四条规则都是「某一层不得越过某条线」，靠源码文本与白名单核对：
//!
//! - 错误边界不得退回字符串（L7 直接调用的层必须持有错误类型）；
//! - 厂商名只许出现在后端实现层（`device`），机制层（`cutile`）只持 `Backend` 缝；
//! - `allow(unsafe_code)` 只许出现在 ADR 0035 §6 列出的越界岛；
//! - 设备后端注册必须在**产品路径**上装配，不能只活在测试里。
//!
//! 白名单都写成常量表：加一条要有据可依，删一条要真删干净（过期条目本身就是判据失败）。

use std::collections::{BTreeMap, BTreeSet};

use crate::architecture::{
    DOC, backticks, contract, rust_files, source_text, workspace, workspace_root,
};
use crate::ownership::{is_public_definition, public_item_name};

/// 允许用 `String` 做**消息片段**的文件（不是"允许用 `String` 报错"）。
///
/// 现在是空的：唯一一份"把失败原因写成字符串"的实现是手写 TOML 解析器，它随 ADR 0022 一起
/// 删除（语料改在构建期解析）。留着这张表是为了让下一条同类豁免有地方登记，而不是散在断言里。
pub(crate) const MESSAGE_FRAGMENTS: &[&str] = &[];

/// 错误边界不得退回字符串（`docs/ARCHITECTURE.md` 第三节）：凡是被 L7 直接调用的层——数据层
/// （`op` / `prof` / `prim`）、内核层（`cutile`，含 `report`）——失败类型必须是
/// `diagnostic::Error`，否则类别与退出码就失去了来源。
///
/// 扫描集**从层表派生**，不手写目录名：手写的清单在成员被合并或删除后会指向不存在的目录，
/// 而 [`rust_files`] 对不存在的目录返回空——那会让本测试在"少扫一个成员"时静默通过。
/// 反过来，层表里每个成员都必须扫出至少一个文件，否则同样失败。
///
/// 扫描的是"`Result<…, String>`"这一形态本身，不是某个函数：新加的函数同样被覆盖。
/// 豁免表 [`MESSAGE_FRAGMENTS`] 现在是空的：随着手写 TOML 解析器删除，最后一处"把失败原因
/// 写成字符串"的实现也没有了。
#[test]
pub(crate) fn the_error_boundary_does_not_fall_back_to_strings() {
    let contract = contract();
    let mut paths = Vec::new();
    for member in contract.layers.keys() {
        let root = workspace_root().join("crates").join(member).join("src");
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
        for (index, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            // 只认错误位是 `String` 的形态；`Result<Vec<String>, Error>` 这类是合法的。
            if line.contains("Result<") && line.contains(", String>") {
                offences.push(format!("{}:{}", path.display(), index + 1));
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
///
/// 三个逻辑岛：storage 的 mmap、device 的 Metal FFI、`op` 的 CPU 微内核。最后一个跨五个文件
/// （`fast.rs` 拆成目录后按微内核族分落），是同一个 P1 待搬出的欠账岛：拆文件时每个微内核各自
/// 重建切片，总数由 17 涨到 21，这属于要往下削的重复而不是新增许可。
pub(crate) const UNSAFE_ISLANDS: &[(&str, usize)] = &[
    ("crates/device/src/metal/executor.rs", 1),
    ("crates/op-kernel/src/fast/i8.rs", 8),
    ("crates/op-kernel/src/fast/micro.rs", 6),
    ("crates/op-kernel/src/fast/pool.rs", 3),
    ("crates/op-kernel/src/fast/sme.rs", 4),
    ("crates/storage/src/mmap.rs", 1),
];

/// 文档正文点名的 `crate::名字` 必须在源码里真的存在。
///
/// 三张契约表由 `cargo metadata` 守着，但正文里的引用没人管：模块改名或删掉后，
/// `device::metal_pipeline` 这类引用可以一直指向一个不存在的名字，而门禁全绿。
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
///
/// 判据是**公开项的名字**，不是"文件里出现过厂商字面量"——`device` 正是该写 `metal_*` / `cuda_*`
/// 的那一层，因此豁免。`cutile` 原先那 17 个按厂商命名的 `*_program` 已收成一个泛型入口
/// （`execution::program`，后端与形态作参数），基线随之清空。基线现在必须为空：新增厂商名
/// 就报"多出"，与源码不同步的残留就报"过期"，两个方向都卡住。
#[test]
pub(crate) fn the_mechanism_layer_exposes_no_vendor_names() {
    let (members, _) = workspace();
    let mut found = BTreeSet::new();
    for member in &members {
        if VENDOR_OWNERS.contains(&member.as_str()) {
            continue;
        }
        let root = workspace_root().join("crates").join(member).join("src");
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
///
/// 原先这张表只活在 ADR 正文里：`op::fast` 的 17 处越界（`fast.rs` 14 处 + `fast/pool.rs` 3 处）
/// 既不在 §2 的禁令豁免里、也不在 §1 的 FFI 类别里，却能长期存在——因为没有任何东西比对
/// "允许集"与"实际集"。
#[test]
pub(crate) fn unsafe_islands_are_the_ones_the_adr_lists() {
    let mut found = BTreeMap::new();
    for member in workspace().0 {
        let root = workspace_root().join("crates").join(&member).join("src");
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
///
/// 上一轮的形态是：`device::backends::install()` 只出现在 `crates/cli/tests/` 里，于是
/// `llm run --device cuda` 把"没装配"报成"本机没有 cuda 后端"，而门禁全绿。
///
/// 装配点有两类，各自要有一条：`llm` 二进制的 `main` 只转调 `llm_cli::entry`，装配在库入口
/// 那一处；`cutile` 的 `main` 不经过 `llm_cli::entry`，得自己装配。任何一处缺 `install()`，
/// 用户拿到的都是“本机没有该后端”，而不是“装了没生效”。判据按**入口形态**认，不预设目录：
/// 二进制入口从 `src/bin/` 平铺回 `src/` 之后，硬编码的路径会让门静默扫不到东西。
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
    let root = workspace_root().join("crates").join(crate_name).join("src");
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
