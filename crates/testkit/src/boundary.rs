//! 层边界规则：错误边界、厂商名归属、unsafe 岛与设备后端装配。
//!
//! 四条规则都是「某一层不得越过某条线」，靠源码文本与白名单核对：
//!
//! - 错误边界不得退回字符串（L7 直接调用的层必须持有错误类型）；
//! - 厂商名只许出现在后端实现层（`engine`），机制层（`cutile`）只持 `Backend` 缝；
//! - `allow(unsafe_code)` 只许出现在 ADR 0035 §6 列出的越界岛；
//! - 设备后端注册必须在**产品路径**上装配，不能只活在测试里。
//!
//! 白名单都写成常量表：加一条要有据可依，删一条要真删干净（过期条目本身就是判据失败）。

#[cfg(test)]
mod tests {
use std::path::Path;

use crate::architecture::tests::{DOC, rust_files, workspace_root};
use crate::ownership::tests::{declares_public, public_item_name};

/// 允许用 `String` 做**消息片段**的文件（不是"允许用 `String` 报错"）。
///
/// 现在是空的：唯一一份"把失败原因写成字符串"的实现是手写 TOML 解析器，它随 ADR 0022 一起
/// 删除（语料改在构建期解析）。留着这张表是为了让下一条同类豁免有地方登记，而不是散在断言里。
const MESSAGE_FRAGMENTS: &[&str] = &[];

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
fn the_error_boundary_does_not_fall_back_to_strings() {
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

/// 厂商名基线：`cutile` 的 17 个 `*_program`（P1 收成一个泛型入口后逐条删掉这里）。
const VENDOR_PUBLIC_BASELINE: &[(&str, &str)] = &[
    ("cutile", "cuda_gemm_program"),
    ("cutile", "cuda_grid_program"),
    ("cutile", "cuda_reduce_program"),
    ("cutile", "cuda_rmsnorm_program"),
    ("cutile", "cuda_scan_program"),
    ("cutile", "cuda_softmax_program"),
    ("cutile", "cuda_wmma_gemm_program"),
    ("cutile", "metal_flash_attention_program"),
    ("cutile", "metal_grid_program"),
    ("cutile", "metal_int8_gemm_program"),
    ("cutile", "metal_rmsnorm_program"),
    ("cutile", "metal_scan_program"),
    ("cutile", "metal_simdgroup_gemm_program"),
    ("cutile", "metal_softmax_program"),
    ("cutile", "metal_tensor_gemm_program"),
    ("cutile", "metal_tiled_attention_program"),
    ("cutile", "metal_tiled_gemm_program"),
];

/// 允许公开项带厂商名的成员：后端实现层（L6），`cutile` 的 `Backend` 缝由它填。
const VENDOR_OWNERS: [&str; 1] = ["engine"];

/// 厂商名清单；判据是"公开项以 `厂商名_` 起头"。
const VENDORS: [&str; 8] = [
    "ascend", "cuda", "hip", "metal", "musa", "rocm", "sycl", "vulkan",
];

/// 越界岛白名单：`docs/adr/0035-unsafe-ffi-boundary.md` §6 的表格，路径 → `allow(unsafe_code)` 条数。
///
/// 四个文件、三个逻辑岛：storage 的 mmap、engine 的 Metal FFI、`op` 的 CPU 微内核（后者跨
/// `fast.rs` 与 `fast/pool.rs` 两文件，是同一个 P1 待搬出的欠账岛；`fast.rs` 的 14 处含 NEON
/// 与 AVX2 两段 SIMD 微内核，同一组不变量）。
const UNSAFE_ISLANDS: &[(&str, usize)] = &[
    ("crates/engine/src/metal/executor.rs", 1),
    ("crates/op/src/fast.rs", 14),
    ("crates/op/src/fast/pool.rs", 3),
    ("crates/storage/src/lib.rs", 1),
];


/// 文档正文点名的 `crate::名字` 必须在源码里真的存在。
///
/// 三张契约表由 `cargo metadata` 守着，但正文里的引用没人管：模块改名或删掉后，
/// `engine::metal_pipeline` 这类引用可以一直指向一个不存在的名字，而门禁全绿。
#[test]
fn the_document_names_only_existing_items() {
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

/// 厂商名不得出现在机制层的公开面上：`cutile` 只持 `Backend` 缝，厂商实现归 L6（`engine`）。
///
/// 判据是**公开项的名字**，不是"文件里出现过厂商字面量"——`engine` 正是该写 `metal_*` / `cuda_*`
/// 的那一层，因此豁免。`cutile` 的 17 个 `*_program` 是存量：同一段 resolve → lower → pair 的复制，
/// 消费方只有测试。基线不许增长；P1 把它们收成一个泛型入口后逐条删掉（删了基线也要删，
/// 否则本测试报"过期"）。
#[test]
fn the_mechanism_layer_exposes_no_vendor_names() {
    let (members, _) = workspace();
    let mut found = BTreeSet::new();
    for member in &members {
        if VENDOR_OWNERS.contains(&member.as_str()) {
            continue;
        }
        let root = workspace_root().join("crates").join(member).join("src");
        for path in rust_files(&root) {
            let Ok(text) = std::fs::read_to_string(&path) else {
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
fn unsafe_islands_are_the_ones_the_adr_lists() {
    let mut found = BTreeMap::new();
    for member in workspace().0 {
        let root = workspace_root().join("crates").join(&member).join("src");
        for path in rust_files(&root) {
            let Ok(text) = std::fs::read_to_string(&path) else {
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

/// 设备后端注册表必须在**产品路径**上装配。
///
/// 上一轮的形态是：`engine::backends::install()` 只出现在 `crates/cli/tests/` 里，于是
/// `llm run --device cuda` 把"没装配"报成"本机没有 cuda 后端"，而门禁全绿。
#[test]
fn the_device_backends_are_installed_on_the_product_path() {
    let cli_src = workspace_root().join("crates/cli/src");
    let bin = cli_src.join("bin");
    let files = rust_files(&cli_src);
    assert!(
        files.iter().any(|path| path.starts_with(&bin)),
        "`{}` 下一个源文件都没扫到：本测试必须扫得到二进制入口",
        bin.display()
    );
    // `llm` 走 `main` → `entry`；`cutile` 自己的 `main` 不经过 `llm_cli::entry`，两者都要各自装配。
    let missing = [
        files_without_install(&files, "pub fn entry(", false, &bin),
        files_without_install(&files, "fn main(", true, &bin),
    ]
    .concat();
    assert!(
        missing.is_empty(),
        "这些产品入口没有装配设备后端（`engine::backends::install()`）：{missing:?}——\
         装配点只留在 `tests/` 时，用户拿到的是“本机没有该后端”这条错误信息"
    );
}


/// `` `crate::名字` `` → (`crate`, `名字`)。
///
/// 首段必须全是小写/数字/下划线，才当成"点名的符号"；`cutile::reference::tolerance` 这类多段
/// 路径只看第一段（后面的段由该模块自己负责）。
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

/// `名字` 是否在 `crate_name` 的源码里可被点名：公开定义、任意可见性的模块、`macro_rules!`。
///
/// 私有模块也算：文档会点名 `serve::protocol` 的 `//!`（线格式的唯一说明处），
/// 存在性才是这条判据要守的东西。
fn declares_any(crate_name: &str, name: &str) -> bool {
    const KINDS: [&str; 8] = [
        "struct", "enum", "trait", "type", "fn", "const", "union", "mod",
    ];
    let root = workspace_root().join("crates").join(crate_name).join("src");
    rust_files(&root).into_iter().any(|path| {
        std::fs::read_to_string(path).is_ok_and(|text| {
            text.lines().any(|line| {
                is_public_definition(line, name, &KINDS) || is_module_or_macro(line, name)
            })
        })
    })
}

/// 一行是否为 `mod 名字`（含私有）或 `macro_rules! 名字`。
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

/// `text` 是否就是标识符 `name`（后面不能再接字母、数字或下划线）。
fn ident_is(text: &str, name: &str) -> bool {
    text.strip_prefix(name).is_some_and(|after| {
        after
            .chars()
            .next()
            .is_none_or(|next| !(next.is_alphanumeric() || next == '_'))
    })
}


/// `name` 是否以厂商名起头（`metal_*` / `cuda_*` …）。
fn vendor_prefix(name: &str, vendor: &str) -> bool {
    name.strip_prefix(vendor)
        .is_some_and(|after| after.starts_with('_'))
}

/// `files` 里定义了 `marker` 却没装设备后端的源文件；`in_bin` 把二进制入口与库入口分开。
fn files_without_install(
    files: &[PathBuf],
    marker: &str,
    in_bin: bool,
    bin_root: &Path,
) -> Vec<String> {
    files
        .iter()
        .filter(|path| path.starts_with(bin_root) == in_bin)
        .filter_map(|path| {
            let text = std::fs::read_to_string(path).ok()?;
            let missing = text.contains(marker) && !text.contains("backends::install()");
            missing.then(|| path.display().to_string())
        })
        .collect()
}
}
