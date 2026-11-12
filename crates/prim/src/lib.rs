//! 片段层：`<vendor>/<prim>.<ext>`——某台设备上某个原语的具体写法，以及它的见证源码。
//!
//! **拥有**：`Builtin`（内嵌片段表）、`Fragment` / `FragmentSet`（装载后的片段）、`Device` /
//! `Image` / `Status`（设备侧代码：源码 + 可选的镜像）。**不拥有**：设备能力区间（`prof`）、
//! 拼接与 emit（`cutile`）、运行时缓冲与数值（各自的层）。**依赖**：`prof`（原语闭集与档案）、
//! `diagnostic`（错误类别由本层给，装载失败因此带类别而不是字符串）。
//! **被依赖**：`cutile`。
//!
//! 这一层是**内容**而不是机制：一台设备的某条原语怎么写，全部是 `<vendor>/<prim>.<ext>` 里的
//! 文本；`cutile` 只负责把它拼进模板、emit 成源文件并编译执行。因此新增设备、替换一段实现都不
//! 必改机制代码，片段也不会与拼接器混在同一个 crate 里。
//!
//! 本 crate 就是数据根的 `prim/` 那一层：`<root>/<vendor>/<prim>.<ext>`。仓库里 `<root>` 是
//! [`dir`]，换 `data.root` 时是 `<data.root>/prim`（见 [`layer_root`]）。文件名必须落在原语
//! 闭集 [`Prim`] 内，否则装载即报错——片段与词汇表同源，写错名字不会静默生效。
//!
//! # 内嵌表是编译期生成的
//!
//! [`BUILTIN`] / [`WITNESS`] / [`DEVICES`] 不是手写登记：`build.rs` 在编译期扫描 `<vendor>/`，
//! 按 [`Prim`] 校验文件名、按档案校验每段片段声明的签名（`/* cutile-sign: <prim>/<impl> */`），
//! 再把三张表写进 `$OUT_DIR/tables.rs`。于是"发布出去的二进制里没有它"这类漂移不再靠人记得补
//! 一行 `include_str!`：加设备只需要放文件，`<vendor>` 与 `prof/devices/<vendor>.toml` 同名。
//!
//! # 两种模式：源码调试 / 设备镜像发布
//!
//! 默认（调试）只有源码：片段文本 + 见证源码，运行期照旧 emit 后编译。打开 `device-image`
//! feature（发布）时，能用本机工具链编出设备镜像的设备（如今是 `metal`）额外带上二进制；
//! [`Device::status`] 说明每台设备这次走到了哪一步，没有镜像也只是"这一步没走"，不是配置错误。
//!
//! # 签名声明
//!
//! 每段片段的第一行是**签名声明** `/* cutile-sign: <prim>/<impl> */`：同一原语可以有多条签名
//! （`dot` 的 `dp4a` 与标量 `fma`），片段只说"这段文本写的是哪一条"，"这台设备有没有这条签名"
//! 仍由档案回答。两边的对齐由 `cutile` 在装载期判定，所以"协商选中 `dp4a`、文本写的却是 `fma`"
//! 不会晚到构建期、更不会静默用错算法。
//!
//! `witness.<ext>` 是见证源码（L2 证据），**不是片段**：装载时跳过它，档案里的 `witness` 用
//! `{ file, func, expect }` 指向同一目录下的这个文件、入口函数与期望输出。
//!
//! # 例子
//!
//! ```
//! let set = llm_prim::builtin("cpu", "c").expect("内置 cpu 片段");
//! assert_eq!(set.vendor(), "cpu");
//! assert!(set.get(llm_prim::Prim::Load).is_some());
//!
//! let device = llm_prim::device("metal").expect("内置设备");
//! assert_eq!(device.ext, "metal", "扩展名由档案的 emit 语言决定");
//! assert!(device.entry(llm_prim::Prim::Dot).is_some(), "每个原语都有入口");
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod device;
mod tables;

pub use device::{Device, Image, Status, device, devices};
pub use prof::atom::Prim;
pub use tables::{BUILTIN, DEVICES, WITNESS};

/// 见证源码的文件名（不含扩展名）。
///
/// 它与片段同目录、同扩展名，但语义不同：片段是"某原语怎么写"，见证是"证据能跑"。
/// 装载片段时按这个名字跳过它。
pub const WITNESS_STEM: &str = "witness";

/// 一台 vendor、一种输出语言的内嵌片段。
#[derive(Debug, Clone, Copy)]
pub struct Builtin {
    /// vendor 名，与档案里的 `meta.vendor` 一致。
    pub vendor: &'static str,
    /// 源文件扩展名（由 `emit` 语言决定，`c` → `.c`）。
    pub ext: &'static str,
    /// 片段：原语 → 文本。
    pub fragments: &'static [(Prim, &'static str)],
}

/// 按 `(vendor, ext)` 查内嵌见证；没有登记时返回 `None`（调用方改读磁盘）。
pub fn builtin_witness(vendor: &str, ext: &str) -> Option<&'static str> {
    WITNESS
        .iter()
        .find(|(have_vendor, have_ext, _)| *have_vendor == vendor && *have_ext == ext)
        .map(|(_, _, source)| *source)
}

/// 签名声明的标记：片段第一行 `/* cutile-sign: <prim>/<impl> */` 里的 `cutile-sign:`。
pub const SIGN_MARKER: &str = "cutile-sign:";

/// 一段片段：文本，以及它声明实现的那条签名。
///
/// 片段不能只说明"怎么写"：同一原语在档案里可以有多条签名，"这段文本写的是哪一条"必须
/// 自己声明，否则 "协商可行" 与 "能生成" 可以同时为真却指向不同的实现。
///
/// 签名声明是**元数据**：它不参与 emit（[`Fragment::body`] 才是拼进模板的文本），也不参与
/// 缓存指纹，因此补上这一行不会改动已落盘的源文本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    source: String,
    body: String,
    implementation: String,
}

impl Fragment {
    /// 片段全文（含第一行的签名声明）。
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 真正拼进模板的文本：声明行之后的部分。
    pub fn body(&self) -> &str {
        &self.body
    }

    /// 声明实现的 `impl` 名：档案里那条 `provides.impl`。
    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    /// 解析第一行的签名声明。
    ///
    /// # Errors
    ///
    /// 第一行不是签名声明、声明里的原语与文件名不一致、或缺少 `impl` 名时返回 `invalid`。
    fn parse(prim: Prim, source: impl Into<String>) -> diagnostic::Result<Self> {
        let source = source.into();
        let (first, body) = source.split_once('\n').ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("片段 `{prim}` 只有签名声明，没有实现"),
            )
        })?;
        let declared = first
            .strip_prefix("/*")
            .and_then(|rest| rest.strip_suffix("*/"))
            .map(str::trim)
            .and_then(|rest| rest.strip_prefix(SIGN_MARKER))
            .map(str::trim)
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "片段 `{prim}` 的第一行必须是 `/* {SIGN_MARKER} {prim}/<impl> */`，实际是 `{first}`"
                    ),
                )
            })?;
        let (declared_prim, implementation) = declared.split_once('/').ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("`{prim}` 的签名声明 `{declared}` 不是 `<prim>/<impl>`"),
            )
        })?;
        if Prim::parse(declared_prim.trim()) != Some(prim) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的签名声明写的是 `{}`：文件名即原语名，两者必须一致",
                    declared_prim.trim()
                ),
            ));
        }
        let implementation = implementation.trim();
        if implementation.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                format!("`{prim}` 的签名声明缺少 `impl` 名"),
            ));
        }
        // 先把两处都落成 `String`：它们借的是 `source`，随后 `source` 要整体移进结构体。
        let implementation = implementation.to_string();
        let body = body.to_string();
        Ok(Self {
            source,
            body,
            implementation,
        })
    }
}

/// 某个 vendor 的片段集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSet {
    vendor: String,
    ext: String,
    fragments: BTreeMap<Prim, Fragment>,
}

impl FragmentSet {
    /// 新建空集合。
    pub fn new(vendor: &str, ext: &str) -> Self {
        Self {
            vendor: vendor.to_string(),
            ext: ext.to_string(),
            fragments: BTreeMap::new(),
        }
    }

    /// 加入一段片段。
    ///
    /// # Errors
    ///
    /// 第一行不是签名声明、声明里的原语与文件名不一致、或缺少 `impl` 名时返回 `invalid`。
    pub fn insert(&mut self, prim: Prim, source: impl Into<String>) -> diagnostic::Result<()> {
        self.fragments.insert(prim, Fragment::parse(prim, source)?);
        Ok(())
    }

    /// 取一段片段。
    pub fn get(&self, prim: Prim) -> Option<&str> {
        self.fragments.get(&prim).map(Fragment::body)
    }

    /// 取一段片段（含它的签名声明）。
    pub fn fragment(&self, prim: Prim) -> Option<&Fragment> {
        self.fragments.get(&prim)
    }

    /// 片段声明实现的 `impl` 名；没有这段片段时 `None`。
    pub fn implementation(&self, prim: Prim) -> Option<&str> {
        self.fragments.get(&prim).map(Fragment::implementation)
    }

    /// 设备名。
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// 片段所在的扩展名。
    pub fn ext(&self) -> &str {
        &self.ext
    }

    /// 片段数。
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }
}

/// 本 crate 的片段层根目录（仓库内）。
///
/// 档案里的 `witness` 与 [`load`] 的 `<root>` 都相对它解析；换 `data.root` 时用
/// [`layer_root`]。
pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// 数据根对应的片段层根目录：`<data.root>/prim`。
///
/// 一个 `data.root` 同时含 `ops/`、`devices/`、`prim/` 三层，本 crate 只解释最后一层。
pub fn layer_root(data_root: &Path) -> PathBuf {
    data_root.join("prim")
}

/// 按 `(vendor, ext)` 查内嵌片段；没有登记时返回 `None`（调用方改读磁盘）。
///
/// # Panics
///
/// 内嵌片段的第一行不是签名声明时 panic：`BUILTIN` 是编译期数据，写成非法形态就是 crate
/// 缺陷，由 `builtin_matches_disk` 与 `builtin_signs_are_declared` 兜住。
pub fn builtin(vendor: &str, ext: &str) -> Option<FragmentSet> {
    let entry = BUILTIN
        .iter()
        .find(|entry| entry.vendor == vendor && entry.ext == ext)?;
    let mut set = FragmentSet::new(entry.vendor, entry.ext);
    for (prim, source) in entry.fragments {
        set.insert(*prim, *source).expect("内置片段必须声明签名");
    }
    Some(set)
}

/// 从 `<root>/<vendor>/<prim>.<ext>` 读取所有片段。
///
/// `<root>` 是片段层根目录（见 [`dir`] 与 [`layer_root`]）；`witness.<ext>` 被跳过，
/// 文件名不在 [`Prim`] 闭集内的其余文件报错，第一行不是签名声明的片段也报错。
///
/// # Errors
///
/// 目录不存在是 `missing`（`data.root` 指错时最常发生），不可读是 `io`；文件名不在闭集内、
/// 或片段的第一行不是签名声明是 `invalid`。
pub fn load(root: &Path, vendor: &str, ext: &str) -> diagnostic::Result<FragmentSet> {
    let mut set = FragmentSet::new(vendor, ext);
    let dir = root.join(vendor);
    let entries = std::fs::read_dir(&dir).map_err(|error| Error::from_io(&dir, &error))?;
    for entry in entries {
        let entry = entry.map_err(|error| Error::new(Code::Io, error.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some(ext) {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        if stem == WITNESS_STEM {
            continue;
        }
        let Some(prim) = Prim::parse(stem) else {
            return Err(Error::new(
                Code::Invalid,
                format!("`{}` 的文件名 `{stem}` 不在原语闭集内", path.display()),
            ));
        };
        let source =
            std::fs::read_to_string(&path).map_err(|error| Error::from_io(&path, &error))?;
        set.insert(prim, source)?;
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 内嵌表与磁盘一致：生成器漏扫一个目录，这里就会失败。
    #[test]
    fn builtin_matches_disk() {
        for entry in BUILTIN {
            let disk = load(&dir(), entry.vendor, entry.ext).expect("磁盘片段可装载");
            assert_eq!(
                disk.len(),
                entry.fragments.len(),
                "`{}` 的磁盘片段数与内嵌不一致",
                entry.vendor
            );
            for (prim, source) in entry.fragments {
                assert_eq!(
                    disk.fragment(*prim).map(Fragment::source),
                    Some(*source),
                    "`{}` 的 `{prim}` 与磁盘内容不一致",
                    entry.vendor
                );
            }
        }
    }

    #[test]
    fn lookup_needs_both_vendor_and_ext() {
        assert!(builtin("cpu", "c").is_some());
        assert!(builtin("cpu", "cu").is_none(), "扩展名也参与登记键");
        assert!(builtin("nvidia", "c").is_none(), "未登记的设备走磁盘");
    }

    #[test]
    fn builtin_witness_matches_disk() {
        for (vendor, ext, source) in WITNESS {
            let path = dir().join(vendor).join(format!("witness.{ext}"));
            let disk = std::fs::read_to_string(&path).expect("磁盘见证可读");
            assert_eq!(disk, source, "`{vendor}` 的内嵌见证与磁盘不一致");
        }
    }

    #[test]
    fn witness_is_not_a_fragment() {
        let cpu = BUILTIN
            .iter()
            .find(|entry| entry.vendor == "cpu")
            .expect("内置 cpu 片段");
        let set = load(&dir(), "cpu", "c").expect("witness.c 必须被跳过");
        assert_eq!(set.len(), cpu.fragments.len());
        assert!(set.get(Prim::Load).is_some());
    }

    /// 每段片段都声明自己的签名：一个原语可以有多条签名，文本必须说清写的是哪一条。
    #[test]
    fn builtin_signs_are_declared() {
        for entry in BUILTIN {
            let set = builtin(entry.vendor, entry.ext).expect("内嵌片段可装载");
            for (prim, _) in entry.fragments {
                assert!(
                    set.implementation(*prim).is_some_and(|imp| !imp.is_empty()),
                    "`{}` 的 `{prim}` 没有声明签名",
                    entry.vendor
                );
            }
        }
    }

    #[test]
    fn a_file_name_must_be_in_the_primitive_closed_set() {
        let root = std::env::temp_dir().join(format!("prim-test-{}", std::process::id()));
        let vendor_dir = root.join("fake");
        std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
        std::fs::write(
            vendor_dir.join("load.c"),
            "/* cutile-sign: load/memcpy */\n",
        )
        .expect("写片段");
        std::fs::write(vendor_dir.join("not_a_prim.c"), "").expect("写非法片段");
        let error = load(&root, "fake", "c").expect_err("闭集外的文件名必须报错");
        assert_eq!(
            error.code(),
            Code::Invalid,
            "文件名是声明，不是 io：{error}"
        );
        assert!(error.to_string().contains("not_a_prim"), "{error}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// 片段第一行必须是签名声明：没有它的片段无法与档案交叉校验，所以装载期就拒绝。
    #[test]
    fn a_fragment_must_declare_its_signature() {
        let root = std::env::temp_dir().join(format!("prim-sign-test-{}", std::process::id()));
        let vendor_dir = root.join("fake");
        std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
        std::fs::write(vendor_dir.join("load.c"), "{ $0 = $1; }\n").expect("写片段");
        let error = load(&root, "fake", "c").expect_err("没有签名声明的片段必须报错");
        assert!(error.to_string().contains(SIGN_MARKER), "{error}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// 声明里的原语名与文件名不一致：多半是复制粘贴，装载期就要报。
    #[test]
    fn a_signature_must_name_the_file_primitive() {
        let mut set = FragmentSet::new("fake", "c");
        let error = set
            .insert(Prim::Load, "/* cutile-sign: dot/fma */\n{ $0 += $1; }\n")
            .expect_err("原语名不一致必须报错");
        assert!(error.to_string().contains("dot"), "{error}");
        assert!(error.to_string().contains("load"), "{error}");
    }

    /// 别名不能空着：`impl` 就是档案里那条 `provides.impl`，空名无法对齐。
    #[test]
    fn a_signature_needs_an_implementation_name() {
        let mut set = FragmentSet::new("fake", "c");
        let error = set
            .insert(Prim::Load, "/* cutile-sign: load/ */\n{ $0 = $1; }\n")
            .expect_err("缺 impl 必须报错");
        assert!(error.to_string().contains("impl"), "{error}");
    }
}
