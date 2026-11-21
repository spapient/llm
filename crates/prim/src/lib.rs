//! 片段层：某台设备上某个原语的具体写法，以及它的见证源码。
//!
//! **拥有**：`Builtin`（合成表）、`Fragment` / `FragmentSet`（装载后的片段）、[`Layout`]
//! （片段层在哪）。**不拥有**：片段文本本身（六个设备 crate：`cpu` / `cuda` / `metal` /
//! `musa` / `opencl` / `rocm`）、`Device` / `Image` / `Status` 与设备能力区间（`prof`）、
//! 拼接与 emit（`cutile`）。**依赖**：六个设备 crate、`prof`（原语闭集与档案）、
//! `diagnostic`（错误类别由本层给，装载失败因此带类别而不是字符串）。**被依赖**：`cutile`。
//!
//! 这一层是**机制**：怎么找、怎么装载、怎么校验签名声明。一台设备的某条原语怎么写，全部是
//! 那个设备 crate 里的 `<prim>.<ext>` 文本，本 crate 一行片段都没有。于是"加一台设备"是加
//! 一个设备 crate 再加 [`BUILTIN`] 里一行，机制代码不动；片段也不会与拼接器挤在同一个 crate。
//!
//! # 片段在哪
//!
//! [`Layout::builtin`] 是仓库内置（六台设备各在自己的 crate 里），[`Layout::at`] 是 `data.root`
//! 覆盖（`<data.root>/prim/<vendor>/<prim>.<ext>`）。两种模式走同一个装载入口
//! [`Layout::fragments`]，所以"文件名不在闭集内 / 签名声明写错"在两种模式下报同一类错误。
//! 文件名必须落在原语闭集 [`Prim`] 内，否则装载即报错——片段与词汇表同源。
//!
//! # 内嵌表是编译期生成的
//!
//! 每个设备 crate 的 `build.rs` 都 `include!` 同一份生成器（`crates/prim/generate.rs`）：它在
//! 编译期扫描那个 crate 的目录，按 [`Prim`] 校验文件名、按档案校验每段片段声明的签名
//! （`/* cutile-sign: <prim>/<impl> */`），生成该 crate 的 `FRAGMENTS` / `PRIMS` / `DEVICE`。
//! 本 crate 的 [`BUILTIN`] / [`WITNESS`] / [`DEVICES`] 只是六份的合成，没有自己的构建脚本，
//! 因此"发布出去的二进制里没有它"这类漂移不再靠人记得补一行 `include_str!`。
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
//! `{ file, func, expect }` 指向同一个目录下的这个文件、入口函数与期望输出。
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
//!
//! let layout = llm_prim::Layout::builtin();
//! assert!(layout.witness("cpu/witness.c").is_some(), "内置见证在设备 crate 里");
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 签名声明的纯字符串规则；与生成器 `generate.rs` 经 `include!` 共用同一份源码（ADR 0027 §5）。
mod signature;
mod tables;

pub use prof::atom::Prim;
pub use prof::{Device, Image, Status};
pub use tables::{BUILTIN, DEVICES, ROOTS, WITNESS, device, devices};

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
///
/// 与 `signature` 模块内的 `MARKER` 同值；共享规则见该模块。
pub const SIGN_MARKER: &str = signature::MARKER;

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
        let (first, body) = split_declaration(&source, prim)?;
        let (declared_prim, implementation) = declaration_parts(first, prim)?;
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

/// 片段文本 → 「声明行」与「实现体」：只有声明没有实现的片段不是片段。
fn split_declaration(source: &str, prim: Prim) -> diagnostic::Result<(&str, &str)> {
    source.split_once('\n').ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("片段 `{prim}` 只有签名声明，没有实现"),
        )
    })
}

/// 声明行 → `<prim>/<impl>` 两半：注释外壳与 [`SIGN_MARKER`] 标记都要在这里对上。
///
/// 剥壳与切分走 [`signature`] 共享规则；这里只补原语上下文与诊断错误。
fn declaration_parts(first: &str, prim: Prim) -> diagnostic::Result<(&str, &str)> {
    let declared = signature::strip_marker(first).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!(
                "片段 `{prim}` 的第一行必须是 `/* {SIGN_MARKER} {prim}/<impl> */`，实际是 `{first}`"
            ),
        )
    })?;
    let (declared_prim, implementation) = signature::split_slash(declared).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{prim}` 的签名声明 `{declared}` 不是 `<prim>/<impl>`"),
        )
    })?;
    Ok((declared_prim, implementation))
}

/// 某个 vendor 的片段集合。
///
/// 同一原语可以挂多条片段：每条声明自己的 `impl`（[`Fragment::implementation`]），协商按选中的
/// 实现名取对应那一段。未按实现筛选时，[`FragmentSet::get`] 返回登记的首份，保留单片段设备的旧行为。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSet {
    vendor: String,
    ext: String,
    fragments: BTreeMap<Prim, Vec<Fragment>>,
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
    /// 第一行不是签名声明、声明里的原语与文件名不一致、缺少 `impl` 名，或同一原语同名实现重复登记时
    /// 返回 `invalid`。
    pub fn insert(&mut self, prim: Prim, source: impl Into<String>) -> diagnostic::Result<()> {
        let fragment = Fragment::parse(prim, source)?;
        let list = self.fragments.entry(prim).or_default();
        if list
            .iter()
            .any(|existing| existing.implementation == fragment.implementation)
        {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 已登记 `{}` 实现：同一原语的同名片段不能放两份",
                    fragment.implementation
                ),
            ));
        }
        list.push(fragment);
        Ok(())
    }

    /// 取登记的首份片段正文（单片段设备的兼容入口）。
    pub fn get(&self, prim: Prim) -> Option<&str> {
        self.first(&prim).map(Fragment::body)
    }

    /// 取某条实现对应的片段正文；没有这条实现时返回 `None`。
    pub fn get_impl(&self, prim: Prim, implementation: &str) -> Option<&str> {
        self.fragment_impl(prim, implementation).map(Fragment::body)
    }

    /// 取登记的首份片段（含它的签名声明）。
    pub fn fragment(&self, prim: Prim) -> Option<&Fragment> {
        self.first(&prim)
    }

    /// 取某条实现对应的片段（含签名声明）。
    pub fn fragment_impl(&self, prim: Prim, implementation: &str) -> Option<&Fragment> {
        self.fragments
            .get(&prim)?
            .iter()
            .find(|fragment| fragment.implementation == implementation)
    }

    /// 片段声明实现的 `impl` 名；没有这段片段时 `None`。
    pub fn implementation(&self, prim: Prim) -> Option<&str> {
        self.first(&prim).map(Fragment::implementation)
    }

    /// 遍历某原语登记的全部片段（按登记顺序）。
    pub fn fragments_of(&self, prim: Prim) -> &[Fragment] {
        self.fragments
            .get(&prim)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// 设备名。
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// 片段所在的扩展名。
    pub fn ext(&self) -> &str {
        &self.ext
    }

    /// 片段总数（同一原语的多条实现分别计数）。
    pub fn len(&self) -> usize {
        self.fragments.values().map(Vec::len).sum()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.fragments.values().all(Vec::is_empty)
    }

    /// 取某原语登记的首份片段。
    fn first(&self, prim: &Prim) -> Option<&Fragment> {
        self.fragments.get(prim)?.first()
    }
}

/// 数据根对应的片段层根目录：`<data.root>/prim`。
///
/// 一个 `data.root` 同时含 `ops/`、`devices/`、`prim/` 三层，本 crate 只解释最后一层。
pub fn layer_root(data_root: &Path) -> PathBuf {
    data_root.join("prim")
}

/// 片段层在哪：仓库内置（六台设备各在自己的 crate 里）或某个 `data.root` 的 `prim/`。
///
/// 两种模式的差别只有**目录解析**，装载、校验、报错全部共用，所以"换一份片段"不会换一套
/// 语义。`data.root` 覆盖时布局是 `<data.root>/prim/<vendor>/<prim>.<ext>`，与内置的
/// `<crate 根>/<prim>.<ext>` 只差 vendor 那层目录——因为一个目录一台设备，已经是内置的形态。
#[derive(Debug, Clone, Copy)]
pub struct Layout<'a> {
    /// `Some` 表示读磁盘上的 `data.root`；`None` 表示内置表。
    data: Option<&'a Path>,
}

impl<'a> Layout<'a> {
    /// 仓库内置的片段层：六台设备各自的 crate。
    pub const fn builtin() -> Self {
        Self { data: None }
    }

    /// 某个 `data.root` 下的片段层（`<root>/prim/`）。
    pub fn at(data_root: &'a Path) -> Self {
        Self {
            data: Some(data_root),
        }
    }

    /// 取一台设备的片段集。
    ///
    /// 内置模式下先查 [`BUILTIN`]（已校验过的编译期表），没登记的设备再按 [`ROOTS`] 读盘：
    /// 于是"表里漏了一台设备"不会变成"这台设备凭空消失"，而是照样从源码目录装载。
    ///
    /// # Errors
    ///
    /// 同上 [`load`]；内置模式下 vendor 既不在表里也没有源码目录时是 `missing`。
    pub fn fragments(&self, vendor: &str, ext: &str) -> diagnostic::Result<FragmentSet> {
        if let Some(root) = self.data {
            return load(&layer_root(root).join(vendor), vendor, ext);
        }
        if let Some(set) = builtin(vendor, ext) {
            return Ok(set);
        }
        let Some(dir) = dir_of(vendor) else {
            return Err(Error::new(
                Code::Missing,
                format!("`{vendor}` 既不在内置表里，也没有源码目录：`.{ext}` 片段无处可读"),
            ));
        };
        load(Path::new(dir), vendor, ext)
    }

    /// 见证文件 `witness.file`（`<vendor>/witness.<ext>`）的可读路径。
    ///
    /// 与 [`Layout::fragments`] 同一套解析：`data.root` 下是 `<root>/prim/<file>`，内置是
    /// 那个设备 crate 的源码树。`None` 表示 vendor 不认识或 `file` 没有 `<vendor>/` 前缀。
    pub fn witness(&self, file: &str) -> Option<PathBuf> {
        let (vendor, name) = file.split_once('/')?;
        match self.data {
            Some(root) => Some(layer_root(root).join(vendor).join(name)),
            None => Some(Path::new(dir_of(vendor)?).join(name)),
        }
    }
}

impl prof::prof::WitnessLookup for Layout<'_> {
    fn path(&self, file: &str) -> Option<PathBuf> {
        self.witness(file)
    }
}

/// 按 `(vendor, ext)` 查内嵌片段；没有登记时返回 `None`（调用方改读磁盘）。
///
/// 内嵌片段声明的签名读不出来也算「这台设备没有内嵌片段」：`BUILTIN` 是编译期数据，
/// 写成非法形态就是 crate 缺陷，那与其在运行期 panic，不如让调用方回到磁盘的同一份文本。
pub fn builtin(vendor: &str, ext: &str) -> Option<FragmentSet> {
    let entry = BUILTIN
        .iter()
        .find(|entry| entry.vendor == vendor && entry.ext == ext)?;
    let mut set = FragmentSet::new(entry.vendor, entry.ext);
    for (prim, source) in entry.fragments {
        set.insert(*prim, *source).ok()?;
    }
    Some(set)
}

/// 从 `dir`（一台设备自己的目录）读取所有 `<prim>.<ext>` 片段。
///
/// `witness.<ext>` 被跳过，文件名不在 [`Prim`] 闭集内的其余文件报错，第一行不是签名声明的
/// 片段也报错——"这个文件算不算一段片段"是声明，不是 io。
///
/// # Errors
///
/// 目录不存在是 `missing`（`data.root` 指错时最常发生），不可读是 `io`；文件名不在闭集内、
/// 或片段的第一行不是签名声明是 `invalid`。
pub fn load(dir: &Path, vendor: &str, ext: &str) -> diagnostic::Result<FragmentSet> {
    let mut set = FragmentSet::new(vendor, ext);
    let entries = std::fs::read_dir(dir).map_err(|error| Error::from_io(dir, &error))?;
    for entry in entries {
        let entry = entry.map_err(|error| Error::new(Code::Io, error.to_string()))?;
        add(&mut set, &entry.path(), ext)?;
    }
    Ok(set)
}

/// 把一个目录项装进集合；不是片段就什么都不做。
///
/// 回答「这个文件算不算一段片段」，不回答「目录能不能读」——后者在 [`load`]。
fn add(set: &mut FragmentSet, path: &Path, ext: &str) -> diagnostic::Result<()> {
    let Some(stem) = fragment_stem(path, ext) else {
        return Ok(());
    };
    let prim = Prim::parse(stem).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{}` 的文件名 `{stem}` 不在原语闭集内", path.display()),
        )
    })?;
    let source = std::fs::read_to_string(path).map_err(|error| Error::from_io(path, &error))?;
    set.insert(prim, source)
}

/// 文件名里的原语名：扩展名不对、或叫 `witness` 的文件都不是片段。
fn fragment_stem<'a>(path: &'a Path, ext: &str) -> Option<&'a str> {
    let stem = path.file_stem().and_then(|value| value.to_str())?;
    let right_ext = path.extension().and_then(|value| value.to_str()) == Some(ext);
    (right_ext && stem != WITNESS_STEM).then_some(stem)
}

#[cfg(test)]
mod tests;
