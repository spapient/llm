//! 设备行与片段装载：每台设备这次构建带上的东西（档案元数据、见证源码、可选镜像），
//! 以及一段原语正文「长什么样、从哪读」。
//!
//! **拥有**：设备行 `Device` / `Builtin` / `Image` / `Status`，片段层 `Fragment` / `FragmentSet` /
//! `Layout` / `load` / `layer_root`。**不拥有**：能力档案的解析（[`crate::prof`]）。
//!
//! 两半合住一个文件，因为它们是**双向引用的同一层**：设备行的查询返回 [`FragmentSet`]，片段的装载
//! 又回头查 [`dir_of`]——拆成两个文件只会让同一层在自己的边界上绕一圈。归属由依赖方向定：
//! 片段语料与档案都住在本 crate（`src/<vendor>/`），一次性扫成一批 `&'static` 表。
//!
//! # 单一事实源
//!
//! 字段全部是编译期数据（`&'static`），构造只发生在各设备模块生成的表里，所以"设备"没有第二个
//! 事实源：改设备 = 改目录与档案，重新构建即可。
//!
//! # 两种模式
//!
//! Cargo feature `device-image` 关着（默认，调试）：[`Device::image`] 恒为 `None`，源码就是全部，
//! 运行期照旧 emit + 编译；打开（发布）：能用本机工具链编出镜像的设备带上二进制，`metal` 的产物
//! 是 `metallib`。没有镜像不是错误，[`Device::status`] 给出原因——"缺设备即失败"由
//! `CUTILE_REQUIRE_IMAGE` 显式要求，不由默认构建强加。

use crate::atom::{Epilogue, Prim};

/// 一台设备的设备侧代码：源码 + 可选的镜像。
#[derive(Debug, Clone, Copy)]
pub struct Device {
    /// 设备名，与 `devices/<vendor>.toml` 的文件名和 `meta.vendor` 一致。
    pub vendor: &'static str,
    /// 片段与见证的扩展名，由 `emit` 语言决定（`c` / `cu` / `cl` / `cpp` / `metal`）。
    pub ext: &'static str,
    /// 输出语言名（`Emit::as_str`）。
    pub emit: &'static str,
    /// 档案声明的编译工具（`[meta].tools`）。
    pub tool: &'static str,
    /// 档案声明的工具版本（`[meta].tool_ver`）：**登记值**，不是本机实测值。
    pub tool_ver: &'static str,
    /// 原语枚举与映射表：这台设备有哪些原语，每个原语在镜像里的入口叫什么。
    pub prims: &'static [(Prim, &'static str)],
    /// 写回尾块片段：一份文本里按 `/* epi: <form> */` 分成六段，装配按段取用。
    pub epilogue: &'static str,
    /// 见证源码：调试模式下的设备侧代码，也是镜像的输入。
    pub source: &'static str,
    /// 设备镜像；只在 `device-image` feature 下、且本机工具链真的编出来时存在。
    pub image: Option<Image>,
    /// 镜像的结局（闭集，所以"为什么没有镜像"永远有答案）。
    pub status: Status,
}

impl Device {
    /// 原语枚举：这台设备有片段文本的那些原语。
    pub fn prims(&self) -> impl Iterator<Item = Prim> + '_ {
        self.prims.iter().map(|(prim, _)| *prim)
    }

    /// 原语在镜像（与见证源码）里的入口名。
    pub fn entry(&self, prim: Prim) -> Option<&'static str> {
        self.prims
            .iter()
            .find(|(have, _)| *have == prim)
            .map(|(_, entry)| *entry)
    }

    /// 写回尾块的某一段：`/* epi: <form> */` 之后、下一个标记之前的正文。
    ///
    /// 六种形式在同一份文本里，装配按语法顺序各取一段；找不到标记返回 `None`（构建期已经
    /// 保证六段齐全，这里的 `None` 只可能是调用方用错了 vendor）。
    pub fn epilogue_of(&self, form: Epilogue) -> Option<&'static str> {
        let marker = epi_marker(form);
        let rest = self.epilogue.split_once(marker.as_str())?.1;
        let body = rest.split("/* epi: ").next()?;
        Some(body.trim_matches('\n'))
    }

    /// 是否带上了设备镜像。
    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }
}

/// 一台 vendor、一种输出语言的内嵌片段。
#[derive(Debug, Clone, Copy)]
pub struct Builtin {
    /// vendor 名，与档案里的 `meta.vendor`、目录名三者一致。
    pub vendor: &'static str,
    /// 源文件扩展名（`Emit::ext`）：`c` → `.c`、`metal` → `.metal`。
    pub ext: &'static str,
    /// 片段：原语 → 文本。
    pub fragments: &'static [(Prim, &'static str)],
}

/// 写回尾块某一形式在文本里的段标记：`/* epi: bias */`。
pub fn epi_marker(form: Epilogue) -> String {
    format!("/* epi: {} */", form.as_str())
}

/// 编译好的设备镜像。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Image {
    format: &'static str,
    bytes: &'static [u8],
    symbols: usize,
}

impl Image {
    /// 由格式名、字节与符号数构造（只有生成的表调用）。
    pub const fn new(format: &'static str, bytes: &'static [u8], symbols: usize) -> Self {
        Self {
            format,
            bytes,
            symbols,
        }
    }

    /// 格式名：`metallib`。
    pub const fn format(&self) -> &'static str {
        self.format
    }

    /// 镜像字节。
    pub const fn bytes(&self) -> &'static [u8] {
        self.bytes
    }

    /// 字节数。
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    /// 是否为空（空的镜像没有意义，生成时不会出现）。
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// 镜像里已定义的文本符号数（编译器数出来的；`0` 表示没数到）。
    pub const fn symbols(&self) -> usize {
        self.symbols
    }
}

/// 设备镜像的结局。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// 未开启 `device-image`：只有源码，运行期再编译（调试路径）。
    Source,
    /// 已编译成设备镜像并内嵌（[`Device::image`] 必然是 `Some`）。
    Compiled,
    /// 这台设备的输出语言还没有构建配方（`buildimage` 的配方表）。
    NoRecipe,
    /// 有配方，但工具链不在本机或编译失败：源码照旧可用。
    Skipped {
        /// 原因（一行、来自工具链的 stderr 或"不可执行"）。
        reason: &'static str,
    },
}

// ── 内嵌表查询 ────────────────────────────────────────────────────────

use std::sync::OnceLock;

/// 生成表查询：表按生成期顺序升序，二分查找。四张表（设备行 / 片段根 / 内嵌片段 / 内嵌见证）
/// 共用这一份口径：哪个表、按什么键、缺了返回 `None`——口径只写一次。
fn position<T, K: Ord>(table: &[T], key: &K, key_of: impl Fn(&T) -> K) -> Option<usize> {
    table.binary_search_by_key(key, key_of).ok()
}

/// 每台设备一行（顺序同 [`crate::DEVICES`]，即生成期按 vendor 升序）。
pub fn devices() -> &'static [crate::Device] {
    &crate::DEVICES
}

/// 按 vendor 查设备行；没有登记时 `None`。表按 vendor 升序（生成序），二分查找。
pub fn device(vendor: &str) -> Option<&'static crate::Device> {
    let idx = position(&crate::DEVICES, &vendor, |entry| entry.vendor)?;
    Some(&crate::DEVICES[idx])
}

/// 按 vendor 查片段源码根（那个设备目录）；没有登记时 `None`。
pub fn dir_of(vendor: &str) -> Option<&'static str> {
    let idx = position(&crate::ROOTS, &vendor, |(have, _)| *have)?;
    Some(crate::ROOTS[idx].1)
}

/// 按 `(vendor, ext)` 查内嵌片段；没有登记时返回 `None`（调用方改读磁盘）。
///
/// 内嵌表构建期已校验（签名声明、`(prim, impl)` 与档案对账），装载不再重复解析；
/// 首次查询把六台设备一次建成缓存，之后每次取用只加一次引用计数（[`FragmentSet`] 内部是 `Arc`，
/// 见 [`FragmentSet`]）：几百段片段不会因为"再取一次"被重新拷一遍。
pub fn builtin(vendor: &str, ext: &str) -> Option<FragmentSet> {
    static CACHE: OnceLock<Vec<Option<FragmentSet>>> = OnceLock::new();
    let sets = CACHE.get_or_init(|| {
        crate::BUILTIN
            .iter()
            .map(|entry| {
                let mut set = FragmentSet::new(entry.vendor, entry.ext);
                for (prim, source) in entry.fragments {
                    set.insert(*prim, *source).ok()?;
                }
                Some(set)
            })
            .collect()
    });
    let idx = position(&crate::BUILTIN, &(vendor, ext), |entry| {
        (entry.vendor, entry.ext)
    })?;
    sets[idx].clone()
}

/// 按 `(vendor, ext)` 查内嵌见证；没有登记时返回 `None`（调用方改读磁盘）。
pub fn builtin_witness(vendor: &str, ext: &str) -> Option<&'static str> {
    let idx = position(&crate::WITNESS, &(vendor, ext), |(have, have_ext, _)| {
        (*have, *have_ext)
    })?;
    Some(crate::WITNESS[idx].2)
}

// ── 片段：一段文本 + 「片段在哪儿」 ────────────────────────────────────
//
// 同一层里的另一半：设备行回答「这台机器这次带上了什么」，片段层回答「一段原语正文长什么样、
// 从哪读」。两者曾经各占一个文件并且互相引用（设备行的查询返回 `FragmentSet`，片段的装载
// 又回头查 `device`/`dir_of`）——那是一个模块被切成两半的证据，现在合成一处。

use crate::sign::{EPILOGUE_STEM, PREAMBLE_STEM};
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 签名声明的标记：片段第一行 `/* cutile-sign: <prim>/<impl> */` 里的 `cutile-sign:`。
pub const SIGN_MARKER: &str = crate::sign::MARKER;

/// 见证源码的文件名（不含扩展名）。
pub use crate::sign::WITNESS_STEM;

/// 一段片段：一份文本，加两个指向它的字节区间——正文（声明行之后）与 `impl` 名。
///
/// 只有一份 `String`：`body` 与 `implementation` 是它的切片区间而不是第二、第三份拷贝，
/// 克隆一段片段只付一份分配。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    source: String,
    /// 正文起始（声明行之后的字节偏移）。
    body: usize,
    /// `impl` 名在第一行内的字节区间。
    implementation: Range<usize>,
}

impl Fragment {
    /// 片段全文（含第一行的签名声明）。
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 真正拼进模板的文本：声明行之后的部分。
    pub fn body(&self) -> &str {
        &self.source[self.body..]
    }

    /// 声明实现的 `impl` 名：档案里那条 `provides.impl`。
    pub fn implementation(&self) -> &str {
        &self.source[self.implementation.clone()]
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
        Ok(Self {
            body: start_of(&source, body),
            implementation: offset(&source, implementation),
            source,
        })
    }
}

/// 一个切片在 `source` 里的起始字节偏移；构造完成后 `source` 整体移入结构体，偏移仍然有效。
fn start_of(source: &str, part: &str) -> usize {
    part.as_ptr() as usize - source.as_ptr() as usize
}

/// 一个切片在 `source` 里的字节区间（`start_of` 加上长度）。
fn offset(source: &str, part: &str) -> Range<usize> {
    let start = start_of(source, part);
    start..start + part.len()
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
fn declaration_parts(first: &str, prim: Prim) -> diagnostic::Result<(&str, &str)> {
    let declared = crate::sign::strip_marker(first).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!(
                "片段 `{prim}` 的第一行必须是 `/* {SIGN_MARKER} {prim}/<impl> */`，实际是 `{first}`"
            ),
        )
    })?;
    let (declared_prim, implementation) = crate::sign::split_slash(declared).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{prim}` 的签名声明 `{declared}` 不是 `<prim>/<impl>`"),
        )
    })?;
    Ok((declared_prim, implementation))
}

/// 某个 vendor 的片段集合。
///
/// 正文放在 [`Arc`] 后面：内置表一次装好六台设备，之后每次取用只加一次引用计数，克隆是 O(1)
/// 而不是把两三百段片段（每段一份 `String`）重新拷一遍。写时复制由 [`Arc::make_mut`] 保证——
/// 克隆之后 [`FragmentSet::insert`] 改的是自己那份，原集合不受影响。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSet {
    inner: Arc<Inner>,
}

/// 片段集合的正文：共享单元（所有读方法都只读它，只有 `insert` 会触发写时复制）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Inner {
    vendor: String,
    ext: String,
    fragments: BTreeMap<Prim, Vec<Fragment>>,
}

impl FragmentSet {
    /// 新建空集合。
    pub fn new(vendor: &str, ext: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                vendor: vendor.to_string(),
                ext: ext.to_string(),
                fragments: BTreeMap::new(),
            }),
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
        // 先判重再取写权限：重复登记是错误路径，没有理由为它先复制一份共享正文。
        let duplicate = self.inner.fragments.get(&prim).is_some_and(|list| {
            list.iter()
                .any(|existing| existing.implementation() == fragment.implementation())
        });
        if duplicate {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 已登记 `{}` 实现：同一原语的同名片段不能放两份",
                    fragment.implementation()
                ),
            ));
        }
        let list = Arc::make_mut(&mut self.inner)
            .fragments
            .entry(prim)
            .or_default();
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
        self.inner
            .fragments
            .get(&prim)?
            .iter()
            .find(|fragment| fragment.implementation() == implementation)
    }

    /// 片段声明实现的 `impl` 名；没有这段片段时 `None`。
    pub fn implementation(&self, prim: Prim) -> Option<&str> {
        self.first(&prim).map(Fragment::implementation)
    }

    /// 遍历某原语登记的全部片段（按登记顺序）。
    pub fn fragments_of(&self, prim: Prim) -> &[Fragment] {
        self.inner
            .fragments
            .get(&prim)
            .map_or_default(Vec::as_slice)
    }

    /// 设备名。
    pub fn vendor(&self) -> &str {
        &self.inner.vendor
    }

    /// 片段所在的扩展名。
    pub fn ext(&self) -> &str {
        &self.inner.ext
    }

    /// 片段总数（同一原语的多条实现分别计数）。
    pub fn len(&self) -> usize {
        self.inner.fragments.values().map(Vec::len).sum()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.inner.fragments.values().all(Vec::is_empty)
    }

    /// 取某原语登记的首份片段。
    fn first(&self, prim: &Prim) -> Option<&Fragment> {
        self.inner.fragments.get(prim)?.first()
    }
}

/// 数据根对应的片段层根目录：`<data.root>/prim`。
pub fn layer_root(data_root: &Path) -> PathBuf {
    data_root.join("prim")
}

/// 片段层在哪：仓库内置（六台设备各在自己的 `src/<vendor>/`）或某个 `data.root` 的 `prim/`。
#[derive(Debug, Clone, Copy)]
pub struct Layout<'a> {
    /// `Some` 表示读磁盘上的 `data.root`；`None` 表示内置表。
    data: Option<&'a Path>,
}

impl<'a> Layout<'a> {
    /// 仓库内置的片段层：六台设备各自的 `src/<vendor>/`。
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
    /// 内置模式下先查 [`crate::BUILTIN`]（已校验过的编译期表），没登记的设备再按 [`crate::ROOTS`]
    /// 读盘：于是"表里漏了一台设备"不会变成"这台设备凭空消失"，而是照样从源码目录装载。
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
    pub fn witness(&self, file: &str) -> Option<PathBuf> {
        let (vendor, name) = file.split_once('/')?;
        match self.data {
            Some(root) => Some(layer_root(root).join(vendor).join(name)),
            None => Some(Path::new(dir_of(vendor)?).join(name)),
        }
    }
}

impl crate::prof::WitnessLookup for Layout<'_> {
    fn path(&self, file: &str) -> Option<PathBuf> {
        self.witness(file)
    }
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

/// 文件名里的原语名：扩展名不对、叫 [`WITNESS_STEM`] / [`EPILOGUE_STEM`] /
/// [`PREAMBLE_STEM`]，或与内置算子同名的特化内核，都不是原语片段。
///
/// 写回尾块由设备行的 `epilogue` 字段带出去，前置声明只供拼装期内联；算子特化正文走
/// `kernel_path` 特化优先链，不进按 [`Prim`] 索引的集合。算子名判据取编译期闭集，不依赖
/// 磁盘上是否有对应 `ops/<stem>.k`（临时对拍目录只复制设备目录，不拷模板）。
fn fragment_stem<'a>(path: &'a Path, ext: &str) -> Option<&'a str> {
    let stem = path.file_stem().and_then(|value| value.to_str())?;
    let right_ext = path.extension().and_then(|value| value.to_str()) == Some(ext);
    (right_ext && !is_reserved_source(stem)).then_some(stem)
}

/// 保留名与算子特化都不占用原语集合；算子名直接取编译期 `OPS` 闭集。
fn is_reserved_source(stem: &str) -> bool {
    matches!(stem, WITNESS_STEM | EPILOGUE_STEM | PREAMBLE_STEM)
        || crate::OPS.iter().any(|view| view.name == stem)
}
