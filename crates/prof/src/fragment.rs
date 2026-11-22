//! 片段层：一段片段（原语名、签名声明、拼进模板的正文），以及"片段在哪"——仓库内置表
//! 或某个 `data.root` 的 `prim/` 目录。

use crate::device::{builtin, dir_of};
use crate::sign::EPILOGUE_STEM;
use crate::Prim;
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

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
            .any(|existing| existing.implementation() == fragment.implementation())
        {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 已登记 `{}` 实现：同一原语的同名片段不能放两份",
                    fragment.implementation()
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
            .find(|fragment| fragment.implementation() == implementation)
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
pub fn layer_root(data_root: &Path) -> PathBuf {
    data_root.join("prim")
}

/// 片段层在哪：仓库内置（六台设备各在自己的 crate 里）或某个 `data.root` 的 `prim/`。
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

/// 文件名里的原语名：扩展名不对、或叫 [`WITNESS_STEM`] / `epilogue` 的文件都不是片段。
///
/// 写回尾块的那份文本由设备行的 `epilogue` 字段直接带出去（`prof::Device`），不经过
/// [`FragmentSet`]——它的键是 [`Prim`]，六种写回形式不是原语。
fn fragment_stem<'a>(path: &'a Path, ext: &str) -> Option<&'a str> {
    let stem = path.file_stem().and_then(|value| value.to_str())?;
    let right_ext = path.extension().and_then(|value| value.to_str()) == Some(ext);
    let named = stem != WITNESS_STEM && stem != EPILOGUE_STEM;
    (right_ext && named).then_some(stem)
}
