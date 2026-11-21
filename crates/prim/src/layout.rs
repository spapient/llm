//! 片段层在哪：仓库内置（六台设备各在自己的 crate 里）或某个 `data.root` 的 `prim/`。
//!
//! 两种模式的差别只有**目录解析**，装载、校验、报错全部共用，所以"换一份片段"不会换一套
//! 语义：[`Layout::builtin`] 是仓库内置，[`Layout::at`] 是 `data.root` 覆盖
//! （`<data.root>/prim/<vendor>/<prim>.<ext>`）。两种模式走同一个装载入口
//! [`Layout::fragments`]，因此"文件名不在闭集内 / 签名声明写错"在两种模式下报同一类错误。
//!
//! `witness.<ext>` 与片段同目录、同扩展名，但**不是片段**：装载时按 [`WITNESS_STEM`] 跳过它。

use crate::builtin::{builtin, dir_of};
use crate::{FragmentSet, Prim};
use diagnostic::{Code, Error};
use std::path::{Path, PathBuf};

/// 见证源码的文件名（不含扩展名）。
///
/// 它与片段同目录、同扩展名，但语义不同：片段是"某原语怎么写"，见证是"证据能跑"。
/// 装载片段时按这个名字跳过它。
pub const WITNESS_STEM: &str = "witness";

/// 数据根对应的片段层根目录：`<data.root>/prim`。
///
/// 一个 `data.root` 同时含 `ops/`、`devices/`、`prim/` 三层，本 crate 只解释最后一层。
pub fn layer_root(data_root: &Path) -> PathBuf {
    data_root.join("prim")
}

/// 片段层在哪：仓库内置（六台设备各在自己的 crate 里）或某个 `data.root` 的 `prim/`。
///
/// `data.root` 覆盖时布局是 `<data.root>/prim/<vendor>/<prim>.<ext>`，与内置的
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

/// 文件名里的原语名：扩展名不对、或叫 [`WITNESS_STEM`] 的文件都不是片段。
fn fragment_stem<'a>(path: &'a Path, ext: &str) -> Option<&'a str> {
    let stem = path.file_stem().and_then(|value| value.to_str())?;
    let right_ext = path.extension().and_then(|value| value.to_str()) == Some(ext);
    (right_ext && stem != WITNESS_STEM).then_some(stem)
}
