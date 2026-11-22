//! 片段层在哪：仓库内置（六台设备各在自己的 crate 里）或某个 `data.root` 的 `prim/`。

use crate::builtin::{builtin, dir_of};
use crate::stems::EPILOGUE_STEM;
use crate::{FragmentSet, Prim};
use diagnostic::{Code, Error};
use std::path::{Path, PathBuf};

/// 见证源码的文件名（不含扩展名）。
pub use crate::stems::WITNESS_STEM;

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
