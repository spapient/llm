//! 编译期内嵌表：`build.rs` 合成后放在这里的四张表，以及把表变成 [`FragmentSet`] 的入口。
//!
//! 表本身不是手写的：`$OUT_DIR/tables.rs` 由 [`build.rs`](../../build.rs) 从 `src/<vendor>/`
//! 的档案与片段生成，本模块只剩查询。所以"加一台设备"不会在这里留下一行。

pub use crate::{BUILTIN, DEVICES, ROOTS, WITNESS};
use crate::FragmentSet;

/// 每台设备一行（顺序同 [`DEVICES`]，即生成期按 vendor 升序）。
pub fn devices() -> &'static [crate::Device] {
    &DEVICES
}

/// 按 vendor 查设备行；没有登记时 `None`。
pub fn device(vendor: &str) -> Option<&'static crate::Device> {
    DEVICES.iter().find(|device| device.vendor == vendor)
}

/// 按 vendor 查片段源码根（那个设备目录）；没有登记时 `None`。
pub fn dir_of(vendor: &str) -> Option<&'static str> {
    ROOTS
        .iter()
        .find(|(have, _)| *have == vendor)
        .map(|(_, root)| *root)
}

/// 按 `(vendor, ext)` 查内嵌片段；没有登记时返回 `None`（调用方改读磁盘）。
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

/// 按 `(vendor, ext)` 查内嵌见证；没有登记时返回 `None`（调用方改读磁盘）。
pub fn builtin_witness(vendor: &str, ext: &str) -> Option<&'static str> {
    WITNESS
        .iter()
        .find(|(have_vendor, have_ext, _)| *have_vendor == vendor && *have_ext == ext)
        .map(|(_, _, source)| *source)
}
