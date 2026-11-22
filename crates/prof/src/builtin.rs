//! 编译期内嵌表：六个设备模块的产出合成的一张表，以及把它变成 [`FragmentSet`] 的唯一入口。

use crate::{FragmentSet, Prim};
use frag::{cpu, cuda, metal, musa, opencl, rocm};

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

/// 六台设备的片段表：`(vendor, ext, 该设备的片段)`。
pub const BUILTIN: [Builtin; 6] = [
    Builtin {
        vendor: rocm::DEVICE.vendor,
        ext: rocm::DEVICE.ext,
        fragments: &rocm::FRAGMENTS,
    },
    Builtin {
        vendor: cpu::DEVICE.vendor,
        ext: cpu::DEVICE.ext,
        fragments: &cpu::FRAGMENTS,
    },
    Builtin {
        vendor: opencl::DEVICE.vendor,
        ext: opencl::DEVICE.ext,
        fragments: &opencl::FRAGMENTS,
    },
    Builtin {
        vendor: metal::DEVICE.vendor,
        ext: metal::DEVICE.ext,
        fragments: &metal::FRAGMENTS,
    },
    Builtin {
        vendor: musa::DEVICE.vendor,
        ext: musa::DEVICE.ext,
        fragments: &musa::FRAGMENTS,
    },
    Builtin {
        vendor: cuda::DEVICE.vendor,
        ext: cuda::DEVICE.ext,
        fragments: &cuda::FRAGMENTS,
    },
];

/// 内嵌的见证源码：`(vendor, ext, 文本)`。
pub const WITNESS: [(&str, &str, &str); 6] = [
    (rocm::DEVICE.vendor, rocm::DEVICE.ext, rocm::DEVICE.source),
    (cpu::DEVICE.vendor, cpu::DEVICE.ext, cpu::DEVICE.source),
    (
        opencl::DEVICE.vendor,
        opencl::DEVICE.ext,
        opencl::DEVICE.source,
    ),
    (
        metal::DEVICE.vendor,
        metal::DEVICE.ext,
        metal::DEVICE.source,
    ),
    (musa::DEVICE.vendor, musa::DEVICE.ext, musa::DEVICE.source),
    (cuda::DEVICE.vendor, cuda::DEVICE.ext, cuda::DEVICE.source),
];

/// 每台设备一行：vendor、输出语言、工具链、入口表、见证源码与（可选的）设备镜像。
pub static DEVICES: [prof::Device; 6] = [
    rocm::DEVICE,
    cpu::DEVICE,
    opencl::DEVICE,
    metal::DEVICE,
    musa::DEVICE,
    cuda::DEVICE,
];

/// 每台设备的源码根目录：`(vendor, 目录)`。
pub const ROOTS: [(&str, &str); 6] = [
    (rocm::DEVICE.vendor, rocm::ROOT),
    (cpu::DEVICE.vendor, cpu::ROOT),
    (opencl::DEVICE.vendor, opencl::ROOT),
    (metal::DEVICE.vendor, metal::ROOT),
    (musa::DEVICE.vendor, musa::ROOT),
    (cuda::DEVICE.vendor, cuda::ROOT),
];

/// 每台设备一行（顺序同 [`DEVICES`]）。
pub fn devices() -> &'static [prof::Device] {
    &DEVICES
}

/// 按 vendor 查设备行；没有登记时 `None`。
pub fn device(vendor: &str) -> Option<&'static prof::Device> {
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
