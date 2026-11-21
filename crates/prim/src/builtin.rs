//! 编译期内嵌表：六个设备 crate 的产出合成的一张表，以及把它变成 [`FragmentSet`] 的唯一入口。
//!
//! 每台设备的片段文本、见证源码与设备行都由那个设备 crate 的 `build.rs` 生成（规则住在
//! [`prof::codegen`]），本 crate 只**合成**：`BUILTIN` / `WITNESS` / `DEVICES` / `ROOTS` 四张表按
//! 同一个下标同源。于是"加一台设备"＝加一个设备 crate，再在这里给四张表各补一行；顺序由
//! `the_tables_line_up_by_index` 守着，漏行编译器会数出来。
//!
//! 本 crate 没有自己的构建脚本：四张表都是普通 Rust 常量，不是 `include_str!` 拼出来的。

use crate::{FragmentSet, Prim};

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
///
/// 每一行的三列都来自设备 crate 自己（`<crate>::DEVICE.vendor` / `.ext` 与
/// `<crate>::FRAGMENTS`），本 crate 不写第二遍——vendor 名与档案、目录名对不上时，
/// 构建期就已经失败了，这里只是把它合成一张表。
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
///
/// 见证不是片段（装载时跳过），但发布出去的二进制里必须有它，否则"证据能跑"这件事
/// 在没有源码树的机器上无从谈起。
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
///
/// 顺序即 [`BUILTIN`] / [`WITNESS`] 的顺序；三张表按下标同源，所以 `devices()[i].vendor`
/// 一定能在 [`BUILTIN`] 里按同一个下标找到。
pub static DEVICES: [prof::Device; 6] = [
    rocm::DEVICE,
    cpu::DEVICE,
    opencl::DEVICE,
    metal::DEVICE,
    musa::DEVICE,
    cuda::DEVICE,
];

/// 每台设备的源码根目录：`(vendor, 目录)`。
///
/// 片段拆成六个 crate 之后，"片段层根"不再是一个目录——`crates/cpu` 与 `crates/cuda` 各有
/// 自己的源码树。本表把 vendor 映到那个 crate 的根，供 [`dir_of`] 与 [`crate::Layout`]
/// 解析见证文件。
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

/// 按 vendor 查片段源码根（那个设备 crate 的目录）；没有登记时 `None`。
pub fn dir_of(vendor: &str) -> Option<&'static str> {
    ROOTS
        .iter()
        .find(|(have, _)| *have == vendor)
        .map(|(_, root)| *root)
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

/// 按 `(vendor, ext)` 查内嵌见证；没有登记时返回 `None`（调用方改读磁盘）。
pub fn builtin_witness(vendor: &str, ext: &str) -> Option<&'static str> {
    WITNESS
        .iter()
        .find(|(have_vendor, have_ext, _)| *have_vendor == vendor && *have_ext == ext)
        .map(|(_, _, source)| *source)
}
