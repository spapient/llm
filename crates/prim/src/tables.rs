/// 六台设备的片段表：`(vendor, ext, 该设备的片段)`。
///
/// 每一行的三列都来自设备 crate 自己（`<crate>::DEVICE.vendor` / `.ext` 与
/// `<crate>::FRAGMENTS`），本 crate 不写第二遍——vendor 名与档案、目录名对不上时，
/// 构建期就已经失败了，这里只是把它合成一张表。
pub const BUILTIN: [crate::Builtin; 6] = [
    crate::Builtin {
        vendor: rocm::DEVICE.vendor,
        ext: rocm::DEVICE.ext,
        fragments: &rocm::FRAGMENTS,
    },
    crate::Builtin {
        vendor: cpu::DEVICE.vendor,
        ext: cpu::DEVICE.ext,
        fragments: &cpu::FRAGMENTS,
    },
    crate::Builtin {
        vendor: opencl::DEVICE.vendor,
        ext: opencl::DEVICE.ext,
        fragments: &opencl::FRAGMENTS,
    },
    crate::Builtin {
        vendor: metal::DEVICE.vendor,
        ext: metal::DEVICE.ext,
        fragments: &metal::FRAGMENTS,
    },
    crate::Builtin {
        vendor: musa::DEVICE.vendor,
        ext: musa::DEVICE.ext,
        fragments: &musa::FRAGMENTS,
    },
    crate::Builtin {
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
/// 自己的源码树。本表把 vendor 映到那个 crate 的根，供 [`crate::dir_of`] 与
/// [`crate::Layout`] 解析见证文件。
pub const ROOTS: [(&str, &str); 6] = [
    (rocm::DEVICE.vendor, rocm::ROOT),
    (cpu::DEVICE.vendor, cpu::ROOT),
    (opencl::DEVICE.vendor, opencl::ROOT),
    (metal::DEVICE.vendor, metal::ROOT),
    (musa::DEVICE.vendor, musa::ROOT),
    (cuda::DEVICE.vendor, cuda::ROOT),
];
