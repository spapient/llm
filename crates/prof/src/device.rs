//! 设备行：每台设备这次构建带上的东西（档案元数据、见证源码，以及可选的设备侧二进制）。
//!
//! **拥有**：`Device`、`Builtin`、`Image`、`Status`。**不拥有**：片段文本与见证的装载（`prim`）、
//! 能力档案的解析（[`crate::prof`]）。
//!
//! 三者的归属由依赖方向定：片段语料与档案都住在本 crate（`src/<vendor>/` 与 `devices/<vendor>.toml`），
//! 一次性把两份语料扫成一批 `&'static` 表；`prim` 依赖本层，把表拼成装载入口。
//! 这些行的类型必须比 `prim` 更低一层——它们就是设备数据，放在档案层。
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
    /// 这台设备的输出语言还没有构建配方（[`crate::image`] 的配方表）。
    NoRecipe,
    /// 有配方，但工具链不在本机或编译失败：源码照旧可用。
    Skipped {
        /// 原因（一行、来自工具链的 stderr 或"不可执行"）。
        reason: &'static str,
    },
}

// ── 内嵌表查询 ────────────────────────────────────────────────────────

use crate::fragment::FragmentSet;
use std::sync::OnceLock;

/// 每台设备一行（顺序同 [`DEVICES`]，即生成期按 vendor 升序）。
pub fn devices() -> &'static [crate::Device] {
    &crate::DEVICES
}

/// 按 vendor 查设备行；没有登记时 `None`。表按 vendor 升序（生成序），二分查找。
pub fn device(vendor: &str) -> Option<&'static crate::Device> {
    let index = crate::DEVICES
        .binary_search_by_key(&vendor, |entry| entry.vendor)
        .ok()?;
    Some(&crate::DEVICES[index])
}

/// 按 vendor 查片段源码根（那个设备目录）；没有登记时 `None`。
pub fn dir_of(vendor: &str) -> Option<&'static str> {
    let index = crate::ROOTS
        .binary_search_by_key(&vendor, |(have, _)| have)
        .ok()?;
    Some(crate::ROOTS[index].1)
}

/// 按 `(vendor, ext)` 查内嵌片段；没有登记时返回 `None`（调用方改读磁盘）。
///
/// 内嵌表构建期已校验（签名声明、`(prim, impl)` 与档案对账），装载不再重复解析；
/// 首次查询把六台设备一次建成缓存，之后按需克隆——克隆只付一份 `String` 分配（见 [`Fragment`]）。
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
    let index = crate::BUILTIN
        .binary_search_by_key(&(vendor, ext), |entry| (entry.vendor, entry.ext))
        .ok()?;
    sets[index].clone()
}

/// 按 `(vendor, ext)` 查内嵌见证；没有登记时返回 `None`（调用方改读磁盘）。
pub fn builtin_witness(vendor: &str, ext: &str) -> Option<&'static str> {
    let index = crate::WITNESS
        .binary_search_by_key(&(vendor, ext), |(have_vendor, have_ext, _)| {
            (*have_vendor, *have_ext)
        })
        .ok()?;
    Some(crate::WITNESS[index].2)
}
