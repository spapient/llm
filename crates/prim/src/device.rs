//! 设备镜像：每台设备这次构建带上的东西（源码，以及可选的设备侧二进制）。
//!
//! **拥有**：`Device`、`Image`、`Status`、[`devices`]、[`device`]。**不拥有**：片段文本
//! （[`crate::BUILTIN`]）、见证源码（[`crate::WITNESS`]）、能力档案（`prof`）、编译配方
//! （`build.rs`）。**被依赖**：`cutile`（经片段层）。
//!
//! 三张表都由 `build.rs` 在编译期从 `<vendor>/` 扫出来，这一层只解释它们——于是"这台设备有哪些
//! 原语"（[`Device::prims`]）与"每个原语的片段在哪"（[`crate::BUILTIN`]）同源，不会各写一份。
//!
//! # 两种模式
//!
//! Cargo feature `device-image` 关着（默认，调试）：[`Device::image`] 恒为 `None`，源码就是全部，
//! 运行期照旧 emit + 编译；打开（发布）：能用本机工具链编出镜像的设备带上二进制，`metal` 的产物
//! 是 `metallib`。没有镜像不是错误，[`Device::status`] 给出原因——"缺设备即失败"由
//! `CUTILE_REQUIRE_IMAGE` 显式要求，不由默认构建强加。
//!
//! # 镜像里是什么
//!
//! 是 `<vendor>/witness.<ext>` 编译出的设备侧库：一份完整翻译单元、每条签名一个入口，所以
//! [`Device::prims`] 能指向具体符号。片段不在里面——片段是语句块，`$0..$3` 的绑定由算子模板的
//! `@prim(...)` 给出，脱离模板编译只能得到与真实调用无关的代码。
//!
//! # 例子
//!
//! ```
//! let device = llm_prim::device("cpu").expect("内置设备");
//! assert_eq!(device.ext, "c");
//! assert!(device.entry(llm_prim::Prim::Load).is_some(), "每个原语都有入口");
//! assert_eq!(device.image.is_some(), device.status == llm_prim::Status::Compiled);
//! ```

use prof::atom::Prim;

/// 一台设备的设备侧代码：源码 + 可选的镜像。
///
/// 字段全部是编译期数据（`&'static`），构造只发生在 [`crate::DEVICES`] 里，所以"设备"没有第二个
/// 事实源：改设备 = 改目录与档案，重新构建即可。
#[derive(Debug, Clone, Copy)]
pub struct Device {
    /// 目录名，与 `prof/devices/<vendor>.toml` 的文件名和 `meta.vendor` 一致。
    pub vendor: &'static str,
    /// 片段与见证的扩展名，由 `emit` 语言决定（`c` / `cu` / `cl` / `cpp` / `metal`）。
    pub ext: &'static str,
    /// 输出语言名（`Emit::as_str`）。
    pub emit: &'static str,
    /// 档案声明的编译工具（`[meta].tool`）。
    pub tool: &'static str,
    /// 档案声明的工具版本（`[meta].tool_ver`）：**登记值**，不是本机实测值。
    ///
    /// 本机实测值由 `cutile::toolchain::probe` 读 `--version` 得到，进产物身份（`命令@版本`）
    /// 与 `verify` 报告；登记值只说明"这条证据按哪个版本记的"。
    pub tool_ver: &'static str,
    /// 原语枚举与映射表：这台设备有哪些原语，每个原语在镜像里的入口叫什么。
    ///
    /// 顺序与 [`Prim::ALL`] 一致；入口名来自档案里 `(原语, impl)` 那条供给的见证函数，
    /// 所以它一定能在这台设备的 [`Device::source`] 里找到。
    pub prims: &'static [(Prim, &'static str)],
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

    /// 是否带上了设备镜像。
    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }
}

/// 编译好的设备镜像。
///
/// 只有三个数：格式、字节、符号数。字节随二进制一起发布，格式决定谁来解释它，符号数是编译器
/// 数出来的——"镜像里有几个入口"因此不靠自述。
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
    /// 这台设备的输出语言还没有构建配方（见 `build.rs` 的 `recipe`）。
    NoRecipe,
    /// 有配方，但工具链不在本机或编译失败：源码照旧可用。
    Skipped {
        /// 原因（一行、来自工具链的 stderr 或"不可执行"）。
        reason: &'static str,
    },
}

/// 全部设备：`build.rs` 扫描 `<vendor>/` 得到的表，顺序按目录名。
pub fn devices() -> &'static [Device] {
    &crate::DEVICES
}

/// 按 vendor 查设备；没有这个目录时 `None`（调用方改读 `data.root`）。
pub fn device(vendor: &str) -> Option<&'static Device> {
    devices().iter().find(|device| device.vendor == vendor)
}

#[cfg(test)]
#[path = "../tests/device_tests.rs"]
mod tests;
