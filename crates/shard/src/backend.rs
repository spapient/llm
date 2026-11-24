//! 设备后端契约：cutile 只依赖这条缝，不认识 Metal / CUDA / C 的具体实现。
//!
//! 本缝只保留「见证片段装配 + 编译执行回读」：设备算子正文已由 prof 档案里的设备语言片段给出，
//! 这里不再按骨架重写/降低算子算法。生产态设备执行走 `prof::runtime` FFI 后端岛。

use crate::toolchain::Toolchain;
use crate::witness::{Scope, Selected};
use cutile::dirs::Dirs;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error, Result};

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub type Readback = (Vec<i64>, Vec<f32>);

/// 一段要编译并执行的程序源。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// 单源文本（C / CUDA）：编译成可执行文件直接跑。
    Single(String),
    /// 双源文本（Metal）：设备源（.metal）编成 metallib，host 驱动（.m）编成可执行文件并发射。
    Pair {
        /// 设备源（MSL）。
        device: String,
        /// host 驱动（ObjC）。
        host: String,
    },
}

/// 见证程序装配：把已选中的设备见证片段拼成「跑一次全部见证」的程序文本。
/// 不生成算子算法，只组合 prof 档案给出的设备正文。
pub trait Codegen {
    /// 这门语言（[`Emit`] 闭集）。
    fn emit(&self) -> Emit;
    /// 生成「跑一次全部见证」的程序文本；写出上限定在 ABI 常量 [`crate::witness::MAX_OUT`]。
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String>;
}

/// 编译 / 发射 / 回读：唯一碰工具链、设备 FFI 与子进程的一半，由设备后端实现。
pub trait Driver {
    /// 设备是否在本机（工具链 / 驱动就位）。
    fn available(&self, profile: &Profile) -> bool;
    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        max: usize,
        scope: Scope,
    ) -> Result<Readback>;
    /// 这条通道实测的工具链（进产物身份 / 报告）；拿不到（缺编译器）时为 `None`。
    fn toolchain(&self) -> Option<Toolchain>;
    /// 编译并执行一段程序源，返回 `(stdout, stderr)` 原文。
    fn compile_run(
        &self,
        _profile: &Profile,
        _dirs: &Dirs,
        _source: &Source,
        _label: &str,
    ) -> Result<(String, String)> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有编译车道".to_string(),
        ))
    }
}

/// 一台设备的执行后端：`Codegen` 装配见证 + `Driver` 跑源。
/// `Sync` 让后端能被写进进程级注册表（[`register`]）后从任意线程取用。
pub trait Backend: Codegen + Driver + Sync {}

/// 装配一台后端：L7（`cli`）在启动时注入 Metal / CUDA；cutile 已内置 host C 参考后端。
pub fn register(backend: &'static dyn Backend) {
    crate::device::register(backend);
}
