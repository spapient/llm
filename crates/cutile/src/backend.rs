//! 设备后端契约：cutile 只依赖这条缝，不认识 Metal / CUDA / C 的具体实现。
//!
//! 见证执行（[`crate::witness`]）按 `emit` 选后端，经 `&dyn Backend` 生成程序、编译、运行、回读。
//! 具体后端实现这条 trait；本模块不出现任何厂商名字面量（`docs/ARCHITECTURE.md` 的 L3 契约）。
//! 后端差异是 **trait 实现**而非表内函数指针：装配点（`cli`）可以注入任意实现，本层不硬编码后端清单。

use crate::dirs::Dirs;
use crate::prof::{Emit, Profile};
use crate::witness::{Scope, Selected};
use diagnostic::Result;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub type Readback = (Vec<i64>, Vec<f32>);

/// 一台设备的执行后端：差异落在实现里，运行期只经 `dyn` 分派一次，没有枚举匹配。
pub trait Backend {
    /// 这门语言（[`Emit`] 闭集）。
    fn emit(&self) -> Emit;
    /// 设备是否在本机（工具链 / 驱动就位）。
    fn available(&self, profile: &Profile) -> bool;
    /// 生成「跑一次全部见证」的程序文本；写出上限定在 ABI 常量 [`crate::witness::MAX_OUT`]。
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String>;
    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        max: usize,
        scope: Scope,
    ) -> Result<Readback>;
}
