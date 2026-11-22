//! 本 crate 的片段 → `$OUT_DIR/tables.rs`：六台设备一次生成。
//!
//! 扫描、签名校验与设备镜像编译都在 `prof::codegen`（`build` feature）一份实现里；这里只说
//! "哪个目录是哪台设备"。构建立即校验，所以片段写错名字、签名声明与档案对不上都在这里失败，
//! 而不是等运行期。
//!
//! vendor 与目录名可以不同：档案文件与 `witness.file` 的前缀都用 vendor（`nvidia` 的目录叫
//! `cuda`，`intel` 的叫 `opencl`，`ascend` 的叫 `rocm`），目录怎么叫是仓库内部的事。

use prof::codegen;

fn main() {
    codegen::devices(&[
        ("rocm", "ascend"),
        ("cpu", "cpu"),
        ("opencl", "intel"),
        ("metal", "metal"),
        ("musa", "musa"),
        ("cuda", "nvidia"),
    ]);
}
