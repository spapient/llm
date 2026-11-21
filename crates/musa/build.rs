//! 本 crate 的片段 → `$OUT_DIR/tables.rs`：一台设备的构建期入口。
//!
//! 扫描、签名校验与设备镜像编译都在 `prof::codegen`（`build` feature）一份实现里，六个设备 crate
//! 共用；这里只说"这是哪台设备"。构建立即校验，所以片段写错名字、签名声明与档案对不上都在这里
//! 失败，而不是等运行期。

fn main() {
    prof::codegen::device("musa");
}
