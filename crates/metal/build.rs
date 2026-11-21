//! 本 crate 的片段 → `$OUT_DIR/tables.rs`：一台设备的构建期入口。
//!
//! 生成器（扫描、签名校验、设备镜像）是 `crates/prim/generate.rs` 一份源码，六个设备 crate 各
//! `include!` 一次；这里只剩"哪台设备、输出到哪"。构建立即校验，所以片段写错名字、签名声明与档案
//! 对不上都在这里失败，而不是等运行期。
include!("../prim/generate.rs");

fn main() {
    generate(
        &PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")),
        "metal",
        &PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")),
        env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some(),
    );
}
