// 本 crate 的片段 → `$OUT_DIR/tables.rs`；生成器与另外五台设备共用一份源码。
include!("../prim/generate.rs");

fn main() {
    generate(
        &PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")),
        "cpu",
        &PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")),
        env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some(),
    );
}
