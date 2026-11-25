//! 构建期装配入口：解析设备档案（`devices`），生成静态表（`buildtables`/`buildops`）与
//! 原生内核绑定（`nativebuild`），产物写入 `$OUT_DIR`。
//!
//! 通用路径/TOML 内核在 [`diagnostic::buildgen`]；本文件只做环境装配、子模块调度与构建期
//! 失败出口。设备扫描与档案校验在 `devices`，表与代码生成在各 `build*` 模块。
//!
//! 构建期校验替代运行期断言：签名对不上文件名、`(prim, impl)` 档案无供给、见证入口缺失、
//! 尾块段不齐，全部在这里失败；运行期只遍历生成后的强类型静态视图。

#[path = "src/sign.rs"]
mod sign;

#[path = "src/buildops.rs"]
mod buildops;

#[path = "src/fficontract.rs"]
mod fficontract;

#[path = "src/nativebuild.rs"]
mod nativebuild;

#[path = "src/buildtables.rs"]
mod buildtables;

#[path = "src/buildimage.rs"]
mod buildimage;

#[path = "src/devices.rs"]
mod devices;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use diagnostic::buildgen;

pub use devices::Device;

fn main() {
    let manifest = variable("CARGO_MANIFEST_DIR");
    let src = PathBuf::from(&manifest).join("src");
    let out = PathBuf::from(variable("OUT_DIR"));

    // 整棵树都算输入：新增一段片段、换一份见证都必须让构建脚本重跑。
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEVICE_IMAGE");
    println!("cargo:rerun-if-env-changed=CUTILE_REQUIRE_IMAGE");

    let devices = devices::devices_of(&src);
    let images = env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some();
    let tables = buildtables::tables_of(&devices, &out, images);
    let (template_list, templates_table) =
        buildops::template_tables_of(Path::new(&manifest), &devices);
    nativebuild::build_native(Path::new(&manifest), &out);

    for (name, text) in [
        ("embedded.rs", buildtables::archives_of(&devices)),
        ("tables.rs", tables),
        ("template_list.rs", template_list),
        ("templates_table.rs", templates_table),
    ] {
        let path = out.join(name);
        fs::write(&path, text)
            .unwrap_or_else(|error| fail(format!("写 {} 失败：{error}", path.display())));
    }
}

/// 读取一份文件正文，I/O 错误直接终止构建；构建期子 mod 经 `super::read` 复用。
fn read(path: &Path) -> String {
    buildgen::read_text(path).unwrap_or_else(|error| fail(format!("{error}")))
}

/// 构建脚本必带的环境变量；缺了说明调用方不是 `build.rs`，直接失败而不是猜一个目录。
fn variable(name: &str) -> String {
    buildgen::env_required(name).unwrap_or_else(|error| fail(format!("{error}")))
}

/// 构建期失败：报给 `cargo` 并以非零码结束，`build.rs` 因此失败。
fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("prof::build: {message}");
    std::process::exit(1)
}

/// 拆开 `Option`，缺值直接以 [`fail`] 终止构建；调用点不必再写 `||` 闭包。
fn expect<T>(value: Option<T>, message: impl std::fmt::Display) -> T {
    match value {
        Some(value) => value,
        None => fail(message),
    }
}
