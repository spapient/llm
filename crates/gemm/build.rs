//! 工具链探测：产出 `cfg(nightly)`，把「需要 nightly 才编得过的通道」与「stable 也能编过的退路」
//! 交给条件编译，而不是交给 `rust-toolchain.toml` 把整个工作区钉在某个 nightly 上。
//!
//! 为什么不用 `rustversion`：这里只需要一个布尔量，一次 `rustc --version` 就够；引一个构建期
//! 依赖会把它拉进每个消费方的构建树，而收益只是省掉下面这十行。

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nightly)");
    println!("cargo::rerun-if-changed=build.rs");
    if nightly() {
        println!("cargo::rustc-cfg=nightly");
    }
}

/// 本机 `rustc` 是不是 nightly：版本串带 `-nightly`，或者 `RUSTC_BOOTSTRAP` 已开。
///
/// 后者也算 nightly：`RUSTC_BOOTSTRAP` 在 stable 上同样放行 `#![feature]`，把它判成 stable
/// 会走"没声明特性"的那条分支，编译结果与运行行为反而对不上。
fn nightly() -> bool {
    if std::env::var_os("RUSTC_BOOTSTRAP").is_some() {
        return true;
    }
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("nightly"))
}
