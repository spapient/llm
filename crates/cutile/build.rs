//! 构建期工具链探测：`cfg(nightly)` 决定 `native::sme` 是否存在。
//!
//! `cutile::native` 是 `prof::native` 的零成本转发面（nn → cutile → prof），因此必须与 `prof`
//! 用同一套 `cfg(nightly)` 语义，才能在不重复工具链探测的前提下让条件 `sme` 转发匹配
//! `prof::native::sme` 的 `#[cfg(all(nightly, target_arch = "aarch64"))]` 可见性。
//! 逻辑与 `prof/build.rs` 的 `nightly()` 保持一致：版本串带 `-nightly`，或 `RUSTC_BOOTSTRAP` 已开。

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nightly)");
    println!("cargo::rerun-if-env-changed=RUSTC");
    println!("cargo::rerun-if-env-changed=RUSTC_BOOTSTRAP");
    if nightly() {
        println!("cargo::rustc-cfg=nightly");
    }
}

/// 本机 `rustc` 是不是 nightly：版本串带 `-nightly`，或者 `RUSTC_BOOTSTRAP` 已开。
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
