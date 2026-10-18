//! Workspace-root build script: emits the `musa_toolkit` cfg for the root
//! package's targets (the cross-crate test and bench targets) when the MUSA
//! toolkit's `mcc` exists, mirroring `crates/gpu/build.rs`.
//!
//! The `gpu` crate's `musa` module compiles out on Linux hosts without the
//! toolkit; the MUSA contract suite and the MUSA benchmark gate on the same
//! cfg so those hosts link and run the other backends' targets instead of
//! failing on `mu*` symbols.
use std::env;

/// Whether `candidate` names an existing compiler.
///
/// Mirrors the GRID-family probe in `crates/gpu/build.rs`: an absolute or
/// multi-component path must exist as written; a bare command name is
/// searched on `PATH` (with `.exe` appended on Windows). Keeping the two
/// probes in step matters: the root cfg gates the cross-crate SYCL tests and
/// benchmarks, which must compile in exactly when the `gpu` backend does.
fn resolve(candidate: &str) -> bool {
    let path = std::path::Path::new(candidate);
    if path.is_absolute() || path.components().count() > 1 {
        return path.is_file();
    }
    let Some(pathvar) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&pathvar).any(|dir| {
        let full = dir.join(candidate);
        full.is_file()
            || (cfg!(windows) && full.extension().is_none() && full.with_extension("exe").is_file())
    })
}

fn main() {
    // Declared on every host so `-D warnings` (check.sh's clippy) accepts the
    // cfg even when the toolkit is absent and the cfg is never emitted.
    println!("cargo::rustc-check-cfg=cfg(musa_toolkit)");
    println!("cargo::rustc-check-cfg=cfg(sycl_toolkit)");
    println!("cargo:rerun-if-env-changed=MUSA_MCC");
    let mcc = env::var("MUSA_MCC").unwrap_or_else(|_| "/usr/local/musa-3.1.0/bin/mcc".into());
    if std::path::Path::new(&mcc).is_file() {
        println!("cargo:rustc-cfg=musa_toolkit");
    }
    // SYCL: probe the oneAPI driver so the sycl contract suite compiles in
    // on Windows hosts (and Linux hosts with icpx on PATH or the env var).
    println!("cargo:rerun-if-env-changed=ICPX");
    let icpx = env::var("ICPX").unwrap_or_else(|_| String::new());
    let sycl_probes = [
        // Same order as `crates/gpu/build.rs`: the Windows driver first, and
        // the bare names resolved through PATH before the install roots.
        "icx.exe",
        "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin\\icx.exe",
        "C:\\Program Files\\Intel\\oneAPI\\compiler\\latest\\bin\\icx.exe",
        "icpx",
        "/opt/intel/oneapi/compiler/latest/bin/icpx",
    ];
    let has_sycl = (!icpx.is_empty() && resolve(&icpx))
        || sycl_probes.iter().any(|candidate| resolve(candidate));
    if has_sycl {
        println!("cargo:rustc-cfg=sycl_toolkit");
    }
}
