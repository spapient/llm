//! Workspace-root build script: emits the `musa_toolkit` cfg for the root
//! package's targets (the cross-crate test and bench targets) when the MUSA
//! toolkit's `mcc` exists, mirroring `crates/musa/build.rs`.
//!
//! The `musa` crate compiles to an empty library on Linux hosts without the
//! toolkit; the MUSA contract suite and the MUSA benchmark gate on the same
//! cfg so those hosts link and run the other backends' targets instead of
//! failing on `mu*` symbols.
use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=MUSA_MCC");
    let mcc = env::var("MUSA_MCC").unwrap_or_else(|_| "/usr/local/musa-3.1.0/bin/mcc".into());
    if std::path::Path::new(&mcc).is_file() {
        println!("cargo:rustc-cfg=musa_toolkit");
    }
}
