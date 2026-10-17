//! Build script for the GPU backends: the MUSA `mcc` precompilation pipeline
//! and the Metal framework link.
//!
//! Kernels are compiled at build time by `mcc` into ELF fatbins and embedded
//! through `include_bytes!` in the generated `registry` (docs/musa-backend-
//! plan.md §5.3). This is the A1 path of the performance analysis: build-time
//! compilation, zero runtime JIT, no MTRTC dependency on MUSA 3.1.0.
//!
//! Platform contract:
//! - On any non-Linux target the MUSA module is compiled out, so the script
//!   emits an empty registry and does not require `mcc`.
//! - On Linux, `mcc` is probed (override with `MUSA_MCC`, default
//!   `/usr/local/musa-3.1.0/bin/mcc`). When it is absent the script emits an
//!   empty registry: every MUSA operator then fails with a device error at run
//!   time, and the workspace stays green on hosts without the toolkit.
//! - The target architecture is `--offload-arch=mp_22` (MTT S4000), overridable
//!   with `MUSA_ARCH`.
//! - On macOS the script links CoreGraphics, which `MTLCreateSystemDefaultDevice`
//!   requires (objc2-metal's documented requirement). Without it the device
//!   probe silently returns `None` on a machine that has a GPU.
//!
//! Each shared `cuda/*.cu` and the MUSA-only `musa/musa.cu`
//! compiles to `OUT_DIR/fatbin/<stem>.o`; the generated registry lists every
//! `(stem, include_bytes!(fatbin))` pair so the runtime module loader can find
//! a fatbin by source id without a filesystem. Duplicate stems are a build
//! error.
//!
//! All failures propagate through `main -> Result` (never `panic!`/`expect`),
//! so an `mcc` failure is a build failure, exactly as the no-runtime-JIT
//! design intends.

use std::env;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::process::Command;

/// A build-script failure: reported by cargo as the build error.
#[derive(Debug)]
struct builderr(String);

impl fmt::Display for builderr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for builderr {}

fn err(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(builderr(message.into()))
}

fn main() -> Result<(), Box<dyn Error>> {
    // Declared on every host so `-D warnings` (check.sh's clippy) accepts the
    // cfg even when the toolkit is absent and the cfg is never emitted.
    println!("cargo::rustc-check-cfg=cfg(musa_toolkit)");
    println!("cargo:rerun-if-changed=build.rs");
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            println!("cargo:rustc-link-lib=framework=CoreGraphics");
            // The MUSA module is compiled out here, but keep the generated
            // registry present so a toolchain-forced `musa_toolkit` cfg still
            // resolves its `include!`.
            emit(&[])?;
            return Ok(());
        }
        Ok("linux") => {}
        _ => {
            emit(&[])?;
            return Ok(());
        }
    }

    let mcc = env::var("MUSA_MCC").unwrap_or_else(|_| "/usr/local/musa-3.1.0/bin/mcc".into());
    let arch = env::var("MUSA_ARCH").unwrap_or_else(|_| "mp_22".into());
    if !std::path::Path::new(&mcc).is_file() {
        // No toolkit on this host: the MUSA module compiles out (the
        // `musa_toolkit` cfg is absent), exactly like on non-Linux targets. The
        // registry stays empty and any device access errors at run time.
        emit(&[])?;
        return Ok(());
    }
    // The toolkit is present: compile the full backend and link the driver.
    println!("cargo:rustc-cfg=musa_toolkit");
    // Link the MUSA driver (`libmusa.so`) for the `mu*` C symbols declared
    // in `musa/ffi.rs`. Verified by `nm` on the MTT S4000 host: `muInit`,
    // `muDeviceGet`, `muModuleLoadData`, `muLaunchKernel` etc. are defined in
    // `libmusa.so` (driver layer), while `libmusart.so` exports the `musa*`
    // runtime API. The driver directory is fixed at the toolkit install
    // location; the loader finds it when
    // `LD_LIBRARY_PATH=/usr/local/musa/lib` (or the same inside the toolkit
    // tree) is exported at run time, documented in docs/musa-backend-plan.md.
    println!("cargo:rustc-link-search=/usr/local/musa-3.1.0/lib");
    println!("cargo:rustc-link-lib=dylib=musa");

    let out = PathBuf::from(env::var("OUT_DIR").map_err(|e| err(format!("OUT_DIR: {e}")))?);
    let fatbin = out.join("fatbin");
    std::fs::create_dir_all(&fatbin).map_err(|e| err(format!("create fatbin dir: {e}")))?;

    // The shared kernels live in `cuda/`; MUSA's Tensor-Core family
    // is its platform-specific source. Both compile into one registry.
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir("cuda")
        .map_err(|e| err(format!("read cuda dir: {e}")))?
        .filter_map(Result::ok)
        .collect();
    files.sort_by_key(|entry| entry.file_name());
    for entry in files {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("cu") {
            paths.push(path);
        }
    }
    paths.push(PathBuf::from("musa/musa.cu"));

    let mut entries: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if seen.iter().any(|existing| existing == stem) {
            return Err(err(format!("duplicate kernel stem {stem}")));
        }
        seen.push(stem.to_string());
        let obj = fatbin.join(format!("{stem}.o"));
        // `mcc` is a clang driver (`/usr/local/musa-3.1.0/bin/mcc` -> clang):
        // like every clang, it accepts `--offload-arch` only in the `=` form,
        // so the value must be embedded rather than passed as a second arg.
        let offload = format!("--offload-arch={arch}");
        // --cuda-device-only produces a pure device cubin that muModuleLoadData
        // can load and muModuleGetFunction can resolve by plain kernel name
        // (verified by a C probe on the MTT S4000 host: the host+device
        // relocatable object from plain -c loads but resolves no kernel).
        let status = Command::new(&mcc)
            .args([
                "-mtgpu",
                &offload,
                "--cuda-device-only",
                "-DMUSA_OUT_STATE=1",
                "-c",
                "-o",
            ])
            .arg(&obj)
            .arg(&path)
            .status()
            .map_err(|e| err(format!("mcc failed to start for {stem}: {e}")))?;
        if !status.success() {
            return Err(err(format!("mcc failed for {stem}")));
        }
        let objstr = obj.to_string_lossy().to_string();
        entries.push(format!("    (\"{stem}\", include_bytes!(\"{objstr}\")),"));
    }
    emit(&entries)?;
    Ok(())
}

/// Writes `OUT_DIR/registry.rs`: the kernel fatbin registry embedded by
/// `crate::musa::registry`.
fn emit(entries: &[String]) -> Result<(), Box<dyn Error>> {
    let mut reg = String::new();
    reg.push_str("// @generated by build.rs. Do not edit.\n");
    reg.push_str("// Build-time kernel registry: `(source id, fatbin bytes)`\n");
    reg.push_str("// pairs embedded from `OUT_DIR/fatbin`, produced by the `mcc`\n");
    reg.push_str("// precompilation pipeline (docs/musa-backend-plan.md §5.3).\n");
    reg.push_str("pub(crate) const KERNELS: &[(&str, &[u8])] = &[\n");
    reg.push_str(&entries.join("\n"));
    reg.push_str("\n];\n");
    let path = env::var("OUT_DIR").map_err(|e| err(format!("OUT_DIR: {e}")))? + "/registry.rs";
    std::fs::write(path, reg).map_err(|e| err(format!("write registry: {e}")))?;
    Ok(())
}
