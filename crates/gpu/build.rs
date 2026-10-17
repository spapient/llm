//! Build script for the GPU backends: the MUSA `mcc` precompilation pipeline,
//! the GRID-family toolchain pipeline (ROCm / SYCL / CANN / MLU), and the
//! Metal framework link.
//!
//! Kernels are compiled at build time into ELF fatbins and embedded through
//! `include_bytes!` in the generated `registry` (docs/musa-backend-plan.md
//! §5.3, docs/gpu-backend-consolidation-assessment.md). This is the A1 path
//! of the performance analysis: build-time compilation, zero runtime JIT.
//!
//! Platform contract:
//! - On any non-Linux target every accelerator module is compiled out, so the
//!   script emits an empty registry and does not require any toolkit.
//! - On Linux, each backend's compiler is probed (override each with its env
//!   variable, default paths below). When a compiler is absent its module
//!   compiles out: the corresponding operator set then fails with a device
//!   error at run time, and the workspace stays green on hosts without the
//!   toolkit.
//! - The shared `cuda/*.cu` kernels are compiled by the MUSA `mcc` driver (the
//!   same clang-based driver that serves `grid/musa.c`), gated on the MUSA
//!   toolkit exactly as before.
//! - On macOS the script links CoreGraphics, which `MTLCreateSystemDefaultDevice`
//!   requires (objc2-metal's documented requirement).
//!
//! Each non-empty kernel source compiles to `OUT_DIR/fatbin/<stem>.o`; the
//! generated registry lists every `(stem, include_bytes!(fatbin))` pair so the
//! runtime module loader can find a fatbin by source id without a filesystem.
//! Duplicate stems are a build error.
//!
//! All failures propagate through `main -> Result` (never `panic!`/`expect`),
//! so a compiler failure is a build failure, exactly as the no-runtime-JIT
//! design intends.

use std::env;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
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

/// One toolchain-gated backend: how to probe its compiler, compile its single
/// kernel translation unit, and link its runtime. Four backends share this one
/// pipeline, so the whole GRID family is one config table.
struct backendcfg<'a> {
    /// The rustc cfg emitted when the toolchain is found.
    cfg: &'a str,
    /// Env variable that overrides the compiler path.
    env: &'a str,
    /// Probe candidates in order (the env value wins when set).
    probes: &'a [&'a str],
    /// Compiler flags; the arch value may embed `$ARCH` and is substituted
    /// from the per-backend `*_ARCH` env variable (default below).
    flags: &'a [&'a str],
    /// A language-forcing flag (`-x hip` / `-x c++`) required because the
    /// repository names every kernel source `.c` even when it is a C++ dialect.
    lang: Option<&'a str>,
    /// The kernel source (relative to the crate root).
    src: &'a str,
    /// The registry stem (and fatbin file stem).
    stem: &'a str,
    /// Default architecture substituted into flags that contain `$ARCH`.
    arch: &'a str,
    /// Env variable that overrides the architecture.
    arch_env: &'a str,
    /// Linker search path; `None` links nothing.
    link_search: Option<&'a str>,
    /// Linker library names (`dylib=` form); `None` links nothing.
    link_lib: Option<&'a [&'a str]>,
    /// Extra `rustc-link-arg`; `None` for none.
    link_arg: Option<&'a str>,
    /// Link the compiled object into the crate as a static archive. Backends
    /// whose runtime resolves kernels as ordinary symbols (SYCL's `sycl_*`
    /// bridge, MLU's `cnrtGetKernel`) need this; backends that load a fatbin
    /// through the registry (ROCm, CANN, MUSA) do not.
    archive: bool,
}

const GRIDFAMILY: [backendcfg<'_>; 4] = [
    backendcfg {
        cfg: "rocm_toolkit",
        env: "HIPCC",
        probes: &["hipcc", "/opt/rocm/bin/hipcc"],
        flags: &["--offload-arch=$ARCH", "-fgpu-rdc"],
        lang: Some("-x hip"),
        // HIP is source-compatible with CUDA, so the ROCm backend compiles the
        // same translation unit as CUDA and MUSA (`-x hip` forces the HIP
        // language for the `.cu` suffix); one source, three toolchains.
        src: "grid/cuda.c",
        stem: "rocm",
        arch: "gfx90a",
        arch_env: "ROCM_ARCH",
        link_search: Some("/opt/rocm/lib"),
        link_lib: Some(&["amdhip64", "hipblas"]),
        link_arg: None,
        archive: false,
    },
    backendcfg {
        cfg: "sycl_toolkit",
        env: "ICPX",
        probes: &["icpx", "/opt/intel/oneapi/compiler/latest/bin/icpx"],
        flags: &["-fsycl", "-DGRID_SYCL"],
        lang: Some("-x c++"),
        src: "grid/sycl.c",
        stem: "sycl",
        arch: "",
        arch_env: "SYCL_ARCH",
        link_search: None,
        link_lib: None,
        // SYCL device code is embedded in the host object by `icpx -fsycl`;
        // the produced archive carries the kernels and `libsycl` provides the
        // host runtime. The exact `-fsycl` driver link is validated on a real
        // oneAPI host (docs/gpu-backend-consolidation-assessment.md).
        link_arg: Some("-fsycl"),
        archive: true,
    },
    backendcfg {
        cfg: "cann_toolkit",
        env: "CANN_CC",
        probes: &["ccec", "bisheng"],
        flags: &["--cce-aicore-arch=$ARCH", "-DGRID_CANN"],
        lang: None,
        src: "grid/cann.c",
        stem: "cann",
        // Ascend 910B; overridable with CANN_ARCH. Older releases use a
        // different spelling of the arch flag — validated on a real host.
        arch: "ai-core-910b",
        arch_env: "CANN_ARCH",
        link_search: Some("/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64"),
        link_lib: Some(&["ascendcl"]),
        link_arg: None,
        archive: false,
    },
    backendcfg {
        cfg: "mlu_toolkit",
        env: "CNCC",
        probes: &["cncc"],
        flags: &["--bang-mlu-arch=$ARCH", "-DGRID_MLU"],
        lang: None,
        src: "grid/mlu.c",
        stem: "mlu",
        // MLU370 / Siyuan 220; overridable with MLU_ARCH.
        arch: "mtp_372",
        arch_env: "MLU_ARCH",
        link_search: Some("/usr/local/neuware/lib64"),
        link_lib: Some(&["cnrt"]),
        link_arg: None,
        archive: true,
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    // Declared on every host so `-D warnings` (check.sh's clippy) accepts the
    // cfgs even when the toolkits are absent and the cfgs are never emitted.
    for cfg in [
        "musa_toolkit",
        "rocm_toolkit",
        "sycl_toolkit",
        "cann_toolkit",
        "mlu_toolkit",
    ] {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
    }
    println!("cargo:rerun-if-changed=build.rs");
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            println!("cargo:rustc-link-lib=framework=CoreGraphics");
            // Every accelerator module is compiled out here, but keep the
            // generated registry present so a toolchain-forced cfg still
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

    let mut entries: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let out = PathBuf::from(env::var("OUT_DIR").map_err(|e| err(format!("OUT_DIR: {e}")))?);
    let fatbin = out.join("fatbin");
    std::fs::create_dir_all(&fatbin).map_err(|e| err(format!("create fatbin dir: {e}")))?;

    // -----------------------------------------------------------------------
    // MUSA pipeline (unchanged): probe `mcc`, compile the shared `cuda/*.cu`
    // and every non-empty `grid/*.cu` into one registry.
    // -----------------------------------------------------------------------
    let mcc = env::var("MUSA_MCC").unwrap_or_else(|_| "/usr/local/musa-3.1.0/bin/mcc".into());
    let arch = env::var("MUSA_ARCH").unwrap_or_else(|_| "mp_22".into());
    if std::path::Path::new(&mcc).is_file() {
        println!("cargo:rustc-cfg=musa_toolkit");
        println!("cargo:rustc-link-search=/usr/local/musa-3.1.0/lib");
        println!("cargo:rustc-link-lib=dylib=musa");
        compile_cuda_family(&mcc, &arch, &fatbin, &mut entries, &mut seen)?;
    }

    // -----------------------------------------------------------------------
    // GRID-family pipeline: one config table drives probe -> cfg -> compile ->
    // link for ROCm / SYCL / CANN / MLU. A backend whose toolchain is absent
    // simply never emits its cfg, so its module compiles out.
    // -----------------------------------------------------------------------
    for backend in GRIDFAMILY {
        let Some(compiler) = probe(&backend) else {
            continue;
        };
        println!("cargo:rustc-cfg={}", backend.cfg);
        println!("cargo:rerun-if-changed={}", backend.src);
        if let Some(search) = backend.link_search {
            println!("cargo:rustc-link-search={search}");
        }
        if let Some(libs) = backend.link_lib {
            for lib in libs {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
        if let Some(arg) = backend.link_arg {
            println!("cargo:rustc-link-arg={arg}");
        }
        compile_one(&compiler, &backend, &fatbin, &mut entries, &mut seen)?;
    }

    emit(&entries)?;
    Ok(())
}

/// Resolves a backend's compiler: the env override wins, then the probe
/// candidates in order (an absolute path must exist; a bare name must resolve
/// through `PATH`).
fn probe(cfg: &backendcfg<'_>) -> Option<String> {
    match env::var(cfg.env) {
        Ok(over) if !over.is_empty() => return Some(over),
        _ => {}
    }
    for candidate in cfg.probes {
        if candidate.contains('/') {
            if std::path::Path::new(candidate).is_file() {
                return Some((*candidate).into());
            }
        } else if Command::new(candidate).arg("--version").output().is_ok() {
            return Some((*candidate).into());
        }
    }
    None
}

/// Compiles the shared CUDA-family kernels (`cuda/*.cu` + non-empty
/// `grid/*.cu`) with the MUSA `mcc` driver. Empty placeholders are skipped so
/// a scaffolded backend does not break the build.
fn compile_cuda_family(
    mcc: &str,
    arch: &str,
    fatbin: &Path,
    entries: &mut Vec<String>,
    seen: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for dir in ["grid"] {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| err(format!("read {dir} dir: {e}")))?
            .filter_map(Result::ok)
            .collect();
        files.sort_by_key(|entry| entry.file_name());
        for entry in files {
            let path = entry.path();
            // Every `*.cu` plus the dedicated `grid/musa.c` and `grid/cuda.c`
            // (CUDA-dialect
            // sources renamed to `.c` by the GRID naming rule; `-x cuda`
            // below forces the dialect back because the `.c` suffix alone
            // would select plain C). Other `.c` kernels (CANN/BANG) are
            // compiled by their own toolchains and must not enter here.
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let is_cuda_dialect = path.extension().and_then(|ext| ext.to_str()) == Some("cu")
                || name == "musa.c"
                || name == "cuda.c";
            if is_cuda_dialect {
                let nonempty = std::fs::metadata(&path)
                    .map_err(|e| err(format!("metadata {}: {e}", path.display())))?
                    .len()
                    > 0;
                if nonempty {
                    paths.push(path);
                }
            }
        }
    }
    // Rebuild when the GRID directory gains or drops a source file.
    println!("cargo:rerun-if-changed=grid");

    let offload = format!("--offload-arch={arch}");
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
        // `mcc` is a clang driver: like every clang, it accepts
        // `--offload-arch` only in the `=` form, so the value must be embedded
        // rather than passed as a second arg. `--cuda-device-only` produces a
        // pure device cubin that muModuleLoadData can load and
        // muModuleGetFunction can resolve by plain kernel name (verified by a
        // C probe on the MTT S4000 host).
        let status = Command::new(mcc)
            .args([
                "-mtgpu",
                "-x",
                "cuda",
                &offload,
                "--cuda-device-only",
                "-DMUSA_OUT_STATE=1",
                "-DGRID_MUSA",
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
        entries.push(format!(
            "    (\"{stem}\", include_bytes!(\"{}\")),",
            obj.to_string_lossy()
        ));
    }
    Ok(())
}

/// Compiles one GRID-family backend's single kernel translation unit into its
/// fatbin and registers it. The arch substitution, the language-forcing flag
/// (`.c` sources in C++ dialects), and the exact flags come from the config
/// row, so no backend gets its own pipeline code.
fn compile_one(
    compiler: &str,
    cfg: &backendcfg<'_>,
    fatbin: &Path,
    entries: &mut Vec<String>,
    seen: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    if seen.iter().any(|existing| existing == cfg.stem) {
        return Err(err(format!("duplicate kernel stem {}", cfg.stem)));
    }
    seen.push(cfg.stem.to_string());
    let obj = fatbin.join(format!("{}.o", cfg.stem));
    let arch = env::var(cfg.arch_env).unwrap_or_else(|_| cfg.arch.into());
    let mut args: Vec<String> = Vec::new();
    for flag in cfg.flags {
        args.push(flag.replace("$ARCH", &arch));
    }
    if let Some(lang) = cfg.lang {
        args.push(lang.into());
    }
    args.extend(["-c".into(), "-o".into()]);
    let status = Command::new(compiler)
        .args(&args)
        .arg(&obj)
        .arg(cfg.src)
        .status()
        .map_err(|e| {
            err(format!(
                "{} failed to start for {}: {e}",
                compiler, cfg.stem
            ))
        })?;
    if !status.success() {
        return Err(err(format!("{} failed for {}", compiler, cfg.stem)));
    }
    if cfg.archive {
        // The runtime resolves the kernels as ordinary linked symbols
        // (SYCL's `sycl_*` bridge, MLU's `cnrtGetKernel`), so pack the object
        // into a static archive next to the fatbin and link it into the crate.
        let archive = fatbin.join(format!("lib{}.a", cfg.stem));
        let status = Command::new("ar")
            .arg("cr")
            .arg(&archive)
            .arg(&obj)
            .status()
            .map_err(|e| err(format!("ar failed to start for {}: {e}", cfg.stem)))?;
        if !status.success() {
            return Err(err(format!("ar failed for {}", cfg.stem)));
        }
        println!("cargo:rustc-link-search={}", fatbin.to_string_lossy());
        println!("cargo:rustc-link-lib=static={}", cfg.stem);
    } else {
        entries.push(format!(
            "    (\"{}\", include_bytes!(\"{}\")),",
            cfg.stem,
            obj.to_string_lossy()
        ));
    }
    Ok(())
}

/// Writes `OUT_DIR/registry.rs`: the kernel fatbin registry embedded by the
/// backend modules (`crate::<backend>::registry`).
fn emit(entries: &[String]) -> Result<(), Box<dyn Error>> {
    let mut reg = String::new();
    reg.push_str("// @generated by build.rs. Do not edit.\n");
    reg.push_str("// Build-time kernel registry: `(source id, fatbin bytes)`\n");
    reg.push_str("// pairs embedded from `OUT_DIR/fatbin`, produced by the\n");
    reg.push_str("// toolchain pipelines (docs/musa-backend-plan.md §5.3,\n");
    reg.push_str("// docs/gpu-backend-consolidation-assessment.md).\n");
    reg.push_str("pub(crate) const KERNELS: &[(&str, &[u8])] = &[\n");
    reg.push_str(&entries.join("\n"));
    reg.push_str("\n];\n");
    let path = env::var("OUT_DIR").map_err(|e| err(format!("OUT_DIR: {e}")))? + "/registry.rs";
    std::fs::write(path, reg).map_err(|e| err(format!("write registry: {e}")))?;
    Ok(())
}
