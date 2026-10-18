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
//! - On macOS every accelerator module is compiled out, so the script emits
//!   an empty registry and does not require any toolkit.
//! - On Linux, each backend's compiler is probed (override each with its env
//!   variable, default paths below). When a compiler is absent its module
//!   compiles out: the corresponding operator set then fails with a device
//!   error at run time, and the workspace stays green on hosts without the
//!   toolkit.
//! - On Windows only SYCL is available. Its bridge is linked into
//!   `sycl_bridge.dll` by `icx-cl` at build time (the driver must own the
//!   link to register the embedded device images), and the Rust crate links
//!   the generated import library with the ordinary MSVC linker. The DLL is
//!   copied next to Cargo's executables so tests and binaries find it.
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

/// A detected oneMKL installation with the SYCL BLAS interface.
struct mklcfg {
    /// The include root that holds `oneapi/mkl/blas.hpp`.
    include: PathBuf,
    /// The directory holding the four import libraries.
    libdir: PathBuf,
    /// The link libraries, in dependency order.
    libs: &'static [&'static str],
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
        probes: &[
            // Windows oneAPI: the clang driver `icx.exe` accepts the same
            // `-fsycl` dialect flags; it is probed before `icpx` because a
            // WSL-interop `icpx` can shadow the Windows driver on PATH.
            "icx.exe",
            "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin\\icx.exe",
            "C:\\Program Files\\Intel\\oneAPI\\compiler\\latest\\bin\\icx.exe",
            "icpx",
            "/opt/intel/oneapi/compiler/latest/bin/icpx",
        ],
        // `-fexceptions`: the Windows icx.exe driver disables exceptions
        // under `-fsycl` by default, but the bridge's host-side SYCLTRY
        // wrapper relies on try/catch to report failures over the C ABI.
        flags: &["-fsycl", "-fexceptions", "-DGRID_SYCL"],
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
    declarecfgs();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    if skip_target()? {
        return Ok(());
    }

    let out = PathBuf::from(env::var("OUT_DIR").map_err(|e| err(format!("OUT_DIR: {e}")))?);
    let fatbin = out.join("fatbin");
    std::fs::create_dir_all(&fatbin).map_err(|e| err(format!("create fatbin dir: {e}")))?;
    let mut entries: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();

    compile_musa(&fatbin, &mut entries, &mut seen)?;

    // oneMKL is independent of the compiler probe: a toolkit install can exist
    // without the MKL component. Detection is all-or-nothing, and a miss keeps
    // the portable reference matmul.
    let mkl = find_mkl();
    for backend in GRIDFAMILY {
        compile_backend(&backend, mkl.as_ref(), &fatbin, &mut entries, &mut seen)?;
    }
    emit(&entries)
}

/// Declares every rustc cfg this script may emit.
///
/// Declared on every host so `-D warnings` (check.sh's clippy) accepts the
/// cfgs even when the toolkits are absent and the cfgs are never emitted.
fn declarecfgs() {
    for cfg in [
        "musa_toolkit",
        "rocm_toolkit",
        "sycl_toolkit",
        "cann_toolkit",
        "mlu_toolkit",
    ] {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
    }
}

/// Returns `true` when this target has no accelerator pipeline.
///
/// On macOS every accelerator module is compiled out, but the generated
/// registry stays present so a toolchain-forced cfg still resolves its
/// `include!`. Unsupported targets are treated the same way.
fn skip_target() -> Result<bool, Box<dyn Error>> {
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            println!("cargo:rustc-link-lib=framework=CoreGraphics");
            emit(&[])?;
            Ok(true)
        }
        // Windows runs the GRID family (SYCL at minimum): the icx.exe
        // oneAPI driver compiles `grid/sycl.c` with `-fsycl` and links
        // against `sycl.lib` explicitly, so no `-fsycl` final-link flag is
        // needed (MSVC's link.exe would reject it).
        Ok("linux" | "windows") => Ok(false),
        _ => {
            emit(&[])?;
            Ok(true)
        }
    }
}

/// Probes `mcc` and compiles the shared CUDA-family kernels into the registry.
fn compile_musa(
    fatbin: &Path,
    entries: &mut Vec<String>,
    seen: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let mcc = env::var("MUSA_MCC").unwrap_or_else(|_| "/usr/local/musa-3.1.0/bin/mcc".into());
    let arch = env::var("MUSA_ARCH").unwrap_or_else(|_| "mp_22".into());
    if std::path::Path::new(&mcc).is_file() {
        println!("cargo:rustc-cfg=musa_toolkit");
        println!("cargo:rustc-link-search=/usr/local/musa-3.1.0/lib");
        println!("cargo:rustc-link-lib=dylib=musa");
        compile_cuda_family(&mcc, &arch, fatbin, entries, seen)?;
    }
    Ok(())
}

/// Probes one GRID-family backend and, when present, emits its cfg, links its
/// runtime, and compiles its kernel translation unit.
///
/// A backend whose toolchain is absent simply never emits its cfg, so its
/// module compiles out.
fn compile_backend(
    backend: &backendcfg<'_>,
    mkl: Option<&mklcfg>,
    fatbin: &Path,
    entries: &mut Vec<String>,
    seen: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let Some(compiler) = probe(backend) else {
        return Ok(());
    };
    println!("cargo:rustc-cfg={}", backend.cfg);
    println!("cargo:rerun-if-changed={}", backend.src);
    let mkl = if backend.stem == "sycl" { mkl } else { None };
    if let Some(mkl) = mkl {
        // The switch is the C macro `GRID_MKL` below; no Rust cfg is needed
        // because nothing in the crate branches on it.
        // On Windows the libraries go into `sycl_bridge.dll`; the crate
        // links only the bridge's import library (see `link_sycl_dll`).
        if !target_is_windows() {
            println!("cargo:rustc-link-search={}", mkl.libdir.display());
            for lib in mkl.libs {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
    }
    if let Some(search) = backend.link_search {
        println!("cargo:rustc-link-search={search}");
    }
    if let Some(libs) = backend.link_lib {
        for lib in libs {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
    if let Some(arg) = backend.link_arg {
        // On Linux `-fsycl` reaches the final link, where the oneAPI driver
        // registers the embedded device images. On Windows the SYCL bridge is
        // linked into `sycl_bridge.dll` by `icx-cl` in `compile_one`, and the
        // final Rust link is plain MSVC `link.exe`, which has no `-fsycl`.
        if !(target_is_windows() && backend.stem == "sycl") {
            println!("cargo:rustc-link-arg={arg}");
        }
    }
    compile_one(&compiler, backend, fatbin, entries, seen, mkl)
}

/// Locates an installed oneMKL with the SYCL BLAS interface.
///
/// `MKLROOT` wins, then the oneAPI default install roots. Detection is
/// all-or-nothing: the header and all four libraries must exist, so a partial
/// install never produces a half-linked bridge. A miss is not an error — the
/// bridge keeps the portable reference matmul.
fn find_mkl() -> Option<mklcfg> {
    const LIBS: &[&str] = &["mkl_sycl", "mkl_intel_lp64", "mkl_sequential", "mkl_core"];
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(root) = env::var("MKLROOT")
        && !root.is_empty()
    {
        roots.push(PathBuf::from(root));
    }
    if target_is_windows() {
        roots.push(PathBuf::from(
            "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest",
        ));
        roots.push(PathBuf::from(
            "C:\\Program Files\\Intel\\oneAPI\\mkl\\latest",
        ));
    } else {
        roots.push(PathBuf::from("/opt/intel/oneapi/mkl/latest"));
    }
    for root in roots {
        if !root.join("include/oneapi/mkl/blas.hpp").is_file() {
            continue;
        }
        // Windows keeps the import libraries in `<root>\lib`; Linux oneAPI
        // installs them under `<root>/lib/intel64` (older layouts use
        // `<root>/lib` directly).
        for libdir in [root.join("lib"), root.join("lib/intel64")] {
            let present = LIBS.iter().all(|lib| {
                let name = if target_is_windows() {
                    format!("{lib}.lib")
                } else {
                    format!("lib{lib}.so")
                };
                libdir.join(name).is_file()
            });
            if present {
                return Some(mklcfg {
                    include: root.join("include"),
                    libdir,
                    libs: LIBS,
                });
            }
        }
    }
    None
}

/// Resolves a backend's compiler: the env override wins, then the probe
/// candidates in order. The returned path is absolute (a bare command name is
/// resolved through `PATH`), because callers derive toolchain directories and
/// sibling drivers from it.
fn probe(cfg: &backendcfg<'_>) -> Option<String> {
    match env::var(cfg.env) {
        Ok(over) if !over.is_empty() => return resolve_command(&over),
        _ => {}
    }
    cfg.probes
        .iter()
        .find_map(|candidate| resolve_command(candidate))
}

/// Resolves a probe candidate to an absolute path.
///
/// An absolute path must name an existing file; a relative path containing a
/// separator is resolved against the working directory; a bare command name
/// is searched on `PATH` (with `.exe` appended on Windows when the candidate
/// has no extension).
fn resolve_command(candidate: &str) -> Option<String> {
    let path = Path::new(candidate);
    if path.is_absolute() || path.components().count() > 1 {
        return path.is_file().then(|| candidate.to_string());
    }
    let pathvar = env::var_os("PATH")?;
    for dir in env::split_paths(&pathvar) {
        let full = dir.join(candidate);
        if full.is_file() {
            return Some(full.to_string_lossy().into_owned());
        }
        if cfg!(windows) && full.extension().is_none() {
            let exe = full.with_extension("exe");
            if exe.is_file() {
                return Some(exe.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// True when Cargo is building for Windows (`CARGO_CFG_TARGET_OS`).
fn target_is_windows() -> bool {
    env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
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
    mkl: Option<&mklcfg>,
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
    // Windows icx.exe runs in clang-cl (MSVC-compatible) mode, where the
    // exception control is spelled `/EHsc`; the SYCL bridge's host side
    // catches `sycl::exception` over the C ABI, so exceptions must stay on.
    // `GRID_SYCL_DLL` marks every bridge entry as `dllexport` for the DLL
    // link below (the static Linux archive needs no export attributes).
    if target_is_windows() {
        args.push("/EHsc".into());
        if cfg.stem == "sycl" {
            args.push("-DGRID_SYCL_DLL".into());
        }
    }
    // The dual-track matmul is selected at compile time: with oneMKL present
    // the bridge calls `oneapi::mkl::blas` (Arc XMX path); otherwise it keeps
    // the portable reference kernel.
    if cfg.stem == "sycl"
        && let Some(mkl) = mkl
    {
        args.push("-DGRID_MKL".into());
        args.push(format!("-I{}", mkl.include.display()));
    }
    if let Some(lang) = cfg.lang {
        for part in lang.split_whitespace() {
            args.push(part.into());
        }
    }
    // A build script does not inherit rustc's optimization flags, and the
    // clang driver defaults to `-O0`, so without this every profile would
    // compile unoptimized device code. Map Cargo's `OPT_LEVEL` onto the
    // driver's spelling (0/1/2/3/s/z all exist) for the SYCL bridge; the
    // other GRID backends keep their existing flags.
    if cfg.stem == "sycl" {
        let opt = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".into());
        args.push(format!("-O{opt}"));
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
        if target_is_windows() && cfg.stem == "sycl" {
            // Windows ships the bridge as a DLL: only the oneAPI driver can
            // register the embedded device images at link time. The crate
            // then links the generated import library with plain link.exe.
            link_sycl_dll(compiler, &obj, fatbin, mkl)?;
            println!("cargo:rustc-link-search={}", fatbin.to_string_lossy());
            println!("cargo:rustc-link-lib=dylib=sycl_bridge");
        } else {
            // The runtime resolves the kernels as ordinary linked symbols
            // (SYCL's `sycl_*` bridge on Linux, MLU's `cnrtGetKernel`), so
            // pack the object into a static archive next to the fatbin and
            // link it into the crate.
            let archive = fatbin.join(format!("lib{}.a", cfg.stem));
            let artool =
                if target_is_windows() && Command::new("ar").arg("--version").output().is_err() {
                    "llvm-ar"
                } else {
                    "ar"
                };
            let status = Command::new(artool)
                .arg("cr")
                .arg(&archive)
                .arg(&obj)
                .status()
                .map_err(|e| err(format!("{artool} failed to start for {}: {e}", cfg.stem)))?;
            if !status.success() {
                return Err(err(format!("ar failed for {}", cfg.stem)));
            }
            println!("cargo:rustc-link-search={}", fatbin.to_string_lossy());
            println!("cargo:rustc-link-lib=static={}", cfg.stem);
        }
    } else {
        entries.push(format!(
            "    (\"{}\", include_bytes!(\"{}\")),",
            cfg.stem,
            obj.to_string_lossy()
        ));
    }
    Ok(())
}

/// Links the Windows SYCL bridge into `sycl_bridge.dll` and its import
/// library, then copies the DLL next to Cargo's executables.
///
/// A plain MSVC `link.exe` invocation keeps the host code but drops the
/// device-image registration, so every kernel submission fails with a SYCL
/// exception. `icx-cl -fsycl /LD` performs the device link and writes the
/// import library the Rust crate links against. The DLL lives in `fatbin`
/// (like the other build artifacts); copies in `<profile>` and
/// `<profile>/deps` make it discoverable by `cargo run` binaries and by
/// test/bench executables without a PATH override.
fn link_sycl_dll(
    compiler: &str,
    obj: &Path,
    fatbin: &Path,
    mkl: Option<&mklcfg>,
) -> Result<(), Box<dyn Error>> {
    let bin = Path::new(compiler)
        .parent()
        .ok_or_else(|| err(format!("SYCL compiler path has no parent: {compiler}")))?;
    let cl = bin.join("icx-cl.exe");
    if !cl.is_file() {
        return Err(err(format!(
            "Windows SYCL needs the cl-compatible driver {} (not found)",
            cl.display()
        )));
    }
    let libdir = bin.parent().map(|root| root.join("lib")).ok_or_else(|| {
        err(format!(
            "SYCL compiler path has no toolchain root: {compiler}"
        ))
    })?;
    let sycl_lib = libdir.join("sycl.lib");
    if !sycl_lib.is_file() {
        return Err(err(format!(
            "oneAPI SYCL runtime import library missing: {}",
            sycl_lib.display()
        )));
    }
    let dll = fatbin.join("sycl_bridge.dll");
    let implib = fatbin.join("sycl_bridge.lib");
    let mut command = Command::new(&cl);
    command.arg("-fsycl").arg("/LD").arg(obj).arg(&sycl_lib);
    if let Some(mkl) = mkl {
        command.arg(format!("/LIBPATH:{}", mkl.libdir.display()));
        for lib in mkl.libs {
            command.arg(mkl.libdir.join(format!("{lib}.lib")));
        }
    }
    let status = command
        .arg(format!("/Fe:{}", dll.display()))
        .arg("/link")
        .arg(format!("/IMPLIB:{}", implib.display()))
        .arg(format!("/LIBPATH:{}", libdir.display()))
        .status()
        .map_err(|e| {
            err(format!(
                "{} failed to start for sycl DLL: {e}",
                cl.display()
            ))
        })?;
    if !status.success() {
        return Err(err("icx-cl failed to link sycl_bridge.dll"));
    }
    // OUT_DIR is `<profile>/build/gpu-<hash>/out`; `fatbin` adds one level.
    let profile = fatbin
        .ancestors()
        .nth(4)
        .ok_or_else(|| err(format!("unexpected OUT_DIR layout: {}", fatbin.display())))?;
    for dir in [profile.to_path_buf(), profile.join("deps")] {
        std::fs::create_dir_all(&dir).map_err(|e| err(format!("create {}: {e}", dir.display())))?;
        std::fs::copy(&dll, dir.join("sycl_bridge.dll"))
            .map_err(|e| err(format!("copy sycl_bridge.dll to {}: {e}", dir.display())))?;
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
