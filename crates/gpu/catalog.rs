//! Kernel catalog: every kernel whose source is shared, plus the dispatch
//! geometry.
//!
//! `matmul` has a genuine platform split — cuBLAS and the NVRTC reference on
//! CUDA, the Tensor-Core family on MUSA — so its constants are grouped here
//! with the rest and the MUSA-only entries sit in the `musa` submodule. Every
//! other difference is expressed inside the shared source (the
//! `MUSA_OUT_STATE` macro in `scan.cu`) or in the backend's primitive. Metal's
//! language (MSL) cannot share source text with the CUDA family, so `msl`
//! mirrors the same catalog over the single `grid/metal.c` translation unit.

// ---------------------------------------------------------------------------
// GRID-family catalog generator.
// ---------------------------------------------------------------------------
// The four toolchain-gated backends (ROCm, SYCL, CANN, MLU) are single
// translation units with the same entry set and tiling constants, so one macro
// generates all four catalog modules from a `(module, toolkit cfg, source,
// entry prefix)` quadruple. The prefix mangles kernel names where the backend
// exports them with a prefix (SYCL's `sycl_*` C-ABI functions); CUDA-family
// backends pass "". Each module compiles out when its toolkit cfg is absent,
// exactly like the `musa` catalog module.

macro_rules! GENENTRIES {
    ($prefix:literal) => {
        pub(crate) const ATTENTIONENTRY: &str = concat!($prefix, "attention");
        pub(crate) const ATTENTIONCACHEDENTRY: &str = concat!($prefix, "attention_cached");
        pub(crate) const MATMULENTRY: &str = concat!($prefix, "matmul");
        pub(crate) const ADDENTRY: &str = concat!($prefix, "add");
        pub(crate) const MULENTRY: &str = concat!($prefix, "mul");
        pub(crate) const SILUENTRY: &str = concat!($prefix, "silu");
        pub(crate) const RMSNORMENTRY: &str = concat!($prefix, "rmsnorm");
        pub(crate) const ROPEENTRY: &str = concat!($prefix, "rope");
        pub(crate) const ROPEATENTRY: &str = concat!($prefix, "rope_at");
        pub(crate) const EMBEDDINGENTRY: &str = concat!($prefix, "embedding");
        pub(crate) const SOFTMAXENTRY: &str = concat!($prefix, "softmax");
        pub(crate) const SCANENTRY: &str = concat!($prefix, "scan");
        pub(crate) const AGGENTRY: &str = concat!($prefix, "scan_agg");
        pub(crate) const PREFIXENTRY: &str = concat!($prefix, "scan_prefix");
        pub(crate) const APPLYENTRY: &str = concat!($prefix, "scan_apply");
    };
}

macro_rules! GENCATALOG {
    ($mod:ident, $toolkit:ident, $source:literal, $prefix:literal, $attnlimit:expr) => {
        #[cfg(all(any(target_os = "linux", target_os = "windows"), $toolkit))]
        pub(crate) mod $mod {
            /// The single translation unit; every kernel entry is a function in
            /// this one source.
            pub(crate) const SOURCE: &str = include_str!($source);

            GENENTRIES!($prefix);

            pub(crate) const ATTENTIONLANES: u32 = 256;
            /// The attention `headdim` ceiling enforced on launch: the kernel
            /// buffers the per-key accumulators on chip, so an over-long head
            /// dimension is rejected instead of overflowing the tile.
            pub(crate) const ATTNLIMIT: u32 = $attnlimit;
            pub(crate) const FLATTILE: u32 = 256;
            pub(crate) const RMSNORMTILE: u32 = 256;
            pub(crate) const SOFTMAXTILE: u32 = 256;
            pub(crate) const SCANMAXN: u32 = 256;
            pub(crate) const CHUNK: u32 = 256;
        }
    };
}

GENCATALOG!(rocm, rocm_toolkit, "grid/cuda.c", "", 4096);
GENCATALOG!(sycl, sycl_toolkit, "grid/sycl.c", "sycl_", 4096);
GENCATALOG!(cann, cann_toolkit, "grid/cann.c", "", 4096);
GENCATALOG!(mlu, mlu_toolkit, "grid/mlu.c", "", 256);

#[cfg(target_os = "linux")]
pub(crate) mod cu {
    /// The single CUDA-family translation unit with all fifteen entries
    /// (`grid/cuda.c`). Compiled by `mcc` (MUSA) and `hipcc` (ROCm) at build
    /// time; compiled at run time by NVRTC on hosts without `nvcc`. NVRTC
    /// compiles from a string with no filesystem, so grid/grid.h is spliced
    /// in front here with the dialect macro (the `#if !defined(GRID_CUDA)`
    /// guard in cuda.c then skips the file-system `#include`).
    pub(crate) const SOURCE: &str = concat!(
        "#define GRID_CUDA\n",
        include_str!("grid/grid.h"),
        "\n",
        include_str!("grid/cuda.c"),
    );

    GENENTRIES!("");

    pub(crate) const ATTENTIONLANES: u32 = 256;
    /// Attention headdim ceiling. CUDA sizes the shared buffer dynamically
    /// and the driver rejects an over-large launch; 4096 is the defensive
    /// contract ceiling (LLM headdims are 64-128).
    pub(crate) const ATTNLIMIT: u32 = 4096;
    pub(crate) const FLATTILE: u32 = 256;
    pub(crate) const RMSNORMTILE: u32 = 256;
    pub(crate) const SOFTMAXTILE: u32 = 256;
    pub(crate) const SCANMAXN: u32 = 256;
    pub(crate) const SCANBLOCKDIM: u32 = 256;
    pub(crate) const CHUNK: u32 = 256;
    pub(crate) const THREADEDGE: u32 = 16;
}

#[cfg(target_os = "macos")]
pub(crate) mod msl {
    /// The single MSL translation unit; every Metal kernel entry is a
    /// `kernel void` function in this one source. Metal compiles at run time
    /// from a string, so grid/grid.h is spliced in front here with the
    /// dialect macro (same mechanism as `cu::SOURCE` above).
    pub(crate) const SOURCE: &str = concat!(
        "#define GRID_METAL\n",
        include_str!("grid/grid.h"),
        "\n",
        include_str!("grid/metal.c"),
    );

    GENENTRIES!("");

    pub(crate) const ATTENTIONLANES: usize = 256;
    /// Attention headdim ceiling matching `accum[4096]` in grid/metal.c; an
    /// over-long head dimension is rejected instead of overflowing the
    /// threadgroup buffer.
    pub(crate) const ATTNLIMIT: usize = 4096;
    pub(crate) const FLATTILE: usize = 256;
    pub(crate) const RMSNORMTILE: usize = 256;
    pub(crate) const SOFTMAXTILE: usize = 256;
    pub(crate) const SCANMAXN: usize = 256;
    pub(crate) const CHUNK: usize = 256;
}

/// MUSA's Tensor-Core `matmul` family, compiled from a separate source file.
#[cfg(all(target_os = "linux", musa_toolkit))]
pub(crate) mod musa {
    /// The fatbin registry key for `grid/musa.c`.
    pub(crate) const WMMASOURCEID: &str = "musa";
    pub(crate) const WMMAENTRY: &str = "matmul_wmma";
    pub(crate) const RMSINVENTRY: &str = "rmsinv";
    pub(crate) const MATMULRMSENTRY: &str = "matmul_rms";
    pub(crate) const WMMABM: u32 = 64;
    pub(crate) const WMMABN: u32 = 32;
    pub(crate) const WMMATHREADS: u32 = 128;
}
