//! Kernel catalog: every kernel whose source is shared, plus the dispatch
//! geometry.
//!
//! `matmul` has a genuine platform split — cuBLAS and the NVRTC reference on
//! CUDA, the Tensor-Core family on MUSA — so its constants are grouped here
//! with the rest and the MUSA-only entries sit in the `musa` submodule. Every
//! other difference is expressed inside the shared source (the
//! `MUSA_OUT_STATE` macro in `scan.cu`) or in the backend's primitive. Metal's
//! language (MSL) cannot share source text with the CUDA family, so `msl`
//! mirrors the same catalog over the single `metal/metal.c` translation unit.

#[cfg(target_os = "linux")]
pub(crate) mod cu {
    /// Fused scaled-dot-product attention.
    pub(crate) const ATTENTION: &str = include_str!("cuda/attention.cu");
    pub(crate) const ATTENTIONENTRY: &str = "attention";
    pub(crate) const ATTENTIONLANES: u32 = 256;

    /// Incremental-decode attention over a cached key/value buffer.
    pub(crate) const ATTENTIONCACHED: &str = include_str!("cuda/cached.cu");
    pub(crate) const ATTENTIONCACHEDENTRY: &str = "attention_cached";

    /// Elementwise `add`, `mul`, and `silu`.
    pub(crate) const ELEMENTWISE: &str = include_str!("cuda/elementwise.cu");
    pub(crate) const ADDENTRY: &str = "add";
    pub(crate) const MULENTRY: &str = "mul";
    pub(crate) const SILUENTRY: &str = "silu";
    pub(crate) const ELEMENTWISETILE: u32 = 256;

    /// Row-wise RMS normalization.
    pub(crate) const RMSNORM: &str = include_str!("cuda/rmsnorm.cu");
    pub(crate) const RMSNORMENTRY: &str = "rmsnorm";
    pub(crate) const ROWTILE: u32 = 256;

    /// Rotary position embedding, relative and absolute.
    pub(crate) const ROPE: &str = include_str!("cuda/rope.cu");
    pub(crate) const ROPEENTRY: &str = "rope";
    pub(crate) const ROPEAT: &str = include_str!("cuda/at.cu");
    pub(crate) const ROPEATENTRY: &str = "rope_at";

    /// Embedding gather.
    pub(crate) const EMBEDDING: &str = include_str!("cuda/embedding.cu");
    pub(crate) const EMBEDDINGENTRY: &str = "embedding";

    /// Last-axis softmax.
    pub(crate) const SOFTMAX: &str = include_str!("cuda/softmax.cu");
    pub(crate) const SOFTMAXENTRY: &str = "softmax";

    /// Serial per-channel scan and its compile-time state-width bound.
    pub(crate) const SCAN: &str = include_str!("cuda/scan.cu");
    pub(crate) const SCANENTRY: &str = "scan";
    pub(crate) const SCANMAXN: u32 = 256;

    /// Batched reference matmul shared by CUDA and MUSA. `blockIdx.z` selects
    /// the batch slice; a single-slice caller passes `batch = 1`.
    pub(crate) const MATMUL: &str = include_str!("cuda/matmul.cu");
    pub(crate) const MATMULENTRY: &str = "matmul";
    pub(crate) const THREADEDGE: u32 = 16;

    /// Three-stage parallel scan family.
    pub(crate) const PARALLEL: &str = include_str!("cuda/parallel.cu");
    pub(crate) const AGGENTRY: &str = "scan_agg";
    pub(crate) const PREFIXENTRY: &str = "scan_prefix";
    pub(crate) const APPLYENTRY: &str = "scan_apply";
    pub(crate) const BLOCKDIM: u32 = 256;
    pub(crate) const CHUNK: u32 = 256;
}

#[cfg(target_os = "macos")]
pub(crate) mod msl {
    /// The single MSL translation unit; every Metal kernel entry is a
    /// `kernel void` function in this one source.
    pub(crate) const SOURCE: &str = include_str!("metal/metal.c");

    pub(crate) const ATTENTIONENTRY: &str = "attention";
    pub(crate) const ATTENTIONLANES: usize = 256;

    pub(crate) const ATTENTIONCACHEDENTRY: &str = "attention_cached";

    pub(crate) const MATMULENTRY: &str = "matmul";

    pub(crate) const ADDENTRY: &str = "add";
    pub(crate) const MULENTRY: &str = "mul";
    pub(crate) const SILUENTRY: &str = "silu";
    pub(crate) const FLATTILE: usize = 256;

    pub(crate) const RMSNORMENTRY: &str = "rmsnorm";
    pub(crate) const RMSNORMTILE: usize = 256;

    pub(crate) const ROPEENTRY: &str = "rope";
    pub(crate) const ROPEATENTRY: &str = "rope_at";

    pub(crate) const EMBEDDINGENTRY: &str = "embedding";

    pub(crate) const SOFTMAXENTRY: &str = "softmax";
    pub(crate) const SOFTMAXTILE: usize = 256;

    pub(crate) const SCANENTRY: &str = "scan";
    pub(crate) const SCANMAXN: usize = 256;

    pub(crate) const AGGENTRY: &str = "scan_agg";
    pub(crate) const PREFIXENTRY: &str = "scan_prefix";
    pub(crate) const APPLYENTRY: &str = "scan_apply";
    pub(crate) const CHUNK: usize = 256;
}

/// MUSA's Tensor-Core `matmul` family, compiled from a separate source file.
#[cfg(all(target_os = "linux", musa_toolkit))]
pub(crate) mod musa {
    /// The fatbin registry key for `musa/musa.cu`.
    pub(crate) const WMMASOURCEID: &str = "musa";
    pub(crate) const WMMAENTRY: &str = "matmul_wmma";
    pub(crate) const RMSINVENTRY: &str = "rmsinv";
    pub(crate) const MATMULRMSENTRY: &str = "matmul_rms";
    pub(crate) const WMMABM: u32 = 64;
    pub(crate) const WMMABN: u32 = 32;
    pub(crate) const WMMATHREADS: u32 = 128;
}
