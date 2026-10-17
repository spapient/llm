//! Kernel source metadata for the MUSA backend.
//!
//! Kernels live in `kernels/*.cu` and are compiled **at build time** by `mcc`
//! into ELF fatbins embedded with `include_bytes!` (see [`crate::registry`])
//! and loaded at run time through `muModuleLoadData` — no runtime JIT, no
//! NVRTC, no MTRTC (docs/musa-backend-plan.md §5.3). The kernel sources
//! themselves are therefore never embedded in the binary; this module only
//! declares the stable entry-point symbols and the dispatch geometry the host
//! resolves with `muModuleGetFunction`.
//!
//! Apart from the Tensor-Core kernel, which includes `<mma.h>` for the
//! `mtmusa::wmma` intrinsics, kernels are written to need no headers:
//! `mcc -mtgpu` provides the device math functions (`sqrtf`, `expf`, `powf`)
//! and the thread-index environment (`threadIdx`/`blockIdx`) implicitly, which
//! the MTT S4000 smoke run verified for the parallel-scan family
//! (scripts/musa/README.md).

/// The stable symbol name of the reference matmul kernel.
pub(crate) const MATMULENTRY: &str = "matmul";

/// The stable symbol name of the Tensor-Core (TF32) matmul kernel.
pub(crate) const MATMULWMMAENTRY: &str = "matmul_wmma";

/// The threadgroup edge used when dispatching the reference matmul kernel.
pub(crate) const THREADEDGE: u32 = 16;

/// The output tile row count of the Tensor-Core matmul kernel (one block per
/// tile).
pub(crate) const MATMULWMMABM: u32 = 64;

/// The output tile column count of the Tensor-Core matmul kernel.
pub(crate) const MATMULWMMABN: u32 = 32;

/// The thread count of the Tensor-Core matmul kernel's threadgroup.
///
/// The `m32n32k8` TF32 MMA is a wave128 instruction: its four 32-lane warps
/// cooperate on one 32x32 tile, so the block is a full 128-thread wave.
pub(crate) const MATMULWMMATHREADS: u32 = 128;

/// The symbol of the elementwise addition kernel.
pub(crate) const ADDENTRY: &str = "add";

/// The symbol of the elementwise multiplication kernel.
pub(crate) const MULENTRY: &str = "mul";

/// The symbol of the sigmoid-linear-unit kernel.
pub(crate) const SILUENTRY: &str = "silu";

/// The symbol of the max-subtracting softmax kernel.
pub(crate) const SOFTMAXENTRY: &str = "softmax";

/// The flat element count per threadgroup for the elementwise kernels.
pub(crate) const ELEMENTWISETILE: u32 = 256;

/// The stable symbol name of the RMS-normalization kernel.
pub(crate) const RMSNORMENTRY: &str = "rmsnorm";

/// The threadgroup width of the RMS-normalization reduction.
pub(crate) const ROWTILE: u32 = 256;

/// The stable symbol name of the rotary-position-embedding kernel.
pub(crate) const ROPEENTRY: &str = "rope";

/// The stable symbol name of the absolute-position rotary kernel.
pub(crate) const ROPEATENTRY: &str = "rope_at";

/// The stable symbol name of the embedding-gather kernel.
pub(crate) const EMBEDDINGENTRY: &str = "embedding";

/// The stable symbol name of the attention kernel.
pub(crate) const ATTENTIONENTRY: &str = "attention";

/// The threadgroup width of the attention kernel.
///
/// The kernel's reduction tree and shared-memory partition are both sized by
/// this value, so it is a hard requirement rather than a tuning knob.
pub(crate) const ATTENTIONLANES: u32 = 256;

/// The stable symbol name of the incremental-decode attention kernel.
pub(crate) const ATTENTIONCACHEDENTRY: &str = "attention_cached";

/// The stable symbol name of the serial scan kernel.
pub(crate) const SCANENTRY: &str = "scan";

/// The stable symbol name of the block-aggregation kernel.
pub(crate) const SCANAGGENTRY: &str = "scan_agg";

/// The stable symbol name of the inter-block prefix kernel.
pub(crate) const SCANPREFIXENTRY: &str = "scan_prefix";

/// The stable symbol name of the per-block apply kernel.
pub(crate) const SCANAPPLYENTRY: &str = "scan_apply";

/// The sequence-chunk length of the parallel scan, and the compile-time bound
/// on the per-channel state width. The apply kernel keeps its state in a
/// thread-local array, so widths above this bound fall back to the serial
/// kernel.
pub(crate) const SCANMAXN: u32 = 256;

/// The per-threadgroup thread count of the parallel scan's block passes.
pub(crate) const SCANBLOCKDIM: u32 = 256;

/// The sequence-chunk length of the parallel scan (one block per chunk).
pub(crate) const SCANCHUNK: u32 = 256;
