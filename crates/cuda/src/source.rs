//! The CUDA C++ sources compiled at run time through NVRTC.
//!
//! Each source lives in `kernels/*.cu` and is embedded here with
//! [`include_str!`], then compiled to PTX by NVRTC on the first call that needs
//! it, so the build never requires `nvcc` and no PTX is committed. Each source
//! declares one or more `extern "C"` kernels, and the matching `*ENTRY` constant
//! names the symbol the host resolves with `CudaModule::load_function`.
//!
//! Every kernel is written to need no headers: NVRTC provides the device math
//! functions (`sqrtf`, `expf`, `powf`, `sinf`, `cosf`) implicitly, which is the
//! same property `cudarc`'s own `sin.cu` example relies on.

/// CUDA C source of the reference `f32` matrix-multiply kernel
/// (`kernels/matmul.cu`).
pub(crate) const MATMULCUDA: &str = include_str!("../kernels/matmul.cu");

/// The stable symbol name of the kernel in [`MATMULCUDA`].
pub(crate) const MATMULENTRY: &str = "matmul";

/// The threadgroup edge used when dispatching the matmul kernel.
pub(crate) const THREADEDGE: u32 = 16;

/// CUDA C source of the elementwise `f32` kernels (`kernels/elementwise.cu`).
pub(crate) const ELEMENTWISE: &str = include_str!("../kernels/elementwise.cu");

/// The symbol of the elementwise addition kernel in [`ELEMENTWISE`].
pub(crate) const ADDENTRY: &str = "add";

/// The symbol of the elementwise multiplication kernel in [`ELEMENTWISE`].
pub(crate) const MULENTRY: &str = "mul";

/// The symbol of the sigmoid-linear-unit kernel in [`ELEMENTWISE`].
pub(crate) const SILUENTRY: &str = "silu";

/// The flat element count per threadgroup for the elementwise kernels.
pub(crate) const ELEMENTWISETILE: u32 = 256;

/// CUDA C source of the `f32` RMS-normalization kernel (`kernels/rmsnorm.cu`).
pub(crate) const RMSNORMCUDA: &str = include_str!("../kernels/rmsnorm.cu");

/// The stable symbol name of the kernel in [`RMSNORMCUDA`].
pub(crate) const RMSNORMENTRY: &str = "rmsnorm";

/// The threadgroup width of the RMS-normalization reduction.
pub(crate) const ROWTILE: u32 = 256;

/// CUDA C source of the `f32` rotary-position-embedding kernel
/// (`kernels/rope.cu`).
pub(crate) const ROPECUDA: &str = include_str!("../kernels/rope.cu");

/// The stable symbol name of the kernel in [`ROPECUDA`].
pub(crate) const ROPEENTRY: &str = "rope";

/// CUDA C source of the absolute-position rotary embedding
/// (`kernels/rope_at.cu`).
pub(crate) const ROPEATCUDA: &str = include_str!("../kernels/rope_at.cu");

/// The stable symbol name of the kernel in [`ROPEATCUDA`].
pub(crate) const ROPEATENTRY: &str = "rope_at";

/// CUDA C source of the `f32` embedding-gather kernel (`kernels/embedding.cu`).
pub(crate) const EMBEDDINGCUDA: &str = include_str!("../kernels/embedding.cu");

/// The stable symbol name of the kernel in [`EMBEDDINGCUDA`].
pub(crate) const EMBEDDINGENTRY: &str = "embedding";

/// CUDA C source of the fused scaled-dot-product attention kernel
/// (`kernels/attention.cu`).
pub(crate) const ATTENTIONCUDA: &str = include_str!("../kernels/attention.cu");

/// The stable symbol name of the kernel in [`ATTENTIONCUDA`].
pub(crate) const ATTENTIONENTRY: &str = "attention";

/// CUDA C source of the incremental-decode attention kernel
/// (`kernels/attention_cached.cu`).
pub(crate) const ATTENTIONCACHEDCUDA: &str = include_str!("../kernels/attention_cached.cu");

/// The stable symbol name of the kernel in [`ATTENTIONCACHEDCUDA`].
pub(crate) const ATTENTIONCACHEDENTRY: &str = "attention_cached";

/// CUDA C source of the `f32` semigroup scan kernel (`kernels/scan.cu`).
pub(crate) const SCANCUDA: &str = include_str!("../kernels/scan.cu");

/// The stable symbol name of the kernel in [`SCANCUDA`].
pub(crate) const SCANENTRY: &str = "scan";

/// CUDA C source of the parallel three-stage scan kernel family
/// (`kernels/scan_parallel.cu`): block aggregation, inter-block prefix, and
/// per-block apply.
pub(crate) const SCANPARALLELCUDA: &str = include_str!("../kernels/scan_parallel.cu");

/// The stable symbol name of the block-aggregation kernel in
/// [`SCANPARALLELCUDA`].
pub(crate) const SCANAGGENTRY: &str = "scan_agg";

/// The stable symbol name of the inter-block prefix kernel in
/// [`SCANPARALLELCUDA`].
pub(crate) const SCANPREFIXENTRY: &str = "scan_prefix";

/// The stable symbol name of the per-block apply kernel in [`SCANPARALLELCUDA`].
pub(crate) const SCANAPPLYENTRY: &str = "scan_apply";

/// The sequence-chunk length of the parallel scan, and the compile-time bound
/// on the per-channel state width. The apply kernel keeps its state in a
/// thread-local array, so widths above this bound fall back to the serial
/// kernel in [`SCANCUDA`].
pub(crate) const SCANMAXN: u32 = 256;

/// The per-threadgroup thread count of the parallel scan's block passes.
pub(crate) const SCANBLOCKDIM: u32 = 256;

/// The sequence-chunk length of the parallel scan (one block per chunk).
pub(crate) const SCANCHUNK: u32 = 256;

/// The threadgroup width of the attention kernel.
///
/// The kernel's reduction tree and shared-memory partition are both sized by
/// this value, so it is a hard requirement rather than a tuning knob.
pub(crate) const ATTENTIONLANES: u32 = 256;
