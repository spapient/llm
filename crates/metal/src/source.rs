//! The Metal Shading Language sources compiled at run time.
//!
//! Each source lives in `kernels/*.metal` and is embedded here with
//! [`include_str!`], then compiled to an `MTLLibrary` by `newLibraryWithSource`
//! on the first call that needs it, so the build never requires `xcrun` or a
//! Metal toolchain and no `.metallib` is committed. Every source is
//! self-contained — it includes `<metal_stdlib>` and pulls the device math
//! functions from it — and declares one or more `kernel` functions. The
//! matching `*ENTRY` constant names the function the host resolves with
//! `MTLLibrary::newFunctionWithName`.
//!
//! Scalar parameters are passed as a single `constant` buffer whose element
//! type matches the Rust array the host uploads with `setBytes`; floats cross
//! the boundary as their `u32` bit pattern and are recovered device-side with
//! `as_type<float>`. Keeping every kernel's parameter buffer to one array keeps
//! the host's parameter binding a single call.

/// Metal Shading Language source of the `f32` matrix-multiply kernel
/// (`kernels/matmul.metal`).
pub(crate) const MATMULMSL: &str = include_str!("../kernels/matmul.metal");

/// The stable MSL function name of the kernel in [`MATMULMSL`].
pub(crate) const MATMULENTRY: &str = "matmul";

/// Metal Shading Language source of the elementwise `f32` kernels
/// (`kernels/elementwise.metal`).
pub(crate) const ELEMENTWISE: &str = include_str!("../kernels/elementwise.metal");

/// The symbol of the elementwise addition kernel in [`ELEMENTWISE`].
pub(crate) const ADDENTRY: &str = "add";

/// The symbol of the elementwise multiplication kernel in [`ELEMENTWISE`].
pub(crate) const MULENTRY: &str = "mul";

/// The symbol of the sigmoid-linear-unit kernel in [`ELEMENTWISE`].
pub(crate) const SILUENTRY: &str = "silu";

/// The flat element count per threadgroup for the elementwise kernels.
pub(crate) const FLATTILE: usize = 256;

/// Metal Shading Language source of the `f32` RMS-normalization kernel
/// (`kernels/rmsnorm.metal`).
pub(crate) const RMSNORMMSL: &str = include_str!("../kernels/rmsnorm.metal");

/// The stable symbol name of the kernel in [`RMSNORMMSL`].
pub(crate) const RMSNORMENTRY: &str = "rmsnorm";

/// The threadgroup width of the RMS-normalization reduction.
pub(crate) const RMSNORMTILE: usize = 256;

/// Metal Shading Language source of the `f32` last-axis softmax kernel
/// (`kernels/softmax.metal`).
pub(crate) const SOFTMAXMSL: &str = include_str!("../kernels/softmax.metal");

/// The stable symbol name of the kernel in [`SOFTMAXMSL`].
pub(crate) const SOFTMAXENTRY: &str = "softmax";

/// The threadgroup width of the softmax reduction.
pub(crate) const SOFTMAXTILE: usize = 256;

/// Metal Shading Language source of the `f32` rotary-position-embedding kernel
/// (`kernels/rope.metal`).
pub(crate) const ROPEMSL: &str = include_str!("../kernels/rope.metal");

/// The stable symbol name of the kernel in [`ROPEMSL`].
pub(crate) const ROPEENTRY: &str = "rope";

/// Metal Shading Language source of the absolute-position rotary embedding
/// (`kernels/rope_at.metal`).
pub(crate) const ROPEATMSL: &str = include_str!("../kernels/rope_at.metal");

/// The stable symbol name of the kernel in [`ROPEATMSL`].
pub(crate) const ROPEATENTRY: &str = "rope_at";

/// Metal Shading Language source of the `f32` embedding-gather kernel
/// (`kernels/embedding.metal`).
pub(crate) const EMBEDDINGMSL: &str = include_str!("../kernels/embedding.metal");

/// The stable symbol name of the kernel in [`EMBEDDINGMSL`].
pub(crate) const EMBEDDINGENTRY: &str = "embedding";

/// Metal Shading Language source of the fused scaled-dot-product attention
/// kernel (`kernels/attention.metal`).
pub(crate) const ATTENTIONMSL: &str = include_str!("../kernels/attention.metal");

/// The stable symbol name of the kernel in [`ATTENTIONMSL`].
pub(crate) const ATTENTIONENTRY: &str = "attention";

/// Metal Shading Language source of the incremental-decode attention kernel
/// (`kernels/attention_cached.metal`).
pub(crate) const ATTENTIONCACHEDMSL: &str = include_str!("../kernels/attention_cached.metal");

/// The stable symbol name of the kernel in [`ATTENTIONCACHEDMSL`].
pub(crate) const ATTENTIONCACHEDENTRY: &str = "attention_cached";

/// Metal Shading Language source of the `f32` semigroup scan kernel
/// (`kernels/scan.metal`). Serial per-channel reference and the fallback for
/// state widths above [`SCANMAXN`].
pub(crate) const SCANMSL: &str = include_str!("../kernels/scan.metal");

/// The stable symbol name of the kernel in [`SCANMSL`].
pub(crate) const SCANENTRY: &str = "scan";

/// Metal Shading Language source of the parallel three-stage scan kernel
/// family (`kernels/scan_parallel.metal`): block aggregation, inter-block
/// prefix, and per-block apply.
pub(crate) const SCANPARALLELMSL: &str = include_str!("../kernels/scan_parallel.metal");

/// The stable symbol name of the block-aggregation kernel in
/// [`SCANPARALLELMSL`].
pub(crate) const SCANAGGENTRY: &str = "scan_agg";

/// The stable symbol name of the inter-block prefix kernel in
/// [`SCANPARALLELMSL`].
pub(crate) const SCANPREFIXENTRY: &str = "scan_prefix";

/// The stable symbol name of the per-block apply kernel in
/// [`SCANPARALLELMSL`].
pub(crate) const SCANAPPLYENTRY: &str = "scan_apply";

/// The maximum state width the parallel scan kernels support.
///
/// The apply kernel keeps one channel's `N` state components in a
/// compile-time-sized thread-local array; a wider state falls back to the
/// serial [`SCANMSL`] kernel.
pub(crate) const SCANMAXN: usize = 256;

/// The sequence steps per block in the parallel scan.
///
/// The number of blocks is `n.div_ceil(SCANCHUNK)`; the inter-block prefix
/// (Pass 2) is serial over the blocks, so a large block count would make that
/// pass the bottleneck — 256 steps keeps P ≤ 64 for the 16K M1 probe.
pub(crate) const SCANCHUNK: usize = 256;

/// The threadgroup width of the attention kernel.
///
/// The kernel's reduction tree and shared-memory partition are both sized by
/// this value, so it is a hard requirement rather than a tuning knob.
pub(crate) const ATTENTIONLANES: usize = 256;
