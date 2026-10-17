//! Safe SIMD primitive library for the CPU backend.
//!
//! This crate is the workspace-internal replacement for the third-party
//! `archmage` dependency: it exposes narrow, slice-level kernels behind a fully
//! safe API. Every kernel takes [`Isa`] explicitly; the runtime feature check
//! is paid once per call in [`detect`] and the `#[target_feature]` boundary is
//! crossed inside `intrinsics.rs`.
//!
//! # Unsafe policy
//!
//! The only `unsafe` code in the CPU SIMD path lives in `src/intrinsics.rs`,
//! a registered exemption island under docs/engineering.md (rule 14) and
//! docs/design.md §21.8. This file (`lib.rs`) declares
//! `#![deny(unsafe_code)]` and contains no `unsafe` construct; the dispatch
//! arms that call `#[target_feature]` kernels are confined to the island and
//! covered by `SAFETY:` comments.
//!
//! # Numerical invariants
//!
//! The kernel bodies are bit-for-bit identical to the former `archmage`
//! `#[arcane(import_intrinsics)]` implementations: fused accumulation order,
//! lane-tree reductions and the degree-7 Taylor exp2 approximation are
//! preserved exactly, so CPU results are unchanged by this migration.
#![deny(unsafe_code)]

mod detect;
mod intrinsics;
mod scalar;

pub use detect::{Isa, detect};
pub use intrinsics::{
    attentiondot, attentionqblock, matmuldirect, matmultile, rmsnormrow, rowinv, softmaxrowspan,
};

/// The row-major layout of one grouped-query attention call, shared with the
/// CPU backend's blocked kernel. Pure data; constructed by the backend.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    /// Number of query rows.
    pub qseq: usize,
    /// Number of key/value rows.
    pub kvseq: usize,
    /// Row stride of the query and output tensors.
    pub qstride: usize,
    /// Row stride of the key tensor.
    pub kstride: usize,
    /// Row stride of the value tensor.
    pub vstride: usize,
    /// Number of query heads.
    pub heads: usize,
    /// Number of key/value heads (grouped-query).
    pub kvheads: usize,
    /// Head dimension.
    pub headdim: usize,
    /// Query heads per key/value head (heads / kvheads).
    pub group: usize,
    /// `1 / sqrt(headdim)`.
    pub scale: f32,
    /// Key offset for causal attention (`kvseq - qseq`).
    pub offset: usize,
    /// Whether attention is causal.
    pub causal: bool,
}
