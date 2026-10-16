#![forbid(unsafe_code)]
//! Reusable benchmark workloads for the LLM runtime.
//!
//! `bench` owns benchmark workload setup, execution harnesses, and
//! reproducibility metadata. It MUST NOT own production operator
//! implementations or alternative implementations that diverge from production
//! behavior, and it MUST NOT become a production dependency (see
//! `docs/architecture.md`).
//!
//! Workloads are exposed as deterministic constructors so that every benchmark
//! target builds identical inputs. No allocation happens inside a timed region
//! beyond the output that the measured contract returns.
//!
//! # Layout
//!
//! Benchmark executables live under `benches/`. This library holds the shared
//! workload generators they compose; it depends only on `lib`, `tensor`, and
//! `cpu`, so the crate remains a dependency leaf.

use ::tensor::{shape, tensor, tensorerror};
use cpu::cpuexecutor;
use lib::limits;

/// Element count for the elementwise activation workloads.
pub const ELEMENTWISE: usize = 1 << 20;

/// Row count for the softmax workload.
pub const SOFTMAXROWS: usize = 512;

/// Row width for the softmax workload.
pub const SOFTMAXCOLS: usize = 4096;

/// Resource budget that admits every benchmark workload without rejecting it.
///
/// The budget is deliberately large: benchmarks measure kernel cost, not limit
/// enforcement, which is covered by the `cpu` contract tests. It is an internal
/// detail of [`cpuexec`], so it stays crate-private (A4 interface minimality).
pub(crate) const OPLIMITS: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

/// Builds a CPU executor with the benchmark resource budget.
#[must_use]
pub fn cpuexec() -> cpuexecutor {
    cpuexecutor::new(OPLIMITS)
}

/// Builds a CPU executor with the benchmark resource budget and a worker
/// budget of `workers` for the internally parallel kernels.
///
/// # Errors
///
/// Returns the executor's rejection when `workers` is zero.
pub fn cpuexecw(workers: usize) -> Result<cpuexecutor, cpu::cpuerror> {
    cpuexecutor::newwithworkers(OPLIMITS, workers)
}

/// Builds the fixed elementwise input of [`ELEMENTWISE`] `f32` values.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn elementwiseinput() -> Result<tensor<f32>, tensorerror> {
    tensor::new(shape::new(vec![ELEMENTWISE])?, vec![0.5_f32; ELEMENTWISE])
}

/// Builds the fixed `SOFTMAXROWS` by `SOFTMAXCOLS` softmax input.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn softmaxinput() -> Result<tensor<f32>, tensorerror> {
    let elements = SOFTMAXROWS * SOFTMAXCOLS;
    tensor::new(
        shape::new(vec![SOFTMAXROWS, SOFTMAXCOLS])?,
        vec![0.25_f32; elements],
    )
}

/// Square dimension of the matrix-multiplication workload.
pub const MATMULDIM: usize = 256;

/// Row count of the RMS normalization workload.
pub const NORMROWS: usize = 512;

/// Row width of the RMS normalization workload.
pub const NORMCOLS: usize = 4096;

/// Sequence length of the RoPE and attention workloads.
pub const SEQLEN: usize = 128;

/// Query head count of the RoPE and attention workloads.
pub const HEADS: usize = 8;

/// Per-head width of the RoPE and attention workloads.
pub const HEADDIM: usize = 64;

/// Vocabulary size of the embedding workload.
///
/// It is only read by [`embeddinginput`], so it stays crate-private (A4).
pub(crate) const VOCAB: usize = 4096;

/// Row width of the embedding workload.
pub const EMBEDDIM: usize = 1024;

/// Number of rows gathered per embedding call.
pub const GATHER: usize = 128;

/// Builds the two fixed `[MATMULDIM, MATMULDIM]` matmul operands.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn matmulinput() -> Result<(tensor<f32>, tensor<f32>), tensorerror> {
    let elements = MATMULDIM * MATMULDIM;
    let lhs = tensor::new(
        shape::new(vec![MATMULDIM, MATMULDIM])?,
        vec![0.5_f32; elements],
    )?;
    let rhs = tensor::new(
        shape::new(vec![MATMULDIM, MATMULDIM])?,
        vec![0.25_f32; elements],
    )?;
    Ok((lhs, rhs))
}

/// Block count of the batched matrix-multiplication workload.
///
/// It is what a `matmul` pays per-block dispatch cost on, so the benchmark
/// suite measures it separately from the single square case. It stays
/// crate-private (A4 interface minimality): callers read the workload's shape
/// from the tensor they are handed, exactly as they do for the other fixed
/// workloads whose extents are not exported either.
const MATMULBATCH: usize = 8;

/// Builds the two fixed `[MATMULBATCH, MATMULDIM, MATMULDIM]` batched matmul
/// operands.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn matmulbatchinput() -> Result<(tensor<f32>, tensor<f32>), tensorerror> {
    let elements = MATMULBATCH * MATMULDIM * MATMULDIM;
    let lhs = tensor::new(
        shape::new(vec![MATMULBATCH, MATMULDIM, MATMULDIM])?,
        vec![0.5_f32; elements],
    )?;
    let rhs = tensor::new(
        shape::new(vec![MATMULBATCH, MATMULDIM, MATMULDIM])?,
        vec![0.25_f32; elements],
    )?;
    Ok((lhs, rhs))
}

/// Builds the fixed `[NORMROWS, NORMCOLS]` RMS normalization input.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn norminput() -> Result<(tensor<f32>, tensor<f32>), tensorerror> {
    let elements = NORMROWS * NORMCOLS;
    let x = tensor::new(
        shape::new(vec![NORMROWS, NORMCOLS])?,
        vec![0.5_f32; elements],
    )?;
    let w = tensor::new(shape::new(vec![NORMCOLS])?, vec![1.0_f32; NORMCOLS])?;
    Ok((x, w))
}

/// Builds the fixed `[SEQLEN, HEADS * HEADDIM]` RoPE input.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn ropeinput() -> Result<tensor<f32>, tensorerror> {
    let width = HEADS * HEADDIM;
    tensor::new(
        shape::new(vec![SEQLEN, width])?,
        vec![0.5_f32; SEQLEN * width],
    )
}

/// Builds the fixed `[VOCAB, EMBEDDIM]` table and gather indices.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn embeddinginput() -> Result<(tensor<f32>, Vec<usize>), tensorerror> {
    let table = tensor::new(
        shape::new(vec![VOCAB, EMBEDDIM])?,
        vec![0.5_f32; VOCAB * EMBEDDIM],
    )?;
    let indices = (0..GATHER).map(|i| (i * 7) % VOCAB).collect();
    Ok((table, indices))
}

/// The three attention operands returned by [`attentioninput`].
/// The three operands of one attention call. Crate-private because callers
/// destructure the tuple rather than naming the type (A4 interface minimality).
type attentionoperands = (tensor<f32>, tensor<f32>, tensor<f32>);

/// Builds the fixed query/key/value attention operands.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn attentioninput() -> Result<attentionoperands, tensorerror> {
    let width = HEADS * HEADDIM;
    let elements = SEQLEN * width;
    let q = tensor::new(shape::new(vec![SEQLEN, width])?, vec![0.1_f32; elements])?;
    let k = tensor::new(shape::new(vec![SEQLEN, width])?, vec![0.2_f32; elements])?;
    let v = tensor::new(shape::new(vec![SEQLEN, width])?, vec![0.3_f32; elements])?;
    Ok((q, k, v))
}

/// Channel count of the scan workload.
pub const SCANCHANNELS: usize = 256;

/// State width of the scan workload.
pub const SCANSTATE: usize = 64;

/// The six operands of one scan call, returned by [`scaninput`].
pub type scanoperands = (
    tensor<f32>,
    tensor<f32>,
    tensor<f32>,
    tensor<f32>,
    tensor<f32>,
    tensor<f32>,
);

/// Builds the fixed-channel scan operands for `seq` timesteps.
///
/// The recurrence kernel `a` stays within `(-1, 1)` so the spectral precondition
/// holds; `init` and `skip` are zero.
///
/// # Errors
///
/// Returns a [`tensorerror`] if a fixed shape is rejected by `tensor`.
pub fn scaninput(seq: usize) -> Result<scanoperands, tensorerror> {
    let d = SCANCHANNELS;
    let n = SCANSTATE;
    let input = tensor::new(shape::new(vec![seq, d])?, vec![0.5_f32; seq * d])?;
    let init = tensor::new(shape::new(vec![d, n])?, vec![0.0_f32; d * n])?;
    let a = tensor::new(shape::new(vec![d, n])?, vec![0.5_f32; d * n])?;
    let b = tensor::new(shape::new(vec![d, n])?, vec![0.25_f32; d * n])?;
    let c = tensor::new(shape::new(vec![d, n])?, vec![0.125_f32; d * n])?;
    let skip = tensor::new(shape::new(vec![d])?, vec![0.0_f32; d])?;
    Ok((input, init, a, b, c, skip))
}

/// Square dimension of the prefill-scale matrix-multiplication workload.
///
/// At this size the right operand (16 MiB) exceeds the on-chip caches of the
/// documented baseline machine, so the workload measures the operand reuse
/// pattern of a prefill-sized multiplication rather than an L2-resident one.
pub const MATMULPREFILL: usize = 2048;

/// Builds the two fixed `[MATMULPREFILL, MATMULPREFILL]` matmul operands.
///
/// # Errors
///
/// Returns a [`tensorerror`] if the fixed shape is rejected by `tensor`.
pub fn matmulprefillinput() -> Result<(tensor<f32>, tensor<f32>), tensorerror> {
    let elements = MATMULPREFILL * MATMULPREFILL;
    let lhs = tensor::new(
        shape::new(vec![MATMULPREFILL, MATMULPREFILL])?,
        vec![0.5_f32; elements],
    )?;
    let rhs = tensor::new(
        shape::new(vec![MATMULPREFILL, MATMULPREFILL])?,
        vec![0.25_f32; elements],
    )?;
    Ok((lhs, rhs))
}

/// Row count of the decode-scale matrix-multiplication workload: one token.
pub const MATMULDECODEM: usize = 1;

/// Inner dimension of the decode-scale matrix-multiplication workload.
pub const MATMULDECODEK: usize = 4096;

/// Column count of the decode-scale matrix-multiplication workload.
pub const MATMULDECODEN: usize = 4096;

/// Builds the decode-scale `[1, K] * [K, N]` matmul operands.
///
/// # Errors
///
/// Returns a [`tensorerror`] if a fixed shape is rejected by `tensor`.
pub fn matmuldecodeinput() -> Result<(tensor<f32>, tensor<f32>), tensorerror> {
    let lhs = tensor::new(
        shape::new(vec![MATMULDECODEM, MATMULDECODEK])?,
        vec![0.5_f32; MATMULDECODEM * MATMULDECODEK],
    )?;
    let rhs = tensor::new(
        shape::new(vec![MATMULDECODEK, MATMULDECODEN])?,
        vec![0.25_f32; MATMULDECODEK * MATMULDECODEN],
    )?;
    Ok((lhs, rhs))
}

/// Query sequence length of the prefill-scale attention workload.
pub const ATTNPREFILLSEQ: usize = 512;

/// Query sequence length of the long-prefill attention workload, where the
/// key/value operands (16 MiB each) exceed the on-chip caches of the
/// documented baseline machine and tile reuse is what pays.
pub const ATTNPREFILLLONG: usize = 2048;

/// Key/value sequence length of the decode-scale attention workload: one query
/// position attending a long cached context.
pub const ATTNDECODEKV: usize = 4096;

/// Builds attention operands with a `[qseq, width]` query and `[kvseq, width]`
/// key/value pair, where `width = HEADS * HEADDIM` and `qseq <= kvseq` so the
/// operands satisfy the causal precondition of the attention contract.
///
/// # Errors
///
/// Returns a [`tensorerror`] if a fixed shape is rejected by `tensor`.
pub fn attentionseqinput(qseq: usize, kvseq: usize) -> Result<attentionoperands, tensorerror> {
    let width = HEADS * HEADDIM;
    let q = tensor::new(shape::new(vec![qseq, width])?, vec![0.1_f32; qseq * width])?;
    let k = tensor::new(
        shape::new(vec![kvseq, width])?,
        vec![0.2_f32; kvseq * width],
    )?;
    let v = tensor::new(
        shape::new(vec![kvseq, width])?,
        vec![0.3_f32; kvseq * width],
    )?;
    Ok((q, k, v))
}
