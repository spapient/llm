//! The Metal Shading Language sources compiled at run time.
//!
//! Nothing here is compiled by the Rust build: each string is compiled to a
//! `MTLLibrary` by `newLibraryWithSource` on the first call that needs it, so
//! the build never requires `xcrun` or a Metal toolchain and no `.metallib` is
//! committed. Every source is self-contained — it includes `<metal_stdlib>` and
//! pulls the device math functions from it — and declares one or more `kernel`
//! functions. The matching `*ENTRY` constant names the function the host
//! resolves with `MTLLibrary::newFunctionWithName`.
//!
//! Scalar parameters are passed as a single `constant` buffer whose element
//! type matches the Rust array the host uploads with `setBytes`; floats cross
//! the boundary as their `u32` bit pattern and are recovered device-side with
//! `as_type<float>`. Keeping every kernel's parameter buffer to one array keeps
//! the host's parameter binding a single call.

/// The Metal Shading Language source of the `f32` matrix-multiply kernel.
///
/// One dispatch carries every batch block: `a` is `[batch, m, k]`, `b` is
/// `[batch, k, n]`, and `c` is `[batch, m, n]`, all row-major, and the third
/// grid dimension is the block index. The parameter buffer `p` carries `m`,
/// `k`, and `n` as three `uint` values at indices `0`, `1`, and `2`; the block
/// count travels in the grid rather than in a parameter, so a block's operands
/// and its result are reached by pointer arithmetic over the same buffers.
///
/// A command buffer round trip costs the same for one dispatch as for many
/// (docs/design.md §21.10.2), so one dispatch per call is what makes a batched
/// multiplication cost one round trip instead of one per block. Threads outside
/// the output bounds return early, so the same source is correct for any
/// positive `m` and `n`, and for `batch == 1` it degenerates to the unbatched
/// case; the host guards `m == 0`, `n == 0`, and `k == 0` before dispatching.
///
/// The source is compiled at run time through `newLibraryWithSource`, so the
/// build never needs `xcrun` or a Metal toolchain.
pub(crate) const MATMULMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint* p [[buffer(3)]],
                   uint3 gid [[thread_position_in_grid]]) {
    uint m = p[0];
    uint k = p[1];
    uint n = p[2];
    if (gid.x >= n || gid.y >= m) {
        return;
    }
    device const float* ab = a + (size_t)gid.z * m * k;
    device const float* bb = b + (size_t)gid.z * k * n;
    device float* cb = c + (size_t)gid.z * m * n;
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += ab[gid.y * k + i] * bb[i * n + gid.x];
    }
    cb[gid.y * n + gid.x] = acc;
}
"#;

/// The stable MSL function name of the kernel in [`MATMULMSL`].
pub(crate) const MATMULENTRY: &str = "matmul";

/// The Metal Shading Language source of the elementwise `f32` kernels.
///
/// `add` and `mul` take two equal-length operands; `silu` takes one. All three
/// stride over the flat element count so a caller may dispatch any grid larger
/// than the element count and rely on the bounds test. The count arrives in the
/// parameter buffer as a single `uint` at index `0`.
///
/// `silu` reproduces `cpu`'s `x / (1 + exp(-x))` exactly rather than using a
/// fused approximation, so the two backends agree to the last bit on the
/// arithmetic and differ only in `exp`'s rounding.
pub(crate) const ELEMENTWISE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint* p [[buffer(3)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void mul(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint* p [[buffer(3)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) {
        c[gid] = a[gid] * b[gid];
    }
}

kernel void silu(device const float* a [[buffer(0)]],
                 device float* c [[buffer(1)]],
                 constant uint* p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) {
        float x = a[gid];
        c[gid] = x / (1.0f + exp(-x));
    }
}
"#;

/// The symbol of the elementwise addition kernel in [`ELEMENTWISE`].
pub(crate) const ADDENTRY: &str = "add";

/// The symbol of the elementwise multiplication kernel in [`ELEMENTWISE`].
pub(crate) const MULENTRY: &str = "mul";

/// The symbol of the sigmoid-linear-unit kernel in [`ELEMENTWISE`].
pub(crate) const SILUENTRY: &str = "silu";

/// The flat element count per threadgroup for the elementwise kernels.
pub(crate) const FLATTILE: usize = 256;

/// The Metal Shading Language source of the `f32` RMS-normalization kernel.
///
/// One threadgroup per row; the sum of squares is reduced across the
/// threadgroup in threadgroup memory. `x` is `[rows, width]`, `w` is `[width]`,
/// and `y` is `[rows, width]`. The parameter buffer carries `width` as a `uint`
/// at index `0` and `eps` as its `u32` bit pattern at index `1`.
///
/// The row extent is taken from `threads_per_threadgroup`, so the caller MUST
/// dispatch exactly [`RMSNORMTILE`] threads per threadgroup; the reduction tree
/// halves that value and therefore requires a power of two.
pub(crate) const RMSNORMMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    constant uint* p [[buffer(3)]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tg [[threads_per_threadgroup]]) {
    threadgroup float partial[256];
    uint width = p[0];
    float eps = as_type<float>(p[1]);
    device const float* row = x + (size_t)tgid * width;
    float sumsq = 0.0f;
    for (uint i = tid; i < width; i += tg) {
        float v = row[i];
        sumsq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(partial[0] / (float)width + eps);
    for (uint i = tid; i < width; i += tg) {
        y[(size_t)tgid * width + i] = row[i] * inv * w[i];
    }
}
"#;

/// The stable symbol name of the kernel in [`RMSNORMMSL`].
pub(crate) const RMSNORMENTRY: &str = "rmsnorm";

/// The threadgroup width of the RMS-normalization reduction.
pub(crate) const RMSNORMTILE: usize = 256;

/// The Metal Shading Language source of the `f32` last-axis softmax kernel.
///
/// One threadgroup per row; the row maximum and the shifted-exponential sum are
/// each reduced across the threadgroup in threadgroup memory. The kernel
/// subtracts the row maximum before exponentiating, matching `cpu`'s
/// numerically stable form and only dividing when the sum is positive, so a
/// zero-sum row keeps the shifted exponential it already wrote instead of
/// dividing by zero. That branch is unreachable for a finite row (the
/// maximum's own shifted exponential is one); a row of negative infinity
/// propagates NaN, because the shift itself is undefined.
/// The parameter buffer carries `width` as a single `uint` at index `0`.
///
/// The row extent is taken from `threads_per_threadgroup`, so the caller MUST
/// dispatch exactly [`SOFTMAXTILE`] threads per threadgroup; the reduction tree
/// halves that value and therefore requires a power of two.
pub(crate) const SOFTMAXMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax(device const float* x [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint* p [[buffer(2)]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tg [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    uint width = p[0];
    device const float* row = x + (size_t)tgid * width;
    device float* dst = out + (size_t)tgid * width;

    float localmax = -INFINITY;
    for (uint i = tid; i < width; i += tg) {
        float v = row[i];
        if (v > localmax) {
            localmax = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = localmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowmax = scratch[0];

    float localsum = 0.0f;
    for (uint i = tid; i < width; i += tg) {
        localsum += exp(row[i] - rowmax);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = localsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = scratch[0];

    for (uint i = tid; i < width; i += tg) {
        float e = exp(row[i] - rowmax);
        dst[i] = (total > 0.0f) ? (e / total) : e;
    }
}
"#;

/// The stable symbol name of the kernel in [`SOFTMAXMSL`].
pub(crate) const SOFTMAXENTRY: &str = "softmax";

/// The threadgroup width of the softmax reduction.
pub(crate) const SOFTMAXTILE: usize = 256;

/// The Metal Shading Language source of the `f32` rotary-position-embedding
/// kernel.
///
/// One thread per `(row, head, pair)`. `x` and `y` are
/// `[rows, heads * headdim]`, the position of a row is `row % seqaxis`, and the
/// angle of pair `i` is `position * base^(-2i / headdim)`. `seqaxis` equal to
/// zero means every row sits at position zero. The parameter buffer carries
/// `rows`, `seqaxis`, `heads`, and `headdim` as four `uint` values at indices
/// `0`..`3` and `base` as its `u32` bit pattern at index `4`.
pub(crate) const ROPEMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 constant uint* p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    uint pairs = p[3] / 2;
    ulong total = (ulong)p[0] * p[2] * pairs;
    if ((ulong)gid >= total) {
        return;
    }
    uint i = (uint)((ulong)gid % pairs);
    uint h = (uint)(((ulong)gid / pairs) % p[2]);
    uint row = (uint)((ulong)gid / ((ulong)pairs * p[2]));
    uint position = (p[1] == 0u) ? 0u : (row % p[1]);
    float angle = (float)position * pow(as_type<float>(p[4]), -2.0f * (float)i / (float)p[3]);
    float s = sin(angle);
    float c = cos(angle);
    ulong offset = (((ulong)row * p[2]) + h) * p[3] + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
"#;

/// The stable symbol name of the kernel in [`ROPEMSL`].
pub(crate) const ROPEENTRY: &str = "rope";

/// The Metal Shading Language source of the `f32` embedding-gather kernel.
///
/// `table` is `[vocab, width]`, `indices` holds `rows` row numbers already
/// validated on the host, and `out` is `[rows, width]`. One thread per output
/// element, so the gather is fully coalesced on the read side. The parameter
/// buffer carries `rows` and `width` as two `ulong` values at indices `0` and
/// `1`.
pub(crate) const EMBEDDINGMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding(device const float* table [[buffer(0)]],
                      device const ulong* indices [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant ulong* p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    ulong width = p[1];
    ulong total = p[0] * width;
    if ((ulong)gid >= total) {
        return;
    }
    ulong row = (ulong)gid / width;
    out[gid] = table[indices[row] * width + ((ulong)gid % width)];
}
"#;

/// The stable symbol name of the kernel in [`EMBEDDINGMSL`].
pub(crate) const EMBEDDINGENTRY: &str = "embedding";

/// The Metal Shading Language source of the fused scaled-dot-product attention
/// kernel.
///
/// One threadgroup per `(query position, head)`. Scores are recomputed rather
/// than materialized: the kernel makes one pass for the row maximum, one for
/// the shifted-exponential sum, then tiles the key axis so each tile of weights
/// is computed once and reused by every output dimension. That keeps
/// threadgroup memory proportional to `2 * ATTENTIONLANES + METALMAXHEADDIM` —
/// independent of the sequence length — so no sequence-length cap is needed.
///
/// `NEGLARGE` is the reduction identity for the maximum. It is the most
/// negative finite `f32` rather than negative infinity because Metal needs no
/// header to produce it; for finite inputs the two are equivalent, and for a
/// non-finite row both yield the all-zero row the scalar contract specifies.
/// `mmax`/`msum` mirror this for the shifted-exponential reduction.
///
/// The kernel runs with exactly `ATTENTIONLANES` threads per threadgroup
/// because the reduction tree halves `threads_per_threadgroup`. The
/// `accum` array is sized by `METALMAXHEADDIM`, so the two bounds MUST stay
/// equal; the host rejects a head dimension beyond that bound before
/// dispatching.
pub(crate) const ATTENTIONMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define NEGLARGE (-3.402823466e+38f)

float attentiondot(device const float* qrow, device const float* krow, uint headdim) {
    float dot = 0.0f;
    for (uint d = 0; d < headdim; ++d) {
        dot += qrow[d] * krow[d];
    }
    return dot;
}

float reducemax(threadgroup float* scratch, float value, uint tid, uint tg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

float reducesum(threadgroup float* scratch, float value, uint tid, uint tg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

kernel void attention(device const float* q [[buffer(0)]],
                      device const float* k [[buffer(1)]],
                      device const float* v [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      constant uint* p [[buffer(4)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tg [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    threadgroup float weights[256];
    threadgroup float accum[4096];

    uint seq = p[0];
    uint heads = p[1];
    uint kvheads = p[2];
    uint headdim = p[3];
    uint causal = p[4];
    uint head = tgid % heads;
    uint position = tgid / heads;
    uint kvhead = head / (heads / kvheads);
    uint limit = (causal != 0u && position + 1u < seq) ? (position + 1u) : seq;
    device const float* qrow = q + ((size_t)position * heads + head) * headdim;
    float scale = 1.0f / sqrt((float)headdim);

    float localmax = NEGLARGE;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = k + ((size_t)key * kvheads + kvhead) * headdim;
        float score = attentiondot(qrow, krow, headdim) * scale;
        if (score > localmax) {
            localmax = score;
        }
    }
    float maxv = reducemax(scratch, localmax, tid, tg);

    float localsum = 0.0f;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = k + ((size_t)key * kvheads + kvhead) * headdim;
        localsum += exp(attentiondot(qrow, krow, headdim) * scale - maxv);
    }
    float total = reducesum(scratch, localsum, tid, tg);

    for (uint d = tid; d < headdim; d += tg) {
        accum[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < limit; base += tg) {
        uint span = limit - base;
        if (span > tg) {
            span = tg;
        }
        if (tid < span) {
            device const float* krow = k + ((size_t)(base + tid) * kvheads + kvhead) * headdim;
            weights[tid] = exp(attentiondot(qrow, krow, headdim) * scale - maxv);
        } else {
            weights[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < headdim; d += tg) {
            float partial = 0.0f;
            for (uint j = 0; j < span; ++j) {
                device const float* vrow = v + ((size_t)(base + j) * kvheads + kvhead) * headdim;
                partial += weights[j] * vrow[d];
            }
            accum[d] += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* orow = out + ((size_t)position * heads + head) * headdim;
    for (uint d = tid; d < headdim; d += tg) {
        orow[d] = (total > 0.0f) ? (accum[d] / total) : 0.0f;
    }
}
"#;

/// The stable symbol name of the kernel in [`ATTENTIONMSL`].
pub(crate) const ATTENTIONENTRY: &str = "attention";

/// The threadgroup width of the attention kernel.
///
/// The kernel's reduction tree and shared-memory partition are both sized by
/// this value, so it is a hard requirement rather than a tuning knob.
pub(crate) const ATTENTIONLANES: usize = 256;
