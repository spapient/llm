//! The CUDA C++ sources compiled at run time through NVRTC.
//!
//! Nothing here is compiled by the Rust build: the strings are compiled to PTX
//! by NVRTC on the first call that needs them, so the build never requires
//! `nvcc` and no PTX is committed. Each source declares one or more
//! `extern "C"` kernels, and the matching `*ENTRY` constant names the symbol
//! the host resolves with `CudaModule::load_function`.
//!
//! Every kernel is written to need no headers: NVRTC provides the device math
//! functions (`sqrtf`, `expf`, `powf`, `sinf`, `cosf`) implicitly, which is the
//! same property `cudarc`'s own `sin.cu` example relies on.

/// The CUDA C source of the reference `f32` matrix-multiply kernel.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]`, all row-major. Threads
/// outside the output bounds return early, so the kernel is correct for any
/// positive `m` and `n`; the host guards the degenerate `m == 0`, `n == 0`, and
/// `k == 0` cases before dispatching.
pub(crate) const MATMULCUDA: &str = r#"
extern "C" __global__ void matmul(const float* a, const float* b, float* c,
                                  unsigned m, unsigned k, unsigned n) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (unsigned i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}
"#;

/// The stable symbol name of the kernel in [`MATMULCUDA`].
pub(crate) const MATMULENTRY: &str = "matmul";

/// The threadgroup edge used when dispatching the matmul kernel.
pub(crate) const THREADEDGE: u32 = 16;

/// The CUDA C source of the elementwise `f32` kernels.
///
/// `add` and `mul` take two equal-length operands; `silu` takes one. All three
/// stride over the flat element count so a caller may dispatch any grid larger
/// than the element count and rely on the bounds test.
///
/// `silu` reproduces `cpu`'s `x / (1 + exp(-x))` exactly rather than using a
/// fused approximation, so the two backends agree to the last bit on the
/// arithmetic and differ only in `expf`'s rounding.
pub(crate) const ELEMENTWISE: &str = r#"
extern "C" __global__ void add(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void mul(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void silu(const float* a, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        c[i] = x / (1.0f + expf(-x));
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
pub(crate) const ELEMENTWISETILE: u32 = 256;

/// The CUDA C source of the `f32` RMS-normalization kernel.
///
/// One threadgroup per row; the sum of squares is reduced across the
/// threadgroup in shared memory. `x` is `[rows, width]`, `w` is `[width]`, and
/// `y` is `[rows, width]`. The row extent is taken from `blockDim.x`, so the
/// caller must pass a power-of-two `ROWTILE`.
pub(crate) const RMSNORMCUDA: &str = r#"
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y,
                                   unsigned width, float eps) {
    extern __shared__ float partial[];
    unsigned tid = threadIdx.x;
    const float* xrow = x + (size_t)blockIdx.x * width;
    float sumsq = 0.0f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        float v = xrow[i];
        sumsq += v * v;
    }
    partial[tid] = sumsq;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(partial[0] / (float)width + eps);
    for (unsigned i = tid; i < width; i += blockDim.x) {
        y[(size_t)blockIdx.x * width + i] = xrow[i] * inv * w[i];
    }
}
"#;

/// The stable symbol name of the kernel in [`RMSNORMCUDA`].
pub(crate) const RMSNORMENTRY: &str = "rmsnorm";

/// The threadgroup width of the RMS-normalization reduction.
pub(crate) const ROWTILE: u32 = 256;

/// The CUDA C source of the `f32` rotary-position-embedding kernel.
///
/// One thread per `(row, head, pair)`. `x` and `y` are
/// `[rows, heads * headdim]`, the position of a row is `row % seqaxis`, and the
/// angle of pair `i` is `position * base^(-2i / headdim)`. `seqaxis` equal to
/// zero means every row sits at position zero.
pub(crate) const ROPECUDA: &str = r#"
extern "C" __global__ void rope(const float* x, float* y, unsigned rows,
                                unsigned seqaxis, unsigned heads, unsigned headdim,
                                float base) {
    unsigned half = headdim / 2;
    unsigned long long total = (unsigned long long)rows * heads * half;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    unsigned i = (unsigned)(index % half);
    unsigned h = (unsigned)((index / half) % heads);
    unsigned row = (unsigned)(index / ((unsigned long long)half * heads));
    unsigned position = (seqaxis == 0u) ? 0u : (row % seqaxis);
    float angle = (float)position * powf(base, -2.0f * (float)i / (float)headdim);
    float s = sinf(angle);
    float c = cosf(angle);
    unsigned long long offset = (((unsigned long long)row * heads) + h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
"#;

/// The stable symbol name of the kernel in [`ROPECUDA`].
pub(crate) const ROPEENTRY: &str = "rope";

/// Absolute-position rotary embedding of a single token. The position is a
/// scalar parameter rather than a row index modulo `seqaxis`.
pub(crate) const ROPEATCUDA: &str = r#"
extern "C" __global__ void rope_at(const float* x, float* y, unsigned heads,
                                   unsigned headdim, unsigned position, float base) {
    unsigned half = headdim / 2;
    unsigned long long total = (unsigned long long)heads * half;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    unsigned i = (unsigned)(index % half);
    unsigned h = (unsigned)(index / half);
    float angle = (float)position * powf(base, -2.0f * (float)i / (float)headdim);
    float s = sinf(angle);
    float c = cosf(angle);
    unsigned long long offset = ((unsigned long long)h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
"#;

/// The stable symbol name of the kernel in [`ROPEATCUDA`].
pub(crate) const ROPEATENTRY: &str = "rope_at";

/// The CUDA C source of the `f32` embedding-gather kernel.
///
/// `table` is `[vocab, width]`, `indices` holds `rows` row numbers already
/// validated on the host, and `out` is `[rows, width]`. One thread per output
/// element, so the gather is fully coalesced on the read side.
pub(crate) const EMBEDDINGCUDA: &str = r#"
extern "C" __global__ void embedding(const float* table,
                                     const unsigned long long* indices,
                                     float* out, unsigned long long rows,
                                     unsigned long long width) {
    unsigned long long total = rows * width;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    unsigned long long row = index / width;
    out[index] = table[indices[row] * width + (index % width)];
}
"#;

/// The stable symbol name of the kernel in [`EMBEDDINGCUDA`].
pub(crate) const EMBEDDINGENTRY: &str = "embedding";

/// The CUDA C source of the fused scaled-dot-product attention kernel.
///
/// One threadgroup per `(query position, head)`. Scores are recomputed rather
/// than materialized: the kernel makes one pass for the row maximum, one for
/// the shifted-exponential sum, then tiles the key axis so each tile of weights
/// is computed once and reused by every output dimension. That keeps shared
/// memory proportional to `ATTENTIONLANES + headdim` — independent of the
/// sequence length — so no sequence-length cap is needed.
///
/// `NEGLARGE` is the reduction identity for the maximum. It is the most
/// negative finite `f32` rather than negative infinity because NVRTC needs no
/// header to produce it; for finite inputs the two are equivalent, and for a
/// non-finite row both yield the all-zero row the scalar contract specifies.
///
/// The kernel runs with exactly `ATTENTIONLANES` threads per threadgroup
/// because the reduction tree halves `blockDim.x`.
pub(crate) const ATTENTIONCUDA: &str = r#"
#define NEGLARGE (-3.402823466e+38f)

__device__ float attentionmax(float* scratch, float value) {
    __syncthreads();
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && scratch[threadIdx.x + s] > scratch[threadIdx.x]) {
            scratch[threadIdx.x] = scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float attentionsum(float* scratch, float value) {
    __syncthreads();
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float attentionscore(const float* qrow, const float* k, unsigned key,
                                unsigned kvheads, unsigned kvhead, unsigned headdim,
                                float scale) {
    const float* krow = k + ((size_t)key * kvheads + kvhead) * headdim;
    float dot = 0.0f;
    for (unsigned d = 0; d < headdim; ++d) {
        dot += qrow[d] * krow[d];
    }
    return dot * scale;
}

extern "C" __global__ void attention(const float* q, const float* k, const float* v,
                                     float* out, unsigned qseq, unsigned kvseq,
                                     unsigned heads, unsigned kvheads, unsigned headdim,
                                     unsigned causal) {
    extern __shared__ float scratch[];
    float* reduce = scratch;
    float* weights = scratch + blockDim.x;
    float* accum = weights + blockDim.x;

    const unsigned tid = threadIdx.x;
    const unsigned head = blockIdx.x % heads;
    const unsigned position = blockIdx.x / heads;
    const unsigned kvhead = head / (heads / kvheads);
    const unsigned offset = kvseq - qseq;
    const unsigned limit = (causal != 0u && offset + position + 1u < kvseq)
        ? (offset + position + 1u) : kvseq;
    const float* qrow = q + ((size_t)position * heads + head) * headdim;
    const float scale = 1.0f / sqrtf((float)headdim);

    float localmax = NEGLARGE;
    for (unsigned key = tid; key < limit; key += blockDim.x) {
        float score = attentionscore(qrow, k, key, kvheads, kvhead, headdim, scale);
        if (score > localmax) {
            localmax = score;
        }
    }
    const float maxv = attentionmax(reduce, localmax);

    float localsum = 0.0f;
    for (unsigned key = tid; key < limit; key += blockDim.x) {
        localsum += expf(attentionscore(qrow, k, key, kvheads, kvhead, headdim, scale) - maxv);
    }
    const float total = attentionsum(reduce, localsum);

    for (unsigned d = tid; d < headdim; d += blockDim.x) {
        accum[d] = 0.0f;
    }
    __syncthreads();

    for (unsigned base = 0; base < limit; base += blockDim.x) {
        unsigned span = limit - base;
        if (span > blockDim.x) {
            span = blockDim.x;
        }
        if (tid < span) {
            float score = attentionscore(qrow, k, base + tid, kvheads, kvhead, headdim, scale);
            weights[tid] = expf(score - maxv);
        } else {
            weights[tid] = 0.0f;
        }
        __syncthreads();
        for (unsigned d = tid; d < headdim; d += blockDim.x) {
            float partial = 0.0f;
            for (unsigned j = 0; j < span; ++j) {
                const float* vrow = v + ((size_t)(base + j) * kvheads + kvhead) * headdim;
                partial += weights[j] * vrow[d];
            }
            accum[d] += partial;
        }
        __syncthreads();
    }

    for (unsigned d = tid; d < headdim; d += blockDim.x) {
        out[((size_t)position * heads + head) * headdim + d] =
            (total > 0.0f) ? (accum[d] / total) : 0.0f;
    }
}
"#;

/// The stable symbol name of the kernel in [`ATTENTIONCUDA`].
pub(crate) const ATTENTIONENTRY: &str = "attention";

/// The incremental-decode attention kernel: one query token attends over a
/// cached key/value buffer followed by one new row, without materialising the
/// concatenation. The query has a single position, so the grid is `heads`.
pub(crate) const ATTENTIONCACHEDCUDA: &str = r#"
#define NEGLARGE (-3.402823466e+38f)

__device__ float cachedmax(float* scratch, float value) {
    __syncthreads();
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && scratch[threadIdx.x + s] > scratch[threadIdx.x]) {
            scratch[threadIdx.x] = scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float cachedsum(float* scratch, float value) {
    __syncthreads();
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float cachedscore(const float* qrow, const float* ck, const float* nk,
                             unsigned key, unsigned pos, unsigned kvheads,
                             unsigned kvhead, unsigned headdim, float scale) {
    const float* krow = (key < pos)
        ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
        : (nk + ((size_t)kvhead) * headdim);
    float dot = 0.0f;
    for (unsigned d = 0; d < headdim; ++d) {
        dot += qrow[d] * krow[d];
    }
    return dot * scale;
}

extern "C" __global__ void attention_cached(
    const float* q, const float* ck, const float* cv,
    const float* nk, const float* nv, float* out,
    unsigned kvseq, unsigned heads, unsigned kvheads, unsigned headdim,
    unsigned causal) {
    extern __shared__ float scratch[];
    float* reduce = scratch;
    float* weights = scratch + blockDim.x;
    float* accum = weights + blockDim.x;

    const unsigned tid = threadIdx.x;
    const unsigned head = blockIdx.x % heads;
    const unsigned kvhead = head / (heads / kvheads);
    const unsigned pos = kvseq - 1u;
    const unsigned limit = kvseq;
    const float* qrow = q + ((size_t)head) * headdim;
    const float scale = 1.0f / sqrtf((float)headdim);

    float localmax = NEGLARGE;
    for (unsigned key = tid; key < limit; key += blockDim.x) {
        float score = cachedscore(qrow, ck, nk, key, pos, kvheads, kvhead, headdim, scale);
        if (score > localmax) {
            localmax = score;
        }
    }
    const float maxv = cachedmax(reduce, localmax);

    float localsum = 0.0f;
    for (unsigned key = tid; key < limit; key += blockDim.x) {
        localsum += expf(cachedscore(qrow, ck, nk, key, pos, kvheads, kvhead, headdim, scale) - maxv);
    }
    const float total = cachedsum(reduce, localsum);

    for (unsigned d = tid; d < headdim; d += blockDim.x) {
        accum[d] = 0.0f;
    }
    __syncthreads();

    for (unsigned base = 0; base < limit; base += blockDim.x) {
        unsigned span = limit - base;
        if (span > blockDim.x) {
            span = blockDim.x;
        }
        if (tid < span) {
            float score = cachedscore(qrow, ck, nk, base + tid, pos, kvheads, kvhead, headdim, scale);
            weights[tid] = expf(score - maxv);
        } else {
            weights[tid] = 0.0f;
        }
        __syncthreads();
        for (unsigned d = tid; d < headdim; d += blockDim.x) {
            float partial = 0.0f;
            for (unsigned j = 0; j < span; ++j) {
                unsigned key = base + j;
                const float* vrow = (key < pos)
                    ? (cv + ((size_t)key * kvheads + kvhead) * headdim)
                    : (nv + ((size_t)kvhead) * headdim);
                partial += weights[j] * vrow[d];
            }
            accum[d] += partial;
        }
        __syncthreads();
    }

    for (unsigned d = tid; d < headdim; d += blockDim.x) {
        out[((size_t)head) * headdim + d] =
            (total > 0.0f) ? (accum[d] / total) : 0.0f;
    }
}
"#;

/// The stable symbol name of the kernel in [`ATTENTIONCACHEDCUDA`].
pub(crate) const ATTENTIONCACHEDENTRY: &str = "attention_cached";

/// The threadgroup width of the attention kernel.
///
/// The kernel's reduction tree and shared-memory partition are both sized by
/// this value, so it is a hard requirement rather than a tuning knob.
pub(crate) const ATTENTIONLANES: u32 = 256;
