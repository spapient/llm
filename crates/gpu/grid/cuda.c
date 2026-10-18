// CUDA-family f32 kernels, one translation unit for all fifteen entries.
//
// This single source is compiled at build time by `mcc` (MUSA backend) and
// `hipcc` (ROCm backend), and at run time by NVRTC (CUDA backend, from the
// Rust-side spliced source in catalog.rs `cu::SOURCE`).
//
// HIP and MUSA are source-compatible with CUDA, so the three backends share
// this text verbatim; every platform difference lives in the build flags (the
// `MUSA_OUT_STATE` macro in `scan` below) or in the Rust island. Metal, SYCL,
// CANN and MLU use their own languages and mirror this catalog over their own
// translation units.
//
// Entries: attention, attention_cached, add, copy, mul, silu, embedding,
// matmul, rmsnorm, rope, rope_at, scan, scan_agg, scan_prefix, scan_apply,
// softmax.
//
// Shared constants (GRID_NEGLARGE, GRID_SCANMAXN) come from grid/grid.h.
// NVRTC compiles this source from a string with no filesystem, so the Rust
// side splices the header in front and defines GRID_CUDA; file-system
// compilers (mcc/hipcc) include the header directly from the grid/ directory.
#if !defined(GRID_CUDA)
#include "grid.h"
#endif
#define LLM_SCAN_BLOCKDIM 256


// ---------------------------------------------------------------------------
// elementwise.cu
// ---------------------------------------------------------------------------


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

// Device-to-device row copy: c[off + i] = a[i]. Used by the KV-cache append
// path, where the driver's cuMemcpyDtoD is unreliable on WSL/NVIDIA virtual
// GPU stacks. Offsets are element-based and caller-validated.
extern "C" __global__ void copy(const float* a, float* c, unsigned n, unsigned off) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[off + i] = a[i];
    }
}

extern "C" __global__ void silu(const float* a, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        c[i] = x / (1.0f + expf(-x));
    }
}


// ---------------------------------------------------------------------------
// matmul.cu
// ---------------------------------------------------------------------------


extern "C" __global__ void matmul(const float* a, const float* b, float* c,
                                  unsigned m, unsigned k, unsigned n,
                                  unsigned batch) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    const float* ab = a + (size_t)blockIdx.z * m * k;
    const float* bb = b + (size_t)blockIdx.z * k * n;
    float* cb = c + (size_t)blockIdx.z * m * n;
    float acc = 0.0f;
    for (unsigned i = 0; i < k; ++i) {
        acc += ab[row * k + i] * bb[i * n + col];
    }
    cb[row * n + col] = acc;
}


// ---------------------------------------------------------------------------
// embedding.cu
// ---------------------------------------------------------------------------


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


// ---------------------------------------------------------------------------
// rmsnorm.cu
// ---------------------------------------------------------------------------


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


// ---------------------------------------------------------------------------
// rope.cu
// ---------------------------------------------------------------------------


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
    float s, c;
    sincosf(angle, &s, &c);
    unsigned long long offset = (((unsigned long long)row * heads) + h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}


// ---------------------------------------------------------------------------
// at.cu
// ---------------------------------------------------------------------------


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
    float s, c;
    sincosf(angle, &s, &c);
    unsigned long long offset = ((unsigned long long)h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}


// ---------------------------------------------------------------------------
// softmax.cu
// ---------------------------------------------------------------------------


extern "C" __global__ void softmax(const float* x, float* y, unsigned width) {
    extern __shared__ float scratch[];
    unsigned tid = threadIdx.x;
    const float* xrow = x + (size_t)blockIdx.x * width;
    float localmax = GRID_NEGLARGE;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        float v = xrow[i];
        if (v > localmax) {
            localmax = v;
        }
    }
    scratch[tid] = localmax;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        __syncthreads();
    }
    float rowmax = scratch[0];
    float localsum = 0.0f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        localsum += expf(xrow[i] - rowmax);
    }
    scratch[tid] = localsum;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        __syncthreads();
    }
    float inv = (scratch[0] > 0.0f) ? (1.0f / scratch[0]) : 0.0f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        y[(size_t)blockIdx.x * width + i] = expf(xrow[i] - rowmax) * inv;
    }
}


// ---------------------------------------------------------------------------
// attention.cu
// ---------------------------------------------------------------------------

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
    // Online softmax: one pass computes the running max and the running
    // sum together, so each QK score is evaluated twice instead of three
    // times (the PV pass below re-evaluates scores for the weights).
    float runningmax = GRID_NEGLARGE;
    float localsum = 0.0f;
    for (unsigned key = tid; key < limit; key += blockDim.x) {
        float score = attentionscore(qrow, k, key, kvheads, kvhead, headdim, scale);
        float nm = fmaxf(score, runningmax);
        localsum = localsum * expf(runningmax - nm) + expf(score - nm);
        runningmax = nm;
    }
    const float maxv = attentionmax(reduce, runningmax);
    // Re-scale each thread's partial sum to the block max, then reduce.
    const float total = attentionsum(reduce, localsum * expf(runningmax - maxv));
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


// ---------------------------------------------------------------------------
// cached.cu
// ---------------------------------------------------------------------------

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
    // Online softmax, same as `attention` above: one pass for the running
    // max and running sum (two score evaluations instead of three).
    float runningmax = GRID_NEGLARGE;
    float localsum = 0.0f;
    for (unsigned key = tid; key < limit; key += blockDim.x) {
        float score = cachedscore(qrow, ck, nk, key, pos, kvheads, kvhead, headdim, scale);
        float nm = fmaxf(score, runningmax);
        localsum = localsum * expf(runningmax - nm) + expf(score - nm);
        runningmax = nm;
    }
    const float maxv = cachedmax(reduce, runningmax);
    const float total = cachedsum(reduce, localsum * expf(runningmax - maxv));
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


// ---------------------------------------------------------------------------
// scan.cu
// ---------------------------------------------------------------------------


extern "C" __global__ void scan(const float* input, const float* a,
                                const float* b, const float* c, const float* skip,
                                float* state, float* out
#ifdef MUSA_OUT_STATE
                                , float* out_state
#endif
                                , unsigned n, unsigned d, unsigned N) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)d) {
        return;
    }
    for (unsigned t = 0; t < n; ++t) {
        float x = input[t * d + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < N; ++j) {
            unsigned idx = i * N + j;
            float h = a[idx] * state[idx] + b[idx] * x;
            state[idx] = h;
            acc += c[idx] * h;
        }
        out[t * d + i] = acc + skip[i] * x;
    }
#ifdef MUSA_OUT_STATE
    for (unsigned j = 0; j < N; ++j) {
        out_state[i * N + j] = state[i * N + j];
    }
#endif
}


// ---------------------------------------------------------------------------
// parallel.cu
// ---------------------------------------------------------------------------


extern "C" __global__ void scan_agg(const float* input, const float* a,
                                    const float* b, const float* c,
                                    const float* skip, const float* init,
                                    float* block_total, unsigned n, unsigned d,
                                    unsigned N, unsigned chunk, unsigned blocks) {
    const unsigned block = blockIdx.x;
    const unsigned i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const unsigned t0 = block * chunk;
    const unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
    const unsigned dN = d * N;
    const unsigned base = block * (2u * dN);
    for (unsigned j = 0; j < N; ++j) {
        const unsigned idx = i * N + j;
        float A = 1.0f;
        float B = 0.0f;
        const float aj = a[idx];
        const float bj = b[idx];
        for (unsigned t = t0; t < t1; ++t) {
            const float x = input[t * d + i];
            B = aj * B + bj * x;
            A = aj * A;
        }
        block_total[base + idx] = A;
        block_total[base + dN + idx] = B;
    }
}

extern "C" __global__ void scan_prefix(const float* block_total,
                                       const float* init, float* state,
                                       float* block_prefix, unsigned n,
                                       unsigned d, unsigned N, unsigned chunk,
                                       unsigned blocks) {
    const unsigned dN = d * N;
    for (unsigned j = threadIdx.x; j < dN; j += blockDim.x) {
        // prefix[0] is the identity (1, 0).
        block_prefix[j] = 1.0f;
        block_prefix[dN + j] = 0.0f;
        float A = 1.0f;
        float B = 0.0f;
        for (unsigned q = 0; q < blocks; ++q) {
            const unsigned base = q * (2u * dN);
            const float aq = block_total[base + j];
            const float bq = block_total[base + dN + j];
            block_prefix[(q + 1u) * (2u * dN) + j] = A * aq;
            block_prefix[(q + 1u) * (2u * dN) + dN + j] = fmaf(aq, B, bq);
            B = fmaf(aq, B, bq);
            A = A * aq;
        }
        state[j] = A * init[j] + B;
    }
}

extern "C" __global__ void scan_apply(const float* input, const float* a,
                                      const float* b, const float* c,
                                      const float* skip, const float* init,
                                      const float* block_prefix, float* out,
                                      unsigned n, unsigned d, unsigned N,
                                      unsigned chunk, unsigned blocks) {
    const unsigned block = blockIdx.x;
    const unsigned i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const unsigned t0 = block * chunk;
    const unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
    const unsigned dN = d * N;
    const unsigned base = block * (2u * dN);
    float h[GRID_SCANMAXN];
    for (unsigned j = 0; j < N; ++j) {
        const unsigned idx = i * N + j;
        h[j] = block_prefix[base + idx] * init[idx]
            + block_prefix[base + dN + idx];
    }
    for (unsigned t = t0; t < t1; ++t) {
        const float x = input[t * d + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < N; ++j) {
            const unsigned idx = i * N + j;
            h[j] = a[idx] * h[j] + b[idx] * x;
            acc += c[idx] * h[j];
        }
        out[t * d + i] = acc + skip[i] * x;
    }
}

