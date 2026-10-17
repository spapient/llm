// Fused scaled-dot-product attention kernel, shared by the CUDA and MUSA backends.
// One threadgroup per (query position, head). Scores are recomputed, not
// materialized: one pass for the row max, one for the shifted-exponential sum,
// then tiles of the key axis are computed once and reused by every output dim.
// Shared memory is proportional to ATTENTIONLANES + headdim, independent of
// sequence length. NEGLARGE is the most negative finite f32 (not -inf), because
// NVRTC needs no header to produce it; for finite inputs the two are equivalent.
// Runs with exactly ATTENTIONLANES threads per threadgroup (the reduction tree
// halves blockDim.x).
// Symbol: attention
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
