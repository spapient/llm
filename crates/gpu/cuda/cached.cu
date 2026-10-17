// Incremental-decode attention kernel, shared by the CUDA and MUSA backends.
// One query token attends over a cached key/value buffer followed by one new
// row, without materialising the concatenation. The query has a single
// position, so the grid is heads.
// Symbol: attention_cached
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
