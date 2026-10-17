// Incremental-decode attention kernel, compiled at run time through
// newLibraryWithSource. One query token attends over a cached key/value buffer
// followed by one new row, without materialising the concatenation. The query
// has a single position, so the grid is heads. p carries kvseq, heads, kvheads,
// headdim as four uint values at indices 0..3.
// Symbol: attention_cached
#include <metal_stdlib>
using namespace metal;

#define NEGLARGE (-3.402823466e+38f)

float attentioncacheddot(device const float* qrow, device const float* krow, uint headdim) {
    float dot = 0.0f;
    for (uint d = 0; d < headdim; ++d) {
        dot += qrow[d] * krow[d];
    }
    return dot;
}

float cachedreducemax(threadgroup float* scratch, float value, uint tid, uint tg) {
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

float cachedreducesum(threadgroup float* scratch, float value, uint tid, uint tg) {
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

kernel void attention_cached(
    device const float* q [[buffer(0)]],
    device const float* ck [[buffer(1)]],
    device const float* cv [[buffer(2)]],
    device const float* nk [[buffer(3)]],
    device const float* nv [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    threadgroup float weights[256];
    threadgroup float accum[4096];

    uint kvseq = p[0];
    uint heads = p[1];
    uint kvheads = p[2];
    uint headdim = p[3];
    uint pos = kvseq - 1u;
    uint head = tgid % heads;
    uint kvhead = head / (heads / kvheads);
    uint limit = kvseq;
    device const float* qrow = q + ((size_t)head) * headdim;
    float scale = 1.0f / sqrt((float)headdim);

    float localmax = NEGLARGE;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = (key < pos)
            ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
            : (nk + ((size_t)kvhead) * headdim);
        float score = attentioncacheddot(qrow, krow, headdim) * scale;
        if (score > localmax) {
            localmax = score;
        }
    }
    scratch[tid] = localmax;
    float maxv = cachedreducemax(scratch, localmax, tid, tg);

    float localsum = 0.0f;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = (key < pos)
            ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
            : (nk + ((size_t)kvhead) * headdim);
        localsum += exp(attentioncacheddot(qrow, krow, headdim) * scale - maxv);
    }
    scratch[tid] = localsum;
    float total = cachedreducesum(scratch, localsum, tid, tg);

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
            uint key = base + tid;
            device const float* krow = (key < pos)
                ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
                : (nk + ((size_t)kvhead) * headdim);
            weights[tid] = exp(attentioncacheddot(qrow, krow, headdim) * scale - maxv);
        } else {
            weights[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < headdim; d += tg) {
            float partial = 0.0f;
            for (uint j = 0; j < span; ++j) {
                uint key = base + j;
                device const float* vrow = (key < pos)
                    ? (cv + ((size_t)key * kvheads + kvhead) * headdim)
                    : (nv + ((size_t)kvhead) * headdim);
                partial += weights[j] * vrow[d];
            }
            accum[d] += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* orow = out + ((size_t)head) * headdim;
    for (uint d = tid; d < headdim; d += tg) {
        orow[d] = (total > 0.0f) ? (accum[d] / total) : 0.0f;
    }
}
