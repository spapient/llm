// Fused scaled-dot-product attention kernel, compiled at run time through
// newLibraryWithSource. One threadgroup per (query position, head); scores are
// recomputed, not materialized. Threadgroup memory is proportional to
// 2 * ATTENTIONLANES + METALMAXHEADDIM, independent of sequence length. NEGLARGE
// is the most negative finite f32 (not -inf). p carries qseq, kvseq, heads,
// kvheads, headdim, causal as six uint values at indices 0..5. Runs with exactly
// ATTENTIONLANES threads per threadgroup; accum is sized by METALMAXHEADDIM, so
// those two bounds MUST stay equal.
// Symbol: attention
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

    uint qseq = p[0];
    uint kvseq = p[1];
    uint heads = p[2];
    uint kvheads = p[3];
    uint headdim = p[4];
    uint causal = p[5];
    uint offset = kvseq - qseq;
    uint head = tgid % heads;
    uint position = tgid / heads;
    uint kvhead = head / (heads / kvheads);
    uint limit = (causal != 0u && offset + position + 1u < kvseq) ? (offset + position + 1u) : kvseq;
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
