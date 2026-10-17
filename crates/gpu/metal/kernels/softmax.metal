// f32 last-axis softmax kernel, compiled at run time through newLibraryWithSource.
// One threadgroup per row; subtracts the row max before exponentiating (cpu's
// numerically stable form) and divides only when the sum is positive. p[0] is
// width as a uint. The caller MUST dispatch exactly SOFTMAXTILE threads per
// threadgroup (a power of two). A row of -inf propagates NaN.
// Symbol: softmax
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
