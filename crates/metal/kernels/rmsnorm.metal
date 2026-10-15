// f32 RMS-normalization kernel, compiled at run time through newLibraryWithSource.
// One threadgroup per row; x is [rows, width], w is [width], y is [rows, width].
// p[0] = width (uint), p[1] = eps (u32 bit pattern of a float). The row extent
// is taken from threads_per_threadgroup, so the caller MUST dispatch exactly
// RMSNORMTILE threads per threadgroup (a power of two).
// Symbol: rmsnorm
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
