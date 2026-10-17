// Elementwise f32 kernels, compiled at run time through newLibraryWithSource.
// add and mul take two equal-length operands; silu takes one. All stride over
// the flat element count and bounds-test. The count is a single uint at p[0].
// silu reproduces cpu's x / (1 + exp(-x)) exactly (no fused approximation).
// Symbols: add, mul, silu
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
