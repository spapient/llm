// Serial per-channel scan kernel, compiled at run time through
// newLibraryWithSource. One thread per channel; each thread runs the serial
// recurrence over the sequence axis for its channel's state vector, so the
// result is bit-for-bit the serial reference (the parallel reorganization is a
// later, measured optimization). The state buffer is initialized to `init` by
// the host and updated in place; `p` carries n, d, N as three uint values at
// indices 0..2. Symbol: scan
#include <metal_stdlib>
using namespace metal;

kernel void scan(device const float* input [[buffer(0)]],
                 device const float* a [[buffer(1)]],
                 device const float* b [[buffer(2)]],
                 device const float* c [[buffer(3)]],
                 device const float* skip [[buffer(4)]],
                 device float* state [[buffer(5)]],
                 device float* out [[buffer(6)]],
                 constant uint* p [[buffer(7)]],
                 uint gid [[thread_position_in_grid]]) {
    uint n = p[0];
    uint d = p[1];
    uint N = p[2];
    uint i = gid;
    if (i >= d) {
        return;
    }
    for (uint t = 0; t < n; ++t) {
        float x = input[t * d + i];
        float acc = 0.0f;
        for (uint j = 0; j < N; ++j) {
            uint idx = i * N + j;
            float h = a[idx] * state[idx] + b[idx] * x;
            state[idx] = h;
            acc += c[idx] * h;
        }
        out[t * d + i] = acc + skip[i] * x;
    }
}
