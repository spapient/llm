// f32 rotary-position-embedding kernel, compiled at run time through
// newLibraryWithSource. One thread per (row, head, pair); x and y are
// [rows, heads * headdim]. p carries rows, seqaxis, heads, headdim as uint at
// indices 0..3 and base as its u32 bit pattern at index 4. seqaxis == 0 places
// every row at position 0.
// Symbol: rope
#include <metal_stdlib>
using namespace metal;

kernel void rope(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 constant uint* p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    uint pairs = p[3] / 2;
    ulong total = (ulong)p[0] * p[2] * pairs;
    if ((ulong)gid >= total) {
        return;
    }
    uint i = (uint)((ulong)gid % pairs);
    uint h = (uint)(((ulong)gid / pairs) % p[2]);
    uint row = (uint)((ulong)gid / ((ulong)pairs * p[2]));
    uint position = (p[1] == 0u) ? 0u : (row % p[1]);
    float angle = (float)position * pow(as_type<float>(p[4]), -2.0f * (float)i / (float)p[3]);
    float s = sin(angle);
    float c = cos(angle);
    ulong offset = (((ulong)row * p[2]) + h) * p[3] + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
