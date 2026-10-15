// Absolute-position rotary embedding of a single token, compiled at run time
// through newLibraryWithSource. Position is a scalar parameter, not row % seqaxis.
// p carries heads, headdim, position, and base's u32 bit pattern at indices 0..3.
// Symbol: rope_at
#include <metal_stdlib>
using namespace metal;

kernel void rope_at(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    constant uint* p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint heads = p[0];
    uint headdim = p[1];
    uint position = p[2];
    float base = as_type<float>(p[3]);
    uint pairs = headdim / 2;
    ulong total = (ulong)heads * pairs;
    if ((ulong)gid >= total) {
        return;
    }
    uint i = (uint)((ulong)gid % pairs);
    uint h = (uint)((ulong)gid / pairs);
    float angle = (float)position * pow(base, -2.0f * (float)i / (float)headdim);
    float s = sin(angle);
    float c = cos(angle);
    ulong offset = ((ulong)h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
