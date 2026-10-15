// f32 embedding-gather kernel, compiled at run time through newLibraryWithSource.
// table is [vocab, width], indices holds rows row numbers validated on the host,
// out is [rows, width]. p carries rows and width as two ulong values at indices
// 0 and 1. One thread per output element; fully coalesced reads.
// Symbol: embedding
#include <metal_stdlib>
using namespace metal;

kernel void embedding(device const float* table [[buffer(0)]],
                      device const ulong* indices [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant ulong* p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    ulong width = p[1];
    ulong total = p[0] * width;
    if ((ulong)gid >= total) {
        return;
    }
    ulong row = (ulong)gid / width;
    out[gid] = table[indices[row] * width + ((ulong)gid % width)];
}
