// f32 matrix-multiply kernel, compiled at run time through newLibraryWithSource.
// One dispatch carries every batch block: a is [batch, m, k], b is [batch, k, n],
// c is [batch, m, n], all row-major; the third grid dimension is the block index.
// The parameter buffer p carries m, k, n as three uint values at indices 0, 1, 2.
// Threads outside output bounds return early; the host guards m == 0, n == 0, k == 0.
// Symbol: matmul
#include <metal_stdlib>
using namespace metal;

kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint* p [[buffer(3)]],
                   uint3 gid [[thread_position_in_grid]]) {
    uint m = p[0];
    uint k = p[1];
    uint n = p[2];
    if (gid.x >= n || gid.y >= m) {
        return;
    }
    device const float* ab = a + (size_t)gid.z * m * k;
    device const float* bb = b + (size_t)gid.z * k * n;
    device float* cb = c + (size_t)gid.z * m * n;
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += ab[gid.y * k + i] * bb[i * n + gid.x];
    }
    cb[gid.y * n + gid.x] = acc;
}
