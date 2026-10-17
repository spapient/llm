// Reference f32 matrix-multiply kernel, compiled at run time through NVRTC.
// a is [m, k], b is [k, n], c is [m, n], all row-major. Threads outside the
// output bounds return early; the host guards m == 0, n == 0, k == 0.
// Symbol: matmul
extern "C" __global__ void matmul(const float* a, const float* b, float* c,
                                  unsigned m, unsigned k, unsigned n) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (unsigned i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}
