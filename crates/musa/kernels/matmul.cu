// Reference f32 matrix-multiply kernel, compiled at build time by mcc.
// a is [batch, m, k], b is [batch, k, n], c is [batch, m, n], all row-major;
// blockIdx.z selects the batch slice so one dispatch covers every slice.
// Threads outside the output bounds return early; the host guards
// m == 0, n == 0, k == 0, batch == 0.
// Symbol: matmul
extern "C" __global__ void matmul(const float* a, const float* b, float* c,
                                  unsigned m, unsigned k, unsigned n,
                                  unsigned batch) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    const float* ab = a + (size_t)blockIdx.z * m * k;
    const float* bb = b + (size_t)blockIdx.z * k * n;
    float* cb = c + (size_t)blockIdx.z * m * n;
    float acc = 0.0f;
    for (unsigned i = 0; i < k; ++i) {
        acc += ab[row * k + i] * bb[i * n + col];
    }
    cb[row * n + col] = acc;
}
