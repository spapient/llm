// Elementwise f32 kernels, compiled at run time through NVRTC.
// add and mul take two equal-length operands; silu takes one. All stride over
// the flat element count and bounds-test, so any larger grid is safe. silu
// reproduces cpu's x / (1 + exp(-x)) exactly (no fused approximation).
// Symbols: add, mul, silu
extern "C" __global__ void add(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void mul(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void silu(const float* a, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        c[i] = x / (1.0f + expf(-x));
    }
}
