// f32 RMS-normalization kernel, shared by the CUDA and MUSA backends.
// One threadgroup per row; x is [rows, width], w is [width], y is [rows, width].
// The row extent is taken from blockDim.x, so the caller must pass a
// power-of-two ROWTILE.
// Symbol: rmsnorm
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y,
                                   unsigned width, float eps) {
    extern __shared__ float partial[];
    unsigned tid = threadIdx.x;
    const float* xrow = x + (size_t)blockIdx.x * width;
    float sumsq = 0.0f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        float v = xrow[i];
        sumsq += v * v;
    }
    partial[tid] = sumsq;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(partial[0] / (float)width + eps);
    for (unsigned i = tid; i < width; i += blockDim.x) {
        y[(size_t)blockIdx.x * width + i] = xrow[i] * inv * w[i];
    }
}
