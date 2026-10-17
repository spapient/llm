// f32 softmax-over-last-axis kernel, shared by the CUDA and MUSA backends.
// One threadgroup per row; x is [rows, width], y is [rows, width].
// The block reduction tree halves blockDim.x, so the caller must pass a
// power-of-two ROWTILE. NEGLARGE is the most negative finite f32 (not -inf),
// because NVRTC needs no header to produce it; for finite inputs the two are
// equivalent. The max-subtracting order is the numerically stable one the
// scalar reference shares. A row whose shifted-exponent sum is not positive
// (non-finite input) is written as zeros; the single-operator path routes
// non-finite input to the host scalar rule (NaN) before launching this kernel,
// while the submission path cannot inspect device-resident operands and keeps
// the kernel's own result.
// Symbol: softmax
#define NEGLARGE (-3.402823466e+38f)

extern "C" __global__ void softmax(const float* x, float* y, unsigned width) {
    extern __shared__ float scratch[];
    unsigned tid = threadIdx.x;
    const float* xrow = x + (size_t)blockIdx.x * width;

    float localmax = NEGLARGE;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        float v = xrow[i];
        if (v > localmax) {
            localmax = v;
        }
    }
    scratch[tid] = localmax;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        __syncthreads();
    }
    float rowmax = scratch[0];

    float localsum = 0.0f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        localsum += expf(xrow[i] - rowmax);
    }
    scratch[tid] = localsum;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        __syncthreads();
    }
    float inv = (scratch[0] > 0.0f) ? (1.0f / scratch[0]) : 0.0f;

    for (unsigned i = tid; i < width; i += blockDim.x) {
        y[(size_t)blockIdx.x * width + i] = expf(xrow[i] - rowmax) * inv;
    }
}
