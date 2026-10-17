// Max-subtracting softmax kernel over the last axis, compiled at build time
// by mcc. One threadgroup per row; x and y are [rows, width]. The two-pass
// max-subtraction form reproduces cpu's softmax exactly (no fused
// approximation), matching the metal backend's softmax contract.
// Symbol: softmax
extern "C" __global__ void softmax(const float* x, float* y, unsigned width) {
    extern __shared__ float scratch[];
    unsigned tid = threadIdx.x;
    const float* xrow = x + (size_t)blockIdx.x * width;
    float* yrow = y + (size_t)blockIdx.x * width;
    float maxv = -3.402823466e+38f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        float value = xrow[i];
        if (value > maxv) {
            maxv = value;
        }
    }
    scratch[tid] = maxv;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (scratch[tid + stride] > scratch[tid]) {
                scratch[tid] = scratch[tid + stride];
            }
        }
        __syncthreads();
    }
    maxv = scratch[0];
    float sum = 0.0f;
    for (unsigned i = tid; i < width; i += blockDim.x) {
        float e = expf(xrow[i] - maxv);
        yrow[i] = e;
        sum += e;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    sum = scratch[0];
    for (unsigned i = tid; i < width; i += blockDim.x) {
        yrow[i] /= sum;
    }
}
