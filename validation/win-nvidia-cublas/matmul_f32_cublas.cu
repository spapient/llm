// NVIDIA cuBLAS FP32 特化：行主序 ABI 不变，严格 FP32，不使用 TF32。
// 进程内缓存按当前设备校验；多线程调用串行化，显存只按高水位增长。
// 任一 CUDA/cuBLAS 失败都写 NaN，不静默回退成主机结果。
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mutex>

namespace {

struct CutileNvidiaMatmulState {
    cublasHandle_t handle = nullptr;
    int device = -1;
    float *a = nullptr;
    float *b = nullptr;
    float *y = nullptr;
    size_t ac = 0, bc = 0, yc = 0;

    void clear_buffers() {
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (y) cudaFree(y);
        a = b = y = nullptr;
        ac = bc = yc = 0;
    }

    void clear() {
        clear_buffers();
        if (handle) cublasDestroy(handle);
        handle = nullptr;
        device = -1;
    }

    ~CutileNvidiaMatmulState() { clear(); }
};

CutileNvidiaMatmulState cutile_nvidia_matmul_state;
std::mutex cutile_nvidia_matmul_mutex;

bool cutile_nvidia_matmul_reserve(size_t ab, size_t bb, size_t yb) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    auto &state = cutile_nvidia_matmul_state;
    if (state.handle && state.device != device) state.clear();
    if (!state.handle) {
        if (cublasCreate(&state.handle) != CUBLAS_STATUS_SUCCESS) return false;
        state.device = device;
        if (cublasSetMathMode(state.handle, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS ||
            cublasSetPointerMode(state.handle, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) {
            state.clear();
            return false;
        }
    }
    if (ab <= state.ac && bb <= state.bc && yb <= state.yc) return true;

    state.clear_buffers();
    const bool ok = cudaMalloc((void **)&state.a, ab) == cudaSuccess &&
                    cudaMalloc((void **)&state.b, bb) == cudaSuccess &&
                    cudaMalloc((void **)&state.y, yb) == cudaSuccess;
    if (!ok) {
        state.clear_buffers();
        return false;
    }
    state.ac = ab;
    state.bc = bb;
    state.yc = yb;
    return true;
}

bool cutile_nvidia_matmul_run(
    const float *a, const float *b, float *y, int M, int N, int K) {
    const size_t ab = (size_t)M * (size_t)K * sizeof(float);
    const size_t bb = (size_t)K * (size_t)N * sizeof(float);
    const size_t yb = (size_t)M * (size_t)N * sizeof(float);
    std::lock_guard<std::mutex> lock(cutile_nvidia_matmul_mutex);
    if (!cutile_nvidia_matmul_reserve(ab, bb, yb)) return false;
    auto &state = cutile_nvidia_matmul_state;
    if (cudaMemcpy(state.a, a, ab, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(state.b, b, bb, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    // 行主序 C(M,N)=A(M,K)B(K,N) 等价于列主序 C'(N,M)=B'(N,K)A'(K,M)。
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(state.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, state.b, N, state.a, K, &beta, state.y, N) !=
        CUBLAS_STATUS_SUCCESS) return false;
    // 同步 D2H 已等待默认流上的 GEMM，并把异步设备错误带回；无需再做全设备同步。
    return cudaMemcpy(y, state.y, yb, cudaMemcpyDeviceToHost) == cudaSuccess;
}

} // namespace

void cutile_matmul_f32_kernel(const float *a, const float *b, float *y, int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    const size_t outputs = (size_t)M * (size_t)N;
    if (K <= 0) {
        for (size_t i = 0; i < outputs; ++i) y[i] = 0.0f;
        return;
    }
    if (cutile_nvidia_matmul_run(a, b, y, M, N, K)) return;
    for (size_t i = 0; i < outputs; ++i) y[i] = NAN;
}
