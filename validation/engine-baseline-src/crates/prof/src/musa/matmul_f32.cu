/* cutile-native: matmul_f32 */
/* cutile-sign: mma/mp_22 */
// MUSA mp_22 特化候选：FP32 16×16 共享内存 tile；不是 WMMA，不提升 wmma.tf32 等级。
// 通过档案的 cuda_runtime.h 垫片调用 MUSA runtime。ABI 与共享参考一致；设备失败写 NaN，
// 让现有对拍显式失败，绝不静默回退成主机结果。当前树只承诺源码/ABI 检查，真机另验。
#include <cuda_runtime.h>

#define CUTILE_MUSA_TILE 16

__global__ void cutile_musa_matmul_f32_tile(
    const float *a, const float *b, float *y, int M, int N, int K) {
    __shared__ float as[CUTILE_MUSA_TILE][CUTILE_MUSA_TILE];
    __shared__ float bs[CUTILE_MUSA_TILE][CUTILE_MUSA_TILE];
    const int row = (int)blockIdx.y * CUTILE_MUSA_TILE + (int)threadIdx.y;
    const int col = (int)blockIdx.x * CUTILE_MUSA_TILE + (int)threadIdx.x;
    float acc = 0.0f;
    for (int base = 0; base < K; base += CUTILE_MUSA_TILE) {
        const int k = base + (int)threadIdx.x;
        as[threadIdx.y][threadIdx.x] =
            row < M && k < K ? a[row * K + k] : 0.0f;
        const int bk = base + (int)threadIdx.y;
        bs[threadIdx.y][threadIdx.x] =
            bk < K && col < N ? b[bk * N + col] : 0.0f;
        __syncthreads();
        /* cutile-unroll */
        for (int p = 0; p < CUTILE_MUSA_TILE; ++p) {
            acc += as[threadIdx.y][p] * bs[p][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) y[row * N + col] = acc;
}

static int cutile_musa_matmul_f32_try(
    const float *a, const float *b, float *y, int M, int N, int K) {
    const size_t ab = (size_t)M * (size_t)K * sizeof(float);
    const size_t bb = (size_t)K * (size_t)N * sizeof(float);
    const size_t yb = (size_t)M * (size_t)N * sizeof(float);
    float *da = 0, *db = 0, *dy = 0;
    cudaError_t status = cudaMalloc((void **)&da, ab);
    if (status == cudaSuccess) status = cudaMalloc((void **)&db, bb);
    if (status == cudaSuccess) status = cudaMalloc((void **)&dy, yb);
    if (status == cudaSuccess) status = cudaMemcpy(da, a, ab, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(db, b, bb, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        const dim3 block(CUTILE_MUSA_TILE, CUTILE_MUSA_TILE);
        const dim3 grid((unsigned)(N + CUTILE_MUSA_TILE - 1) / CUTILE_MUSA_TILE,
                        (unsigned)(M + CUTILE_MUSA_TILE - 1) / CUTILE_MUSA_TILE);
        cutile_musa_matmul_f32_tile<<<grid, block>>>(da, db, dy, M, N, K);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaMemcpy(y, dy, yb, cudaMemcpyDeviceToHost);
    if (da) cudaFree(da);
    if (db) cudaFree(db);
    if (dy) cudaFree(dy);
    return status == cudaSuccess;
}

void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        for (long i = 0; i < (long)M * N; ++i) y[i] = 0.0f;
        return;
    }
    if (cutile_musa_matmul_f32_try(a, b, y, M, N, K)) return;
    for (long i = 0; i < (long)M * N; ++i) y[i] = NAN;
}
