// WMMA v6:64x64 tile、4 warps、每 warp 1 个 32x32x8 原生片段(QY2 硬件形态)、单缓冲 BK8、grid swap。
#include <musa_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace mtmusa::wmma;

#define TM 64
#define TN 64
#define BK 8

__global__ void __launch_bounds__(128, 8) wmma_v6(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int M, int N, int K) {
    __shared__ float As[BK][TM];
    __shared__ float Bs[BK][TN];

    const int warp = threadIdx.x >> 5;
    const int wrow = (warp >> 1) * 32;
    const int wcol = (warp & 1) * 32;

    fragment<accumulator, 32, 32, 8, float> acc;
    fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            int row = threadIdx.x >> 1, v = threadIdx.x & 1;
            float4 val = *(const float4*)(A + (blockIdx.x * TM + row) * K + k0 + v * 4);
            As[v*4+0][row] = __musa_float_to_tf32(val.x);
            As[v*4+1][row] = __musa_float_to_tf32(val.y);
            As[v*4+2][row] = __musa_float_to_tf32(val.z);
            As[v*4+3][row] = __musa_float_to_tf32(val.w);
            int brow = threadIdx.x >> 4, bv = threadIdx.x & 15;
            float4 bval = *(const float4*)(B + (k0 + brow) * N + blockIdx.y * TN + bv * 4);
            Bs[brow][bv*4+0] = __musa_float_to_tf32(bval.x);
            Bs[brow][bv*4+1] = __musa_float_to_tf32(bval.y);
            Bs[brow][bv*4+2] = __musa_float_to_tf32(bval.z);
            Bs[brow][bv*4+3] = __musa_float_to_tf32(bval.w);
        }
        __syncthreads();
        {
            fragment<matrix_a, 32, 32, 8, precision::tf32, col_major> a;
            fragment<matrix_b, 32, 32, 8, precision::tf32, row_major> b;
            load_matrix_sync(a, &As[0][wrow], TM);
            load_matrix_sync(b, &Bs[0][wcol], TN);
            mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    store_matrix_sync(C + (blockIdx.x * TM + wrow) * N + blockIdx.y * TN + wcol,
                      acc, N, mem_row_major);
}

static void run_case(const char* name, int M, int N, int K) {
    float *A, *B, *C;
    musaMalloc(&A, (size_t)M * K * 4);
    musaMalloc(&B, (size_t)K * N * 4);
    musaMalloc(&C, (size_t)M * N * 4);
    musaMemset(A, 0, (size_t)M * K * 4);
    musaMemset(B, 0, (size_t)K * N * 4);

    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    dim3 block(128);

    for (int i = 0; i < 3; i++) wmma_v6<<<grid, block>>>(A, B, C, M, N, K);
    musaDeviceSynchronize();

    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 100;
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) wmma_v6<<<grid, block>>>(A, B, C, M, N, K);
    musaEventRecord(e1, 0);
    musaEventSynchronize(e1);
    float ms = 0;
    musaEventElapsedTime(&ms, e0, e1);

    double flops = 2.0 * M * N * K * iters;
    double tf = flops / (ms * 1e-3) / 1e12;
    printf("%-10s %dx%dx%d: %.3f ms/iter, %.2f TFLOPS\n", name, M, N, K, ms / iters, tf);

    musaFree(A); musaFree(B); musaFree(C);
    musaEventDestroy(e0); musaEventDestroy(e1);
}

int main(void) {
    musaSetDevice(0);
    run_case("cube", 1024, 1024, 1024);
    run_case("prefill", 4096, 512, 14336);
    run_case("n1536", 4096, 1536, 14336);
    run_case("n768", 4096, 768, 14336);
    return 0;
}
