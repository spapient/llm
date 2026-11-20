// WMMA v8:64x64 tile = 4 个 32x32 单元/CTA,每层 4 次 wave128 mma、4 acc 片段、单缓冲 BK8。
#include <musa_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace mtmusa::wmma;

#define TM 64
#define TN 64
#define BK 8

__global__ void __launch_bounds__(128, 8) wmma_v8(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int M, int N, int K) {
    __shared__ float As[TM][BK];
    __shared__ float Bs[TN][BK];

    fragment<accumulator, 32, 32, 8, float> acc[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) fill_fragment(acc[u], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            int row = threadIdx.x >> 1, v = threadIdx.x & 1;
            float4 val = *(const float4*)(A + (blockIdx.y * TM + row) * K + k0 + v * 4);
            As[row][v*4+0] = __musa_float_to_tf32(val.x);
            As[row][v*4+1] = __musa_float_to_tf32(val.y);
            As[row][v*4+2] = __musa_float_to_tf32(val.z);
            As[row][v*4+3] = __musa_float_to_tf32(val.w);
        }
        {
            // B 片转置:float4 全局读(n 连续 4),4 标量写 Bs[n][k](stride 8,无 bank 冲突)
            int k = threadIdx.x >> 4, n0 = (threadIdx.x & 15) * 4;
            float4 bval = *(const float4*)(B + (k0 + k) * N + blockIdx.x * TN + n0);
            Bs[n0 + 0][k] = __musa_float_to_tf32(bval.x);
            Bs[n0 + 1][k] = __musa_float_to_tf32(bval.y);
            Bs[n0 + 2][k] = __musa_float_to_tf32(bval.z);
            Bs[n0 + 3][k] = __musa_float_to_tf32(bval.w);
        }
        __syncthreads();
        {
            fragment<matrix_a, 32, 32, 8, precision::tf32, row_major> a[2];
            fragment<matrix_b, 32, 32, 8, precision::tf32, col_major> b[2];
            load_matrix_sync(a[0], &As[0][0], BK);
            load_matrix_sync(a[1], &As[32][0], BK);
            load_matrix_sync(b[0], &Bs[0][0], BK);
            load_matrix_sync(b[1], &Bs[32][0], BK);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++)
                    mma_sync(acc[i * 2 + j], a[i], b[j], acc[i * 2 + j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            store_matrix_sync(C + (blockIdx.y * TM + i * 32) * N + blockIdx.x * TN + j * 32,
                              acc[i * 2 + j], N, mem_row_major);
}

static void run_case(const char* name, int M, int N, int K) {
    float *A, *B, *C;
    musaMalloc(&A, (size_t)M * K * 4);
    musaMalloc(&B, (size_t)K * N * 4);
    musaMalloc(&C, (size_t)M * N * 4);
    musaMemset(A, 0, (size_t)M * K * 4);
    musaMemset(B, 0, (size_t)K * N * 4);

    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    dim3 block(128);

    for (int i = 0; i < 3; i++) wmma_v8<<<grid, block>>>(A, B, C, M, N, K);
    musaDeviceSynchronize();

    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 100;
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) wmma_v8<<<grid, block>>>(A, B, C, M, N, K);
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
