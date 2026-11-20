// 3xTF32:64x64 tile、4 个 wave128 单元、hi/lo 分解、每层 12 次 mma(4 单元 x 3 项)。
// C = A_hi*B_hi + A_hi*B_lo + A_lo*B_hi(lo*lo 忽略,FP32 级精度)。
#include <musa_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace mtmusa::wmma;

#define TM 64
#define TN 64
#define BK 8

__global__ void __launch_bounds__(128, 8) wmma_3xtf32(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int N, int K) {
    __shared__ float As_hi[TM][BK], As_lo[TM][BK];
    __shared__ float Bs_hi[TN][BK], Bs_lo[TN][BK];

    fragment<accumulator, 32, 32, 8, float> acc[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) fill_fragment(acc[u], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            int row = threadIdx.x >> 1, v = threadIdx.x & 1;
            float4 val = *(const float4*)(A + (blockIdx.y * TM + row) * K + k0 + v * 4);
            float hi = __musa_float_to_tf32(val.x);
            As_hi[row][v*4+0] = hi; As_lo[row][v*4+0] = val.x - hi;
            hi = __musa_float_to_tf32(val.y);
            As_hi[row][v*4+1] = hi; As_lo[row][v*4+1] = val.y - hi;
            hi = __musa_float_to_tf32(val.z);
            As_hi[row][v*4+2] = hi; As_lo[row][v*4+2] = val.z - hi;
            hi = __musa_float_to_tf32(val.w);
            As_hi[row][v*4+3] = hi; As_lo[row][v*4+3] = val.w - hi;
        }
        {
            int k = threadIdx.x >> 4, n0 = (threadIdx.x & 15) * 4;
            float4 bval = *(const float4*)(B + (k0 + k) * N + blockIdx.x * TN + n0);
            float hi = __musa_float_to_tf32(bval.x);
            Bs_hi[n0 + 0][k] = hi; Bs_lo[n0 + 0][k] = bval.x - hi;
            hi = __musa_float_to_tf32(bval.y);
            Bs_hi[n0 + 1][k] = hi; Bs_lo[n0 + 1][k] = bval.y - hi;
            hi = __musa_float_to_tf32(bval.z);
            Bs_hi[n0 + 2][k] = hi; Bs_lo[n0 + 2][k] = bval.z - hi;
            hi = __musa_float_to_tf32(bval.w);
            Bs_hi[n0 + 3][k] = hi; Bs_lo[n0 + 3][k] = bval.w - hi;
        }
        __syncthreads();
        {
            fragment<matrix_a, 32, 32, 8, precision::tf32, row_major> ah[2], al[2];
            fragment<matrix_b, 32, 32, 8, precision::tf32, col_major> bh[2], bl[2];
            load_matrix_sync(ah[0], &As_hi[0][0], BK);
            load_matrix_sync(ah[1], &As_hi[32][0], BK);
            load_matrix_sync(al[0], &As_lo[0][0], BK);
            load_matrix_sync(al[1], &As_lo[32][0], BK);
            load_matrix_sync(bh[0], &Bs_hi[0][0], BK);
            load_matrix_sync(bh[1], &Bs_hi[32][0], BK);
            load_matrix_sync(bl[0], &Bs_lo[0][0], BK);
            load_matrix_sync(bl[1], &Bs_lo[32][0], BK);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int u = i * 2 + j;
                    mma_sync(acc[u], ah[i], bh[j], acc[u]);
                    mma_sync(acc[u], ah[i], bl[j], acc[u]);
                    mma_sync(acc[u], al[i], bh[j], acc[u]);
                }
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

    for (int i = 0; i < 3; i++) wmma_3xtf32<<<grid, block>>>(A, B, C, M, N, K);
    musaDeviceSynchronize();

    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 100;
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) wmma_3xtf32<<<grid, block>>>(A, B, C, M, N, K);
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
