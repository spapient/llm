// FP32 SIMT v4 优化变体:128x128、512t、8acc、BK16、单缓冲。
// 变体 A:grid swap(x=M 方向);变体 B:+launch_bounds(512,4) 保 4 CTA/SM。
#include <musa_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TM 128
#define TN 128
#define BK 16
#define SWAP 1
#ifdef LB
__global__ void __launch_bounds__(512, LB) sgemm_opt(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
#else
__global__ void sgemm_opt(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
#endif
    __shared__ float As[BK][TM];
    __shared__ float Bs[BK][TN];

    const int tid = threadIdx.x;
    const int rid = tid >> 5;
    const int cid = tid & 31;
    #if SWAP
    const int row_base = blockIdx.x * TM + rid * 8;
    const int col_base = blockIdx.y * TN + cid * 4;
    #else
    const int row_base = blockIdx.y * TM + rid * 8;
    const int col_base = blockIdx.x * TN + cid * 4;
    #endif

    float acc[8][4] = {{0}};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // A 片 128x16 = 512 float4,每线程 1 个
        {
            int row = tid >> 2, v = tid & 3;
            #if SWAP
            float4 val = *(const float4*)(A + (blockIdx.x * TM + row) * K + k0 + v * 4);
            #else
            float4 val = *(const float4*)(A + (blockIdx.y * TM + row) * K + k0 + v * 4);
            #endif
            As[v*4+0][row] = val.x;
            As[v*4+1][row] = val.y;
            As[v*4+2][row] = val.z;
            As[v*4+3][row] = val.w;
        }
        {
            int row = tid >> 5, v = tid & 31;
            #if SWAP
            float4 val = *(const float4*)(B + (k0 + row) * N + blockIdx.y * TN + v * 4);
            #else
            float4 val = *(const float4*)(B + (k0 + row) * N + blockIdx.x * TN + v * 4);
            #endif
            Bs[row][v*4+0] = val.x;
            Bs[row][v*4+1] = val.y;
            Bs[row][v*4+2] = val.z;
            Bs[row][v*4+3] = val.w;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float4 a0 = *(const float4*)&As[k][rid * 8];
            float4 a1 = *(const float4*)&As[k][rid * 8 + 4];
            float4 b0 = *(const float4*)&Bs[k][cid * 4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float b = (c == 0) ? b0.x : (c == 1) ? b0.y : (c == 2) ? b0.z : b0.w;
                acc[0][c] += a0.x * b; acc[1][c] += a0.y * b;
                acc[2][c] += a0.z * b; acc[3][c] += a0.w * b;
                acc[4][c] += a1.x * b; acc[5][c] += a1.y * b;
                acc[6][c] += a1.z * b; acc[7][c] += a1.w * b;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int c = 0; c < 4; c++)
            C[(row_base + r) * N + col_base + c] = acc[r][c];
}

static void run_case(const char* name, int M, int N, int K) {
    float *A, *B, *C;
    musaMalloc(&A, (size_t)M * K * 4);
    musaMalloc(&B, (size_t)K * N * 4);
    musaMalloc(&C, (size_t)M * N * 4);
    musaMemset(A, 0, (size_t)M * K * 4);
    musaMemset(B, 0, (size_t)K * N * 4);

    #if SWAP
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    #else
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    #endif
    dim3 block(512);

    for (int i = 0; i < 3; i++) sgemm_opt<<<grid, block>>>(A, B, C, M, N, K);
    musaDeviceSynchronize();

    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 100;
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) sgemm_opt<<<grid, block>>>(A, B, C, M, N, K);
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
