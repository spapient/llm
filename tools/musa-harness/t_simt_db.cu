// FP32 SIMT v6:128x128、512 线程、8x4 块(8 acc)、BK=8、双缓冲(16KB smem)。
// 旧树 3.1 的 10.74(prefill)即 128x128+双缓冲形态;5.2 静态 smem 限 28KB,BK16 双缓冲(32KB)超限,用 BK8(16KB)。
#include <musa_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TM 128
#define TN 128
#define BK 8

__global__ void sgemm_simt_db(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][BK][TM];
    __shared__ float Bs[2][BK][TN];

    const int tid = threadIdx.x;
    const int rid = tid >> 5;        // 0..15 行组(每线程 8 行)
    const int cid = tid & 31;        // 0..31 列组(每线程 4 列)
    const int row_base = blockIdx.y * TM + rid * 8;
    const int col_base = blockIdx.x * TN + cid * 4;

    float acc[8][4] = {{0}};

    // 预取
    if (tid < 256) {
        int row = tid >> 1, v = tid & 1;
        float4 val = *(const float4*)(A + (blockIdx.y * TM + row) * K + v * 4);
        As[0][v*4+0][row] = val.x; As[0][v*4+1][row] = val.y;
        As[0][v*4+2][row] = val.z; As[0][v*4+3][row] = val.w;
        int brow = tid >> 5, bv = tid & 31;
        float4 bval = *(const float4*)(B + brow * N + blockIdx.x * TN + bv * 4);
        Bs[0][brow][bv*4+0] = bval.x; Bs[0][brow][bv*4+1] = bval.y;
        Bs[0][brow][bv*4+2] = bval.z; Bs[0][brow][bv*4+3] = bval.w;
    }

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int next = buf ^ 1;
        int k1 = k0 + BK;
        if (k1 < K && tid < 256) {
            int row = tid >> 1, v = tid & 1;
            float4 val = *(const float4*)(A + (blockIdx.y * TM + row) * K + k1 + v * 4);
            As[next][v*4+0][row] = val.x; As[next][v*4+1][row] = val.y;
            As[next][v*4+2][row] = val.z; As[next][v*4+3][row] = val.w;
            int brow = tid >> 5, bv = tid & 31;
            float4 bval = *(const float4*)(B + (k1 + brow) * N + blockIdx.x * TN + bv * 4);
            Bs[next][brow][bv*4+0] = bval.x; Bs[next][brow][bv*4+1] = bval.y;
            Bs[next][brow][bv*4+2] = bval.z; Bs[next][brow][bv*4+3] = bval.w;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float4 a0 = *(const float4*)&As[buf][k][rid * 8];
            float4 a1 = *(const float4*)&As[buf][k][rid * 8 + 4];
            float4 b0 = *(const float4*)&Bs[buf][k][cid * 4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float b = (c == 0) ? b0.x : (c == 1) ? b0.y : (c == 2) ? b0.z : b0.w;
                acc[0][c] += a0.x * b; acc[1][c] += a0.y * b;
                acc[2][c] += a0.z * b; acc[3][c] += a0.w * b;
                acc[4][c] += a1.x * b; acc[5][c] += a1.y * b;
                acc[6][c] += a1.z * b; acc[7][c] += a1.w * b;
            }
        }
        buf = next;
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

    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    dim3 block(512);

    for (int i = 0; i < 3; i++) sgemm_simt_db<<<grid, block>>>(A, B, C, M, N, K);
    musaDeviceSynchronize();

    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 100;
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) sgemm_simt_db<<<grid, block>>>(A, B, C, M, N, K);
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
