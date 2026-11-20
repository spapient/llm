// 32x32x8 片段正确性:64x64x8 单 CTA,A[m][k]=m+k、B[k][n]=k+n。
// 手算:C[m][n] = 140 + 28(m+n) + 8mn。
#include <musa_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace mtmusa::wmma;

#define TM 64
#define TN 64
#define BK 8

__global__ void __launch_bounds__(128, 8) wmma_dbg2(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C, int N) {
    __shared__ float As[BK][TM];
    __shared__ float Bs[BK][TN];
    const int warp = threadIdx.x >> 5;
    const int wrow = (warp >> 1) * 32;
    const int wcol = (warp & 1) * 32;
    fragment<accumulator, 32, 32, 8, float> acc;
    fill_fragment(acc, 0.0f);
    {
        int row = threadIdx.x >> 1, v = threadIdx.x & 1;
        float4 val = *(const float4*)(A + row * BK + v * 4);
        As[v*4+0][row] = val.x;
        As[v*4+1][row] = val.y;
        As[v*4+2][row] = val.z;
        As[v*4+3][row] = val.w;
        int brow = threadIdx.x >> 4, bv = threadIdx.x & 15;
        float4 bval = *(const float4*)(B + brow * TN + bv * 4);
        Bs[brow][bv*4+0] = bval.x;
        Bs[brow][bv*4+1] = bval.y;
        Bs[brow][bv*4+2] = bval.z;
        Bs[brow][bv*4+3] = bval.w;
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
    store_matrix_sync(C + wrow * N + wcol, acc, N, mem_row_major);
}

int main(void) {
    musaSetDevice(0);
    const int M = 64, N = 64, K = 8;
    float *hA = (float*)malloc(M * K * 4), *hB = (float*)malloc(K * N * 4);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) hA[m * K + k] = (float)(m + k);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) hB[k * N + n] = (float)(k + n);

    float *A, *B, *C;
    musaMalloc(&A, M * K * 4);
    musaMalloc(&B, K * N * 4);
    musaMalloc(&C, M * N * 4);
    musaMemcpy(A, hA, M * K * 4, musaMemcpyHostToDevice);
    musaMemcpy(B, hB, K * N * 4, musaMemcpyHostToDevice);

    wmma_dbg2<<<1, 128>>>(A, B, C, N);
    musaDeviceSynchronize();
    float* hC = (float*)malloc(M * N * 4);
    musaMemcpy(hC, C, M * N * 4, musaMemcpyDeviceToHost);

    int bad = 0;
    for (int m = 0; m < M; m += 7)
        for (int n = 0; n < N; n += 7) {
            float want = 140.0f + 28.0f * (m + n) + 8.0f * m * n;
            if (hC[m * N + n] != want) {
                if (bad < 8) printf("C[%d][%d] = %.1f, want %.1f\n", m, n, hC[m * N + n], want);
                bad++;
            }
        }
    printf(bad ? "错误 %d 处\n" : "全部正确\n", bad);
    return 0;
}
