// wave128 形态:CTA 128 线程共持 1 个 32x32x8 片段,tile 32x32、K=8。
#include <musa_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace mtmusa::wmma;

#define TM 32
#define TN 32
#define BK 8

__global__ void __launch_bounds__(128, 1) wmma_dbg7(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C, int N) {
    __shared__ float As[TM][BK];
    __shared__ float Bs[TN][BK];
    fragment<accumulator, 32, 32, 8, float> acc;
    fill_fragment(acc, 0.0f);
    {
        // A 片 32x8 = 64 float4,128 线程:前 64 线程各 1 个
        if (threadIdx.x < 64) {
            int row = threadIdx.x >> 1, v = threadIdx.x & 1;
            float4 val = *(const float4*)(A + row * 8 + v * 4);
            As[row][v*4+0] = val.x; As[row][v*4+1] = val.y;
            As[row][v*4+2] = val.z; As[row][v*4+3] = val.w;
        }
        // B 片 32x8 转置存(Bs[n][k]):标量载入(跨 k 不能用 float4)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = threadIdx.x * 4 + i;    // 0..511 超 256,只用前 256
            if (idx < 256) {
                int n = idx >> 3, k = idx & 7;
                Bs[n][k] = B[k * 32 + n];
            }
        }
    }
    __syncthreads();
    {
        fragment<matrix_a, 32, 32, 8, precision::tf32, row_major> a;
        fragment<matrix_b, 32, 32, 8, precision::tf32, col_major> b;
        load_matrix_sync(a, &As[0][0], BK);
        load_matrix_sync(b, &Bs[0][0], BK);
        mma_sync(acc, a, b, acc);
    }
    __syncthreads();
    store_matrix_sync(C, acc, N, mem_row_major);
}

int main(void) {
    musaSetDevice(0);
    const int M = 32, N = 32, K = 8;
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

    wmma_dbg7<<<1, 128>>>(A, B, C, N);
    musaDeviceSynchronize();
    float* hC = (float*)malloc(M * N * 4);
    musaMemcpy(hC, C, M * N * 4, musaMemcpyDeviceToHost);

    int bad = 0, shown = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float want = 140.0f + 28.0f * (m + n) + 8.0f * m * n;
            if (hC[m * N + n] != want) {
                if (shown < 8) { printf("C[%d][%d] = %.1f, want %.1f\n", m, n, hC[m * N + n], want); shown++; }
                bad++;
            }
        }
    printf("%s (%d/%d 错)\n", bad ? "FAIL" : "OK", bad, M * N);
    return 0;
}
