// 诊断:载入+屏障 only(无 mma)与 mma+屏障 only(无载入)的每层裸开销。
#include <musa_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace mtmusa::wmma;

#define TM 64
#define TN 64
#define BK 8

// 载入+屏障 only
__global__ void __launch_bounds__(128, 8) load_only(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ out,
                                                    int M, int N, int K) {
    __shared__ float As[BK][TM];
    __shared__ float Bs[BK][TN];
    float sink = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
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
        __syncthreads();
        sink += As[threadIdx.x & 7][threadIdx.x & 63];
        __syncthreads();
    }
    if (sink == 12345.0f) out[0] = sink;
}

// mma+屏障 only(数据预填一次)
__global__ void __launch_bounds__(128, 8) mma_barrier_only(float* out, int layers) {
    __shared__ float As[BK][TM];
    __shared__ float Bs[BK][TN];
    for (int i = threadIdx.x; i < BK * TM; i += 128) As[0][i] = 1.0f;
    for (int i = threadIdx.x; i < BK * TN; i += 128) Bs[0][i] = 1.0f;
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int wrow = (warp >> 1) * 32;
    const int wcol = (warp & 1) * 32;
    fragment<accumulator, 16, 16, 8, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) fill_fragment(acc[i][j], 0.0f);
    for (int l = 0; l < layers; l++) {
        __syncthreads();
        fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a[2];
        fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b[2];
        #pragma unroll
        for (int i = 0; i < 2; i++) load_matrix_sync(a[i], &As[0][wrow + i * 16], TM);
        #pragma unroll
        for (int j = 0; j < 2; j++) load_matrix_sync(b[j], &Bs[0][wcol + j * 16], TN);
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++) mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        __syncthreads();
    }
    float s = 0;
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            for (int e = 0; e < acc[i][j].num_elements; e++) s += acc[i][j].x[e];
    if (s == 12345.0f) out[0] = s;
}

int main(void) {
    musaSetDevice(0);
    int M = 4096, N = 768, K = 14336;
    float *A, *B, *out;
    musaMalloc(&A, (size_t)M * K * 4);
    musaMalloc(&B, (size_t)K * N * 4);
    musaMalloc(&out, 4);
    musaMemset(A, 0, (size_t)M * K * 4);
    musaMemset(B, 0, (size_t)K * N * 4);

    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);  // swap 版:y 是 N 方向
    dim3 block(128);
    musaEvent_t e0, e1; musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 50;
    load_only<<<grid, block>>>(A, B, out, M, N, K);
    musaDeviceSynchronize();
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) load_only<<<grid, block>>>(A, B, out, M, N, K);
    musaEventRecord(e1, 0); musaEventSynchronize(e1);
    float ms = 0; musaEventElapsedTime(&ms, e0, e1);
    printf("载入+屏障 only: %.3f ms/iter\n", ms / iters);

    int layers = K / BK;
    dim3 grid8(448);
    mma_barrier_only<<<grid8, block>>>(out, layers);
    musaDeviceSynchronize();
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++) mma_barrier_only<<<grid8, block>>>(out, layers);
    musaEventRecord(e1, 0); musaEventSynchronize(e1);
    musaEventElapsedTime(&ms, e0, e1);
    printf("mma+屏障 only(448 CTA): %.3f ms/iter\n", ms / iters);
    return 0;
}
