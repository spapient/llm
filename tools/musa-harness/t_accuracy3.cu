// 精度对拍:WMMA(tf32) 与 muBLAS 5.2 各自对 f64 标量参考的 max_rel_err。
// 形状 512x512x1024(行主序),确定性数据(LCG 均匀 [-1,1])。
#include <musa_runtime.h>
#include <mublas.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace mtmusa::wmma;

#define TM 64
#define TN 64
#define BK 8

__global__ void __launch_bounds__(128, 8) wmma_64x64(const float* __restrict__ A,
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


// f64 标量参考(host,行主序)
static void reference(const float* A, const float* B, double* C, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)A[m * K + k] * (double)B[k * N + n];
            C[m * N + n] = s;
        }
}

static void accuracy(const char* tag, const float* got, const double* want, int len) {
    double max_abs = 0, max_want = 0;
    for (int i = 0; i < len; i++) {
        double d = fabs((double)got[i] - want[i]);
        if (d > max_abs) max_abs = d;
        double w = fabs(want[i]);
        if (w > max_want) max_want = w;
    }
    printf("%-12s max_abs_diff = %.3e, norm = %.3e\n", tag, max_abs, max_abs / max_want);
}

static void fill(float* A, int len, unsigned seed) {
    unsigned x = seed;
    for (int i = 0; i < len; i++) {
        x = x * 1664525u + 1013904223u;          // LCG
        A[i] = ((float)(x >> 8) / 8388608.0f) - 1.0f;  // [-1, 1)
    }
}

int main(void) {
    musaSetDevice(0);
    const int M = 512, N = 512, K = 1024;
    const size_t ab = (size_t)M * K, bb = (size_t)K * N, cb = (size_t)M * N;
    float *hA = (float*)malloc(ab * 4), *hB = (float*)malloc(bb * 4);
    double *ref = (double*)malloc(cb * 8);
    fill(hA, ab, 12345);
    fill(hB, bb, 67890);

    float *A, *B, *Cw, *Cm;
    musaMalloc(&A, ab * 4);
    musaMalloc(&B, bb * 4);
    musaMalloc(&Cw, cb * 4);
    musaMalloc(&Cm, cb * 4);
    musaMemcpy(A, hA, ab * 4, musaMemcpyHostToDevice);
    musaMemcpy(B, hB, bb * 4, musaMemcpyHostToDevice);

    reference(hA, hB, ref, M, N, K);
    printf("参考完成\n");

    // 我的 WMMA
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    dim3 block(128);
    wmma_64x64<<<grid, block>>>(A, B, Cw, M, N, K);
    musaDeviceSynchronize();

    // muBLAS 5.2(handle 风格,行主序按 (N,M,K) 传参)
    mublasHandle_t handle = NULL;
    mublasCreate(&handle);
    float one = 1.0f, zero = 0.0f;
    mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, Cm, N);
    musaDeviceSynchronize();

    float *hw = (float*)malloc(cb * 4), *hm = (float*)malloc(cb * 4);
    musaMemcpy(hw, Cw, cb * 4, musaMemcpyDeviceToHost);
    musaMemcpy(hm, Cm, cb * 4, musaMemcpyDeviceToHost);

    accuracy("WMMA(tf32)", hw, ref, cb);
    accuracy("muBLAS5.2", hm, ref, cb);
    return 0;
}
