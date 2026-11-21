// muBLAS 5.2 SGEMM 对照(handle 风格,cuBLAS v2 同构)。
// 行主序按 (N,M,K) 传参(2026-10-20 口径)。
#include <musa_runtime.h>
#include <mublas.h>
#include <stdio.h>
#include <stdlib.h>

static void run_case(mublasHandle_t handle, const char* name, int M, int N, int K) {
    float *A, *B, *C;
    musaMalloc(&A, (size_t)M * K * 4);
    musaMalloc(&B, (size_t)K * N * 4);
    musaMalloc(&C, (size_t)M * N * 4);
    musaMemset(A, 0, (size_t)M * K * 4);
    musaMemset(B, 0, (size_t)K * N * 4);

    float one = 1.0f, zero = 0.0f;
    for (int i = 0; i < 3; i++)
        mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N);
    musaDeviceSynchronize();

    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    int iters = 100;
    musaEventRecord(e0, 0);
    for (int i = 0; i < iters; i++)
        mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N);
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
    mublasHandle_t handle = NULL;
    mublasCreate(&handle);
    run_case(handle, "cube", 1024, 1024, 1024);
    run_case(handle, "prefill", 4096, 512, 14336);
    run_case(handle, "n1536", 4096, 1536, 14336);
    run_case(handle, "n768", 4096, 768, 14336);
    mublasDestroy(handle);
    return 0;
}
