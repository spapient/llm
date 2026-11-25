#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <math.h>
#include "matmul_f32.cu"

void check(cudaError_t status, const char *where) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char *where) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + ": cuBLAS error " + std::to_string((int)status));
}

float value(uint64_t index, uint64_t stream) {
    const uint64_t mixed = index * (stream == 0 ? 17 : 7) + stream * 3;
    return ((int)(mixed % 13) - 6) * 0.03125f;
}

std::vector<float> input(int rows, int cols, uint64_t stream) {
    std::vector<float> out((size_t)rows * cols);
    for (size_t i = 0; i < out.size(); ++i) out[i] = value(i, stream);
    return out;
}

double error(const std::vector<float> &got, const std::vector<float> &expected) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, (double)std::abs(got[i] - expected[i]) / (1.0 + std::abs((double)expected[i])));
    }
    return worst;
}

template <typename Launch>
double time_kernel(Launch launch, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) launch();
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup synchronize");
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "event create begin");
    check(cudaEventCreate(&end), "event create end");
    check(cudaEventRecord(begin), "event record begin");
    for (int i = 0; i < reps; ++i) launch();
    check(cudaEventRecord(end), "event record end");
    check(cudaEventSynchronize(end), "event synchronize");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, end), "event elapsed");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed / reps;
}

double gflops(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e6);
}

void run(std::ofstream &csv, cublasHandle_t handle, int M, int N, int K, int warmup, int reps) {
    auto ha = input(M, K, 0), hb = input(K, N, 1);
    const size_t ab = ha.size() * sizeof(float), bb = hb.size() * sizeof(float), yb = (size_t)M * N * sizeof(float);
    float *da, *db, *dy, *dv;
    check(cudaMalloc((void **)&da, ab), "cudaMalloc A");
    check(cudaMalloc((void **)&db, bb), "cudaMalloc B");
    check(cudaMalloc((void **)&dy, yb), "cudaMalloc candidate");
    check(cudaMalloc((void **)&dv, yb), "cudaMalloc cuBLAS");
    check(cudaMemcpy(da, ha.data(), ab, cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(db, hb.data(), bb, cudaMemcpyHostToDevice), "H2D B");

    const dim3 block(16, 16), grid((unsigned)(N + 15) / 16, (unsigned)(M + 15) / 16);
    auto candidate = [&] { cutile_musa_matmul_f32_tile<<<grid, block>>>(da, db, dy, M, N, K); };
    const float alpha = 1.0f, beta = 0.0f;
    auto vendor = [&] {
        check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                          &alpha, db, N, da, K, &beta, dv, N), "cublasSgemm");
    };
    candidate();
    vendor();
    check(cudaGetLastError(), "initial launches");
    check(cudaDeviceSynchronize(), "initial synchronize");
    std::vector<float> got((size_t)M * N), expected((size_t)M * N);
    check(cudaMemcpy(got.data(), dy, yb, cudaMemcpyDeviceToHost), "D2H candidate");
    check(cudaMemcpy(expected.data(), dv, yb, cudaMemcpyDeviceToHost), "D2H cuBLAS");
    const double err = error(got, expected);
    if (err > 1e-6) throw std::runtime_error("cuBLAS output mismatch");

    const double candidate_ms = time_kernel(candidate, warmup, reps);
    const double cublas_ms = time_kernel(vendor, warmup, reps);
    csv << M << ',' << N << ',' << K << ',' << reps << ',' << err << ','
        << candidate_ms << ',' << cublas_ms << ','
        << gflops(M, N, K, candidate_ms) << ',' << gflops(M, N, K, cublas_ms) << '\n';
    cudaFree(da); cudaFree(db); cudaFree(dy); cudaFree(dv);
}

int main() {
    try {
        check(cudaSetDevice(0), "cudaSetDevice");
        cublasHandle_t handle;
        check(cublasCreate(&handle), "cublasCreate");
        check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode FP32");
        std::ofstream csv("cublas.csv");
        csv << std::setprecision(12);
        csv << "M,N,K,reps,max_scaled_error,candidate_ms,cublas_fp32_ms,candidate_gflops,cublas_fp32_gflops\n";
        run(csv, handle, 255, 257, 127, 20, 100);
        run(csv, handle, 512, 512, 512, 5, 30);
        run(csv, handle, 1024, 1024, 1024, 3, 10);
        run(csv, handle, 4096, 256, 4096, 2, 5);
        cublasDestroy(handle);
        std::cout << "PASS: 4 cuBLAS FP32 comparisons\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
