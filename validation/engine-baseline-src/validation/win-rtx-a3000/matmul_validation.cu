#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

namespace {

void check(cudaError_t status, const char *where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

__global__ void naive_matmul(
    const float *a, const float *b, float *y, int M, int N, int K) {
    const int row = (int)blockIdx.y * 16 + (int)threadIdx.y;
    const int col = (int)blockIdx.x * 16 + (int)threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int p = 0; p < K; ++p) acc += a[(size_t)row * K + p] * b[(size_t)p * N + col];
    y[(size_t)row * N + col] = acc;
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

void reference(const float *a, const float *b, float *y, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += a[(size_t)i * K + p] * b[(size_t)p * N + j];
            }
            y[(size_t)i * N + j] = acc;
        }
    }
}

double error(const std::vector<float> &got, const std::vector<float> &expected) {
    if (got.size() != expected.size()) throw std::runtime_error("output length mismatch");
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, (double)std::abs(got[i] - expected[i]) /
                                    (1.0 + std::abs((double)expected[i])));
    }
    return worst;
}

void validate_case(std::ofstream &csv, int M, int N, int K) {
    auto a = input(M, K, 0);
    auto b = input(K, N, 1);
    std::vector<float> got((size_t)M * N, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> expected(got.size(), 0.0f);
    reference(a.data(), b.data(), expected.data(), M, N, K);
    cutile_matmul_f32_kernel(a.data(), b.data(), got.data(), M, N, K);
    const double err = error(got, expected);
    csv << M << ',' << N << ',' << K << ',' << got.size() << ',' << err << '\n';
    if (err > 1e-6) throw std::runtime_error("validation tolerance exceeded");
}

struct DeviceBuffers {
    float *a = nullptr;
    float *b = nullptr;
    float *candidate = nullptr;
    float *naive = nullptr;

    DeviceBuffers(const std::vector<float> &ha, const std::vector<float> &hb, size_t outputs) {
        check(cudaMalloc((void **)&a, ha.size() * sizeof(float)), "cudaMalloc A");
        check(cudaMalloc((void **)&b, hb.size() * sizeof(float)), "cudaMalloc B");
        check(cudaMalloc((void **)&candidate, outputs * sizeof(float)), "cudaMalloc candidate");
        check(cudaMalloc((void **)&naive, outputs * sizeof(float)), "cudaMalloc naive");
        check(cudaMemcpy(a, ha.data(), ha.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
        check(cudaMemcpy(b, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
    }

    ~DeviceBuffers() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(candidate);
        cudaFree(naive);
    }
};

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

void benchmark_case(std::ofstream &csv, int M, int N, int K, int warmup, int reps) {
    auto ha = input(M, K, 0);
    auto hb = input(K, N, 1);
    const size_t outputs = (size_t)M * N;
    DeviceBuffers buffers(ha, hb, outputs);
    const dim3 block(16, 16);
    const dim3 grid((unsigned)(N + 15) / 16, (unsigned)(M + 15) / 16);
    auto candidate = [&] {
        cutile_musa_matmul_f32_tile<<<grid, block>>>(buffers.a, buffers.b, buffers.candidate, M, N, K);
    };
    auto naive = [&] {
        naive_matmul<<<grid, block>>>(buffers.a, buffers.b, buffers.naive, M, N, K);
    };

    candidate();
    naive();
    check(cudaGetLastError(), "initial benchmark launches");
    check(cudaDeviceSynchronize(), "initial benchmark synchronize");
    std::vector<float> got(outputs), expected(outputs);
    check(cudaMemcpy(got.data(), buffers.candidate, outputs * sizeof(float), cudaMemcpyDeviceToHost), "D2H candidate");
    check(cudaMemcpy(expected.data(), buffers.naive, outputs * sizeof(float), cudaMemcpyDeviceToHost), "D2H naive");

    std::vector<float> cpu;
    double cpu_err = 0.0;
    double cpu_ms = -1.0;
    const double work = 2.0 * M * N * K;
    if (work <= 2.2e9) {
        cpu.resize(outputs);
        reference(ha.data(), hb.data(), cpu.data(), M, N, K);
        cpu_err = std::max(error(got, cpu), error(expected, cpu));
        const auto begin = std::chrono::steady_clock::now();
        reference(ha.data(), hb.data(), cpu.data(), M, N, K);
        const auto end = std::chrono::steady_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    }
    const double gpu_err = error(got, expected);
    const double worst_err = std::max(gpu_err, cpu_err);
    if (worst_err > 1e-6) throw std::runtime_error("benchmark output mismatch");

    const double candidate_ms = time_kernel(candidate, warmup, reps);
    const double naive_ms = time_kernel(naive, warmup, reps);
    std::vector<float> host_output(outputs);
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        cutile_matmul_f32_kernel(ha.data(), hb.data(), host_output.data(), M, N, K);
    }
    const auto end = std::chrono::steady_clock::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(end - begin).count() / reps;
    if (error(host_output, expected) > 1e-6) throw std::runtime_error("public ABI benchmark mismatch");

    csv << M << ',' << N << ',' << K << ',' << reps << ',' << worst_err << ','
        << candidate_ms << ',' << naive_ms << ',' << cpu_ms << ',' << e2e_ms << ','
        << gflops(M, N, K, candidate_ms) << ',' << gflops(M, N, K, naive_ms) << ',';
    if (cpu_ms > 0) csv << gflops(M, N, K, cpu_ms); else csv << -1;
    csv << ',' << gflops(M, N, K, e2e_ms) << '\n';
}

} // namespace

int main() {
    try {
        check(cudaSetDevice(0), "cudaSetDevice");
        std::ofstream validation("validation.csv");
        validation << std::setprecision(12);
        validation << "M,N,K,outputs,max_scaled_error\n";
        const int cases[][3] = {
            {0, 3, 2}, {3, 0, 2}, {1, 1, 0}, {3, 5, 0}, {1, 1, 1},
            {1, 17, 3}, {17, 1, 19}, {15, 17, 31}, {16, 16, 16},
            {17, 17, 17}, {31, 33, 35}, {64, 65, 63}, {65, 64, 129},
            {128, 3, 65}, {2, 4096, 1},
        };
        for (const auto &c : cases) validate_case(validation, c[0], c[1], c[2]);
        validation.close();

        std::ofstream benchmark("benchmark.csv");
        benchmark << std::setprecision(12);
        benchmark << "M,N,K,reps,max_scaled_error,candidate_ms,naive_cuda_ms,cpu_ms,public_abi_e2e_ms,"
                     "candidate_gflops,naive_cuda_gflops,cpu_gflops,public_abi_e2e_gflops\n";
        benchmark_case(benchmark, 255, 257, 127, 20, 100);
        benchmark_case(benchmark, 512, 512, 512, 5, 30);
        benchmark_case(benchmark, 1024, 1024, 1024, 3, 10);
        benchmark_case(benchmark, 4096, 256, 4096, 2, 5);
        benchmark.close();
        std::cout << "PASS: 15 validation cases and 4 benchmark cases\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
