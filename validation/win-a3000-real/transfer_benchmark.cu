#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr size_t AB = 4096ULL * 4096ULL * sizeof(float);
constexpr size_t BB = 4096ULL * 256ULL * sizeof(float);
constexpr size_t YB = 4096ULL * 256ULL * sizeof(float);

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

template <typename F>
double measure(F fn, int warmup = 3, int reps = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() / reps;
}

int main() {
    std::vector<float> a(AB / sizeof(float), 1.0f), b(BB / sizeof(float), 2.0f), y(YB / sizeof(float));
    float *da, *db, *dy, *pa, *pb, *py;
    check(cudaMalloc((void **)&da, AB));
    check(cudaMalloc((void **)&db, BB));
    check(cudaMalloc((void **)&dy, YB));
    check(cudaMallocHost((void **)&pa, AB));
    check(cudaMallocHost((void **)&pb, BB));
    check(cudaMallocHost((void **)&py, YB));
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    auto pageable = [&] {
        check(cudaMemcpy(da, a.data(), AB, cudaMemcpyHostToDevice));
        check(cudaMemcpy(db, b.data(), BB, cudaMemcpyHostToDevice));
        check(cudaMemcpy(y.data(), dy, YB, cudaMemcpyDeviceToHost));
    };
    auto pinned = [&] {
        std::memcpy(pa, a.data(), AB);
        std::memcpy(pb, b.data(), BB);
        check(cudaMemcpyAsync(da, pa, AB, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(db, pb, BB, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(py, dy, YB, cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        std::memcpy(y.data(), py, YB);
    };
    const double pageable_ms = measure(pageable);
    const double pinned_ms = measure(pinned);
    const double bytes = (double)AB + BB + YB;
    std::printf("pageable_ms=%.6f pinned_staging_ms=%.6f pageable_GBps=%.3f pinned_staging_GBps=%.3f speedup=%.4f\n",
                pageable_ms, pinned_ms, bytes / (pageable_ms * 1e6),
                bytes / (pinned_ms * 1e6), pageable_ms / pinned_ms);

    cudaStreamDestroy(stream);
    cudaFreeHost(pa); cudaFreeHost(pb); cudaFreeHost(py);
    cudaFree(da); cudaFree(db); cudaFree(dy);
    return 0;
}
