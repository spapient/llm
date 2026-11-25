#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) return 1;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 2;
    std::printf("name=%s\n", p.name);
    std::printf("cc=%d.%d\n", p.major, p.minor);
    std::printf("multiProcessorCount=%d\n", p.multiProcessorCount);
    std::printf("warpSize=%d\n", p.warpSize);
    std::printf("sharedMemPerBlock=%zu\n", p.sharedMemPerBlock);
    std::printf("sharedMemPerMultiprocessor=%zu\n", p.sharedMemPerMultiprocessor);
    std::printf("regsPerBlock=%d\n", p.regsPerBlock);
    std::printf("regsPerMultiprocessor=%d\n", p.regsPerMultiprocessor);
    std::printf("maxThreadsPerMultiProcessor=%d\n", p.maxThreadsPerMultiProcessor);
    std::printf("memoryBusWidth=%d\n", p.memoryBusWidth);
    int memory_clock = 0, core_clock = 0;
    if (cudaDeviceGetAttribute(&memory_clock, cudaDevAttrMemoryClockRate, 0) != cudaSuccess) return 3;
    if (cudaDeviceGetAttribute(&core_clock, cudaDevAttrClockRate, 0) != cudaSuccess) return 4;
    std::printf("memoryClockRate=%d\n", memory_clock);
    std::printf("clockRate=%d\n", core_clock);
    std::printf("totalGlobalMem=%zu\n", p.totalGlobalMem);
    return 0;
}
