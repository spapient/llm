// 纯屏障成本:每层 2 次 syncthreads(128 线程),无载入无 mma。
#include <musa_runtime.h>
#include <stdio.h>
__global__ void __launch_bounds__(128, 8) barrier_only(float* out, int layers) {
    __shared__ float s;
    s = threadIdx.x;
    for (int l = 0; l < layers; l++) {
        __syncthreads();
        s += 1.0f;
        __syncthreads();
    }
    if (s == 12345.0f) out[0] = s;
}
int main(void) {
    musaSetDevice(0);
    float* out; musaMalloc(&out, 4);
    int layers = 1792;
    dim3 grid8(448), block(128);
    barrier_only<<<grid8, block>>>(out, layers);
    musaDeviceSynchronize();
    musaEvent_t e0, e1; musaEventCreate(&e0); musaEventCreate(&e1);
    musaEventRecord(e0, 0);
    barrier_only<<<grid8, block>>>(out, layers);
    musaEventRecord(e1, 0); musaEventSynchronize(e1);
    float ms = 0; musaEventElapsedTime(&ms, e0, e1);
    printf("纯屏障(448 CTA, 2/层): %.3f ms\n", ms);
    return 0;
}
