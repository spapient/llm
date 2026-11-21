// 纯 FMA 峰值微基准:每线程 16 累加器,无全局访存,确定 mcc 5.2 的代码生成上限。
#include <musa_runtime.h>
#include <stdio.h>

__global__ void fma_peak(float* out, int iters) {
    float a0=1.0001f,a1=1.0001f,a2=1.0001f,a3=1.0001f,a4=1.0001f,a5=1.0001f,a6=1.0001f,a7=1.0001f;
    float b0=0.9999f,b1=0.9999f,b2=0.9999f,b3=0.9999f,b4=0.9999f,b5=0.9999f,b6=0.9999f,b7=0.9999f;
    float c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    for (int i = 0; i < iters; i++) {
        c0=fmaf(a0,b0,c0); c1=fmaf(a1,b1,c1); c2=fmaf(a2,b2,c2); c3=fmaf(a3,b3,c3);
        c4=fmaf(a4,b4,c4); c5=fmaf(a5,b5,c5); c6=fmaf(a6,b6,c6); c7=fmaf(a7,b7,c7);
        d0=fmaf(a0,b0,d0); d1=fmaf(a1,b1,d1); d2=fmaf(a2,b2,d2); d3=fmaf(a3,b3,d3);
        d4=fmaf(a4,b4,d4); d5=fmaf(a5,b5,d5); d6=fmaf(a6,b6,d6); d7=fmaf(a7,b7,d7);
    }
    if (c0 == 12345.0f) out[0] = c0+c1+c2+c3+c4+c5+c6+c7+d0+d1+d2+d3+d4+d5+d6+d7;
}

int main(void) {
    musaSetDevice(0);
    float* out; musaMalloc(&out, 4);
    int iters = 200000;
    for (int g = 0; g < 4; g++) {
        int grid = (g == 0) ? 56 : (g == 1) ? 112 : (g == 2) ? 448 : 1120;
        dim3 block(128);
        fma_peak<<<grid, block>>>(out, iters);
        musaDeviceSynchronize();
        musaEvent_t e0, e1; musaEventCreate(&e0); musaEventCreate(&e1);
        musaEventRecord(e0, 0);
        fma_peak<<<grid, block>>>(out, iters);
        musaEventRecord(e1, 0); musaEventSynchronize(e1);
        float ms = 0; musaEventElapsedTime(&ms, e0, e1);
        double fma = (double)grid * 128 * 16 * iters * 2;  // 16 FMA/线程/迭代,每 FMA 2 flop
        printf("grid=%4d: %.3f ms, %.2f TFLOPS\n", grid, ms, fma / (ms*1e-3) / 1e12);
        musaEventDestroy(e0); musaEventDestroy(e1);
    }
    return 0;
}
