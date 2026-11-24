#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define N 96
#define BLOCK 256

__global__ void sigmoid_f32_grad(const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ dout, float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dx[i] = dout[i] * (y[i] * (1.0f - y[i]));
}
__global__ void tanh_f32_grad(const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ dout, float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dx[i] = dout[i] * (1.0f - y[i] * y[i]);
}
__global__ void silu_f32_grad(const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ dout, float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 1.0f / (1.0f + expf(-x[i]));
    dx[i] = dout[i] * (s + x[i] * s * (1.0f - s));
}
__global__ void gelu_f32_grad(const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ dout, float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float az = abs(x[i] * 0.70710678f);
    float t = 1.0f / (1.0f + 0.3275911f * az);
    float series = t * (0.2548296f + t * (-0.28449672f + t * (1.4214138f + t * (-1.4531521f + t * 1.0614054f))));
    float erf = (x[i] < 0.0f ? -1.0f : 1.0f) * (1.0f - series * expf(-az * az));
    float phi = 0.5f * (1.0f + erf);
    float pdf = x[i] * expf(-x[i] * x[i] * 0.5f) * 0.39894228f;
    dx[i] = dout[i] * (phi + pdf);
}

static double fwd(int op, double v) {
    switch (op) {
        case 0: return 1.0 / (1.0 + exp(-v));
        case 1: return 2.0 / (1.0 + exp(-2.0 * v)) - 1.0;
        case 2: { double s = 1.0 / (1.0 + exp(-v)); return v * s; }
        case 3: {
            double z = v * 0.7071067811865476;
            double az = fabs(z);
            double tt = 1.0 / (1.0 + 0.3275911 * az);
            double ser = tt * (0.2548296 + tt * (-0.28449672 + tt * (1.4214138 + tt * (-1.4531521 + tt * 1.0614054))));
            double e = (z < 0 ? -1.0 : 1.0) * (1.0 - ser * exp(-az * az));
            return 0.5 * v * (1.0 + e);
        }
    }
    return 0;
}

typedef void (*Kernel)(const float*, const float*, const float*, float*);

static void run_one(int op, const char* name, Kernel kern) {
    float hx[N], hy[N], hd[N], hdxc[N];
    double ref[N];
    for (int i = 0; i < N; i++) {
        hx[i] = (float)(-3.0 + 0.0625 * i);
        hd[i] = (float)(0.5 + 0.1 * i);
        hy[i] = (float)fwd(op, hx[i]);
        double h = 1e-6;
        double central = (fwd(op, (double)hx[i] + h) - fwd(op, (double)hx[i] - h)) / (2.0 * h);
        ref[i] = (double)hd[i] * central;
    }
    float *dx, *dy, *dd, *ddx;
    cudaMalloc(&dx, N*4); cudaMalloc(&dy, N*4); cudaMalloc(&dd, N*4); cudaMalloc(&ddx, N*4);
    cudaMemcpy(dx, hx, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, hd, N*4, cudaMemcpyHostToDevice);
    kern<<<1, BLOCK>>>(dx, dy, dd, ddx);
    cudaMemcpy(hdxc, ddx, N*4, cudaMemcpyDeviceToHost);
    double maxrel = 0; int worst = -1;
    for (int i = 0; i < N; i++) {
        double a = hdxc[i], b = ref[i];
        double denom = fabs(b) > 1e-3 ? fabs(b) : 1e-3;
        double rel = fabs(a - b) / denom;
        if (rel > maxrel) { maxrel = rel; worst = i; }
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dd); cudaFree(ddx);
    printf("%s maxrel=%.5f worst=%d %s\n", name, maxrel, worst, maxrel < 5e-2 ? "PASS" : "FAIL");
}

int main() {
    run_one(0, "sigmoid_f32", sigmoid_f32_grad);
    run_one(1, "tanh_f32", tanh_f32_grad);
    run_one(2, "silu_f32", silu_f32_grad);
    run_one(3, "gelu_f32", gelu_f32_grad);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device=%s\n", p.name);
    return 0;
}
