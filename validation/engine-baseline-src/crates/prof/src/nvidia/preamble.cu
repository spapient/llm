#include <math.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
__host__ __device__ float _cutile_exp_f32(float x) { return expf(x); }
__host__ __device__ float _cutile_log_f32(float x) { return logf(x); }
__host__ __device__ float _cutile_sum_f32(const float *a, int n, int stride) {
    if (n == 1) return a[0];
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i * stride];
    return acc;
}
__host__ __device__ float _cutile_sumsq_f32(const float *a, int n, int stride) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) { float x = a[i * stride]; acc += x * x; }
    return acc;
}
__host__ __device__ float _cutile_max_f32(const float *a, int n, int stride) {
    if (n == 1) return a[0];
    float acc = -3.402823466e38f;
    for (int i = 0; i < n; ++i) { float x = a[i * stride]; if (x > acc) acc = x; }
    return acc;
}
__host__ __device__ float _cutile_exp_sum_f32(const float *x, int n, int stride, float max) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += _cutile_exp_f32(x[i * stride] - max);
    return acc;
}
__host__ __device__ void _cutile_exp_norm_f32(float *y, const float *x, int n, int stride, float max, float sum) {
    for (int i = 0; i < n; ++i) y[i * stride] = _cutile_exp_f32(x[i * stride] - max) / sum;
}
__host__ __device__ float _cutile_sqrt_f32(float x) { return sqrtf(x); }
__host__ __device__ float _cutile_sin_f32(float x) { return sinf(x); }
__host__ __device__ float _cutile_cos_f32(float x) { return cosf(x); }
__host__ __device__ float _cutile_pow_f32(float x, float y) { return powf(x, y); }
__host__ __device__ float _cutile_ceil_f32(float x) { return ceilf(x); }
__host__ __device__ float _cutile_floor_f32(float x) { return floorf(x); }
__host__ __device__ float _cutile_round_f32(float x) { return roundf(x); }
__host__ __device__ float _cutile_trunc_f32(float x) { return truncf(x); }
