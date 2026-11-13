// cutile 模板：rmsnorm_f32 —— RMSNorm
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} gamma={{gamma}}
//
// 两趟：先求平方和得均方根，再归一化乘权重。复用 load/store 片段，sqrtf 来自数学库。
#include <math.h>

void cutile_{{op}}_kernel(const float *x, const float *gamma, float *y, int N) {
    float sum_sq = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float *xp = 0;
        @load(xp, x + n);
        sum_sq += xp[0] * xp[0];
    }
    const float rms = sqrtf(sum_sq / (float)N + 1e-6f);
    for (int n = 0; n < N; ++n) {
        const float *xp = 0;
        const float *gp = 0;
        @load(xp, x + n);
        @load(gp, gamma + n);
        const float out = gp[0] * xp[0] / rms;
        @store(y, n, out);
    }
}
