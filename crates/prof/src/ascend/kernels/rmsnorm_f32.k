// cutile ascend 微内核：rmsnorm_f32 —— RMSNorm
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} gamma={{gamma}}
//
// 两趟：先求平方和（x·x 即 Σx²，交 @dot 原语）得均方根，再归一化乘权重。
// 复用 load/store，sqrt 走 preamble 软件标量接口。

void cutile_{{op}}_kernel(const float *x, const float *gamma, float *y, int N) {
    float sum_sq = 0.0f;
    @dot(sum_sq, x, x, N);
    const float rms = _cutile_sqrt_f32(sum_sq / (float)N + 1e-6f);
    for (int n = 0; n < N; ++n) {
        const float *xp = 0;
        const float *gp = 0;
        @load(xp, x + n);
        @load(gp, gamma + n);
        const float out = gp[0] * xp[0] / rms;
        @store(y, n, out);
    }
}
