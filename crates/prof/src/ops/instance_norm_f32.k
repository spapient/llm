// cutile 共享参考微内核：instance_norm_f32 —— 实例归一化（NHWC 布局）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} gamma={{gamma}} beta={{beta}} y={{y}}
//
// 等价于 GroupNorm(G=C)：每个样本每个通道独立统计。

void cutile_{{op}}_kernel(const float *x, const float *gamma, const float *beta,
                         float *y, int N, int C, int S) {
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        float sq = 0.0f;
        int count = 0;
        for (int i = 0; i < N; ++i) {
            if (i % C != c) continue;
            const float v = x[i];
            sum += v;
            sq += v * v;
            ++count;
        }
        const float mean = sum / (float)count;
        const float var = sq / (float)count - mean * mean;
        const float inv = 1.0f / _cutile_sqrt_f32(var + 1e-5f);
        for (int i = 0; i < N; ++i) {
            if (i % C != c) continue;
            y[i] = (x[i] - mean) * inv * gamma[c] + beta[c];
        }
    }
}
