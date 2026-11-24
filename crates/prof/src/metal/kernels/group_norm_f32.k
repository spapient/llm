// cutile metal 微内核：group_norm_f32 —— 组归一化（NHWC 布局）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} gamma={{gamma}} beta={{beta}} y={{y}}
//
// 每组在 (C/G, S) 上统计；通道在组内的归属由 (idx % C) 决定。

void cutile_{{op}}_kernel(const float *x, const float *gamma, const float *beta,
                         float *y, int N, int G, int C, int S) {
    const int per = C / G;
    const int count = per * S;
    for (int g = 0; g < G; ++g) {
        const int c0 = g * per;
        float sum = 0.0f;
        float sq = 0.0f;
        for (int i = 0; i < N; ++i) {
            if (i % C < c0 || i % C >= c0 + per) continue;
            const float v = x[i];
            sum += v;
            sq += v * v;
        }
        const float mean = sum / (float)count;
        const float var = sq / (float)count - mean * mean;
        const float inv = 1.0f / _cutile_sqrt_f32(var + 1e-5f);
        for (int i = 0; i < N; ++i) {
            if (i % C < c0 || i % C >= c0 + per) continue;
            const int c = i % C;
            y[i] = (x[i] - mean) * inv * gamma[c] + beta[c];
        }
    }
}
