// cutile 模板：batch_norm_f32 —— 批归一化（NHWC 布局，(n*S+s)*C+c）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} gamma={{gamma}} beta={{beta}} y={{y}}
//
// 批内统计：每通道在两趟里求均值与方差，再按 gamma/beta 仿射。

void cutile_{{op}}_kernel(const float *x, const float *gamma, const float *beta,
                         float *y, int N, int C, int S) {
    const int per = N / C;
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        float sq = 0.0f;
        for (int i = 0; i < per; ++i) {
            const float v = x[i * C + c];
            sum += v;
            sq += v * v;
        }
        const float mean = sum / (float)per;
        const float var = sq / (float)per - mean * mean;
        const float inv = 1.0f / _cutile_sqrt_f32(var + 1e-5f);
        for (int i = 0; i < per; ++i) {
            const int idx = i * C + c;
            y[idx] = (x[idx] - mean) * inv * gamma[c] + beta[c];
        }
    }
}
