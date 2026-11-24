// cutile nvidia 微内核：layer_norm_f32 —— 层归一化（行主序 R×W）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} gamma={{gamma}} beta={{beta}} y={{y}}
//
// 逐行在最后一维统计，gamma/beta 按列广播。

void cutile_{{op}}_kernel(const float *x, const float *gamma, const float *beta,
                         float *y, int N, int R, int W) {
    for (int r = 0; r < R; ++r) {
        const int base = r * W;
        float sum = 0.0f;
        float sq = 0.0f;
        for (int j = 0; j < W; ++j) {
            const float v = x[base + j];
            sum += v;
            sq += v * v;
        }
        const float mean = sum / (float)W;
        const float var = sq / (float)W - mean * mean;
        const float inv = 1.0f / _cutile_sqrt_f32(var + 1e-5f);
        for (int j = 0; j < W; ++j) {
            y[base + j] = (x[base + j] - mean) * inv * gamma[j] + beta[j];
        }
    }
}
