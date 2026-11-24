// cutile metal 微内核：interpolate_f32 —— 双线性插值（声明档 2x2 -> 4x4, align_corners=false）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
//
// 源坐标 (i+0.5)*h/oh - 0.5 夹到边界后取四邻加权。

static float cutile_clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const int H = 2;
    const int O = 4;
    const float scale = (float)H / (float)O;
    for (int i = 0; i < O; ++i) {
        const float si = cutile_clamp01((((float)i + 0.5f) * scale - 0.5f) / (float)(H - 1));
        const int i0 = si <= 0.0f ? 0 : (si >= 1.0f ? H - 2 : (int)si);
        const float fi = si * (float)(H - 1) - (float)i0;
        for (int j = 0; j < O; ++j) {
            const float sj = cutile_clamp01((((float)j + 0.5f) * scale - 0.5f) / (float)(H - 1));
            const int j0 = sj <= 0.0f ? 0 : (sj >= 1.0f ? H - 2 : (int)sj);
            const float fj = sj * (float)(H - 1) - (float)j0;
            const float a = x[i0 * H + j0];
            const float b = x[i0 * H + j0 + 1];
            const float c = x[(i0 + 1) * H + j0];
            const float d = x[(i0 + 1) * H + j0 + 1];
            y[i * O + j] = a * (1.0f - fi) * (1.0f - fj) + b * (1.0f - fi) * fj
                         + c * fi * (1.0f - fj) + d * fi * fj;
        }
    }
}
