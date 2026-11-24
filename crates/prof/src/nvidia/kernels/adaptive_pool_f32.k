// cutile nvidia 微内核：adaptive_pool_f32 —— 自适应平均池化（声明档 4x4 -> 2x2）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
//
// 窗口按 [floor(i*h/oh), ceil((i+1)*h/oh)) 取，声明档每格恰好 2x2。

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const int H = 4;
    const int O = 2;
    const int R = H / O;
    for (int i = 0; i < O; ++i) {
        for (int j = 0; j < O; ++j) {
            float sum = 0.0f;
            for (int di = 0; di < R; ++di) {
                for (int dj = 0; dj < R; ++dj) {
                    sum += x[(i * R + di) * H + j * R + dj];
                }
            }
            y[i * O + j] = sum / (float)(R * R);
        }
    }
}
