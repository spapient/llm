// cutile cpu 微内核：max_pool2d_f32 —— 二维最大池化（声明档 3x3, s=1, p=1，4x4 输入、same 输出）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
//
// 补零位用 -inf 语义：越界下标直接跳过，等价于不参与比较。

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const int H = 4;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            float best = -3.402823466e38f;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ii = i + di;
                    const int jj = j + dj;
                    if (ii < 0 || jj < 0 || ii >= H || jj >= H) continue;
                    const float v = x[ii * H + jj];
                    if (v > best) best = v;
                }
            }
            y[i * H + j] = best;
        }
    }
}
