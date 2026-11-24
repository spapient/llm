// cutile nvidia 微内核：avg_pool2d_f32 —— 二维平均池化（声明档 3x3, s=1, p=1，补零不计入分母）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
//
// count_include_pad=false：分母是有效元素个数，角落窗口也按真统计。

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const int H = 4;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            float sum = 0.0f;
            int count = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ii = i + di;
                    const int jj = j + dj;
                    if (ii < 0 || jj < 0 || ii >= H || jj >= H) continue;
                    sum += x[ii * H + jj];
                    ++count;
                }
            }
            y[i * H + j] = sum / (float)count;
        }
    }
}
