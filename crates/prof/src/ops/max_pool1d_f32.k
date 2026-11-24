// cutile cpu 微内核：max_pool1d_f32 —— 一维最大池化（声明档 k=2, s=2, p=0，单通道）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
//
// 逐输出位取窗口最大值：窗口不重叠、补零位不进窗口。

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float a = x[2 * n];
        const float b = x[2 * n + 1];
        y[n] = a > b ? a : b;
    }
}
