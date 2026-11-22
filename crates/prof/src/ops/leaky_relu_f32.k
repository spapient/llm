// cutile 模板：leaky_relu_f32 —— 逐元素 LeakyReLU
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、按符号选乘 slope、写一个数。

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        const float out = x >= 0.0f ? x : 0.01f * x;
        @store(y, n, out);
    }
}
