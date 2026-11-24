// cutile 共享参考微内核：relu_f32 —— 逐元素 ReLU
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、负区归零、写一个数。没有数学库依赖。

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        float out = ap[0];
        if (out < 0.0f) { out = 0.0f; }
        @store(y, n, out);
    }
}
