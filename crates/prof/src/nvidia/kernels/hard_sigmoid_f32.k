// cutile nvidia 微内核：hard_sigmoid_f32 —— 逐元素 HardSigmoid
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、算 clamp(x/6+0.5,0,1)、写一个数。

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        float out = x / 6.0f + 0.5f;
        if (out < 0.0f) { out = 0.0f; }
        if (out > 1.0f) { out = 1.0f; }
        @store(y, n, out);
    }
}
