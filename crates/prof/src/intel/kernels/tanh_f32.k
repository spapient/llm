// cutile intel 微内核：tanh_f32 —— 逐元素双曲正切
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、算 2/(1+exp(-2x))-1、写一个数。
// 与 sigmoid 同一条恒等式：tanh(x) = 2·sigmoid(2x) − 1，两端饱和由此自然成立。

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        const float out = 2.0f / (1.0f + _cutile_exp_f32(-2.0f * x)) - 1.0f;
        @store(y, n, out);
    }
}
