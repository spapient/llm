// cutile ascend 微内核：silu_f32 —— 逐元素 SiLU 激活
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、算 x*sigmoid(x)、写一个数。exp 走 preamble 软件标量接口，复用 load/store 片段。

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        const float out = x / (1.0f + _cutile_exp_f32(-x));
        @store(y, n, out);
    }
}
