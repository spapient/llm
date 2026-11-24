// cutile 模板：sigmoid_f32 —— 逐元素 Sigmoid
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、算 1/(1+exp(-x))、写一个数。exp 走 preamble 软件标量接口。
// 两端饱和自然成立：x 极大时 exp(-x) 下溢为 0（得 1），x 极小时 exp 上溢为 inf（得 0）。

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        const float out = 1.0f / (1.0f + _cutile_exp_f32(-x));
        @store(y, n, out);
    }
}
