// cutile musa 微内核：gelu_f32 —— 逐元素 GELU（A&S erf，与 Rust 参考同一份系数）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、算 0.5*x*(1+erf(x/sqrt(2)))、写一个数。
// erf 用 Abramowitz-Stegun 7.1.26（与 nn::gelu_f32::erf 同一份系数），保证两侧对拍零漂移；
// 其中的 exp 走 preamble 软件标量接口，绝对值用条件取，不依赖数学库。

static float gelu_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = sign < 0.0f ? -x : x;
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float t = 1.0f / (1.0f + p * ax);
    float series = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    float y = 1.0f - series * _cutile_exp_f32(-ax * ax);
    return sign * y;
}

void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        const float out = 0.5f * x * (1.0f + gelu_erf(x * 0.7071067811865476f));
        @store(y, n, out);
    }
}
