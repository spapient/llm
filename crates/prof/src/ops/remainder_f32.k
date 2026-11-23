// cutile 模板：remainder_f32 —— 逐元素 IEEE 754 余数。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}  b={{b}}
//
// C99 / OpenCL 都没有 `fmodf` 后缀；先用 `copysign(x - floor(x / y) * y, x)` 写在标量上，
// 结果符号同被除数，各后端 preamble 只取基础内建函数。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        const float *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);
        const float x = ap[0];
        const float v = bp[0];
        const float out = copysign(x - floor(x / v) * v, x);
        @store(y, n, out);
    }
}
