// cutile 模板：mul_f32 —— 逐元素 f32 乘法
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
//
// 与 add_f32 同构：一次循环读两个数、做一次乘法、写一个数，复用 load/store 片段。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        const float *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);

        const float out = ap[0] * bp[0];
        @store(y, n, out);
    }
}
