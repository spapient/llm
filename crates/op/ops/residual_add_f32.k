// cutile 模板：residual_add_f32 —— 残差相加
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
//
// 每元素两读一写。复用 load/store，解引用写在普通 C 里。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        const float *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);
        @store(y, n, ap[0] + bp[0]);
    }
}
