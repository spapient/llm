// cutile ascend 微内核：min_f32 —— 逐元素取小。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}  b={{b}}
//
// 二元骨架：读两个数、取小、写一个数。用三元表达式，不依赖数学库的 fminf（跨设备语义一致）。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        const float *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);
        const float out = ap[0] < bp[0] ? ap[0] : bp[0];
        @store(y, n, out);
    }
}
