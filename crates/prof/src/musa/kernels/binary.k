// cutile musa 微内核：binary —— 逐元素二元运算的共享内核（add/mul 等只差一个运算符）
// 槽位：{{op}} 是算子名，{{binop}} 来自 [consts]；@prim(...) 由片段层逐个替换。
// binop 是逐元素运算符（"+" / "*"），落在 ap[0] 与 bp[0] 之间；其余结构完全一致。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int N) {
    #pragma unroll 4
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        const float *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);
        const float out = ap[0] {{binop}} bp[0];
        @store(y, n, out);
    }
}
