// 四车道二元逐元素内核：N 仍是标量数，循环变量 n 是向量组下标。
void cutile_{{op}}_kernel(const float4 *a, const float4 *b, float4 *y, int N) {
    /* cutile-unroll */
    for (int n = 0; n < N / 4; ++n) {
        const float4 *ap = 0;
        const float4 *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);
        const float4 out = ap[0] {{binop}} bp[0];
        @store(y, n, out);
    }
}
