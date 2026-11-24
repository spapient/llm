// cutile 模板：gemm_packed —— int2/int4 打包量化 GEMM 的共享内核
// 槽位：{{op}} 是算子名，{{cpb}}/{{width}}/{{zero}} 来自 [consts]；@prim(...) 由片段层逐个替换。
// i2 与 i4 的唯一差别是打包宽度：cpb=每字节码数、width=每码位数、zero=零点（i2=4/2/2、i4=2/4/8）。
// 矩阵按行主序打包：K 维度每 cpb 个码压进 1 字节，步长与列内余数都按 cpb 折算。
typedef unsigned char u8;

static int cutile_code(u8 byte, int r) {
    u8 code = (u8)((byte >> ({{width}} * r)) & ((1 << {{width}}) - 1));
    return (int)code - {{zero}};
}

void cutile_{{op}}_kernel(const u8 *a, const u8 *b, float *y, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int acc = 0;
            for (int p = 0; p < K; ++p) {
                const u8 *ap = 0;
                const u8 *bp = 0;
                @load(ap, a + (i * K) / {{cpb}} + p / {{cpb}});
                @load(bp, b + (p * N + j) / {{cpb}});
                acc += cutile_code(ap[0], p & ({{cpb}} - 1)) * cutile_code(bp[0], (p * N + j) & ({{cpb}} - 1));
            }
            float v = 0.0f;
            @cvt(v, acc, sat: off);
            @store(y, i * N + j, v);
        }
    }
}
