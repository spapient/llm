// cutile 模板：gemm_i4 —— 对称 int4 矩阵乘（group-wise 反量化 + 累加的整数路径）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
// nibble 解包后在 int32 上精确累加（对应 mma 整数路径），再 cvt 成 f32；CPU 无 mma，退标量。
typedef unsigned char u8;

// 取一个字节的第 r 个 nibble（r=0 低 4 位、r=1 高 4 位），对称零点 8。
static int cutile_nib(u8 byte, int r) {
    u8 nib = r ? (u8)((byte >> 4) & 0x0F) : (u8)(byte & 0x0F);
    return (int)nib - 8;
}

void cutile_{{op}}_kernel(const u8 *a, const u8 *b, float *y, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int acc = 0;
            for (int p = 0; p < K; ++p) {
                const u8 *ap = 0;
                const u8 *bp = 0;
                @load(ap, a + (i * K) / 2 + p / 2);
                @load(bp, b + (p * N + j) / 2);
                acc += cutile_nib(ap[0], p & 1) * cutile_nib(bp[0], (p * N + j) & 1);
            }
            float v = 0.0f;
            @cvt(v, acc, sat: off);
            @store(y, i * N + j, v);
        }
    }
}
