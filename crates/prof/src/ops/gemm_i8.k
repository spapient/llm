// cutile 共享微内核：gemm_i8 / matmul_i8 —— 对称 int8 矩阵乘（dp4a 签名）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
// 点积在 int32 上精确累加（对应 dp4a），再 cvt 成 f32；CPU 无 dp4a，退化为标量整数点积。
typedef unsigned char u8;

void cutile_{{op}}_kernel(const u8 *a, const u8 *b, float *y, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int acc = 0;
            /* cutile-unroll */
            for (int p = 0; p < K; ++p) {
                const u8 *ap = 0;
                const u8 *bp = 0;
                @load(ap, a + i * K + p);
                @load(bp, b + p * N + j);
                acc += (int)(signed char)ap[0] * (int)(signed char)bp[0];
            }
            float v = 0.0f;
            @cvt(v, acc, sat: off);
            @store(y, i * N + j, v);
        }
    }
}
