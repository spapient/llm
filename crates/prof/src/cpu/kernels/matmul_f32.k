// cutile cpu 微内核：matmul_f32 —— 大 K 稠密矩阵乘（收缩轴 K 为运行时标量，生产档 4096）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
//
// 标量三重循环（CPU 无 MMA 片段）。复用 load/store，M/N/K 是整数标量。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int M, int N, int K) {
    /* cutile-unroll */
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                const float *ap = 0;
                const float *bp = 0;
                @load(ap, a + i * K + p);
                @load(bp, b + p * N + j);
                acc += ap[0] * bp[0];
            }
            @store(y, i * N + j, acc);
        }
    }
}
