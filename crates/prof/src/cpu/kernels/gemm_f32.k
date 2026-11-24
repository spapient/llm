// cutile cpu 微内核：gemm_f32 —— 矩阵乘
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
//
// 矩阵乘收缩轴 K=8；每个输出元素一个 1×1 块，收缩 K 个元素，交给 @mma 的标量 tile。
// MMA 原语与此外层循环均归当前后端；共享声明只约束 ABI 与数值语义。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            @mma(y + i * N + j, a + i * K, b + j, 1, 1, K, K, N, N);
        }
    }
}
