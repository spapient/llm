// cutile 模板：gemm_i2 —— 对称 int2 矩阵乘（解包后整数路径）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}} b={{b}}
// 2 位码解包后在 int32 上精确累加（对应 mma 整数路径），再 cvt 成 f32；CPU 无 mma，退标量。
// 打包前提：K=8 为 4 倍数；每字节四码、低 2 位在前。2 位解包设备片段随 P3 量化内核补齐。
typedef unsigned char u8;

// 取一个字节的第 r 个 2 位段（r=0 最低位），对称零点 2。
static int cutile_i2(u8 byte, int r) {
    u8 code = (u8)((byte >> (2 * r)) & 0x03);
    return (int)code - 2;
}

void cutile_{{op}}_kernel(const u8 *a, const u8 *b, float *y, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int acc = 0;
            for (int p = 0; p < K; ++p) {
                const u8 *ap = 0;
                const u8 *bp = 0;
                @load(ap, a + (i * K) / 4 + p / 4);
                @load(bp, b + (p * N + j) / 4);
                acc += cutile_i2(ap[0], p & 3) * cutile_i2(bp[0], (p * N + j) & 3);
            }
            float v = 0.0f;
            @cvt(v, acc, sat: off);
            @store(y, i * N + j, v);
        }
    }
}
