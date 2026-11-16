// cutile 模板：gemv_i8 —— 对称 int8 解码 GEMV（不打包、零点 0）
// 槽位：{{GMAX}} 来自 [axes]，{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：w={{w}} x={{x}}
// quant.c 只承载 int4 解包；int8 的有符号解释是普通标量转换，这里内联，不占 quant 原语。
typedef unsigned char u8;

void cutile_{{op}}_kernel(const float *x, const u8 *w, const float *scales,
                          float *y, int M, int N, int gsz) {
    int tok = 0;
    for (int n = 0; n < N; ++n) {
        const u8 *row = 0;
        @load(row, w + (long)n * M);

        float acc = 0.0f;
        float deq[{{GMAX}}];

        for (int g = 0; g < M / gsz; ++g) {
            const float *xs = 0;
            const float *sc = scales + n * (M / gsz) + g;

            @load_async(xs, x + g * gsz, tok);
            @commit_wait(tok);
            // i8 反量化：字节按有符号解释（对称零点 0），不乘标度，保持与 Rust 参考同一结合顺序。
            for (int k = 0; k < gsz; ++k) {
                deq[k] = (float)(signed char)row[g * gsz + k];
            }
            float dot = 0.0f;
            @dot(dot, deq, xs, gsz);
            acc += dot * sc[0];
            @bar(tok);
        }

        float v = 0.0f;
        float out = 0.0f;
        @cvt(v, acc, sat: off);
        @reduce(out, &v, 1, 1, op: sum, axis: GMAX, func: _cutile_sum_f32);
        @store(y, n, out);
    }
}
