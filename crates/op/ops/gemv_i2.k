// cutile 模板：gemv_i2 —— weight-only int2 解码 GEMV
// 槽位：{{GMAX}} 来自 [axes]，{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 逻辑只有一份；设备差异全部在 prim/<vendor>/<prim>.<ext> 里。
// 布局契约（由声明层校验）：w={{w}} x={{x}}
// 打包前提：M 与 gsz 均为 4 的倍数、M 整除 gsz；非 4 倍数宽度在 op 前置校验处被拒，模板按 M/4、
// gsz/4 成四元解包，不处理补齐码。2 位解包设备片段随 P3 量化内核补齐，模板只定逻辑与签名。
typedef unsigned char u8;

void cutile_{{op}}_kernel(const float *x, const u8 *w, const float *scales,
                          float *y, int M, int N, int gsz) {
    int tok = 0;
    for (int n = 0; n < N; ++n) {
        const u8 *row = 0;
        @load(row, w + (long)n * (M / 4));

        float acc = 0.0f;
        float deq[{{GMAX}}];

        for (int g = 0; g < M / gsz; ++g) {
            const float *xs = 0;
            const float *sc = scales + n * (M / gsz) + g;

            @load_async(xs, x + g * gsz, tok);
            @commit_wait(tok);
            @quant(deq, row + g * (gsz / 4), sc, gsz);
            @dot(acc, deq, xs, gsz);
            @bar(tok);
        }

        float v = 0.0f;
        float out = 0.0f;
        @cvt(v, acc, sat: off);
        @reduce(out, &v, 1, 1, op: sum, axis: GMAX, func: _cutile_sum_f32);
        @store(y, n, out);
    }
}
