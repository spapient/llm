// cutile 共享参考微内核：gemv_packed —— weight-only int2/int4 解码 GEMV 的共享内核
// 槽位：{{GMAX}} 来自 [axes]，{{op}} 是算子名，{{cpb}}/{{width}}/{{zero}} 来自 [consts]；
// @prim(...) 由片段层逐个替换。共享 ABI 由同目录 TOML 契约固定；后端可用自己的 kernels/<stem>.k 特化。
// 布局契约（由声明层校验）：w={{w}} x={{x}}
// i2 与 i4 的唯一差别是打包宽度：cpb=每字节码数、width=每码位数、zero=零点（i2=4/2/2、i4=2/4/8）。
// 打包前提：M 与 gsz 均为 cpb 的倍数、M 整除 gsz；非整倍数宽度在 op 前置校验处被拒，模板按
// M/cpb、gsz/cpb 成组解包，不处理补齐码。
typedef unsigned char u8;

void cutile_{{op}}_kernel(const float *x, const u8 *w, const float *scales,
                          float *y, int M, int N, int gsz) {
    for (int n = 0; n < N; ++n) {
        const u8 *row = 0;
        @load(row, w + (long)n * (M / {{cpb}}));

        float acc = 0.0f;
        float deq[{{GMAX}}];

        for (int g = 0; g < M / gsz; ++g) {
            const float *xs = 0;
            const float *sc = scales + n * (M / gsz) + g;

            @load_async(xs, x + g * gsz);
            @quant(deq, row + g * (gsz / {{cpb}}), sc, gsz, cpb: {{cpb}}, width: {{width}}, zero: {{zero}});
            @dot(acc, deq, xs, gsz);
        }

        float v = 0.0f;
        float out = 0.0f;
        @cvt(v, acc, sat: off);
        @reduce(out, &v, 1, 1, op: sum, axis: GMAX, func: _cutile_sum_f32);
        @store(y, n, out);
    }
}
