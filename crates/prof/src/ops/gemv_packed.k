// cutile 共享参考微内核：gemv_packed —— weight-only int2/int4 解码 GEMV 的共享内核
// 槽位：{{GMAX}} 来自 [axes]，{{op}} 是算子名，{{cpb}}/{{width}}/{{zero}}/{{nblock}} 来自 [consts]；
// @prim(...) 由片段层逐个替换。共享 ABI 由同目录 TOML 契约固定；后端可用自己的 <stem>.<ext> 特化。
// 布局契约（由声明层校验）：w={{w}} x={{x}}
// i2 与 i4 的唯一差别是打包宽度：cpb=每字节码数、width=每码位数、zero=零点（i2=4/2/2、i4=2/4/8）。
// 打包前提：M 与 gsz 均为 cpb 的倍数、M 整除 gsz；非整倍数宽度在 op 前置校验处被拒，模板按
// M/cpb、gsz/cpb 成组解包，不处理补齐码。
// N 方向按 nblock 行分块：每个 G 组只加载一次 x，组内复用；每个输出的 G 累加与标度顺序不变。
typedef unsigned char u8;

void cutile_{{op}}_kernel(const float *x, const u8 *w, const float *scales,
                          float *y, int M, int N, int gsz) {
    for (int n0 = 0; n0 < N; n0 += {{nblock}}) {
        const int lanes = (N - n0 < {{nblock}}) ? (N - n0) : {{nblock}};
        const u8 *rows[{{nblock}}];
        float acc[{{nblock}}];

        for (int lane = 0; lane < lanes; ++lane) {
            acc[lane] = 0.0f;
            @load(rows[lane], w + (long)(n0 + lane) * (M / {{cpb}}));
        }

        for (int g = 0; g < M / gsz; ++g) {
            const float *xs = 0;
            @load_async(xs, x + g * gsz);

            for (int lane = 0; lane < lanes; ++lane) {
                const float *sc = scales + (n0 + lane) * (M / gsz) + g;
                float deq[{{GMAX}}];
                // 每组先求原始点积、最后只乘一次标度，避免逐元素重复乘 scale。
                const float unit = 1.0f;
                @quant(deq, rows[lane] + g * (gsz / {{cpb}}), &unit, gsz, cpb: {{cpb}}, width: {{width}}, zero: {{zero}});
                float dot = 0.0f;
                @dot(dot, deq, xs, gsz);
                acc[lane] += dot * sc[0];
            }
        }

        for (int lane = 0; lane < lanes; ++lane) {
            float v = 0.0f;
            float out = 0.0f;
            @cvt(v, acc[lane], sat: off);
            @reduce(out, &v, 1, 1, op: sum, axis: GMAX, func: _cutile_sum_f32);
            @store(y, n0 + lane, out);
        }
    }
}
