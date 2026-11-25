// cutile 共享参考微内核：gemv_packed —— weight-only int2/int4 解码 GEMV 的共享内核
// 槽位：{{GMAX}} 来自 [axes]，{{op}} 是算子名，{{cpb}}/{{width}}/{{zero}} 来自 [consts]；
// @prim(...) 由片段层逐个替换。共享 ABI 由同目录 TOML 契约固定；后端可用自己的 <stem>.<ext> 特化。
// 布局契约（由声明层校验）：w={{w}} x={{x}}
// i2 与 i4 的唯一差别是打包宽度：cpb=每字节码数、width=每码位数、zero=零点（i2=4/2/2、i4=2/4/8）。
// 打包前提：M 与 gsz 均为 cpb 的倍数、M 整除 gsz；非整倍数宽度在 op 前置校验处被拒，模板按
// M/cpb、gsz/cpb 成组解包，不处理补齐码。
// ABI 要求输入、权重、标度与输出互不重叠；restrict 让后端保留更激进的寄存器与向量调度。
typedef unsigned char u8;

void cutile_{{op}}_kernel(const float * restrict x, const u8 * restrict w,
                          const float * restrict scales, float * restrict y,
                          int M, int N, int gsz) {
    for (int n = 0; n < N; ++n) {
        const u8 *row = 0;
        @load(row, w + (long)n * (M / {{cpb}}));

        float acc = 0.0f;
        float deq[{{GMAX}}];

        /* cutile-unroll */
        for (int g = 0; g < M / gsz; ++g) {
            const float *xs = 0;
            const float *sc = scales + n * (M / gsz) + g;

            @load_async(xs, x + g * gsz);
            // 每组先求原始点积、最后只乘一次标度，避免逐元素重复乘 scale。
            const float unit = 1.0f;
            @quant(deq, row + g * (gsz / {{cpb}}), &unit, gsz, cpb: {{cpb}}, width: {{width}}, zero: {{zero}});
            float dot = 0.0f;
            @dot(dot, deq, xs, gsz);
            acc += dot * sc[0];
        }

        float v = 0.0f;
        float out = 0.0f;
        @cvt(v, acc, sat: off);
        @reduce(out, &v, 1, 1, op: sum, axis: GMAX, func: _cutile_sum_f32);
        @store(y, n, out);
    }
}
