// cutile cpu 微内核：dequant_packed —— int2/int4 打包反量化（每字节 cpb 个 width 位码）
// 槽位：{{op}} 是算子名，{{cpb}}/{{width}}/{{zero}} 来自 [consts]；@prim(...) 由片段层替换。
// 布局契约（由声明层校验）：q={{q}}
// 前提：N 为 cpb 的倍数，循环按 N/cpb 成组解包；非整倍数宽度由调用侧前置校验拒绝。
// 低 width 位在前、高位在后；i2 与 i4 只差打包宽度（cpb=4/2、width=2/4、zero=2/8）。
// 标量参数框架只支持整数，故 scale/zero 写成与样本一致的常量。
void cutile_{{op}}_kernel(const unsigned char *q, float *y, int N) {
    const float scale = 0.25f;
    const float zero = {{zero}};
    // 循环头写成 `b < N` 而不是 `b < N/cpb`：网格注入只能绑「上界就是标量」的循环，上界是表达式时
    // 每线程各跑一整遍、同一段结果被重复解包。非字节线程（`b >= N/cpb`）在这里直接返回，
    // 语义与原来的 `b < N/cpb` 一致。
    for (int b = 0; b < N; ++b) {
        if (b >= N / {{cpb}}) {
            return;
        }
        const unsigned char *qp = 0;
        @load(qp, q + b);
        const unsigned char packed = qp[0];
        for (int r = 0; r < {{cpb}}; ++r) {
            @store(y, {{cpb}} * b + r, scale * ((float)((packed >> ({{width}} * r)) & ((1 << {{width}}) - 1)) - zero));
        }
    }
}
