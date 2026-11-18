// cutile 模板：dequant_i2 —— 2 位反量化（打包字节，每字节四个 2 位码）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}}
// 前提：N 为 4 的倍数，循环按 N/4 成四解包；非 4 倍数宽度由调用侧前置校验拒绝。
//
// 低 2 位在前、高位在后；标量参数框架只支持整数，故 scale/zero 写成与样本一致的常量。
void cutile_{{op}}_kernel(const unsigned char *q, float *y, int N) {
    const float scale = 0.25f;
    const float zero = 2.0f;
    for (int b = 0; b < N / 4; ++b) {
        const unsigned char *qp = 0;
        @load(qp, q + b);
        const unsigned char packed = qp[0];
        @store(y, 4 * b, scale * ((float)(packed & 0x03) - zero));
        @store(y, 4 * b + 1, scale * ((float)((packed >> 2) & 0x03) - zero));
        @store(y, 4 * b + 2, scale * ((float)((packed >> 4) & 0x03) - zero));
        @store(y, 4 * b + 3, scale * ((float)(packed >> 6) - zero));
    }
}
