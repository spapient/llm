// cutile 模板：dequant_i4 —— 4 位反量化（打包字节，每字节两个 nibble）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}}
// 前提：N 为偶数，循环按 N/2 成对解包；奇数宽度由调用侧前置校验拒绝。
//
// 低 4 位在前、高 4 位在后；标量参数框架只支持整数，故 scale/zero 写成与样本一致的常量。
void cutile_{{op}}_kernel(const unsigned char *q, float *y, int N) {
    const float scale = 0.25f;
    const float zero = 8.0f;
    for (int b = 0; b < N / 2; ++b) {
        const unsigned char *qp = 0;
        @load(qp, q + b);
        const unsigned char packed = qp[0];
        @store(y, 2 * b, scale * ((float)(packed & 0x0F) - zero));
        @store(y, 2 * b + 1, scale * ((float)(packed >> 4) - zero));
    }
}
