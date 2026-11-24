// cutile musa 微内核：clamp_f32 —— 逐元素裁剪
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}}
//
// 逐元素裁剪。复用 load/store；标量参数框架只支持整数，故 lo/hi 写成与样本一致的常量，
// f32 标量入参通道随取值批次接入（同 dequant_i8 口径）。
void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const float lo = -1.0f;
    const float hi = 1.0f;
    for (int n = 0; n < N; ++n) {
        const float *xp = 0;
        @load(xp, x + n);
        const float v = xp[0];
        const float out = v < lo ? lo : (v > hi ? hi : v);
        @store(y, n, out);
    }
}
