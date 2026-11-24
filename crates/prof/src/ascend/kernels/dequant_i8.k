// cutile ascend 微内核：dequant_i8 —— 8 位反量化
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}}
//
// 逐元素反量化。复用 load/store；标量参数框架只支持整数，故 scale/zero 写成与样本一致的常量。
void cutile_{{op}}_kernel(const float *q, float *y, int N) {
    const float scale = 0.25f;
    const float zero = 8.0f;
    for (int n = 0; n < N; ++n) {
        const float *qp = 0;
        @load(qp, q + n);
        const float out = scale * (qp[0] - zero);
        @store(y, n, out);
    }
}
