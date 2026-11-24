// cutile 共享参考微内核：scale_f32 —— 标量缩放
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 逐元素乘常量。复用 load/store。
void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    const float alpha = 0.5f;
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);
        @store(y, n, alpha * ap[0]);
    }
}
