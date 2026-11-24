// cutile nvidia 微内核：reduce_prod_f32 —— 连乘归约。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// reduce 骨架：整段数组折叠成一个标量，最后写一个标量。连乘没有设备侧 `_cutile_prod_f32`
// 原语（与求和/求最大不同），这里先按标量折叠落成档案结构；并行归约原语列入后端 PARTIAL，
// 不假装已有设备内核。语义与 nn::reduce_prod_f32 的标量参考对拍一致。
void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    float out = 1.0f;
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);
        out *= ap[0];
    }
    @store(y, 0, out);
}
