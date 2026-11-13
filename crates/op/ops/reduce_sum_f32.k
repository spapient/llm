// cutile 模板：reduce_sum_f32 —— 求和归约
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// reduce 骨架：循环累加，最后写一个标量。复用 load/store 片段。
void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);
        acc += ap[0];
    }
    @store(y, 0, acc);
}
