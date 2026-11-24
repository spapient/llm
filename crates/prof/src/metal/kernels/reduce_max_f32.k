// cutile metal 微内核：reduce_max_f32 —— 最大值归约
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// reduce 骨架：整段数组交给 reduce 原语归约（长度 N、连续步长 1），最后写一个标量。复用 store 片段。
void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    float out = 0.0f;
    @reduce(out, a, N, 1, op: max, axis: GMAX, func: _cutile_max_f32);
    @store(y, 0, out);
}
