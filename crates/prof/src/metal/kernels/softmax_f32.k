// cutile metal 微内核：softmax_f32 —— Softmax
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}}
//
// 三趟：max 归约交 @reduce；exp 求和与归一化走 preamble 软件向量 exp 接口
// （NEON 4-wide，标量回退），模板不再逐元素循环调数学库 expf。

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    float max = -3.402823466e38f;
    @reduce(max, x, N, 1, op: max, axis: GMAX, func: _cutile_max_f32);
    float sum = _cutile_exp_sum_f32(x, N, 1, max);
    _cutile_exp_norm_f32(y, x, N, 1, max, sum);
}
