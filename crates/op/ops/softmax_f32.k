// cutile 模板：softmax_f32 —— Softmax
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}}
//
// 三趟：max、exp 求和、归一化。max 归约交 @reduce 原语；exp 求和与归一化保留逐元素趟。
// 复用 load/store，expf 来自数学库。
#include <math.h>

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    float max = -3.402823466e38f;
    @reduce(max, x, N, 1, op: max, axis: GMAX, func: _cutile_max_f32);
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float *xp = 0;
        @load(xp, x + n);
        sum += expf(xp[0] - max);
    }
    for (int n = 0; n < N; ++n) {
        const float *xp = 0;
        @load(xp, x + n);
        const float out = expf(xp[0] - max) / sum;
        @store(y, n, out);
    }
}
