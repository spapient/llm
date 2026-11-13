// cutile 模板：rope_f32 —— 旋转位置编码
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}}
//
// 每对分量按位置旋转。复用 load/store，sinf/cosf/powf 来自数学库。
#include <math.h>

void cutile_{{op}}_kernel(const float *x, float *y, int N, int pos) {
    for (int i = 0; i < N / 2; ++i) {
        const float *ep = 0;
        const float *op = 0;
        @load(ep, x + 2 * i);
        @load(op, x + 2 * i + 1);
        const float freq = powf(10000.0f, -(float)(2 * i) / (float)N);
        const float angle = (float)pos * freq;
        const float s = sinf(angle);
        const float c = cosf(angle);
        const float e = ep[0];
        const float odd = op[0];
        @store(y, 2 * i, e * c - odd * s);
        @store(y, 2 * i + 1, e * s + odd * c);
    }
    if (N % 2 == 1) {
        const float *tail = 0;
        @load(tail, x + N - 1);
        @store(y, N - 1, tail[0]);
    }
}
