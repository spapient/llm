// cutile ascend 微内核：rope_f32 —— 旋转位置编码
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}}
//
// 每对分量按位置旋转。复用 load/store，sin/cos/pow 走 preamble 软件标量接口。

void cutile_{{op}}_kernel(const float *x, float *y, int N, int pos) {
    // 循环头写成 `i < N` 而不是 `i < N/2`：网格注入只能绑「上界就是标量」的循环，上界是表达式时
    // 每线程各跑一整遍、结果被写坏。守卫把多出来的线程（同一次循环里）直接送回，语义与 `i < N/2` 一致。
    for (int i = 0; i < N; ++i) {
        if (2 * i + 1 >= N) {
            // 奇数长度时最后一个元素原样拷贝，且只由 `2*i == N-1` 的那一个线程做；其余线程什么都不做。
            if (2 * i == N - 1) {
                const float *tail = 0;
                @load(tail, x + N - 1);
                @store(y, N - 1, tail[0]);
            }
            return;
        }
        const float *ep = 0;
        const float *op = 0;
        @load(ep, x + 2 * i);
        @load(op, x + 2 * i + 1);
        const float freq = _cutile_pow_f32(10000.0f, -(float)(2 * i) / (float)N);
        const float angle = (float)pos * freq;
        const float s = _cutile_sin_f32(angle);
        const float c = _cutile_cos_f32(angle);
        const float e = ep[0];
        const float odd = op[0];
        @store(y, 2 * i, e * c - odd * s);
        @store(y, 2 * i + 1, e * s + odd * c);
    }
}
