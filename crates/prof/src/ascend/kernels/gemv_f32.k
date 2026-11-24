// cutile ascend 微内核：gemv_f32 —— 稠密 f32 投影
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 共享 ABI 由 ops/*.toml 固定；本文件是 ascend 的独立实现，可针对设备演进。
// 布局契约（由声明层校验）：w={{w}} x={{x}}
//
// 与 gemv_i4 的模板逐行对照：少了 `@quant` 与标度指针，多了把权重行本身搬进寄存器的那次
// `@load_async`。累加仍是"每块一个部分和，再加进总量"——与标量参考的分块顺序一致，因此
// 对拍比的是同一份舍入顺序，而不是运气。
//
// 尾部用 `len` 截断而不是 `M / gsz`：稠密权重没有"必须是分组整数倍"的约束，输入宽度
// 是 100 也应当能算，只是最后一块短一点。
void cutile_{{op}}_kernel(const float *x, const float *w, float *y,
                          int M, int N, int gsz) {
    for (int n = 0; n < N; ++n) {
        const float *row = 0;
        @load(row, w + (long)n * M);

        float acc = 0.0f;
        for (int base = 0; base < M; base += gsz) {
            const int len = (M - base < gsz) ? (M - base) : gsz;
            const float *xs = 0;
            const float *wp = 0;
            @load_async(xs, x + base);
            @load_async(wp, row + base);
            @dot(acc, wp, xs, len);
        }

        float v = 0.0f;
        float out = 0.0f;
        @cvt(v, acc, sat: off);
        @reduce(out, &v, 1, 1, op: sum, axis: GMAX, func: _cutile_sum_f32);
        @store(y, n, out);
    }
}
