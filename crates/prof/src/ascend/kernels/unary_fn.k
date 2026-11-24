// cutile ascend 微内核：unary_fn —— 逐元素一元函数的共享内核（abs / neg / square 等只差一个右值表达式）。
// 槽位：{{op}} 是算子名，{{unexpr}} 来自 [consts]，作用于读入的 x；@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}
//
// 一元骨架：读一个数、算 {{unexpr}}、写一个数。表达式只用到寄存器里的 x，不依赖设备数学库，
// 跨设备语义与 nn::unary 的标量核由执行对拍钉住。
void cutile_{{op}}_kernel(const float *a, float *y, int N) {
    #pragma unroll 4
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        @load(ap, a + n);

        const float x = ap[0];
        const float out = {{unexpr}};
        @store(y, n, out);
    }
}
