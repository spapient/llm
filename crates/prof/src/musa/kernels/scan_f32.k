// cutile 模板：scan_f32 —— β 指数梯形扫描
// 槽位：{{op}} 是算子名，{{a}}/{{b}} 是模板常量（离散对角衰减与输入系数）。
//
// scan 骨架：沿时间轴串行递推 s = a·s + b·x[t]，扫描没有可借用的跨元素归约原语，
// 正文就是一条串行循环；衰减与输入系数在声明层 [consts] 里固定，内核不调数学库。

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        s = (float)({{a}}) * s + (float)({{b}}) * x[t];
        y[t] = s;
    }
}
