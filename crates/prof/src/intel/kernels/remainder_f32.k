// cutile intel 微内核：remainder_f32 —— 逐元素 fmod（截断商余数）。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：a={{a}}  b={{b}}
//
// 语义对齐 nn 参考闭包 `x % y`：`y = x - trunc(x / v)·v`，商向零截断，结果符号同被除数
// （C99 fmod / Rust f32 Rem 口径）。取整走后端 preamble 的 `_cutile_trunc_f32`，与一元
// `trunc_f32` 同一可移植接缝；不在模板里裸写 floor / copysign，避免负商非整数时与参考分歧。
void cutile_{{op}}_kernel(const float *a, const float *b, float *y, int N) {
    for (int n = 0; n < N; ++n) {
        const float *ap = 0;
        const float *bp = 0;
        @load(ap, a + n);
        @load(bp, b + n);
        const float x = ap[0];
        const float v = bp[0];
        const float out = x - _cutile_trunc_f32(x / v) * v;
        @store(y, n, out);
    }
}
