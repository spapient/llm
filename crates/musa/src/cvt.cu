/* cutile-sign: cvt/cvt.rn */
/* cvt: 累加器收敛。参数对是 i32 → f32：模板里 `@cvt(v, acc, sat: off)` 的 `v` 是 `float`、
 * `acc` 是 `int`。mcc 的 CUDA 前端把 C 的就近舍入 int→float 发成 `cvt.rn.f32.s32`，正文与
 * 签名指同一条指令。`$sat` 不进正文：`.sat` 落在浮点目的地上表示夹到 [0, 1]，而 `acc` 是可能
 * 为负的带符号点积，夹过之后与 CPU 参考不再逐位相同——档案因此只登记 `off`。 */
($0) = (float)($1);
