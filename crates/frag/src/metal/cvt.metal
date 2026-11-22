/* cutile-sign: cvt/float_cast */
/* cvt: 累加器收敛。参数对是 i32 → f32：模板里 `@cvt(v, acc, sat: off)` 的 `v` 是 `float`、
 * `acc` 是 `int`。MSL 的 int→float 转换同样按就近舍入，没有更窄的一条内建可用（`as_type` 是
 * 位重解释，改语义）。`$sat` 不进正文：`saturate()` 把结果夹到 [0, 1]，而 `acc` 是可能为负的
 * 带符号点积，夹过之后与 CPU 参考不再逐位相同——档案因此只登记 `off`。 */
($0) = (float)($1);
