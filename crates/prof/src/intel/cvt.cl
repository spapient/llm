/* cutile-sign: cvt/convert_float */
/* cvt: 累加器收敛。参数对是 i32 → f32：模板里 `@cvt(v, acc, sat: off)` 的 `v` 是 `float`、
 * `acc` 是 `int`。正文写 `convert_float` 而不是 C 的隐式转换：它是 OpenCL C 的显式转换，舍入
 * 方向由标准钉死（int→float 就近舍入，正是不表态时的默认），签名与正文于是指同一条指令。
 * `$sat` 不进正文：`convert_float_sat` 在浮点目的地上把结果夹到 [0, 1]，而 `acc` 是可能为负的
 * 带符号点积，夹过之后与 CPU 参考不再逐位相同——档案只登记 `off` 这一侧。 */
($0) = convert_float(($1));
