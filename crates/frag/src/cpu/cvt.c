/* cutile-sign: cvt/c_cast */
/* cvt: 累加器收敛。参数对是 i32 → f32：模板里 `@cvt(v, acc, sat: off)` 的 `v` 是 `float`、
 * `acc` 是 `int`。C 的 int→float 按标准就近舍入，标量单元编出来就是那一条转换指令，没有第二
 * 条更窄的路。`$sat` 不进正文：夹到 [0, 1] 会改掉可能为负的点积结果——档案因此只登记 `off`。 */
($0) = (float)($1);
