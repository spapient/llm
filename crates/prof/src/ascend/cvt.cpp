/* cutile-sign: cvt/vcvt */
/* cvt: 累加器收敛。参数对是 i32 → f32：模板里 `@cvt(v, acc, sat: off)` 的 `v` 是 `float`、
 * `acc` 是 `int`。ccec 在 `-xcce` 下把这条就地就近舍入的转换直接落到向量单元的类型转换上；
 * 正文不走 `AscendC::Cast` 是因为那条路要 `LocalTensor` + `TPipe`，与模板的裸指针 ABI 不兼容。
 * `$sat` 不进正文：饱和落在浮点目的地上是夹到 [0, 1]，而 `acc` 是可能为负的带符号点积，
 * 夹过之后与 CPU 参考不再逐位相同——档案因此只登记 `off`。 */
($0) = (float)($1);
