/* cutile-sign: cvt/vcvt */
/* cvt: 累加器收敛到 f32。模板里 `@cvt(v, acc, sat: off)` 的 `$0` 是目的 `float`、
 * `$1` 是累加器——i8/i32 点积模板里是 `int`，f32 模板里已是 `float`。
 *
 * ccec 在 `-xcce` 下把这条就地就近舍入的转换直接落到向量单元的类型转换上；正文不走
 * `AscendC::Cast` 是因为那条路要 `LocalTensor` + `TPipe`，与模板的裸指针 ABI 不兼容。
 * `char` 输入先符号展宽再转换，`float` 输入直通、不改变位模式。
 *
 * `$sat` 不进正文：饱和落在浮点目的地上是夹到 [0,1]，而 `acc` 是可能为负的带符号点积，
 * 夹过之后与 CPU 参考不再逐位相同——档案因此只登记 `off`。 */
{
    const float _v = (float)($1);
    ($0) = _v;
}
