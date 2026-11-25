/* cutile-sign: cvt/cvt.rn */
/* cvt: 累加器收敛到 f32。模板里 `@cvt(v, acc, sat: off)` 的 `$0` 是目的 `float`、
 * `$1` 是累加器——i8/i32 点积模板里是 `int`，f32 模板里已是 `float`。
 *
 * C 的 int→float 按标准就近舍入，nvcc 对它发的就是 `cvt.rn.f32.s32`，正文与签名指同一条
 * 指令；`char` 输入先符号展宽再转换，`float` 输入直通、不改变位模式。
 *
 * `$sat` 不进正文：`.sat` 落在浮点目的地上表示夹到 [0,1]，而 `acc` 是可能为负的带符号
 * 点积，夹过之后与 CPU 参考不再逐位相同——档案因此只登记 `off`。 */
{
    const float _v = (float)($1);
    ($0) = _v;
}
