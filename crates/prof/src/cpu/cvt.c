/* cutile-sign: cvt/c_cast */
/* cvt: 累加器收敛到 f32。模板里 `@cvt(v, acc, sat: off)` 的 `$0` 是目的 `float`、
 * `$1` 是累加器——i8/i32 点积模板里是 `int`，f32 模板里已是 `float`。
 *
 * C 的 int→float 按标准就近舍入，标量单元编出来就是那一条转换指令，没有第二条更窄的路；
 * `char` 输入先符号展宽再转换，`float` 输入直通、不改变位模式。
 *
 * `$sat` 不进正文：夹到 [0,1] 会改掉可能为负的点积结果——档案因此只登记 `off`。 */
{
    const float _v = (float)($1);
    ($0) = _v;
}
