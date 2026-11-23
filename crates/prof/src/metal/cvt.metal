/* cutile-sign: cvt/float_cast */
/* cvt: 累加器收敛到 f32。模板里 `@cvt(v, acc, sat: off)` 的 `$0` 是目的 `float`、
 * `$1` 是累加器——i8/i32 点积模板里是 `int`，f32 模板里已是 `float`。
 *
 * MSL 的 `(float)x` 对三种输入是同一条重载：`int`/`char` → `float` 按 IEEE 就近舍入，
 * 有符号窄类型先符号展宽再转换；`float` 输入直通、不改变位模式。没有更窄的内建可走
 * （`as_type` 是位重解释，改的是语义，不是转换）。
 *
 * `$sat` 不进正文：`saturate()` 把结果夹到 [0,1]，而 `acc` 是可能为负的带符号点积，
 * 夹过之后与 CPU 参考不再逐位相同——档案因此只登记 `off` 这一侧。 */
{
    const float _v = (float)($1);
    ($0) = _v;
}
