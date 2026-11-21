/* cutile-sign: cvt/convert_float */
/* cvt: 累加器收敛。用 `convert_float` 而不是 C 的隐式转换：它是 OpenCL C 的显式转换，舍入方向
 * 由标准钉死（两侧同为 f32 时是恒等），签名与正文于是指同一条指令。
 * `sat = $sat` 不在正文里分支：饱和只在整型目标上有可饱和的端点，源与目标都是 f32 时
 * `convert_float_sat` 与 `convert_float` 逐位相同，两条设置走同一行才是正确的那一行。 */
($0) = convert_float(($1));
