/* cutile-sign: cvt/vcvt */
/* cvt: 累加器收敛（`sat = $sat`）；P0 只请求 `sat = off`，用 C 的隐式转换，
 * vector 单元的 `vcvt` 饱和模式留给 P3。 */
($0) = (float)($1);
