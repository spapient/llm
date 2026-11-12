/* cutile-sign: cvt/saturate_convert */
/* cvt: 累加器收敛（`sat = $sat`）；P0 只请求 `sat = off`，用 C 的隐式转换，
 * 饱和 f32 → i8 的 `saturate` 留给 P3。 */
($0) = (float)($1);
