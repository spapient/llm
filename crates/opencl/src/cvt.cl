/* cutile-sign: cvt/convert_float */
/* cvt: 累加器收敛（`sat = $sat`）；P0 只请求 `sat = off`，用 C 的隐式转换，
 * 饱和转换的 `convert_sat` 留给 P3。 */
($0) = (float)($1);
