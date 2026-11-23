/* cutile-sign: epilogue */
/* metal 写回尾块：六种形式各一段，装配按 prof::epilogue 的语法顺序拼接。
 *
 * 占位符契约与其余设备一致（$0 输出/行主序，$1 第二操作数，$2 元素数或列数，$3 本步参数）。
 * 量化钳位写成显式分支而非 `metal::clamp`：后者只在内核里有定义，而尾块要能被宿主编译器
 * 原样编译并执行，才能与 `gemm` 的口径对拍。
 *
 * 数值口径以 crates/prof/src/epilogue.rs 为准。
 */

/* epi: bias */
{ for (int _r = 0; _r < ($3); ++_r) {
      const int _base = _r * ($2);
      for (int _c = 0; _c < ($2); ++_c) { ($0)[_base + _c] += ($1)[_c]; }
  } }

/* epi: act */
/* `$3` 是按 ActKind 注入的表达式，操作数名 `_v`。 */
{ for (int _i = 0; _i < ($2); ++_i) { const float _v = ($0)[_i]; ($0)[_i] = ($3); } }

/* epi: residual */
{ for (int _i = 0; _i < ($2); ++_i) { ($0)[_i] += ($1)[_i]; } }

/* epi: scale */
{ for (int _i = 0; _i < ($2); ++_i) { ($0)[_i] *= ($1); } }

/* epi: quant */
{ const float _s = ($1);
  const int _bits = ($3);
  const float _lo = (float)(-(1 << (_bits - 1)));
  const float _hi = (float)((1 << (_bits - 1)) - 1);
  for (int _i = 0; _i < ($2); ++_i) {
      const float _x = ($0)[_i] / _s;
      const float _r = (_x >= 0.0f) ? (float)(int)(_x + 0.5f) : (float)(int)(_x - 0.5f);
      ($0)[_i] = ((_r < _lo) ? _lo : ((_r > _hi) ? _hi : _r)) * _s;
  } }

/* epi: store */
{ (void)($0); }
