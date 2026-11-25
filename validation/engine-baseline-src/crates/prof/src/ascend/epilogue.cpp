/* cutile-sign: epilogue */
/* ascend（AscendC / rocm emit）写回尾块：六种形式各一段，装配按 prof::epilogue 的语法顺序拼接。
 *
 * 占位符契约与其余设备一致（$0 输出/行主序，$1 第二操作数，$2 元素数或列数，$3 本步参数）。
 * AscendC 侧只有 C++ 前端可用，钳位走 `<math.h>` 的 `fminf`/`fmaxf`。
 *
 * 数值口径以 crates/prof/src/epilogue.rs 为准。
 */

/* epi: bias */
{ for (int _c = 0; _c < ($2); ++_c) {
      const float _b = ($1)[_c];
      for (int _r = 0; _r < ($3); ++_r) { ($0)[_r * ($2) + _c] += _b; }
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
  const float _lo = static_cast<float>(-(1 << (_bits - 1)));
  const float _hi = static_cast<float>((1 << (_bits - 1)) - 1);
  for (int _i = 0; _i < ($2); ++_i) {
      const float _x = ($0)[_i] / _s;
      const float _r = (_x >= 0.0f) ? static_cast<float>(static_cast<int>(_x + 0.5f))
                                    : static_cast<float>(static_cast<int>(_x - 0.5f));
      ($0)[_i] = fminf(fmaxf(_r, _lo), _hi) * _s;
  } }

/* epi: store */
{ static_cast<void>($0); }
