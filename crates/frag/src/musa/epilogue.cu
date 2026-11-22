/* cutile-sign: epilogue */
/* musa 写回尾块：六种形式各一段，装配按 spec::epilogue 的语法顺序拼接。
 *
 * 占位符契约与 nvidia 一致（$0 输出 tile/行主序，$1 第二操作数，$2 元素数或列数，
 * $3 本步参数）。mcc 的展开粒度比 nvcc 粗，这里给固定 4 路展开，避免模板再乘一遍。
 *
 * 数值口径以 crates/gemm/src/epilogue.rs 为准。
 */

/* epi: bias */
{ for (int _c = 0; _c < ($2); ++_c) {
      const float _b = ($1)[_c];
      for (int _r = 0; _r < ($3); ++_r) { ($0)[_r * ($2) + _c] += _b; }
  } }

/* epi: act */
{ int _i = 0;
  for (; _i + 3 < ($2); _i += 4) {
      { const float _v = ($0)[_i];     ($0)[_i]     = ($3); }
      { const float _v = ($0)[_i + 1]; ($0)[_i + 1] = ($3); }
      { const float _v = ($0)[_i + 2]; ($0)[_i + 2] = ($3); }
      { const float _v = ($0)[_i + 3]; ($0)[_i + 3] = ($3); }
  }
  for (; _i < ($2); ++_i) { const float _v = ($0)[_i]; ($0)[_i] = ($3); } }

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
      ($0)[_i] = fminf(fmaxf(_r, _lo), _hi) * _s;
  } }

/* epi: store */
{ (void)($0); }
