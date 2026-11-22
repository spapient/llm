/* cutile-sign: epilogue */
/* cpu 写回尾块：六种形式各一段，装配按 spec::epilogue 的语法顺序拼接。
 *
 * 占位符契约（$0 输出/行主序，$1 第二操作数，$2 元素数或列数，$3 本步参数），与其余设备一致。
 * 标量单元没有 warp 级归约，列偏置走行外层（每行整段连续，利于预取）。
 *
 * 数值口径以 crates/gemm/src/epilogue.rs 为准；这一段是它的可拼接化，不是另一套算法。
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
/* 量化往返：先除 SCALE、按 f32::round（半数远离零）取整、按位宽钳位、再乘回。 */
{ const float _s = ($1);
  const int _bits = ($3);
  const float _lo = (float)(-(1 << (_bits - 1)));
  const float _hi = (float)((1 << (_bits - 1)) - 1);
  for (int _i = 0; _i < ($2); ++_i) {
      const float _x = ($0)[_i] / _s;
      const float _r = (_x >= 0.0f) ? (float)(int)(_x + 0.5f) : (float)(int)(_x - 0.5f);
      float _v = (_r < _lo) ? _lo : _r;
      if (_v > _hi) { _v = _hi; }
      ($0)[_i] = _v * _s;
  } }

/* epi: store */
/* 写回标记：输出缓冲就是 `$0`，这一步只标注装配顺序。 */
{ (void)($0); }
