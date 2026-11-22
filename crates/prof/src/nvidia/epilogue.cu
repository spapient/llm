/* cutile-sign: epilogue */
/* nvidia 写回尾块：六种形式各一段，装配按 prof::epilogue 的语法顺序拼接。
 *
 * 占位符契约（六台设备一致，由模板在拼接时替换）：
 *   $0 输出 tile（f32，行主序）  $2 元素数（`bias` 段是列数）
 *   $1 本步第二个操作数          $3 本步参数（`bias`=行数 / `act`=激活表达式 / `quant`=位宽）
 *
 * 数值口径以 crates/gemm/src/epilogue.rs 为准：列偏置按列广播，残差同形相加，缩放乘标量，
 * 量化先除 SCALE、按 f32::round（半数远离零）取整、按位宽钳位再乘回，写回只是标记
 * （寄存器 tile 就是输出）。`quant` 的钳位边界由位宽派生，i2/i4/i8 共用这一段。
 */

/* epi: bias */
/* 列偏置：列在外层，偏置值一次 load 摊到整列。 */
{ for (int _c = 0; _c < ($2); ++_c) {
      const float _b = ($1)[_c];
      for (int _r = 0; _r < ($3); ++_r) { ($0)[_r * ($2) + _c] += _b; }
  } }

/* epi: act */
/* 逐元素激活：`$3` 是按 ActKind 注入的表达式，操作数名 `_v`。 */
{ for (int _i = 0; _i < ($2); ++_i) {
      const float _v = ($0)[_i];
      ($0)[_i] = ($3);
  } }

/* epi: residual */
/* 残差：同形逐元素相加。 */
{ for (int _i = 0; _i < ($2); ++_i) { ($0)[_i] += ($1)[_i]; } }

/* epi: scale */
/* 标量缩放。 */
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
/* 写回标记：寄存器 tile 已经在 `$0`，这一步只标注装配顺序。 */
{ (void)($0); }
