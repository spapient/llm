/* cutile-sign: epilogue */
/* cpu 写回尾块：六种形式各一段，装配按 spec::epilogue 的语法顺序拼接。
 *
 * 占位符契约（$0 输出/行主序，$1 第二操作数，$2 元素数或列数，$3 本步参数），与其余设备一致。
 * 标量单元没有 warp 级归约，列偏置走行外层（每行整段连续，利于预取）。
 *
 * 数值口径以 crates/gemm/src/epilogue.rs 为准；这一段是它的可拼接化，不是另一套算法。
 *
 * 向量化口径：`bias`/`residual`/`scale`/`quant` 四步在 aarch64 上走 4-wide NEON，余量按标量收口。
 * 这四步彼此无依赖、逐元素独立，没有归约依赖链，所以要的是「一次搬够宽度」而不是 `dot` 那种
 * 多累加器破链——补的是向量宽度，不是 ILP 深度。`act` 的 `$3` 是注入的不透明表达式、`store`
 * 无副作用，两步都保持标量。加/乘与标量路径逐位同源，`quant` 的取整口径见该段注释。
 * 非 aarch64 全部退回标量，与 `dot.c` 同一套分路写法。
 */

/* epi: bias */
{ for (int _r = 0; _r < ($3); ++_r) {
      float *_o = ($0) + _r * ($2);
      const float *_b = ($1);
      int _c = 0;
#if defined(__aarch64__)
      for (; _c + 4 <= ($2); _c += 4) {
          vst1q_f32(_o + _c, vaddq_f32(vld1q_f32(_o + _c), vld1q_f32(_b + _c)));
      }
#endif
      for (; _c < ($2); ++_c) { _o[_c] += _b[_c]; }
  } }

/* epi: act */
/* `$3` 是按 ActKind 注入的表达式，操作数名 `_v`。表达式对这段代码不透明（可能是 `expf`／`erff`），
 * 所以这一形式保持标量：要向量化得先让装配层按形式给出向量体，那是契约层的改动，不在这里假装。 */
{ for (int _i = 0; _i < ($2); ++_i) { const float _v = ($0)[_i]; ($0)[_i] = ($3); } }

/* epi: residual */
{ float *_o = ($0); const float *_r = ($1); int _i = 0;
#if defined(__aarch64__)
  for (; _i + 4 <= ($2); _i += 4) {
      vst1q_f32(_o + _i, vaddq_f32(vld1q_f32(_o + _i), vld1q_f32(_r + _i)));
  }
#endif
  for (; _i < ($2); ++_i) { _o[_i] += _r[_i]; } }

/* epi: scale */
{ float *_o = ($0); const float _s = ($1); int _i = 0;
#if defined(__aarch64__)
  const float32x4_t _sv = vdupq_n_f32(_s);
  for (; _i + 4 <= ($2); _i += 4) {
      vst1q_f32(_o + _i, vmulq_f32(vld1q_f32(_o + _i), _sv));
  }
#endif
  for (; _i < ($2); ++_i) { _o[_i] *= _s; } }

/* epi: quant */
/* 量化往返：先除步长、按 f32::round（半数远离零）取整、按位宽钳位、再乘回。
 * 取整不用 `(int)(x ± 0.5f)`：`x + 0.5f` 本身要先按 f32 舍入，在半整数邻域会把
 * `0.49999997` 抬成 1，与 `f32::round` 不符；且超出 int 范围时是未定义行为。
 * 标量尾部因此改用 `roundf`（C99 与 `f32::round` 同口径），NEON 用 `vcvtaq_s32_f32`
 * （FCVTAS，同样半数远离零，越界饱和），钳位放在整数域，与先钳 f32 再乘回等价。 */
{ const float _s = ($1);
  const int _bits = ($3);
  float *_o = ($0);
  int _i = 0;
#if defined(__aarch64__)
  const int32x4_t _lo = vdupq_n_s32(-(1 << (_bits - 1)));
  const int32x4_t _hi = vdupq_n_s32((1 << (_bits - 1)) - 1);
  const float32x4_t _sv = vdupq_n_f32(_s);
  for (; _i + 4 <= ($2); _i += 4) {
      int32x4_t _q = vcvtaq_s32_f32(vdivq_f32(vld1q_f32(_o + _i), _sv));
      _q = vmaxq_s32(vminq_s32(_q, _hi), _lo);
      vst1q_f32(_o + _i, vmulq_f32(vcvtq_f32_s32(_q), _sv));
  }
#endif
  const float _flo = (float)(-(1 << (_bits - 1)));
  const float _fhi = (float)((1 << (_bits - 1)) - 1);
  for (; _i < ($2); ++_i) {
      float _v = roundf(_o[_i] / _s);
      _v = (_v < _flo) ? _flo : _v;
      if (_v > _fhi) { _v = _fhi; }
      _o[_i] = _v * _s;
  } }

/* epi: store */
/* 写回标记：输出缓冲就是 `$0`，这一步只标注装配顺序。 */
{ (void)($0); }
