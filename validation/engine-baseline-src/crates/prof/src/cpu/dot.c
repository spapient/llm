/* cutile-sign: dot/scalar_f32 */
/* dot: f32 点积，$0 += dot($1, $2)，长度 $3。
   aarch64 走 4 个 NEON 累加器、16-wide 步进，打破单条 FMA 延迟链；余下元素按
   4-wide 再收口，尾部标量补齐。其余架构回退顺序标量循环。
   块内求和顺序与标量参考不同（向量分路累加），数值以算子内核对拍为准（f32 阈值 1e-5）。 */
#if defined(__aarch64__)
{
    float32x4_t _a0 = vdupq_n_f32(0.0f);
    float32x4_t _a1 = vdupq_n_f32(0.0f);
    float32x4_t _a2 = vdupq_n_f32(0.0f);
    float32x4_t _a3 = vdupq_n_f32(0.0f);
    int _k = 0;
    for (; _k + 16 <= ($3); _k += 16) {
        _a0 = vmlaq_f32(_a0, vld1q_f32(($1) + _k),      vld1q_f32(($2) + _k));
        _a1 = vmlaq_f32(_a1, vld1q_f32(($1) + _k + 4),  vld1q_f32(($2) + _k + 4));
        _a2 = vmlaq_f32(_a2, vld1q_f32(($1) + _k + 8),  vld1q_f32(($2) + _k + 8));
        _a3 = vmlaq_f32(_a3, vld1q_f32(($1) + _k + 12), vld1q_f32(($2) + _k + 12));
    }
    _a0 = vaddq_f32(vaddq_f32(_a0, _a1), vaddq_f32(_a2, _a3));
    for (; _k + 4 <= ($3); _k += 4) {
        _a0 = vmlaq_f32(_a0, vld1q_f32(($1) + _k), vld1q_f32(($2) + _k));
    }
    float32x2_t _lo = vget_low_f32(_a0);
    float32x2_t _hi = vget_high_f32(_a0);
    _lo = vadd_f32(_lo, _hi);
    _lo = vpadd_f32(_lo, _lo);
    ($0) += vget_lane_f32(_lo, 0);
    for (; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; }
}
#else
{ for (int _k = 0; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; } }
#endif
