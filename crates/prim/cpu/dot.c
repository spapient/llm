/* cutile-sign: dot/scalar_f32 */
/* dot: f32 点积，$0 += dot($1, $2)，长度 $3。
   aarch64 走 NEON 4 路向量累加，余下元素标量收口；其余架构回退顺序标量循环。
   块内求和顺序与标量参考不同（向量分路累加），数值以算子内核对拍为准（f32 阈值 1e-5）。 */
#if defined(__aarch64__)
{
    float32x4_t _v = vdupq_n_f32(0.0f);
    int _k = 0;
    for (; _k + 4 <= ($3); _k += 4) {
        _v = vmlaq_f32(_v, vld1q_f32(($1) + _k), vld1q_f32(($2) + _k));
    }
    float32x2_t _lo = vget_low_f32(_v);
    float32x2_t _hi = vget_high_f32(_v);
    _lo = vadd_f32(_lo, _hi);
    _lo = vpadd_f32(_lo, _lo);
    ($0) += vget_lane_f32(_lo, 0);
    for (; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; }
}
#else
{ for (int _k = 0; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; } }
#endif
