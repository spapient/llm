/* cutile-sign: dot/scalar_f32 */
/* dot: f32 顺序点积（lanes=1 的退化实现），$0 += dot($1, $2)，长度 $3 */
{ for (int _k = 0; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; } }
