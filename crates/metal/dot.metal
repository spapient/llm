/* cutile-sign: dot/simd_fma */
/* dot: f32 顺序点积（`lanes = 1`），`$0 += dot($1, $2)`，长度 `$3`。
 *
 * 与档案里 `dot` 的 `simd_fma` 签名同构；`simd_dp4a`（i8×4 → i32）是另一条签名，
 * 属 P3 的 tile 落点。 */
{ for (int _k = 0; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; } }
