/* cutile-sign: dot/fma */
/* dot: f32 顺序点积（`lanes = 1`），`$0 += dot($1, $2)`，长度 `$3`。
 *
 * 档案里 `dot/fma` 这条签名的承诺是「标量 f32 累加」，于是正文必须与 `nvidia`/`metal`/`intel`
 * 的同一签名逐字同源（`same_primitive_bodies_are_isomorphic_across_vendors`）。S4000 的 FMA
 * 延迟可以用四链 ILP 摊开，但那改变了求和次序，属于另一条签名——要先在档案里另立一条并提供
 * 对应见证，而不是让同名片段在两张卡上算出两份数。
 *
 * `dp4a`（i8×4 → i32）是另一条签名，见同目录的 `dot` 整数段与 `witness.cu`。 */
{ for (int _k = 0; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; } }
