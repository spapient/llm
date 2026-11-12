/* cutile-sign: quant/unpack_i4 */
/* quant: int4 解包 + 按组缩放（`unpack_i4` 的语义，$0=缓冲，$1=打包权重，$2=scale 指针，$3=组宽）。
 *
 * P0 只请求 `group = 32`、`space = REG|GM` 的标量版本；`.metal` 里它是纯 C 语义，
 * 低位宽解包交给 Metal 编译器做向量化。 */
{ const float _s = *($2);
  for (int _i = 0; _i < ($3) / 2; ++_i) {
      const u8 _b = ($1)[_i];
      ($0)[2 * _i]     = (float)((int)(_b & 0x0Fu) - 8) * _s;
      ($0)[2 * _i + 1] = (float)((int)((_b >> 4) & 0x0Fu) - 8) * _s;
  } }
