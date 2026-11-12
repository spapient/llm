/* cutile-sign: quant/i4_unpack */
/* quant: int4 解包 + 按组缩放（`i4_unpack` 的语义，$0=缓冲，$1=打包权重，$2=scale 指针，$3=组宽）。
 *
 * P0 只请求 `group = 32`、`space = REG|GM` 的标量版本；`.cl` 里它是 OpenCL C 的逐元素循环。 */
{ const float _s = *($2);
  for (int _i = 0; _i < ($3) / 2; ++_i) {
      const u8 _b = ($1)[_i];
      ($0)[2 * _i]     = (float)((int)(_b & 0x0Fu) - 8) * _s;
      ($0)[2 * _i + 1] = (float)((int)((_b >> 4) & 0x0Fu) - 8) * _s;
  } }
