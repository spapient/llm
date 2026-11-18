/* cutile-sign: quant/unpack */
/* quant: 低位宽解包 + 按组缩放（$0=缓冲，$1=打包权重，$2=scale 指针，$3=组宽；$cpb/$width/$zero 由模板按位宽注入，i2/i4 共用一段，掩码由位宽派生）。 */
{ const float _s = *($2);
  const int _mask = (1 << $width) - 1;
  for (int _i = 0; _i < ($3) / $cpb; ++_i) {
      const u8 _b = ($1)[_i];
      for (int _j = 0; _j < $cpb; ++_j) {
          ($0)[$cpb * _i + _j] = (float)(((_b >> ($width * _j)) & _mask) - $zero) * _s;
      }
  } }
