/* cutile-sign: quant/i4_unpack */
/* quant: 低位宽解包 + 按组缩放（`$0`=缓冲，`$1`=打包权重，`$2`=scale，`$3`=组宽）。
 * 位宽不在正文里写死：`$cpb` 字节一包、每元素 `$width` 位、零点 `$zero`，掩码从位宽派生——
 * 与 `cpu` 同一段代码覆盖 i2/i4，加一种位宽不需要新片段。
 * 整数减零点先做，浮点只留一次乘：解包是整数管线，`* _s` 是唯一一条浮点乘，别把零点也搬进
 * 浮点侧（那要多一条 `convert` 和一次减法）。 */
{ const float _s = *($2);
  const int _mask = (1 << $width) - 1;
  const int _packed = ($3) / $cpb;
  for (int _i = 0; _i < _packed; ++_i) {
      const u8 _b = ($1)[_i];
      for (int _j = 0; _j < $cpb; ++_j) {
          ($0)[$cpb * _i + _j] = (float)(((_b >> ($width * _j)) & _mask) - $zero) * _s;
      }
  } }
