/* cutile-sign: quant/unpack */
/* quant: 低位宽解包 + 按组缩放（$0=缓冲，$1=打包权重，$2=scale 指针，$3=组宽；$cpb/$width/$zero
 * 由模板按位宽注入，i2/i4 共用一段，掩码由位宽派生）。
 *
 * 与 nvidia 版的差别只有一处：**内层按 4 位一组整体展开**。`cpb` 在 i4 下是 2、i2 下是 4，
 * 展开后每字节只算一次 `>> / &` 的公共部分，编译器还能把 `(float)` 转换合并成一条 `cvt`；
 * 语义（含 `zero` 偏移与 scale 乘序）与另外两个后端逐位一致，`witness_quant` 对拍的就是这一点。 */
{ const float _s = *($2);
  const int _mask = (1 << $width) - 1;
  for (int _i = 0; _i < ($3) / $cpb; ++_i) {
      const u8 _b = ($1)[_i];
      for (int _j = 0; _j < $cpb; ++_j) {
          ($0)[$cpb * _i + _j] = (float)(((_b >> ($width * _j)) & _mask) - $zero) * _s;
      }
  } }
