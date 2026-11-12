/* cutile-sign: reduce/simd_bfly */
/* reduce: `$op` along `$axis` —— 单线程下只有一个部分和，`simd_bfly`（`simd_shuffle_xor`）
 * 的宽度契约在这里退化。 */
($0) = ($1);
