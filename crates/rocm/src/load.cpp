/* cutile-sign: load/scalar_copy */
/* load: GM → 片上指针（`scalar_copy` 的地址语义）。
 *
 * P0 的 emit 是 scalar 单元的 C 语义：片段只做指针赋值，真正的全局读取发生在随后的
 * 下标访问里。与档案里选中的那条 `scalar_copy` 签名同构（`src = GM`、`dst = GM`）；
 * MTE 的 `copy_gm_to_l1` 落 `dst = FAST`，是另一条签名。 */
$0 = $1;
