/* cutile-sign: load/vload */
/* load: GM → 私有指针（`vload` 的地址语义）。
 *
 * P0 的 emit 是单 work-item 的 C 语义：片段只做指针赋值，真正的全局读取发生在随后的
 * 下标访问里。与档案里选中的那条 `vload` 签名同构（`src = GM`、`dst = GM`）。 */
$0 = $1;
