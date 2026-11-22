/* cutile-sign: load/ld.global */
/* load: GM → 寄存器指针（`ld.global` 的地址语义）。
 *
 * P0 的 emit 是单线程 C 语义：片段只做指针赋值，真正的全局读取发生在随后的下标访问里。
 * 与档案里选中的那条 `ld.global` 签名同构（`src = GM`、`dst = GM`）。 */
$0 = $1;
