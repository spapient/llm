/* cutile-sign: load/device_load */
/* load: `device` 地址空间 → 线程私有指针（`device_load` 的地址语义）。
 *
 * P0 的 emit 是单线程 C 语义：片段只做指针赋值，真正的全局读取发生在随后的下标访问里。
 * 与档案里选中的那条 `device_load` 签名同构（`src = GM`、`dst = GM`）。 */
$0 = $1;
