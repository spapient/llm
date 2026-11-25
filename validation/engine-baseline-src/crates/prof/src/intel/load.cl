/* cutile-sign: load/bind */
/* load: GM → 寄存器指针（`bind` 的地址语义），`src = GM`、`dst = GM`。
 *
 * P0 的 emit 是单线程 C 语义：片段只做指针赋值，模板紧接着用 `row + base` 继续寻址，真正的
 * 全局读取发生在随后的下标访问里——先搬到别处再读一遍只是多一趟。
 * 签名写 `bind` 而不是 `vload`：`vloadN` 要求地址空间已知，本模板的形参未限定（clang 视为
 * `__private`），在这里调它只能凑出"看起来用了向量指令"的代码。 */
$0 = $1;
