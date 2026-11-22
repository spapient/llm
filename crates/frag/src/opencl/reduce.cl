/* cutile-sign: reduce/sub_group_reduce */
/* reduce: 统一数组入口（`$1`=数组、`$2`=长度、`$3`=步长），折叠函数由 `$func` 给定。
 * 长度双通道与遍历都在 preamble 的 `_cutile_*_f32` 里（`n == 1` 走标量快通道），片段只发一次
 * 调用——与 `cpu` / `nvidia` 同一句话，折叠语义只有一处。
 * 别退回 `($0) = ($1)`：那条只在"长度 = 1"时成立，`softmax` / `reduce_*` 的多元素归约会静默
 * 返回首元素且编译期不报错。子组归约要等 tile 落点（`sub_group_reduce_add` 还要求 OpenCL 的
 * 子组扩展打开），在那之前 `$func` 是唯一正确的宽度契约。 */
($0) = $func(($1), ($2), ($3));
