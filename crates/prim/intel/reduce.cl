/* cutile-sign: reduce/sub_group_reduce */
/* reduce: 统一数组入口 $1=数组、$2=长度、$3=步长；折叠函数由 $func 给定（软件标量接口，
 * 挂在设备 preamble）。长度双通道与遍历封装在 $func 里，片段只做一次调用，与 cpu / nvidia 同构。
 *
 * 早先这里是 `($0) = ($1)`：`sub_group_reduce` 的宽度契约在单 work-item 下退化，这句话只在
 * "归约长度 = 1"时成立（gemv 的 `&v, 1, 1` 恰好如此），对 `softmax` / `reduce_*` 的多元素归约
 * 会**静默返回首元素**且编译期不报错。子组归约等 P3 的 tile 落点再用 `$func` 之外的签名表达。 */
($0) = $func(($1), ($2), ($3));
