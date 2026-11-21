/* cutile-sign: reduce/vcreduce */
/* reduce: 统一数组入口 $1=数组、$2=长度、$3=步长；折叠函数由 $func 给定（软件标量接口，
 * 挂在设备 preamble）。长度双通道与遍历封装在 $func 里，片段只做一次调用，与 cpu / nvidia 同构。
 *
 * 早先这里是 `($0) = ($1)`：它只在"归约长度 = 1"时等价（gemv 的 `&v, 1, 1` 恰好如此），
 * 对 `softmax` / `reduce_*` / `rmsnorm` 的多元素归约会**静默返回首元素**，且编译期不报错。
 * `vcreduce` 是这条签名的设备署名：`vcadd` / `vcmax` 特化等 tile 落点再落在这里，
 * 不用"看起来更快"的写法换语义。 */
($0) = $func(($1), ($2), ($3));
