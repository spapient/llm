/* cutile-sign: reduce/shfl.bfly */
/* reduce: 统一数组入口 $1=数组、$2=长度、$3=步长；折叠函数由 $func 给定（软件标量接口，
 * 挂在设备 preamble）。长度双通道与遍历封装在 $func 里，片段只做一次调用，与 cpu 同构。 */
($0) = $func(($1), ($2), ($3));
