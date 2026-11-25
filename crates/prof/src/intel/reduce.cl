/* cutile-sign: reduce/scalar_loop */
/* reduce: 统一数组入口 $1=数组, $2=长度, $3=步长；折叠函数由 $func 给定。
   长度双通道（标量快通道 / 数组归约通道）与遍历都在 preamble 独立接口内，片段只做一次调用。 */
($0) = $func(($1), ($2), ($3));
