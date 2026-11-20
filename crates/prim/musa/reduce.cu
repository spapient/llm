/* cutile-sign: reduce/shfl.bfly */
/* reduce: 统一数组入口 $1=数组、$2=长度、$3=步长；折叠函数由 $func 给定（软件标量接口，
 * 挂在设备 preamble），片段只做一次调用，与 cpu / nvidia 同构。
 *
 * MUSA 的 wave 是 **128 lane**（S4000 实测 `warpSize = 128`），但这一族要求的折叠是
 * **一次有序标量求和**：把 128 lane 的 `__shfl_down_sync` 蝶形塞进来的话，单线程 emit 下
 * 只有 1 个 lane 有数据，其余 127 个 lane 是空的，结果不变、代价照付。因此片段保持标量入口，
 * 波内归约等 P3 的 tile 落点再用 `$func` 之外的签名表达，不在这里用"看起来更快"的写法换语义。 */
($0) = $func(($1), ($2), ($3));
