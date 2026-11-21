/* cutile-sign: bar/noop */
/* bar: 不发屏障是定论，不是欠账。P0 的 emit 里没有 `__local` 缓冲、也没有 `get_local_id`——`[grid]
 * dims` 把 `n` 绑到工作项下标，每个工作项只写自己的 `y[n]`，读的全是 GM 上的 `x` / `row`，组内没有
 * 共享量：`barrier` 要挡的那件事在这里不存在，发它就是给 K 轴每一块加一个同步点外加一次 GM 围栏，
 * 换不到任何正确性。要它的是把 tile 搬进 SLM 的那条路径，另立签名。 */
(void)($0);
