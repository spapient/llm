/* cutile-sign: bar/noop */
/* bar: 不发屏障是定论，不是欠账。P0 的 emit 是"一行一线程"——`[grid] dims` 把 `n` 绑到线程下标，
 * 每个线程只写自己的 `y[n]`，读的全是 GM 上的 `x` / `row`，没有 `threadgroup` 落点，也没有跨线程共享
 * 量：`simdgroup_barrier` 要挡的那件事在这里不存在。真发它只会给 K 轴每一块加一个同步点，换不到任何
 * 正确性。要它的是把 tile 搬进 `threadgroup` 的那条路径，另立签名。 */
(void)($0);
