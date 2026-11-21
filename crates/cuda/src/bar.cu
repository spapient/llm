/* cutile-sign: bar/noop */
/* bar: 不发屏障是定论，不是欠账。P0 的 emit 是"一行一线程"——`[grid] dims` 把 `n` 绑到线程下标，
 * 块内每个线程只写自己的 `y[n]`，读的全是 GM 上的 `x` / `row`，没有 shared 落点，也没有跨线程共享
 * 量：屏障要挡的那件事在这里不存在。真发 `__syncthreads()` 只会给 K 轴每一块加一个同步点，换不到
 * 任何正确性。GEMM 要它，是因为它把 tile 搬进了 shared；那条路径带自己的 `bar.sync` 签名。 */
(void)($0);
