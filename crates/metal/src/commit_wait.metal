/* cutile-sign: commit_wait/noop */
/* commit_wait: 没有在飞的搬运可等。`load_async` 在 M5 上落成同步加载（没有 DMA 可发，见那一头），
 * 值在上一句就已经到位，所以等待是空操作——不是"留到 P3"，而是同步降级下等待的对象不存在。
 * 令牌照旧消费掉，让模板的 `@load_async … @commit_wait` 配对在任何设备上是同一形状。 */
(void)($0);
