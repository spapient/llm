/* cutile-sign: commit_wait/noop */
/* commit_wait: 没有在飞的搬运可等。能等的只有 `wait_group_events`，而它只配平
 * `async_work_group_copy`——本拼接点没有发出过任何 DMA（见 `load_async`：目标是私有指针），
 * 值在上一句就已经到位，所以等待是空操作。不加 `mem_fence` 也是同一条理由：没有第二个工作项
 * 会看到这里的写，围栏挡的可见性/重排都不涉及。令牌照旧消费掉，让模板的
 * `@load_async … @commit_wait` 配对在任何设备上是同一形状。 */
(void)($0);
