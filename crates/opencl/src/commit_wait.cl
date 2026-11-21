/* cutile-sign: commit_wait/mem_fence */
/* commit_wait: 等一次搬运落地。OpenCL 里能等的只有 `wait_group_events`，而它只配平
 * `async_work_group_copy`——本拼接点没有发出过任何 DMA（见 `load_async`），于是同步点由内存
 * 围栏给出：GLOBAL 围栏让之前的读不被重排到之后的 `@dot` 之后，重排会把"预取到的值"换回旧值。
 * `$0` 是令牌，`mem_fence` 不接收令牌，消费掉它只是让模板的 `@load_async ... @commit_wait` 配对
 * 在两种设备上形状一致。 */
mem_fence(CLK_GLOBAL_MEM_FENCE);
(void)($0);
