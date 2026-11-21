/* cutile-sign: bar/barrier */
/* bar: 组内屏障。OpenCL C 允许 `barrier` 出现在"被 kernel 调用"的函数里——模板正是把片段拼进
 * `void cutile_<op>_kernel`，所以这里直接发屏障，而不是当它不存在。
 * 两个围栏位各挡一件事：GLOBAL 侧保证上一轮读到的权重对同组工作项可见，LOCAL 侧保证落在 SLM
 * 的部分和已经写完。`$0` 是模板的令牌，`barrier` 没有令牌参数，照旧消费掉以保持模板侧配对。 */
barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
(void)($0);
