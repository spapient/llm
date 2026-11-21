/* cutile-sign: bar/barrier */
/* bar: P0 的 emit 是单 work-item，组内没有需要同步的工作项；`barrier` 属 P3 的 tile 落点
 * （`barrier` 只能在 kernel 里调用，片段被拼进的模板不是 `__kernel`）。 */
(void)($0);
