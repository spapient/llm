/* cutile-sign: bar/simdgroup_barrier */
/* bar: P0 的 emit 是单线程，simdgroup 内没有需要同步的 lane；`simdgroup_barrier` 属 P3 的落点。 */
(void)($0);
