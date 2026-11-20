/* cutile-sign: bar/bar.sync */
/* bar: P0 的 emit 是单线程，块内没有需要同步的线程；S4000 的 wave 是 128 lane，波内同步属 P3 的 tile
 * 落点（届时用 `__syncthreads()`，与 `bar.sync` 一一对应）。 */
(void)($0);
