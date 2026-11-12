/* cutile-sign: load_async/async_threadgroup_copy */
/* load_async: `async_threadgroup_copy` 只能把数据搬进 `threadgroup`，P0 的目标是 GM ——
 * 退化为预取 + 普通加载。
 *
 * 令牌照旧传回调用方（`$2` 是 `tok`），`commit_wait` 因此仍然配平；真正的
 * `async_copy` 流水线（Metal 3 的双缓冲）是 P3 的落点。 */
$0 = $1; (void)($2);
