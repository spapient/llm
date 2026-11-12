/* cutile-sign: load_async/async_work_group_copy+prefetch */
/* load_async: `async_work_group_copy` 只搬 GM ↔ local，P0 的目标是 GM —— 退化为预取 + 普通加载。
 *
 * 令牌照旧传回调用方（`$2` 是 `tok`），`commit_wait` 因此仍然配平；真正的
 * `wait_group_events` 是 P3 的流水线落点。 */
$0 = $1; (void)($2);
