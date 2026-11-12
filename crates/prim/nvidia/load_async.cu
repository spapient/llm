/* cutile-sign: load_async/cp.async+prefetch */
/* load_async: `cp.async` 只能把数据搬进 shared，P0 的目标是 GM —— 退化为预取 + 普通加载。
 *
 * 令牌照旧传回调用方（`$2` 是 `tok`），`commit_wait` 因此仍然配平；真正的
 * `cp.async.commit_group` 是 P3 的流水线落点。 */
$0 = $1; (void)($2);
