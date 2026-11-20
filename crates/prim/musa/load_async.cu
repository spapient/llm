/* cutile-sign: load_async/cp.async+prefetch */
/* load_async: MUSA 的异步搬运同样只能落 shared（`musa::cp.async` 一脉），P0 的目标是 GM —— 退化为
 * 预取 + 普通加载。令牌照旧传回调用方（`$2` 是 `tok`），`commit_wait` 因此仍然配平。 */
$0 = $1; (void)($2);
