/* cutile-sign: load_async/scalar_stage */
/* load_async: `copy_gm_to_l1_async` 只搬 GM → L1，P0 的目标是 GM —— 退化为预取 + 普通加载。
 *
 * 令牌照旧传回调用方（`$2` 是 `tok`），`commit_wait` 因此仍然配平；真正的
 * `wait_flag` 流水是 P3 的落点。 */
$0 = $1; (void)($2);
