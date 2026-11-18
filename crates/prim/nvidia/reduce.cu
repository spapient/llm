/* cutile-sign: reduce/shfl.bfly */
/* reduce: `$op` along `$axis` —— 单线程下只有一个部分和，`shfl.bfly` 的宽度契约在这里退化。
 * `$1` 是数组入口（gemv 传 `&v`），单元素归约取 `*($1)`。 */
($0) = *($1);
