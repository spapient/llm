/* cutile-sign: load_async/sync_load */
/* load_async: 同步加载。`cp.async` 只能把数据搬进 shared，本拼接点的目标是 GM 上的私有指针，
 * 没有 DMA 可发；于是绑指针 + 交还令牌，值当场到位——语义与异步等价，少的只是重叠。
 * 重叠要有东西才立得住（把 tile 搬进 shared 再算），那条路径是一条独立签名，与本片段并存而非替换。 */
$0 = $1;
