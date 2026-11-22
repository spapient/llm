/* cutile-sign: load_async/prefetch */
/* load_async: 把整行喊进缓存，令牌立即交还。OpenCL 的异步搬运 `async_work_group_copy` 只搬
 * GM ↔ SLM，本拼接点的目标是私有指针，没有 DMA 可发；能发的是 `prefetch`：不改语义、不阻塞，
 * 硬件不理会也只是退化成不预取。128 字节是 gen12 的一整条 L1 行宽。
 * 那一次地址空间收敛是本行的前提：模板形参未限定，clang 视为 `__private`，运行期它指向 GM，
 * `-cl-std=CL2.0`（档案声明的那一版）允许这次收敛，CL3.0 起不允许——换到 CL3.0 时这一句要
 * 删掉，而不是把签名改回 `async_work_group_copy` 去凑。 */
$0 = $1; prefetch((const __global float *)($1), 128);
