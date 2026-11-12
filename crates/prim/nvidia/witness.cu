/* prim/nvidia/witness.cu —— sm100 每条 provide 的最小见证。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），
 * 值按 f32 读回后与档案里的 `witness.expect` 对拍（DESIGN §11.3）。
 * 比较口径固定为 max|got-want| / max|want|，f32 阈值 1e-5（DESIGN §11.1）。
 *
 * 这些函数是**证据**而不是片段：不参与 `@prim` 拼接，也不进 kernel。
 * 本机没有 CUDA 设备，P0 的见证执行器只执行 `emit = c` 的档案，所以下面的
 * 等级仍是 L1——文件先把"证据是什么"写成数据，真机与工具链到位后再执行。
 */

/* dot/dp4a: 4 个 i8 分量乘加（dp4a 的语义），累加器是 i32 → 10.0 */
int witness_dot_i8(float *out) {
    const signed char a[4] = {1, 2, 3, 4};
    const signed char b[4] = {1, 1, 1, 1};
    int acc = 0;
    for (int k = 0; k < 4; ++k) {
        acc += (int)a[k] * (int)b[k];
    }
    out[0] = (float)acc;
    return 1;
}

/* dot/fma: 标量 f32 点积 {1,2,3,4} · {5,6,7,8} = 70 */
int witness_dot(float *out) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
        acc += a[k] * b[k];
    }
    out[0] = acc;
    return 1;
}

/* quant/lop3_i4: 0xF0 解包 + scale=0.5 → {(0-8)*0.5, (15-8)*0.5} = {-4.0, 3.5} */
int witness_quant(float *out) {
    const unsigned char packed = 0xF0u;
    const float scale = 0.5f;
    const int lo = (int)(packed & 0x0Fu) - 8;
    const int hi = (int)((packed >> 4) & 0x0Fu) - 8;
    out[0] = (float)lo * scale;
    out[1] = (float)hi * scale;
    return 2;
}

/* load/ld.global: 指针等同性（同步语义） */
int witness_load(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async/cp.async+prefetch: 目标是 GM 时退化为预取 + 普通加载，等同性不变 */
int witness_load_async(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    int token = 0;
    (void)token;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* commit_wait/cp.async.commit_group: 等待不改变任何值 */
int witness_commit_wait(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

/* store/st.global: y[n] = out */
int witness_store(float *out) {
    float y[2] = {0.0f, 0.0f};
    const float value = 3.25f;
    y[1] = value;
    out[0] = y[1];
    return 1;
}

/* bar/bar.sync: 屏障本身不改变值 */
int witness_bar(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

/* cvt/cvt.rn: (float)(int)3 = 3.0 */
int witness_cvt(float *out) {
    int value = 3;
    out[0] = (float)value;
    return 1;
}

/* reduce/shfl.bfly: 单线程部分和即总和 */
int witness_reduce(float *out) {
    const float partial = 7.5f;
    out[0] = partial;
    return 1;
}

/* atomic/red.global: 2.0 + 3.0（add 语义） */
int witness_atomic(float *out) {
    float value = 2.0f;
    value += 3.0f;
    out[0] = value;
    return 1;
}
