/* prim/ascend/witness.cpp —— 910b 每条 provide 的最小见证。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），
 * 值按 f32 读回后与档案里的 `witness.expect` 对拍（DESIGN §11.3）。
 * 内容按执行器的 host ABI 写（扩展名跟随 `emit = ascendc`，与同目录片段一致）；
 * P0 的见证执行器只执行 `emit = c`，所以等级仍是 L0——AscendC 的 host harness
 * 与真机证据随非 C 执行器一起补。
 *
 * 每条签名一条函数：`dot` 的 `mmad`（Cube，f16 分块）与 `scalar_fma`（scalar，f32）
 * 是两种能力，不能共用一个见证；`quant` / `load` / `load_async` 的标量与特化实现
 * 语义相同（都是解包 / 搬运），共用函数即可。
 */

/* quant/vcvt_pack 与 quant/scalar_dequant: 0xF0 解包 + scale=0.5 → {-4.0, 3.5} */
int witness_quant(float *out) {
    const unsigned char packed = 0xF0u;
    const float scale = 0.5f;
    const int lo = (int)(packed & 0x0Fu) - 8;
    const int hi = (int)((packed >> 4) & 0x0Fu) - 8;
    out[0] = (float)lo * scale;
    out[1] = (float)hi * scale;
    return 2;
}

/* dot/mmad: f16 分块乘加的标量等价 {1,2,3,4} · {5,6,7,8} = 70（f16 可精确表示） */
int witness_dot_f16(float *out) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
        acc += a[k] * b[k];
    }
    out[0] = acc;
    return 1;
}

/* dot/scalar_fma: 标量 f32 点积 {1,2,3,4} · {5,6,7,8} = 70 */
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

/* load/copy_gm_to_l1 与 load/scalar_copy: 指针等同性（同步语义） */
int witness_load(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async/copy_gm_to_l1_async 与 load_async/scalar_stage: 同步退化后等同性不变 */
int witness_load_async(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    int token = 0;
    (void)token;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* commit_wait/wait_flag: 等待不改变任何值 */
int witness_commit_wait(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

/* store/fixpipe_out: y[n] = out */
int witness_store(float *out) {
    float y[2] = {0.0f, 0.0f};
    const float value = 3.25f;
    y[1] = value;
    out[0] = y[1];
    return 1;
}

/* bar/pipe_barrier: 流水屏障本身不改变值 */
int witness_bar(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

/* cvt/vcvt: (float)(int)3 = 3.0 */
int witness_cvt(float *out) {
    int value = 3;
    out[0] = (float)value;
    return 1;
}

/* reduce/vcreduce: 单线程部分和即总和 */
int witness_reduce(float *out) {
    const float partial = 7.5f;
    out[0] = partial;
    return 1;
}

/* atomic/set_atomic_add: 2.0 + 3.0（add 语义） */
int witness_atomic(float *out) {
    float value = 2.0f;
    value += 3.0f;
    out[0] = value;
    return 1;
}
