/* prim/intel/witness.cl —— UHD 750 每条 provide 的真机或主机参考见证。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），值与档案 expect 对拍。
 * OpenCL 后端定义 CUTILE_INTEL_DEVICE，强制走对应设备指令；host 自检保留同语义标量参考。
 * 一条签名一个函数，执行结论才能唯一归属；UHD 750 没有 XMX/DPAS，不保留虚假占位。
 */

/* dot/dp4a: 真机四分量整数 dot；host 自检用同语义标量参考。 */
int witness_dot_i8(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    const char4 a = (char4)(1, 2, 3, 4);
    const char4 b = (char4)(1, 1, 1, 1);
    out[0] = (float)dot(a, b);
#else
    const signed char a[4] = {1, 2, 3, 4};
    const signed char b[4] = {1, 1, 1, 1};
    int acc = 0;
    for (int k = 0; k < 4; ++k) acc += (int)a[k] * (int)b[k];
    out[0] = (float)acc;
#endif
    return 1;
}

/* dot/fma4: 与片段相同的四条独立 FMA 链和固定收口顺序。 */
int witness_dot(float *out) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
#ifdef CUTILE_INTEL_DEVICE
#define CUTILE_FMA fma
#else
#define CUTILE_FMA(x, y, acc) ((x) * (y) + (acc))
#endif
    float a0 = CUTILE_FMA(a[0], b[0], 0.0f);
    float a1 = CUTILE_FMA(a[1], b[1], 0.0f);
    float a2 = CUTILE_FMA(a[2], b[2], 0.0f);
    float a3 = CUTILE_FMA(a[3], b[3], 0.0f);
    out[0] = (a0 + a1) + (a2 + a3);
#undef CUTILE_FMA
    return 1;
}

/* quant/i4_unpack: 0xF0 解包 + scale=0.5 → {-4.0, 3.5}。 */
int witness_quant(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    const uchar packed = (uchar)0xF0u;
#else
    const unsigned char packed = 0xF0u;
#endif
    const float scale = 0.5f;
    out[0] = (float)((int)(packed & 0x0Fu) - 8) * scale;
    out[1] = (float)((int)((packed >> 4) & 0x0Fu) - 8) * scale;
    return 2;
}

/* load/bind: 指针等同性（同步语义）。 */
int witness_load(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async/prefetch: OpenCL 设备直接预取全局输出页，host 只验证同步绑定语义。 */
int witness_load_async(float *out) {
    const float *copy = out;
#ifdef CUTILE_INTEL_DEVICE
    prefetch((const __global float *)copy, 2);
#endif
    out[0] = (copy == out) ? 1.0f : 0.0f;
    return 1;
}

/* commit_wait/noop: P0 direct/pipe 没有在途异步搬运，无可等；见证返回名义 0.0。 */
int witness_commit_wait(float *out) {
    out[0] = 0.0f;
    return 1;
}

/* bar/noop: P0 direct/pipe 没有共享落点，无可挡；见证返回名义 0.0。 */
int witness_bar(float *out) {
    out[0] = 0.0f;
    return 1;
}

/* store/global_store: 执行器直接把 out 指向 GM，见证写的正是最终回读页。 */
int witness_store(float *out) {
    out[1] = 3.25f;
    out[0] = out[1];
    return 1;
}

/* cvt/convert_float: 设备用显式 OpenCL 转换，host 用同规则 C 转换。 */
int witness_cvt(float *out) {
    const int value = 3;
#ifdef CUTILE_INTEL_DEVICE
    out[0] = convert_float(value);
#else
    out[0] = (float)value;
#endif
    return 1;
}

/* reduce/scalar_loop: 多元素标量归约，禁止退回首元素。 */
int witness_reduce(float *out) {
    const float values[4] = {1.0f, 2.0f, 3.0f, 1.5f};
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) acc += values[i];
    out[0] = acc;
    return 1;
}

/* atomic/atomic_add: 真机对全局 atomic_int 发 2 + 3，host 保持标量参考。 */
int witness_atomic(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    volatile __global atomic_int *word = (volatile __global atomic_int *)out;
    atomic_init(word, 2);
    const int previous = atomic_fetch_add(word, 3);
    out[0] = (float)(previous + 3);
#else
    out[0] = 2.0f + 3.0f;
#endif
    return 1;
}

/* mma/scalar_tile：2×2 × 2×2（单位阵右侧，C = A）→ {1, 2, 3, 4}。 */
static void witness_mma_2x2(float *out) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float acc = 0.0f;
            for (int t = 0; t < 2; ++t) acc += a[i * 2 + t] * b[t * 2 + j];
            out[i * 2 + j] = acc;
        }
    }
}

int witness_mma_scalar(float *out) {
    witness_mma_2x2(out);
    return 4;
}

/* shuffle/sub_group_shuffle: 全子组交换，前四条 lane 回读 {4,3,2,1}。 */
int witness_shuffle(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    const uint lane = get_sub_group_local_id();
    const float value = (float)(lane + 1);
    const float reversed = intel_sub_group_shuffle(value, 3 - (lane & 3));
    if (lane < 4) out[lane] = reversed;
#else
    const float lane[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; ++i) out[i] = lane[3 - i];
#endif
    return 4;
}

/* broadcast/sub_group_broadcast: 全子组广播 lane 2 的 7。 */
int witness_broadcast(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    const uint lane = get_sub_group_local_id();
    const float value = lane == 2 ? 7.0f : (float)(lane + 1);
    const float broadcasted = sub_group_broadcast(value, 2);
    if (lane < 4) out[lane] = broadcasted;
#else
    for (int i = 0; i < 4; ++i) out[i] = 7.0f;
#endif
    return 4;
}

/* scan/sub_group_scan_inclusive_add: 全子组含前缀和 {1,3,6,10}。 */
int witness_scan(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    const uint lane = get_sub_group_local_id();
    const float scanned = sub_group_scan_inclusive_add((float)(lane + 1));
    if (lane < 4) out[lane] = scanned;
#else
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) { acc += (float)(i + 1); out[i] = acc; }
#endif
    return 4;
}

/* exp2/native_exp2: 真机 native_exp2，host 用整数次幂参考。 */
int witness_exp2(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    out[0] = native_exp2(3.0f);
    out[1] = native_exp2(-1.0f);
#else
    const float exponent[2] = {3.0f, -1.0f};
    for (int i = 0; i < 2; ++i) {
        float value = 1.0f;
        int steps = (int)exponent[i];
        const int negative = steps < 0;
        if (negative) steps = -steps;
        for (int k = 0; k < steps; ++k) value *= 2.0f;
        out[i] = negative ? 1.0f / value : value;
    }
#endif
    return 2;
}

/* rcp/native_recip + native_rsqrt: 真机原生倒数与平方根倒数。 */
int witness_rcp(float *out) {
#ifdef CUTILE_INTEL_DEVICE
    out[0] = native_recip(4.0f);
    out[1] = native_rsqrt(4.0f);
#else
    const float x = 4.0f;
    out[0] = 1.0f / x;
    float root = x;
    for (int k = 0; k < 8; ++k) root = 0.5f * (root + x / root);
    out[1] = 1.0f / root;
#endif
    return 2;
}
