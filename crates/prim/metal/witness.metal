/* prim/metal/witness.metal —— M5 Pro 每条 provide 的最小见证。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），
 * 值按 f32 读回后与档案里的 `witness.expect` 对拍（DESIGN §11.3）。
 * 比较口径固定为 max|got-want| / max|want|，f32 阈值 1e-5（DESIGN §11.1）。
 *
 * 这些函数是**证据**而不是片段：不参与 `@prim` 拼接，也不进 kernel。见证的 ABI 是 host C，
 * 与片段语言无关，所以本机自检用 `cc` 逐条执行；`cutile` 的执行后端目前只有 C，`emit = metal`
 * 的档案得不到设备执行，下面的等级因此仍是 L1——文件先把"证据是什么"写成数据，等 Metal
 * 执行后端接入后再升级。
 *
 * 一条签名一个函数：见证回答"这条签名写的是什么算法"，两条签名共用一份见证时，
 * 执行通过的结论无法归属到哪一条（DESIGN §11.3）。
 *
 * 末尾六条是 P0 模板不用、但 M 系列 GPU 确实具备的能力（simdgroup 矩阵单元、lane 交换/广播/
 * 扫描、SFU 的 `exp2`/`rcp`）。它们的等级是 L0：有见证源码，没有片段文本，也没有设备执行。
 */

/* dot/simd_dp4a: 4 个 i8 分量乘加（dp4a 的语义），累加器是 i32 → 10.0 */
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

/* dot/simd_fma: 标量 f32 点积 {1,2,3,4} · {5,6,7,8} = 70 */
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

/* quant/unpack_i4: 0xF0 解包 + scale=0.5 → {(0-8)*0.5, (15-8)*0.5} = {-4.0, 3.5} */
int witness_quant(float *out) {
    const unsigned char packed = 0xF0u;
    const float scale = 0.5f;
    const int lo = (int)(packed & 0x0Fu) - 8;
    const int hi = (int)((packed >> 4) & 0x0Fu) - 8;
    out[0] = (float)lo * scale;
    out[1] = (float)hi * scale;
    return 2;
}

/* load/device_load: 指针等同性（同步语义） */
int witness_load(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async/async_threadgroup_copy: 目标是 GM 时退化为预取 + 普通加载，等同性不变 */
int witness_load_async(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    int token = 0;
    (void)token;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* commit_wait/async_copy_commit: 等待不改变任何值 */
int witness_commit_wait(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

/* store/device_store: y[n] = out */
int witness_store(float *out) {
    float y[2] = {0.0f, 0.0f};
    const float value = 3.25f;
    y[1] = value;
    out[0] = y[1];
    return 1;
}

/* bar/simdgroup_barrier: 屏障本身不改变值 */
int witness_bar(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

/* cvt/saturate_convert: (float)(int)3 = 3.0 */
int witness_cvt(float *out) {
    int value = 3;
    out[0] = (float)value;
    return 1;
}

/* reduce/simd_bfly: 单线程部分和即总和 */
int witness_reduce(float *out) {
    const float partial = 7.5f;
    out[0] = partial;
    return 1;
}

/* atomic/atomic_fetch: 2.0 + 3.0（add 语义） */
int witness_atomic(float *out) {
    float value = 2.0f;
    value += 3.0f;
    out[0] = value;
    return 1;
}

/* mma/simdgroup_matrix: 2×2 × 2×2 分块乘加（单位阵右侧，C = A）→ {1, 2, 3, 4} */
int witness_mma(float *out) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float acc = 0.0f;
            for (int t = 0; t < 2; ++t) {
                acc += a[i * 2 + t] * b[t * 2 + j];
            }
            out[i * 2 + j] = acc;
        }
    }
    return 4;
}

/* shuffle/simd_shuffle: lane i 取 lane 3-i 的值，{1,2,3,4} → {4,3,2,1} */
int witness_shuffle(float *out) {
    const float lane[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; ++i) {
        out[i] = lane[3 - i];
    }
    return 4;
}

/* broadcast/simd_broadcast: lane 2 的 7 广播到四条 lane → {7, 7, 7, 7} */
int witness_broadcast(float *out) {
    const float lane[4] = {3.0f, 1.0f, 7.0f, 5.0f};
    const float value = lane[2];
    for (int i = 0; i < 4; ++i) {
        out[i] = value;
    }
    return 4;
}

/* scan/simd_prefix_sum: 含前缀和 {1,2,3,4} → {1, 3, 6, 10} */
int witness_scan(float *out) {
    const float lane[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        acc += lane[i];
        out[i] = acc;
    }
    return 4;
}

/* exp2/exp2_fast: 2^3 = 8、2^-1 = 0.5 */
int witness_exp2(float *out) {
    const float exponent[2] = {3.0f, -1.0f};
    for (int i = 0; i < 2; ++i) {
        float value = 1.0f;
        int steps = (int)exponent[i];
        const int negative = steps < 0;
        if (negative) {
            steps = -steps;
        }
        for (int k = 0; k < steps; ++k) {
            value *= 2.0f;
        }
        out[i] = negative ? 1.0f / value : value;
    }
    return 2;
}

/* rcp/rcp_fast: 1/4 = 0.25、rsqrt(4) = 0.5（牛顿迭代求 √x 再取倒数） */
int witness_rcp(float *out) {
    const float x = 4.0f;
    out[0] = 1.0f / x;
    float root = x;
    for (int k = 0; k < 8; ++k) {
        root = 0.5f * (root + x / root);
    }
    out[1] = 1.0f / root;
    return 2;
}
