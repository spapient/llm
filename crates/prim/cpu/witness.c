/* prim/cpu/witness.c —— 每段只调用一个原语的最小见证 + 期望值。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），
 * 值按 f32 读回后与档案里的 `witness.expect` 对拍（DESIGN §11.3）。
 * 比较口径固定为 max|got-want| / max|want|，f32 阈值 1e-5（DESIGN §11.1）。
 * 本文件不依赖模板，也不参与 `@prim` 拼接。
 *
 * 两个范围互不重叠、合起来覆盖全部声明：`grade ≥ L2` 的见证由**设备执行**（emit = c 时
 * 设备就是本机）跑，结论是 L2 证据；其余（`atomic` 与末尾四条 L0）由**本机自检**跑，
 * 结论只说明"源码可编译、期望值算得对"，不改写任何等级。
 */

/* dot: {1,2,3,4} · {5,6,7,8} = 70 */
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

/* quant: 0xF0 解包 + scale=0.5 → {(0-8)*0.5, (15-8)*0.5} = {-4.0, 3.5} */
int witness_quant(float *out) {
    const unsigned char packed = 0xF0u;
    const float scale = 0.5f;
    const int lo = (int)(packed & 0x0Fu) - 8;
    const int hi = (int)((packed >> 4) & 0x0Fu) - 8;
    out[0] = (float)lo * scale;
    out[1] = (float)hi * scale;
    return 2;
}

/* load: 指针等同性（同步语义） */
int witness_load(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async: 同步退化后指针同样等同 */
int witness_load_async(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    int token = 0;
    (void)token;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* commit_wait / bar: 空操作不改变任何值 */
int witness_commit_wait(float *out) {
    int token = 0;
    (void)token;
    out[0] = 0.0f;
    return 1;
}

int witness_bar(float *out) {
    int token = 0;
    (void)(token);
    out[0] = 0.0f;
    return 1;
}

/* store: y[n] = out */
int witness_store(float *out) {
    float y[2] = {0.0f, 0.0f};
    const float value = 3.25f;
    y[1] = value;
    out[0] = y[1];
    return 1;
}

/* cvt: (float)(int)3 = 3.0 */
int witness_cvt(float *out) {
    int value = 3;
    out[0] = (float)value;
    return 1;
}

/* reduce: 单线程部分和即总和 */
int witness_reduce(float *out) {
    const float partial = 7.5f;
    float total = partial;
    out[0] = total;
    return 1;
}

/* atomic/rmw_scalar: 单线程读改写，2.0 + 3.0 = 5.0（没有多线程原子语义） */
int witness_atomic(float *out) {
    float value = 2.0f;
    value += 3.0f;
    out[0] = value;
    return 1;
}

/* ── 闭集里 CPU 真正具备、但 P0 模板不调用的四条 ───────────────────────────
 *
 * 它们是 L0：有见证源码，没有片段文本，协商照样当作不具备。`shuffle` / `broadcast`
 * 不在此列——`[simt].group = 1` 的单 lane 设备没有可交换的对象，声明它只会造出一份
 * 自欺的证据。
 */

/* mma/scalar_tile: 2×2 分块乘加，右侧单位阵 → {1, 2, 3, 4} */
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

/* scan/prefix_loop: 含前缀和 {1,2,3,4} → {1, 3, 6, 10} */
int witness_scan(float *out) {
    const float lane[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        acc += lane[i];
        out[i] = acc;
    }
    return 4;
}

/* exp2/pow2_loop: 2^3 = 8、2^-1 = 0.5（重复加倍，不依赖 libm） */
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

/* rcp/recip_newton: 1/4 = 0.25、1/√4 = 0.5（牛顿迭代求 √x 再取倒数） */
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
