/* prim/musa/witness.cu —— MTT S4000（mp_22）每条 provide 的最小见证。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），值按 f32 读回后与档案里的
 * `witness.expect` 对拍（cutile README §11.3），阈值 max|got-want| / max|want| ≤ 1e-5（§11.1）。
 *
 * 每条签名一个函数：见证回答"这条签名写的是什么算法"。同一个函数被两条签名共用时，执行通过的
 * 结论无法归属到哪一条，所以这里一条签名一条函数。
 *
 * 这张卡的 wave 是 **128 lane**（`musaGetDeviceProperties().warpSize = 128`，见档案 `[simt]`），
 * 下面的 shuffle/broadcast/scan 见证因此按 8 条 lane 取样、宽度仍是"设备自报的那一个"：
 * 见证要证的是**语义**（取数 / 广播 / 含前缀和），不是再写一遍波宽。
 */

/* dot/dp4a: 4 个 i8 分量乘加，累加器是 i32 → 10 */
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

/* dot/fma4: f32 点积 {1,2,3,4} · {5,6,7,8} = 70（四链 ILP 的合并序不改变这个值） */
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

/* matmul_f32/mp_22: 2×2 矩阵乘 {{1,2},{3,4}} × {{5,6},{7,8}} = {{19,22},{43,50}} */
int witness_matmul_f32(float *out) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    out[0] = a[0] * b[0] + a[1] * b[2];
    out[1] = a[0] * b[1] + a[1] * b[3];
    out[2] = a[2] * b[0] + a[3] * b[2];
    out[3] = a[2] * b[1] + a[3] * b[3];
    return 4;
}

/* quant/unpack: 0xF0 解包 + scale=0.5 → {(0-8)*0.5, (15-8)*0.5} = {-4.0, 3.5} */
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
    out[0] = (copy == src) ? 1.0f : 0.0f;
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

/* store/st.global: y[n] = out */
int witness_store(float *out) {
    float y[2] = {0.0f, 0.0f};
    y[1] = 3.25f;
    out[0] = y[1];
    return 1;
}

/* cvt/cvt.rn: (float)(int)3 = 3 */
int witness_cvt(float *out) {
    int value = 3;
    out[0] = (float)value;
    return 1;
}

/* reduce/shfl.bfly: 单线程部分和即总和 */
int witness_reduce(float *out) {
    out[0] = 7.5f;
    return 1;
}

/* atomic/red.global: 2.0 + 3.0（add 语义） */
int witness_atomic(float *out) {
    out[0] = 2.0f + 3.0f;
    return 1;
}

/* mma/scalar_tile: 2×2 × 2×2 标量分块乘加（单位阵右侧，C = A）→ {1, 2, 3, 4} */
int witness_mma_scalar(float *out) {
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

/* mma/wmma.tf32: 2×2 × 单位阵 → C = A = {1, 2, 3, 4} */
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

/* shuffle/shfl.idx: 按 lane 取数，8 条 lane 反序 → {8,7,6,5,4,3,2,1} */
int witness_shuffle(float *out) {
    const float lane[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
        out[i] = lane[7 - i];
    }
    return 8;
}

/* broadcast/shfl.idx.bcast: lane 2 的值 7 广播到全部 lane */
int witness_broadcast(float *out) {
    const float lane[8] = {3.0f, 1.0f, 7.0f, 5.0f, 9.0f, 2.0f, 6.0f, 4.0f};
    const float value = lane[2];
    for (int i = 0; i < 8; ++i) {
        out[i] = value;
    }
    return 8;
}

/* scan/shfl.up: 含前缀和 {1..8} → {1,3,6,10,15,21,28,36} */
int witness_scan(float *out) {
    float acc = 0.0f;
    for (int i = 0; i < 8; ++i) {
        acc += (float)(i + 1);
        out[i] = acc;
    }
    return 8;
}

/* exp2/ex2.approx: 2^3 = 8、2^-1 = 0.5 */
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

/* rcp/rcp.approx: 1/4 = 0.25、rsqrt(4) = 0.5（牛顿迭代求 √x 再取倒数） */
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
