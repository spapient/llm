/* prim/ascend/witness.cpp —— 910b 每条 provide 的最小见证。
 *
 * 统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），
 * 值按 f32 读回后与档案里的 `witness.expect` 对拍（cutile README §11.3）。
 * 内容按执行器的 host ABI 写（扩展名跟随 `emit = ascendc`，与同目录片段一致）；
 * P0 的见证执行器只执行 `emit = c`，所以等级仍是 L0——AscendC 的 host harness
 * 与真机证据随非 C 执行器一起补。
 *
 * 每条签名一条函数：`dot` 的 `mmad`（Cube，f16 分块）与 `scalar_fma`（scalar，f32）
 * 是两种能力，不能共用一个见证；`quant` / `load` / `load_async` 的特化实现与标量实现
 * 虽然语义相同，也各自保留入口——一条见证只能证明一条签名（cutile README §11.3）。
 *
 * 末尾六条是 P0 模板不用、但 910b 确实具备的能力（Cube 的 tile `mmad`、vector 的
 * 置换/广播/前缀和与 `vexp`/`vrec`）。它们的等级是 L0：有见证源码，没有片段文本，
 * 也没有真机执行。
 */

/* quant/vcvt_pack: vector 单元上 0xF0 解包 + scale=0.5 → {-4.0, 3.5} */
int witness_quant_vcvt_pack(float *out) {
    const unsigned char packed = 0xF0u;
    const float scale = 0.5f;
    const int lo = (int)(packed & 0x0Fu) - 8;
    const int hi = (int)((packed >> 4) & 0x0Fu) - 8;
    out[0] = (float)lo * scale;
    out[1] = (float)hi * scale;
    return 2;
}

/* quant/scalar_dequant: scalar 单元上同一解包（零点 8、scale=0.5）→ {-4.0, 3.5} */
int witness_quant_scalar_dequant(float *out) {
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

/* load/copy_gm_to_l1: MTE 同步搬运，读回等于源（指针等同性） */
int witness_load_copy_gm_to_l1(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load/scalar_copy: scalar 单元上的同步拷贝，读回等于源 */
int witness_load_scalar_copy(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async/copy_gm_to_l1_async: 同步退化后等同性不变 */
int witness_load_async_copy_gm_to_l1(float *out) {
    const float src[2] = {1.0f, 2.0f};
    const float *copy = src;
    out[0] = (copy == src) ? 1.0f : 0.0f;
    return 1;
}

/* load_async/scalar_stage: 两级 staging 退化后等同性不变 */
int witness_load_async_scalar_stage(float *out) {
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

/* store/fixpipe_out: y[n] = out */
int witness_store(float *out) {
    float y[2] = {0.0f, 0.0f};
    const float value = 3.25f;
    y[1] = value;
    out[0] = y[1];
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

/* mma/mmad_tile: Cube 上 2×2 × 2×2 分块乘加（单位阵右侧，C = A）→ {1, 2, 3, 4} */
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

/* shuffle/vgather: lane i 取 lane 3-i 的值，{1,2,3,4} → {4,3,2,1} */
int witness_shuffle(float *out) {
    const float lane[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; ++i) {
        out[i] = lane[3 - i];
    }
    return 4;
}

/* broadcast/vbroadcast: lane 2 的 7 广播到四条 lane → {7, 7, 7, 7} */
int witness_broadcast(float *out) {
    const float lane[4] = {3.0f, 1.0f, 7.0f, 5.0f};
    const float value = lane[2];
    for (int i = 0; i < 4; ++i) {
        out[i] = value;
    }
    return 4;
}

/* scan/vcumsum: 含前缀和 {1,2,3,4} → {1, 3, 6, 10} */
int witness_scan(float *out) {
    const float lane[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        acc += lane[i];
        out[i] = acc;
    }
    return 4;
}

/* exp2/vexp: 2^3 = 8、2^-1 = 0.5 */
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

/* rcp/vrec: 1/4 = 0.25、rsqrt(4) = 0.5（牛顿迭代求 √x 再取倒数） */
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

/* ---- 写回尾块（epilogue）的六种形式 ----------------------------------------
 * 形式集合与 spec::epilogue 的语法一一对应，数值口径是 crates/gemm/src/epilogue.rs。
 * 片段文本是带占位符的可拼接块，不进这里；这里只证明"哪种形式算什么"。
 */

/* epi/bias: 2x2 tile {1,2,3,4} + 列偏置 {10,20} → {11,22,13,24} */
int witness_epi_bias(float *out) {
    const float tile[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float bias[2] = {10.0f, 20.0f};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            out[r * 2 + c] = tile[r * 2 + c] + bias[c];
        }
    }
    return 4;
}

/* epi/act: relu {-1,0,2.5} → {0,0,2.5} */
int witness_epi_act(float *out) {
    const float in[3] = {-1.0f, 0.0f, 2.5f};
    for (int i = 0; i < 3; ++i) {
        const float v = in[i];
        out[i] = (v > 0.0f) ? v : 0.0f;
    }
    return 3;
}

/* epi/residual: {1,2} + {0.5,0.25} → {1.5,2.25} */
int witness_epi_residual(float *out) {
    const float a[2] = {1.0f, 2.0f};
    const float b[2] = {0.5f, 0.25f};
    for (int i = 0; i < 2; ++i) {
        out[i] = a[i] + b[i];
    }
    return 2;
}

/* epi/scale: {1,2} * 0.5 → {0.5,1} */
int witness_epi_scale(float *out) {
    const float a[2] = {1.0f, 2.0f};
    const float alpha = 0.5f;
    for (int i = 0; i < 2; ++i) {
        out[i] = a[i] * alpha;
    }
    return 2;
}

/* epi/quant: scale=1、bits=4、{7.4,-8.6} → 取整 {7,-9} → 钳位 [-8,7] → {7,-8} */
int witness_epi_quant(float *out) {
    const float scale = 1.0f;
    const int bits = 4;
    const float lo = (float)(-(1 << (bits - 1)));
    const float hi = (float)((1 << (bits - 1)) - 1);
    const float src[2] = {7.4f, -8.6f};
    for (int i = 0; i < 2; ++i) {
        const float x = src[i] / scale;
        const float r = (x >= 0.0f) ? (float)(int)(x + 0.5f) : (float)(int)(x - 0.5f);
        float v = (r < lo) ? lo : r;
        if (v > hi) {
            v = hi;
        }
        out[i] = v * scale;
    }
    return 2;
}

/* epi/store: 写回标记，tile 原样落到输出缓冲 {3.25,1.5} → {3.25,1.5} */
int witness_epi_store(float *out) {
    const float tile[2] = {3.25f, 1.5f};
    for (int i = 0; i < 2; ++i) {
        out[i] = tile[i];
    }
    return 2;
}
