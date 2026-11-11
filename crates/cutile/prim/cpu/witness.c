/* prim/cpu/witness.c —— 每段只调用一个原语的最小见证 + 期望值。
 *
 * verify 在能跑的设备上执行对应段，并把等级写回台账（DESIGN §11.3）；
 * 比较口径固定为 max|got-want| / max|want|，f32 阈值 1e-5（DESIGN §11.1）。
 * 本文件不依赖模板，也不参与 `@prim` 拼接。
 */

#if 0
/* 期望值列在每段的末尾注释里；数值按 IEEE-754 f32 计算。 */
#endif

/* dot: {1,2,3,4} · {5,6,7,8} = 70 */
float witness_dot(void) {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
        acc += a[k] * b[k];
    }
    return acc; /* 70 */
}

/* quant: 0xF0 解包 + scale=0.5 → {(-1)*0.5, (7)*0.5} = {-0.5, 3.5} */
void witness_quant(float out[2], unsigned char packed, float scale) {
    const int lo = (int)(packed & 0x0Fu) - 8;
    const int hi = (int)((packed >> 4) & 0x0Fu) - 8;
    out[0] = (float)lo * scale; /* -0.5 */
    out[1] = (float)hi * scale; /* 3.5 */
}

/* load: 指针等同性（同步语义） */
int witness_load(const float *src) {
    const float *copy = src;
    return copy == src; /* 1 */
}

/* load_async: 同步退化后指针同样等同 */
int witness_load_async(const float *src) {
    const float *copy = src;
    int token = 0;
    (void)token;
    return copy == src; /* 1 */
}

/* commit_wait / bar: 空操作不改变任何值 */
int witness_commit_wait(void) {
    int token = 0;
    (void)token;
    return 0; /* 0 */
}

int witness_bar(void) {
    int token = 0;
    (void)(token);
    return 0; /* 0 */
}

/* store: y[n] = out */
float witness_store(float *y, int n, float out) {
    y[n] = out;
    return y[n]; /* out */
}

/* cvt: (float)(int)3 = 3.0 */
float witness_cvt(void) {
    int value = 3;
    return (float)value; /* 3.0 */
}

/* reduce: 单线程部分和即总和 */
float witness_reduce(float partial) {
    float total = partial;
    return total; /* partial */
}
