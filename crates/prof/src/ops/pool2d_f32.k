// cutile 共享微内核：固定 / 自适应二维池化，NHWC，批与通道均可配置。
// MODE: 0=平均（有效格），1=最大，2=自适应平均；几何约束在发射前拒绝。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
#if {{H}} < 1 || {{W}} < 1 || {{OH}} < 1 || {{OW}} < 1 || {{C}} < 1 || {{BATCH}} < 1
#error "cutile pool2d dimensions must be positive"
#endif
#if {{MODE}} == 2
#if {{OH}} > {{H}} || {{OW}} > {{W}}
#error "cutile adaptive pool2d output cannot exceed input"
#endif
#else
#if {{KH}} < 1 || {{KW}} < 1 || {{SH}} < 1 || {{SW}} < 1 || {{PH}} < 0 || {{PW}} < 0
#error "cutile pool2d kernel/stride/padding is invalid"
#endif
#if {{OH}} != (({{H}} + 2 * {{PH}} - {{KH}}) / {{SH}} + 1) || {{OW}} != (({{W}} + 2 * {{PW}} - {{KW}}) / {{SW}} + 1)
#error "cutile pool2d output geometry is inconsistent"
#endif
#endif

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const int H = {{H}}, W = {{W}}, OH = {{OH}}, OW = {{OW}};
    const int C = {{C}};
    (void)N;
    for (int b = 0; b < {{BATCH}}; ++b) {
        for (int i = 0; i < OH; ++i) {
#if {{MODE}} == 2
            const int h0 = i * H / OH;
            const int h1 = ((i + 1) * H + OH - 1) / OH;
#else
            const int h0 = i * {{SH}} - {{PH}};
            const int h1 = h0 + {{KH}};
#endif
            for (int j = 0; j < OW; ++j) {
#if {{MODE}} == 2
                const int w0 = j * W / OW;
                const int w1 = ((j + 1) * W + OW - 1) / OW;
#else
                const int w0 = j * {{SW}} - {{PW}};
                const int w1 = w0 + {{KW}};
#endif
                for (int c = 0; c < C; ++c) {
#if {{MODE}} == 1
                    float acc = -3.402823466e38f;
#else
                    float acc = 0.0f;
#endif
                    int count = 0;
                    for (int ii = h0; ii < h1; ++ii) {
                        if (ii < 0 || ii >= H) continue;
                        for (int jj = w0; jj < w1; ++jj) {
                            if (jj < 0 || jj >= W) continue;
                            const float value = x[((b * H + ii) * W + jj) * C + c];
#if {{MODE}} == 1
                            if (value > acc) acc = value;
#else
                            acc += value;
#endif
                            ++count;
                        }
                    }
#if {{MODE}} == 0 && {{INCLUDE_PAD}}
                    count = (h1 - h0) * (w1 - w0);
#endif
#if {{MODE}} == 1
                    const float out = acc;
#else
                    const float out = acc / (float)(count > 0 ? count : 1);
#endif
                    y[((b * OH + i) * OW + j) * C + c] = out;
                }
            }
        }
    }
}
