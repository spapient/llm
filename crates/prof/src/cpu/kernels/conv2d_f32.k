// cutile 共享微内核：conv2d_f32 —— NHWC 二维卷积，批量 N 为运行时标量。
// 布局契约（由声明层校验）：x={{x}} weight={{weight}} bias={{bias}} y={{y}}
// 权重按 `[(ki·KW+kj)·CIN+ci][co]` 行主序；越界补零不参与累加。
#if {{H}} < 1 || {{W}} < 1 || {{CIN}} < 1 || {{COUT}} < 1 || {{KH}} < 1 || {{KW}} < 1 || {{SH}} < 1 || {{SW}} < 1 || {{PH}} < 0 || {{PW}} < 0
#error "cutile conv2d geometry is invalid"
#endif

void cutile_{{op}}_kernel(const float *x, const float *weight, const float *bias, float *y, int N) {
    const int H = {{H}}, W = {{W}}, CIN = {{CIN}}, COUT = {{COUT}};
    const int OH = (H + 2 * {{PH}} - {{KH}}) / {{SH}} + 1;
    const int OW = (W + 2 * {{PW}} - {{KW}}) / {{SW}} + 1;
    for (int b = 0; b < N; ++b) {
        for (int i = 0; i < OH; ++i) {
            for (int j = 0; j < OW; ++j) {
                for (int co = 0; co < COUT; ++co) {
                    float acc = bias[co];
                    for (int ki = 0; ki < {{KH}}; ++ki) {
                        const int ii = i * {{SH}} + ki - {{PH}};
                        if (ii < 0 || ii >= H) continue;
                        for (int kj = 0; kj < {{KW}}; ++kj) {
                            const int jj = j * {{SW}} + kj - {{PW}};
                            if (jj < 0 || jj >= W) continue;
                            const int patch = (ki * {{KW}} + kj) * CIN;
                            for (int ci = 0; ci < CIN; ++ci) {
                                acc += x[((b * H + ii) * W + jj) * CIN + ci]
                                     * weight[(patch + ci) * COUT + co];
                            }
                        }
                    }
                    y[((b * OH + i) * OW + j) * COUT + co] = acc;
                }
            }
        }
    }
}
