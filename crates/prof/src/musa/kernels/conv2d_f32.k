// cutile musa 微内核：conv2d_f32 —— 二维卷积（声明档 3×3、s=1、p=1，4×4 输入、2 通道、3 输出通道）
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：x={{x}} weight={{weight}} bias={{bias}} y={{y}}
//
// 权重按 `[(ki·k+kj)·cin+ci][co]` 行主序、NHWC 激活；补零位跳过（等价于 0 参与累加），
// bias 每输出加一次。

void cutile_{{op}}_kernel(const float *x, const float *weight, const float *bias, float *y, int N) {
    const int H = 4;
    const int CIN = 2;
    const int COUT = 3;
    const int K = 3;
    const int PAD = 1;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            for (int co = 0; co < COUT; ++co) {
                float acc = bias[co];
                for (int ki = 0; ki < K; ++ki) {
                    for (int kj = 0; kj < K; ++kj) {
                        const int ii = i + ki - PAD;
                        const int jj = j + kj - PAD;
                        if (ii < 0 || jj < 0 || ii >= H || jj >= H) continue;
                        const int patch = (ki * K + kj) * CIN;
                        for (int ci = 0; ci < CIN; ++ci) {
                            acc += x[(ii * H + jj) * CIN + ci] * weight[(patch + ci) * COUT + co];
                        }
                    }
                }
                y[(i * H + j) * COUT + co] = acc;
            }
        }
    }
}
