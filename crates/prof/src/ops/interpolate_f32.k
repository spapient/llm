// cutile 共享微内核：NHWC 双线性插值，align_corners=false。
// 布局契约（由声明层校验）：x={{x}} y={{y}}
// 源坐标 (i+0.5)*H/OH-0.5 夹到输入边界；单像素轴与批量均有定义。
#if {{H}} < 1 || {{W}} < 1 || {{OH}} < 1 || {{OW}} < 1 || {{C}} < 1 || {{BATCH}} < 1
#error "cutile interpolate dimensions must be positive"
#endif

void cutile_{{op}}_kernel(const float *x, float *y, int N) {
    const int H = {{H}}, W = {{W}}, OH = {{OH}}, OW = {{OW}}, C = {{C}};
    (void)N;
    for (int b = 0; b < {{BATCH}}; ++b) {
        for (int i = 0; i < OH; ++i) {
            const float ri = ((float)i + 0.5f) * (float)H / (float)OH - 0.5f;
            const float si = ri < 0.0f ? 0.0f : (ri > (float)(H - 1) ? (float)(H - 1) : ri);
            const int i0 = (int)si, i1 = i0 + 1 < H ? i0 + 1 : H - 1;
            const float fi = i0 == i1 ? 0.0f : si - (float)i0;
            for (int j = 0; j < OW; ++j) {
                const float rj = ((float)j + 0.5f) * (float)W / (float)OW - 0.5f;
                const float sj = rj < 0.0f ? 0.0f : (rj > (float)(W - 1) ? (float)(W - 1) : rj);
                const int j0 = (int)sj, j1 = j0 + 1 < W ? j0 + 1 : W - 1;
                const float fj = j0 == j1 ? 0.0f : sj - (float)j0;
                for (int c = 0; c < C; ++c) {
                    const float a = x[((b * H + i0) * W + j0) * C + c];
                    const float b0 = x[((b * H + i0) * W + j1) * C + c];
                    const float d = x[((b * H + i1) * W + j0) * C + c];
                    const float e = x[((b * H + i1) * W + j1) * C + c];
                    const float top = a + (b0 - a) * fj;
                    const float bottom = d + (e - d) * fj;
                    y[((b * OH + i) * OW + j) * C + c] = top + (bottom - top) * fi;
                }
            }
        }
    }
}
