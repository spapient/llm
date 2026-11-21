// cutile 模板：latent_attention_f32 —— 潜空间注意力，键行尾部是解耦 RoPE 段。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}} k={{k}} v={{v}} y={{y}}
//
// 行宽 W 是读入宽度（压缩 + RoPE），输出宽 DC 是头内乘积宽度，二者可以不等：DC 由上投影的折叠
// 口径给出，模板只按 DC 写回。分数按整行 W 做一次 @dot，行内最大值交 @reduce 数组归约，
// softmax 走 preamble 的软件向量接口，加权段按 DC 列展开。

void cutile_{{op}}_kernel(const float *q, const float *k, const float *v, float *y, int R, int T,
                         int W, int DC) {
    const float scale = _cutile_sqrt_f32((float)W);
    for (int i = 0; i < R; ++i) {
        float scores[64];
        for (int j = 0; j < T; ++j) {
            float latent = 0.0f;
            @dot(latent, q + i * W, k + j * W, W);
            scores[j] = latent / scale;
        }
        float row_max = -3.402823466e+38f;
        @reduce(row_max, scores, T, 1, op: max, axis: GMAX, func: _cutile_max_f32);
        float prob_sum = _cutile_exp_sum_f32(scores, T, 1, row_max);
        _cutile_exp_norm_f32(scores, scores, T, 1, row_max, prob_sum);
        for (int c = 0; c < DC; ++c) {
            float acc = 0.0f;
            for (int j = 0; j < T; ++j) {
                acc += scores[j] * v[j * DC + c];
            }
            @store(y, i * DC + c, acc);
        }
    }
}
