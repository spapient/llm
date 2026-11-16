// cutile 模板：attention_f32 —— 多头（单头）缩放点积注意力，朴素三阶段。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}} k={{k}} v={{v}} y={{y}}
//
// 先算完整分数 QK^T/sqrt(d)，行内最大值交 @reduce 数组归约，再走数值稳定 softmax；
// D 维内积交 @dot 原语；softmax 的 exp 走 preamble 软件向量接口，sqrtf 来自数学库。
#include <math.h>

void cutile_{{op}}_kernel(const float *q, const float *k, const float *v, float *y, int S, int D) {
    const float scale = sqrtf((float)D);
    for (int i = 0; i < S; ++i) {
        float scores[64];
        for (int j = 0; j < S; ++j) {
            float dot = 0.0f;
            @dot(dot, q + i * D, k + j * D, D);
            scores[j] = dot / scale;
        }
        float row_max = -3.402823466e38f;
        @reduce(row_max, scores, S, 1, op: max, axis: GMAX, func: _cutile_max_f32);
        float prob_sum = _cutile_exp_sum_f32(scores, S, 1, row_max);
        _cutile_exp_norm_f32(scores, scores, S, 1, row_max, prob_sum);
        for (int h = 0; h < D; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < S; ++j) {
                acc += scores[j] * v[j * D + h];
            }
            @store(y, i * D + h, acc);
        }
    }
}
