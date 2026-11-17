// cutile 模板：decode_attention_f32 —— 单条 query 对历史 K/V 的朴素三阶段注意力。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}} k={{k}} v={{v}} y={{y}}
//
// query 行数 Q_ROWS 与历史行数 HIST 解耦；先算历史分数 QK^T/sqrt(d)，行内最大值交 @reduce
// 数组归约，再走数值稳定 softmax；D 维内积交 @dot 原语；softmax 的 exp 走 preamble 软件
// 向量接口，sqrtf 来自数学库。
#include <math.h>

void cutile_{{op}}_kernel(const float *q, const float *k, const float *v, float *y, int Q_ROWS, int HIST, int D) {
    const float scale = sqrtf((float)D);
    for (int i = 0; i < Q_ROWS; ++i) {
        float scores[64];
        for (int j = 0; j < HIST; ++j) {
            float dot = 0.0f;
            @dot(dot, q + i * D, k + j * D, D);
            scores[j] = dot / scale;
        }
        float row_max = -3.402823466e38f;
        @reduce(row_max, scores, HIST, 1, op: max, axis: GMAX, func: _cutile_max_f32);
        float prob_sum = _cutile_exp_sum_f32(scores, HIST, 1, row_max);
        _cutile_exp_norm_f32(scores, scores, HIST, 1, row_max, prob_sum);
        for (int h = 0; h < D; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < HIST; ++j) {
                acc += scores[j] * v[j * D + h];
            }
            @store(y, i * D + h, acc);
        }
    }
}
