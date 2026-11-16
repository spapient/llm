// cutile 模板：flash_attention_f32 —— online softmax 分块，单遍 K/V。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}} k={{k}} v={{v}} y={{y}}
//
// 维护 running max、归一化和与累加向量；新最大值出现时把旧账按比例缩回。不落地 s×s 分数矩阵。
#include <math.h>

void cutile_{{op}}_kernel(const float *q, const float *k, const float *v, float *y, int S, int D) {
    const float scale = sqrtf((float)D);
    for (int i = 0; i < S; ++i) {
        float row_max = -3.402823466e38f;
        float prob_sum = 0.0f;
        float acc[64];
        for (int h = 0; h < D; ++h) { acc[h] = 0.0f; }
        for (int j = 0; j < S; ++j) {
            float dot = 0.0f;
            @dot(dot, q + i * D, k + j * D, D);
            float score = dot / scale;
            float new_max = row_max > score ? row_max : score;
            float reweight = expf(row_max - new_max);
            float prob = expf(score - new_max);
            prob_sum = prob_sum * reweight + prob;
            for (int h = 0; h < D; ++h) {
                acc[h] = acc[h] * reweight + prob * v[j * D + h];
            }
            row_max = new_max;
        }
        for (int h = 0; h < D; ++h) {
            @store(y, i * D + h, acc[h] / prob_sum);
        }
    }
}
