// cutile 共享微内核：decode_flash_attention_f32 —— 任意头宽的 online softmax。
// 槽位：{{op}} 是算子名，{{D_BLOCK}} 是声明配置的本地累加块容量；@prim(...) 由片段层替换。
// 布局契约（由声明层校验）：q={{q}} k={{k}} v={{v}} y={{y}}
//
// 每个 D 块单遍扫描历史 K/V；D 超过块容量时逐块重算分数，以有界本地存储换任意正头宽。
// D <= D_BLOCK 时只有一块，循环次序与原先完全一致，不落地历史分数矩阵。
#if {{D_BLOCK}} < 1
#error "cutile D_BLOCK must be positive"
#endif

void cutile_{{op}}_kernel(const float *q, const float *k, const float *v, float *y, int Q_ROWS, int HIST, int D) {
    const float scale = _cutile_sqrt_f32((float)D);
    for (int i = 0; i < Q_ROWS; ++i) {
        for (int db = 0; db < D; db += {{D_BLOCK}}) {
            const int dn = (D - db < {{D_BLOCK}}) ? D - db : {{D_BLOCK}};
            float row_max = -3.402823466e38f;
            float prob_sum = 0.0f;
            float acc[{{D_BLOCK}}];
            for (int h = 0; h < dn; ++h) { acc[h] = 0.0f; }
            for (int j = 0; j < HIST; ++j) {
                float dot = 0.0f;
                @dot(dot, q + i * D, k + j * D, D);
                float score = dot / scale;
                float new_max = row_max > score ? row_max : score;
                float reweight = _cutile_exp_f32(row_max - new_max);
                float prob = _cutile_exp_f32(score - new_max);
                prob_sum = prob_sum * reweight + prob;
                for (int h = 0; h < dn; ++h) {
                    acc[h] = acc[h] * reweight + prob * v[j * D + db + h];
                }
                row_max = new_max;
            }
            for (int h = 0; h < dn; ++h) {
                @store(y, i * D + db + h, acc[h] / prob_sum);
            }
        }
    }
}
