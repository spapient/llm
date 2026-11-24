// cutile musa 微内核：decode_attention_f32 —— 单条 query 对历史 K/V 的朴素三阶段注意力。
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：q={{q}} k={{k}} v={{v}} y={{y}}
//
// query 行数 Q_ROWS 与历史行数 HIST 解耦；先算历史分数 QK^T/sqrt(d)，行内最大值交 @reduce
// 数组归约，再走数值稳定 softmax；D 维内积交 @dot 原语；softmax 的 exp 走 preamble 软件
// 向量接口，sqrt 走 preamble 软件标量接口。
//
// HIST 是运行期参数，分数不能整条落栈：按固定块容量 64 扫两遍历史。第一遍只在线合并出整行
// 最大值与归一化和，第二遍重算块内分数、乘回 V 后累加进 y。块容量固定，HIST 任意大都在栈上
// 占同样空间；HIST <= 64 时与不分块逐位同序。

void cutile_{{op}}_kernel(const float *q, const float *k, const float *v, float *y, int Q_ROWS, int HIST, int D) {
    const float scale = _cutile_sqrt_f32((float)D);
    for (int i = 0; i < Q_ROWS; ++i) {
        float scores[64];
        float row_max = -3.402823466e38f;
        float prob_sum = 0.0f;
        for (int base = 0; base < HIST; base += 64) {
            const int n = (HIST - base) < 64 ? (HIST - base) : 64;
            for (int j = 0; j < n; ++j) {
                float dot = 0.0f;
                @dot(dot, q + i * D, k + (base + j) * D, D);
                scores[j] = dot / scale;
            }
            float block_max = -3.402823466e38f;
            @reduce(block_max, scores, n, 1, op: max, axis: GMAX, func: _cutile_max_f32);
            const float merged = row_max > block_max ? row_max : block_max;
            prob_sum = prob_sum * _cutile_exp_f32(row_max - merged)
                     + _cutile_exp_sum_f32(scores, n, 1, merged);
            row_max = merged;
        }
        for (int h = 0; h < D; ++h) {
            @store(y, i * D + h, 0.0f);
        }
        for (int base = 0; base < HIST; base += 64) {
            const int n = (HIST - base) < 64 ? (HIST - base) : 64;
            for (int j = 0; j < n; ++j) {
                float dot = 0.0f;
                @dot(dot, q + i * D, k + (base + j) * D, D);
                scores[j] = dot / scale;
            }
            _cutile_exp_norm_f32(scores, scores, n, 1, row_max, prob_sum);
            for (int h = 0; h < D; ++h) {
                float acc = 0.0f;
                for (int j = 0; j < n; ++j) {
                    acc += scores[j] * v[(base + j) * D + h];
                }
                @store(y, i * D + h, y[i * D + h] + acc);
            }
        }
    }
}
