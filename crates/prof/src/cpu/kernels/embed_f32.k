// cutile cpu 微内核：embed_f32 —— 行向量 gather
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：idx={{idx}} table={{table}} y={{y}}
//
// 每个索引取表中一整行 D 个元素；索引以 f32 承载，下标转整数。行内逐元素拷贝，行宽 D 是标量。
void cutile_{{op}}_kernel(const float *idx, const float *table, float *y,
                          int N, int V, int D) {
    for (int n = 0; n < N; ++n) {
        const float *ip = 0;
        @load(ip, idx + n);
        const int k = (int)ip[0];
        const float *row = table + k * D;
        for (int d = 0; d < D; ++d) {
            @store(y, n * D + d, row[d]);
        }
    }
}
