// cutile 模板：embed_f32 —— 查表 gather
// 槽位：{{op}} 是算子名，@prim(...) 由片段层逐个替换。
// 布局契约（由声明层校验）：idx={{idx}} table={{table}}
//
// 每个索引取表中一项。复用 load/store，索引以 f32 承载，下标转整数。
void cutile_{{op}}_kernel(const float *idx, const float *table, float *y, int N, int V) {
    for (int n = 0; n < N; ++n) {
        const float *ip = 0;
        @load(ip, idx + n);
        const int k = (int)ip[0];
        const float *tp = 0;
        @load(tp, table + k);
        @store(y, n, tp[0]);
    }
}
