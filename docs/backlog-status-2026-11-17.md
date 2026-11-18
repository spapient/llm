# Backlog 收口矩阵（2026-11-17 活树复核）

> 口径：状态分 **已落地 / 进行中 / 阻塞**；测试证据按 `crate + 测试数或用例名` 给出，计数为
> `cargo test --workspace` 收口实测。门禁不削弱、负例不回写；历史快照刀次记录不回填。
> 计划编号与验收门见 [`implementation-task-plan.md`](implementation-task-plan.md) §3；
> 不退化清单见 [`llmu-parity-optimization-design.md`](llmu-parity-optimization-design.md) §1.5；
> M2 整批几何 / M3 设备后端 / M4 生成服务化设计见 [`batch-backend-serving-design.md`](batch-backend-serving-design.md)。

## 0. 收口门禁实测（合并门，全仓口径）

| 门 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | 干净，0 error / 0 warning |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 零告警 |
| `cargo fmt --all --check` | 0 diff |
| `cargo nextest run --workspace --all-features --locked` | **1,480/1,480 passed，0 skipped**（nextest 口径，9 个 libfuzzer harness + gen_corpus 二进制被 `.config/nextest.toml` 排除） |
| `cargo test --workspace --all-features --locked --exclude llm-fuzz`（libtest 口径；fuzz libFuzzer harness 由 CI fuzz job 单独引导，不进 libtest） | 收口实测 **1,503 passed / 0 failed / 1 ignored**；与 nextest 是两种口径，计数不可混用 |
| architecture | **8/8**；contracts **6/6**；cross_layer **4/4**；静态 gate **1/1**（testkit 合计 36/36） |
| `cargo deny check` | advisories / bans / licenses / sources 四项 ok（许可白名单含 NCSA、Unicode-3.0） |
| `cargo machete` | 无未用依赖 |
| jscpd | 1.49% 行 / 1.79% token，135 clones，425 文件，84,252 行，1,258 重复行 / 9,879 重复 token，exit 0（行阈 1.5%；token 未设硬阈） |
| `cargo bench --workspace --all-features --locked --no-run` | exit 0，全部基准目标编译通过 |
| `unsafe` | 生产唯一岛 `crates/storage/src/mmap.rs`；本轮新增 unsafe 为零；Metal/CUDA 后端未落地，设备侧零 unsafe |

各成员 lib 测试数（`cargo test -p <pkg> --lib` 实测）：annotation 34、autograd 65、cli 25、
cutile 167、data 16、diagnostic 12、dtype 18、engine 77、fuzz 17、nn 79、op 216、prim 19、prof 56、
serve 130、shard 29、storage 13、tensor 25、testkit 17、tokenizer 44、tool 57、train 82、weights 129、wire 13。
集成 / 文档补充：fuzz parsers 2；tokenizer tokens 16 + doc 1；op 集成 2；prof doc 1（`tests/atom_tests.rs` 经 `#[path]` 编进 lib 子模块，计入 lib 56）；benches regress 8；
serve protocol 8 + doc 3；engine schedule 9 + doc 3；nn 集成 2；train doc 3；cli cutile_bin 9 + 集成 23 + doc 2；tool doc 1；weights import_tests 3 + round_trip 12。

---

## 1. P0 架构 / 质量加固

| ID | 状态 | 测试证据（crate + 用例 / 计数） | 未完成原因 / 暂缓 |
|---|---|---|---|
| P0-1 | 已落地 | 文档计数与指针回填：design §1.2/§1.5、plan D-1…D-7、roadmap §3、code-audit；gate 1/1、architecture 8/8 守住文档链接门 | 历史快照刀次记录不回写：`backend-runtime-design.md:472` 仍记该刀 1325 / contracts 5，属当时快照 |
| P0-2 | 已落地 | annotation：`OP_OVERRIDES` 一行三列 `(名, 参考覆盖, 模块覆盖)`；annotation lib 34 + audit 集成 2；无第二份按名查找循环 | — |
| P0-3 | 已落地 | 大文件按职责轴拆分：weights `bin.rs` → `bin_zip.rs` / `bin_pickle.rs`；cutile `main.rs` → `report.rs`；train `snapshot.rs` → `cursor/entry/state`；入口编排拆 `cmd/run.rs` / `cmd/status.rs` / `cmd/train.rs`；gate 行数 ≤899、ccn/nloc 过门 | 其余大文件随后续阶段内拆；入口编排函数已重构回 ccn/nloc 阈值内，`[gate.debt]` 存量为零，无待删基线条目 |
| P0-4 | 已落地 | 报告 / 排版收敛 `tool::Report` 单一报告器；tool lib 51（含 ccn 1、gate 1） | — |
| P0-5 | 已落地 | 手写解析收敛 diagnostic 单源；ADR 0039 草案记依赖策略；diagnostic lib 12 + 集成 1，解析负例全绿 | toml+serde 唯一解析依赖未拍板前不接（D5） |
| P0-6 | 部分落地 | 三族纯函数表驱动边界用例落在各自仓内单测：tokenizer 44 lib + tokens 16；op 198 lib + 2 集成；prof 53 lib + atom 1；op 边界新增 mul / scale / embed / rope / dequant_i4 / dequant_i8 六条空值/非对齐/越界/饱和/重复对拍。prof 表达式边界本轮补三条表驱动用例：`formulas_evaluate_priority_chains_and_parenthesized_groups`（9 条合法链式/优先级/嵌套括号）、`formulas_reject_each_degenerate_shape_with_a_message`（18 条拒绝：除零/括号不闭合/右括号不成对/运算符续接/连续运算符/非法分隔字符/非法字面量，每条 `Code::Invalid` + 非空消息）、`formulas_handle_saturated_large_literals_without_panic`（1e308、可折叠 inf、0、饱和整数等不 panic 不产 NaN）。fuzz 侧只保留不可信字节入口解析器目标（parsers 2 + 9 个 libfuzzer harness，`harness=false` + `required-features=["engine"]` 门控）；三族内部纯函数 / 结构化张量接进 sweep 的 p06 接线范式与 fuzz 契约边界错配，已按规范移除，`tokenizer`/`op`/`prof` 不进 fuzz `[dev-dependencies]`，零新生产依赖；本机 unix（macOS arm64）`cargo test -p llm-fuzz --all-features --no-run` 9 个 harness 全部链接产出 Mach-O，逐个 `nm` 核到 `T _LLVMFuzzerTestOneInput` 与 `T _main`，libfuzzer.a 静态链接 | prof `eval_expression` 仍 `pub(super)` 未接 fuzz（仓内单测覆盖）；`DType` 名字往返由 `dtype/tests/properties.rs` 覆盖；Windows/MSVC 仍走空 main 桩口径，未引导运行时 |
| P0-7 | 已落地 | 注解 / 声明 / 模板三方参数类型签名交叉校验；注册表 24 条全过；codegen 检查模块 `the_annotation_still_matches_the_declared_operators` | 跨设备文本同构校验随后端阶段 |
| P0-8 | 已落地 | 已具备能力常驻回归：weights `tests/round_trip.rs` 12 条多格式往返；prof Grade L0–L3 闭集往返（prof lib 48 + 1） | — |

## 2. M1-A 装载与驻留权重

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-A1 | 已落地 | weights lib 116；`Reader`/`Loader` 收口，边界检查单点；装载负例（缺条目 / 非 F32 / 尾半字 / 长度不符）全绿 | 远端分块源未接 |
| M1-A2 | 已落地 | `weights::MmapSource`（mmap feature 双门控）复用 `storage::Mmap` 唯一 unsafe 岛；`mmap_source_tests.rs`；mmap 口径 weights 116 lib / storage 13 lib；压缩段保持 `Owned` 断言 | 设备权重驻留、热更新暂缓 |
| M1-A3 | 已落地 | `load_f32` / `decode_entry_f32` 驻留解码；`load_f32_decodes_an_entry_into_the_resident_buffer`、`a_non_f32_entry_is_not_unpacked_into_the_resident_buffer`、`decode_entry_f32_passes_through_the_partial_tail_word_rejection`；`StubLoader` 测试件 | 真实权重端到端装载属 M4-1 |
| M1-A4 | 已落地 | `regress.rs` 装载 8 条 Op 记录归档（File 4 + mmap 4，`#[cfg(unix)]`）；`registered_load_records_archive_when_measured`、`load_records_carry_every_field_a_reader_needs_to_judge_it`；Harness / Geometry / Roofline / observed 全字段入台账；Roofline 双 None，不采 RSS；criterion `weights_load.rs` 测量循环保留 | 未采 RSS 进程读数；无跨机基线结论；tree 内无 criterion `estimates.json`，归档测试走 skip 路径；冷热缓存对照入 regress 基线待后续刀 |
| M1-A5 | 已落地 | 自有 `.u` 容器 v2（ADR-0045，魔数 `.u\0\0` / VERSION 2）：REF 跨段去重——写入期哈希预检 + 逐位比对定代表段，引用段只带量化方案与代表段下标，打开期单色 DFS 拒悬空 / 越界 / 自引用 / 代表段是引用 / 元数据不一致 / 成环；中性条目流导入按扩展名分派：safetensors File/Archive seek+read 借用整份字节、GGUF 标量 F32/F16 直映、GGML 块量化显式 `Code::Unsupported`（不反量化、不塞伪装段）、bin/ZIP_STORED+pickle 栈机、`.pt` 走 bin 通路、未登记后缀拒绝；打开后现场现算 `Metrics`（条目数 / 参数 / V_artifact / V_FP16 / ratio / lossless / REF 命中 / 节省字节，不读文件内保存字段）；CLI `import --in/--out`、`inspect --in`、`check --in`。weights lib 129 + import_tests 3 + round_trip 12；cli lib 25 + cutile_bin 9 + 集成 23 + doc 2；fuzz parsers 2；doc weights 3。保留名 `llm.mode` / `llm.vocab`（字节占位，不接 tokenizer）/ `opt.` 前缀 | 码本 / 格码低位宽压缩、分片多文件清单与显式随机访问索引、`llm.vocab`→`VocabSource` 解析、真实权重端到端生成 / HTTP 后端 / 同机对照基准、strided / block-sparse 布局见 M4-1 与 ADR-0045 未决节；v1 LLPK 字节不被 v2 读端兼容，无 v1→v2 解码器 |

## 3. M1-B CPU 单序列前向闭环

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-B1 | 已落地 | serve `KERNELS` const 子集表 + 注册表子集断言；未覆盖名查表返回 `Unsupported`，无静默空入口；serve lib 124 | 标量槽通道归 B6 |
| M1-B2 | 已落地 | op lib 198 条标量参考对拍 + 2 集成；model_tests：`gemv_runs_through_the_registered_kernel`、`rmsnorm_runs_...`、`binary_elementwise_runs_...`、`unary_elementwise_runs_...`、`softmax_and_reduce_run_...`、`attention_runs_...`；`testkit::close` 双算预算 | 注意力主链、量化 i4/i8 lane 暂缓（见 M2-1 / M3-3）；rope 装配链落点见 M1-B3 |
| M1-B3 | 已落地 | `plan.rs` 闭集补 Projection / Embedding / Unary / Norm / Binary / Rope；`assembly.rs` 旁路输入 1 + rope 输出槽；`block.rs` q/k 各插一路 rope；`kernels.rs` 透传 Launch 并登记 `rope_f32`；`model_host` 首层 Binary/Rope 拒 Unsupported；两层合成模型标量参考对拍 + 宽度断裂 / 权重缺失 / 嵌入缺表三负例。serve 124 lib + protocol 8 + doc 3；用例 `a_binary_layer_takes_a_bypass_activation_matches_the_reference`、`a_binary_mul_layer_takes_a_bypass_activation_matches_the_reference`、`a_two_layer_composite_model_logits_match_the_reference`、`a_binary_layer_without_a_bypass_activation_is_rejected_at_assembly`、`a_composite_width_break_is_rejected_at_assembly`、`a_composite_missing_projection_weight_is_reported_at_assembly`、`a_composite_missing_embedding_table_is_reported_at_assembly` | 多 token 矩阵主链、序列并行、设备后端泛型暂缓；未预埋 |
| M1-B4 | 已落地 | `testkit::close` 绝对 + 相对双预算；model_tests `assert_close` 双算；testkit lib 17 + 1 | 第二份独立闭式参考暂缓 |
| M1-B5 | 已落地 | `Classical` 经典采样（温度缩放 → top-k → top-p 核采样 → argmax，不接 RNG）接入 `Completion`；坏温度 / 空候选 / top-p 越界负例；`nucleus_keeps_the_probability_prefix` | top-p 校准表细化暂缓 |
| M1-B6 | 已落地 | engine `ScalarSlots` 定长类型化闭集（首批 `rope_position: Option<u32>`），无字符串键 / 无 HashMap；ADR 0043；`scalar_slots_start_empty`、`a_rope_position_round_trips_through_the_node` | heads / head_dim 槽位随 M2-1 多头通路补；多 token 矩阵几何暂缓 |
| M1-B7 | 已落地 | `BindingDesc{id, entry, shape}` 留 serve 装配侧；engine `Bindings` 纯 `BTreeMap<u32, Vec<f32>>`；ADR 0044；`descriptors_map_ids_back_to_entries_and_shapes` | 设备侧形状元数据暂缓；只读形状访问器待后端实现证明需要再补 |

## 4. M1-C 基准 harness 骨架

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-C1 | 已落地 | `scripts/mechanism_lines.sh` 可复跑；`mechanism-lines-baseline.md` 分账表 233 文件 / 50,713 机制行（L1–L7 分层）；annotation / fuzz / benches / tests / build.rs / vendor 排除口径入档 | 严格口径（逐行扣除内联 `#[cfg(test)]` 体）本轮（2026-11-17 收口刀）评估后仍暂缓：`tool::strip_tests` 仅被 `ccn.rs` 静态门内部调用，tool 无独立可执行计数入口（`ccn`/`gate` 是 `[[test]]` 目标，不是可执行子命令）；bash 脚本要出严格口径必须新增 helper 二进制或扩 cli `Command` 枚举，按"不加新 helper 二进制"规则不硬塞。待 tool 行计数子命令（或 gate 输出附带 stripped 行数）落地后再接；对标框架同口径分母未采集 |
| M1-C2 | 已落地 | `tool::bench`：`EntryKind` / `Harness` / `Geometry` / `Roofline` / `BenchRecord`，构造自检；tool bench 结构用例 6 条；regress 算子报告与上界报告元数据已迁入统一记录，`weights::json` 自检保留；benches regress 8 用例（装载 4 Op + 模型 2 Model + 既有 4 条基线 / 上界 / 主机 / 字段） | 整机峰值 BW / GFLOPS 未测得，Roofline 留 `None`；批次 TPOT / 吞吐 / 内存曲线入表属 M2-4 后续；跨机绝对值结论不出 |

## 5. T1 训练最小闭环

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| T1-1 | 已落地 | `Optimizer` 对象安全 trait + SGD / Adam（矩估计、偏差修正、AdamW 解耦衰减、全局裁剪）；train lib 80；`sgd_takes_one_direct_step`、`adam_moves_the_param_with_momentum`、`adam_moments_round_trip_through_restore`、`restore_rejects_mismatched_moment_tables`、`global_norm_clipping_scales_every_slot`、`a_length_mismatch_is_rejected_before_update`、`weight_decay_applies_only_outside_exempt_groups`、`optimizer_is_object_safe` | 分布式优化器暂缓 |
| T1-2 | 已落地 | `ContractSpec.role: ParamRole` 闭集；`group_mapping_follows_the_role`；wd 豁免 / 可训练判定只从角色 `wd_exempt()` 派生，优化器不猜字段名 | — |
| T1-3 | 已落地 | tape → 反传计划 → `Requirement::Grad` 终点 → 裁剪 → 优化器步；梯度缓冲归 train `Buffers` 自持，autograd 只出 tape/plan/Replay；`a_fixture_dataset_runs_the_declared_steps`、`a_mismatched_batch_shape_is_rejected`、`stepping_without_a_gradient_is_rejected`；autograd 65 lib + gradcheck 3 + 2 集成；data 16 + 4 | 真实大语料通路暂缓 |
| T1-4 | 已落地 | CLI `train` 命令；快照 v5：Facts 五字段（step/world/layers/batch_rows/batch_cols）+ 矩状态段 + 累积梯度段 + ties 段，30 字节 golden 钉版；`a_golden_vector_pins_the_version_five_layout`、`another_version_says_which_side_is_old`、`optimizer_moments_round_trip_bit_for_bit`、`accumulated_gradients_round_trip_bit_for_bit`、`a_slot_with_uneven_moments_is_rejected_before_encoding`；cli 25 lib + 集成 | offload 真实换入换出通路未接（接缝见 T1-5） |
| T1-5 | 已落地 | 快照格式 v4→v5，新增 ties 段；`accum_snapshot()` / `ties()` / `ties_snapshot()` 导出接缝；offload 只留 `WeightOffload` trait + `DeferredOffload` 占位（一律 `Code::Unsupported`）；两层共享权重夹具回填对拍。本轮补回载接缝：`Session::restore_snapshot(&mut self, &Snapshot) -> Result<()>` 顺序矩状态 → 累积窗口 → ties → 步号；矩走 `optimizer.restore`（trait 提升，Sgd 空表 Ok/非空 Invalid，Adam 校验消息保留），accum 按槽位装回缓冲（空表跳过不覆盖），ties 编号越界 / 两端长度不符 `Code::Invalid`，step 从 `facts.step()` 回载；私有助手 `restore_moments` / `check_moment_slots` / `check_moment_slot_length` / `restore_accum` / `check_accum_slot_count` / `load_accum_slot` / `restore_ties`（CCN ≤5）。cli `report_recovered` 补两行：累积窗口 `{count} / {window} micro-batch`、权重绑定 `{n} 对`（无 ties 为 0）。train 82 lib + doc 3；cli 25 lib + cutile_bin 9 + 集成 18 + doc 2；用例 `session_state_round_trips_through_restore_snapshot`、`restore_snapshot_rejects_mismatched_state`、`train_load_reports_accumulation_window_and_tie_pairs`，既有 `session_window_state_round_trips_through_snapshot`、`tied_parameter_copies_source_across_a_two_layer_fixture`、`adam_moments_round_trip_through_restore`、`restore_rejects_mismatched_moment_tables` 等全部保留 | 会话侧矩状态导出通路未接：快照 moments 目前由测试 / fixture 构造，cli 组装尚未读 `optimizer.momentum()`/`variance()`；Adam 非空矩往返只在优化器层对拍。回载不重建 tape / 参数槽 / 内核 / 权重（M4-1 真实权重通路）；offload 仅接口接缝；ZeRO / 分布式优化器 / 序列并行暂缓 |

## 6. M2 推理运行时（KV 与连续批）

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M2-1 | 阻塞 | 守护用例在案：`a_multi_head_block_is_not_assembled_yet`；B6 标量槽、B7 描述表已就位；M2-2 批量结果视图与页表守护已落地 | executor 图仍是单序列几何：多层层偏移行切片、多头展平头内切片、注意力 Program 接 KV 行映射均未接；前置 B6 槽位 + M2-2 视图已就位，剩内核级多序列矩阵主链；头内并行特化更后 |
| M2-2 | 已落地 | `Program::run` 已是通用视图通道，无单序列硬编码；补 `SequenceOutcome<'a>` 只读结果视图、`Batch::sequence_at` / `outcome` / `outcomes`；页表引用计数守护。engine 77 lib + schedule 9 + doc 3；用例 `multi_sequence_batch_runs_their_own_history_and_progress`、`batch_outcomes_pair_each_progress_with_its_own_sequence`、`admission_full_handles_back_without_touching_the_page_accounts`、`finishing_a_sequence_releases_all_of_its_pages`、`shared_prefix_reference_counting_is_correct_across_finish_order`、`shared_prefix_refcount_supports_three_holders` | op 侧 attention decode 批量段编排入口已就位：`decode_forward_batch`/`decode_flash_forward_batch` 显式 heads/batch 守卫、段内复用多头内核；gemm 行批、embed 多行查表可复用；rope `forward_batch` 段内逐序列独立位置、段内复用多头入口已就位；`rmsnorm` `forward_batch` 按序列段分组归一化、gamma 按行宽复用已就位；`Predict::logits_batch` 批量视图接口已定稿（行序对齐准入序，默认逐条回退 `logits`，对拍 `logits_batch_default_runs_each_sequence_in_input_order`）；embed→投影无块链整批前向已落地：投影行循环 gemv 方案 A 适配、输入/激活缓冲按批长重绑、批量末位 token 收集、输出按行宽拆回，batch=2 对拍 `logits_batch_runs_two_sequences_through_the_flat_projection_chain`；含注意力块的链整批 K/V 搬运、rope/rmsnorm 批段 serve 适配、`step_batch` 整批一次提交仍待接；内核级多序列矩阵主链暂缓。serve/engine 已落地的前置：① 投影行批方案 A 已实现（批内逐行循环调 `gemv_f32` 共享驻留权重，不动 op gemm 契约；`gemm_f32` 收缩轴 `K=8` 固定档，不接任意 `in_features` 真实权重）；② `rmsnorm` 行分组 op 入口已就位（按序列段独立归一化、gamma 复用），serve 内核适配接行批分段仍待；③ 激活/历史缓冲装配期存单序列段长，运行期按本轮批长重绑；行循环 gemv 已把 in_features 登记进节点 Launch 第二维几何通道；④ embed 多行查表就位；plan `tokens!=1` 是层形状契约（拒 prefill 多 token 层），运行期多行几何由内核长度反推，放开不需动该装配检查；⑤ batch=2 端到端回归基线夹具已补（`logits_batch_runs_two_sequences_in_input_order_and_keeps_their_own_kv`，默认逐条回退下对拍两行 logits 与各自 KV 归属），整批几何接线后沿用同一份对拍 |
| M2-3 | 已落地 | `Batch::prefix_share` + `Scheduler::admit_shared`，经既有 `Pager::share/release` 引用计数；`identical_prefix_is_adopted_as_shared_pages`、`shared_prefix_row_survives_source_finish_until_last_holder_releases`、`diverging_prefix_takes_fresh_pages`、`prefix_shorter_than_one_page_is_not_split` | 跨请求前缀复用预热暂缓 |
| M2-4 | 已落地 | `regress.rs` 模型 2 条 `EntryKind::Model` 记录（batch=1 / batch=4），Geometry 批次维入台账；`registered_model_records_archive_when_measured`、`model_records_carry_every_field_a_reader_needs_to_judge_it`；criterion `model_decode.rs` 单序列 / 四条序列同批变体测量循环保留；mem_stream 三档 roofline 分母在位；Roofline 双 None。本轮补 BenchRecord 手写 JSON 归档：`tool/src/bench_json.rs` `to_json(&BenchRecord) -> String` + 受限回读 `from_json(&str) -> Result<BenchRecord>`，零外部 JSON 依赖（tool 运行期仍只依赖 diagnostic）；写出 kind/name/backend、geometry(shape/batch/dtype)、harness 六身份字段、roofline 双可选、observed 可选，None 写 null，字符串转义 `"`/`\`/控制字符；回读深度上限 16、重复键拒绝（顶层+嵌套）、截断/文法错误带字节位置、有限数校验（NaN/Infinity 词法层拒）、维度非负范围校验、缺必填/类型不符/未知 kind `Code::Invalid`；读取器为固定 schema 流式草稿，无通用 Value 树。tool 57 lib + doc 1；用例 `json_round_trips_record_without_observed`、`json_round_trips_record_with_observed_and_roofline`、`json_rejects_malformed_and_truncated_input`、`json_rejects_duplicate_keys`、`json_rejects_non_finite_and_wrong_typed_values`、`json_rejects_missing_required_fields` | 未采 RSS；tree 内仍不生成 criterion `estimates.json`；不接 llama.cpp / vLLM 同机对照、不出吞吐倍数；批次 TPOT / 吞吐 / 内存曲线入表暂缓；设备侧数字待 M3；读取器按受限口径拒绝 `\uXXXX` 转义与未知顶层键 |

---

## 7. M3 后端边界 / 量化 / 性能 —— 阻塞边界与前置条件

**总前置（未满足前 M3 不动工）**：
1. **ADR-0035 仍为 Proposed（待评审）**：本稿只定 unsafe/FFI 边界规则，**不是 M3 实现放行**；不得标 Accepted，不得据此展开后端真实 FFI / 设备内核。评审通过后作为 M3-1 前置约束。
2. **硬件前提（两侧实测，不是硬件缺失）**：本机 MBP = Apple M5 Pro，20 核 / Metal 4 / arm64 / 24GB，rustc 1.100.0-nightly；Apple Silicon GPU 与 Metal 4 可用。远端 Windows（SSH 别名 `win`，192.168.0.65，user fei）实测 NVIDIA T600 4096MiB、compute capability 7.5（sm_75，Turing）、driver 596.71、CUDA 13.3（nvcc V13.3.33，`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`）；rustup 全局默认已切到 `nightly-2026-09-01-x86_64-pc-windows-msvc`，与本机同一快照（rustc/cargo/rustfmt/clippy/rust-std 均 1.100-nightly），stable 1.99 仅作备用通道；home 目录 `rust-toolchain.toml` 覆盖也已对齐 nightly（备份 `.bak`）。两侧硬件都不缺：Metal 在本地，CUDA 在 win。
3. **真正剩余门槛（M3 动工前置）**：① ADR-0035 仍 Proposed 待评审，workspace 全局 `unsafe_code=deny`，真实设备 FFI / 设备句柄 / kernel launch 必须等边界规则评审通过后落进边界 crate，不得自行标 Accepted；② D5 生产零外部依赖：不引 cuda-sys / metal-rs 之类 FFI crate，要么手写 FFI 绑定，要么走 ADR 评审放宽策略；③ 源码同步与跨机构建未通：本地 main 与 origin 齐平，win 上仅 `C:\Users\fei\llm` 一份无 `.git` 的源码副本，尚未建 git 工作树、未在 win 跑过 `cargo check`，CUDA kernel / 命令队列 / 缓冲映射 / launch 管线未建；④ T600 是 sm_75 / 4GB，后续内核与批量尺寸按此能力约束，占位测试标签不写死任何 SM 版本。
4. **D6 形状约束**：第二后端出现前不预埋设备泛型参数；四 trait（`Executor`/`Allocator`/`Collective`/`Program`）已定形。

| ID | 状态 | 现状（已落地骨架） | 阻塞边界 / 前置条件 |
|---|---|---|---|
| M3-1 | 阻塞 | 边界 crate `llm-backend` 占位骨架（`UnsupportedBackend`/`MetalBackend`/`CudaBackend`）已从工作树移除，当前 24 成员中无后端边界 crate；trait 端口仍在运行时一侧：`Executor`（`engine::exec`）、`Allocator`（`storage`），CPU 侧 `CpuExecutor`/`CpuAllocator` 在位，设备后端实现未落地；零 FFI crate、零设备 unsafe | 前置 0035 评审通过；若重建边界成员，依赖方向只能从边界 crate 指向运行时 trait 并守门；边界 crate 之外零新增 unsafe；第三加速链路暂缓 |
| M3-2 | 阻塞 | `Executor`/`Allocator`/`Collective` 端口契约在位；CPU 实现 `CpuExecutor`/`CpuAllocator`/`shard::Local`，`shard::Remote` 为 `Collective` 占位；无边界 crate 的设备实现与跨后端行为对拍 | 前置 M3-1；Metal 本机跑通 M2 同一前向、CUDA 在 win（T600 sm_75 / CUDA 13.3）验；同输入同形状对拍一致；两侧未建设备内核前不出吞吐数字；图 capture 特化暂缓 |
| M3-3 | 部分落地 | `gemv_i2/i4/i8`、`gemm_i2/i4/i8`、`dequant_i2/i4/i8` 骨架在 `gemv.rs`/`gemm.rs`/`dequant.rs` 泛型内；注册表 27 条含 9 条量化 lane。本轮落地：①权重对拍矩阵——gemm 三档按 `(1,1)/(5,1)/(1,7)/(16,12)` 形状配置化对拍 + 测试侧独立 i32 闭式点积重算（不经过 `Gemm::dot`）；gemv/dequant 既有参考对拍保留，极值/饱和字节不产 NaN。②零点 / 分组负例——`dequant::check_affine` 单点收口零标度、非有限标度/零点、零点越码域（i2 0..3、i4 0..15）；`gemv::finite_scales` 收口非有限标度；分组不整除、分组非打包倍数、权重长度不足既有骨架负例保留。③lane Program 设备适配——serve 桥接按 `Gemm` trait 泛型化（位宽/收缩轴取标记类型常量，码数 `codes_per_byte` 查表），gemv/dequant/gemm 各一条配置化宏展开；补齐原漏接的 `gemv_i2`/`dequant_i2`/`gemm_i2/i4/i8` 注册 Program，执行器路径对拍全绿 | 前置 M3-1；设备后端（Metal/CUDA）内核与跨后端同输入对拍未落地；FP8 / AWQ / GPTQ 新格式在 M3-3 之后按格式 ADR 接入；反量化内核的文件侧零点 / 半精度解码仍留后（内存夹具对拍口径） |
| M3-4 | 阻塞 | `shard::Remote` 占位后端与 `Local` 同一 `Collective` 端口；`all_reduce` 返回 `Unsupported` 且拒绝时不改调用方缓冲；`all_reduce_is_unsupported_and_leaves_the_buffer_untouched`、`local_and_remote_share_the_collective_port` | 前置 M3-1；真实 NCCL / MPI / Gloo / Metal 对等：组内握手、通信句柄池、拓扑绑定、心跳；集合对拍；拓扑簿记不改；大规模分布式暂缓 |
| M3-5 | 阻塞 | mem_stream 三档（l1/l2/dram）读写实测速率作 roofline 分母；regress 达成率 / 瓶颈分类分析件在位 | 前置 M3-2；按 decode 瓶颈序逐算子优化；同后端对比 burn/candle 需同机对照数字；wmma/wgmma 手写张量核暂缓 |

## 8. M4 模型与服务化 —— 阻塞边界与前置条件

| ID | 状态 | 现状 | 阻塞边界 / 前置条件 |
|---|---|---|---|
| M4-1 | 阻塞 | M1-A 源抽象 / Reader / Loader 在位；`.u` v2 容器读写 / REF 去重 / 外部格式导入 / 现场度量在位；中性 `VocabSource`、`Vocab::try_new`、`Fingerprint` 在位；Tokenizer 现由外部注入 | 前置 M1-A + M2 通路；`.u` v2 容器字节（含预留 `llm.vocab` 段）→ `VocabSource` 解析未接；给定真实权重跑通生成、词表指纹对拍；vision / 多模态暂缓 |
| M4-2 | 阻塞 | `Router` / `Endpoint` / `Request` / `Response` 契约与桩路由在位；serve lib 124 + protocol 8 | 前置 D8 端点契约稳定；真实 HTTP 后端替换桩路由；并发 / 流式通路；gRPC 在 HTTP 通路稳定后 |
| M4-3 | 阻塞 | 基准记录结构（M1-C2）与 mem_stream 分母在位 | 前置 M4-2；并发与 P50/P99、TTFT/TPOT、prefix 命中入表；同负载对照 vLLM / TRT-LLM 口径登记；通用 GPU 峰值竞争暂缓 |

**§3.8 排除边界（本阶段不做）**：投机解码 / beam / MoE / vision / RL / LoRA / FP8 / AWQ / GPTQ / KV 量化 / chunked prefill / 第三后端（MUSA/OpenCL/SYCL/ANE）/ gRPC；无同机对照不写性能倍数。

---

## 9. 优化轮度量对比（同口径脚本：`fn`、`for x in` 标量循环；基线 `baseline_metrics.json`）

| crate | 行 前→后 | 函数 前→后 | 标量循环 前→后 | 备注 |
|---|---|---|---|---|
| annotation | 1,641 → 1,641 | 98 → 98 | 3 → 3 | 解析游标串行依赖保留 |
| autograd | 2,094 → 2,094 | 154 → 154 | 23 → 23 | 未改算术 |
| backend | 256 → 313 | 27 → 31 | 0 → 2 | 占位骨架 + 守护测试；**该成员后已从工作树移除，本行仅为历史快照**；2 条循环在 `#[cfg(test)]` 内（BTreeSet 去重判定），生产路径零循环 |
| cli | 1,201 → 1,203 | 71 → 72 | 4 → 4 | `parse_slots` 合一 run/train 槽位解析 |
| cutile | 7,077 → 7,090 | 367 → 369 | 40 → 41 | 复用合一新增私有助手；1 条循环为台账分组归组 |
| data | 384 → 384 | 28 → 28 | 0 → 0 | 零改动 |
| diagnostic | 730 → 738 | 62 → 63 | 2 → 2 | 错误上下文单源收口 |
| dtype | 1,806 → 1,806 | 48 → 48 | 0 → 0 | 未改 |
| engine | 2,612 → 2,612 | 200 → 200 | 12 → 12 | 未改算术 |
| fuzz | 849 → 849 | 69 → 69 | 6 → 6 | 未改 |
| llm-bin | 1,889 → 1,911 | 61 → 61 | 14 → 14 | 根入口拆 run/status/train 三模块 |
| nn | 2,123 → 2,093 | 135 → 123 | 6 → 6 | `spec_accessors!` 宏合一五处手写委托，−30 行 / −12 函数 |
| op | 7,267 → 7,267 | 463 → 463 | 51 → 51 | 未改算术 |
| prim | 793 → 793 | 46 → 46 | 4 → 4 | 未改 |
| prof | 2,626 → 2,626 | 212 → 212 | 14 → 14 | 未改 |
| serve | 3,602 → 3,638 | 217 → 217 | 13 → 13 | `assembly.rs` 855→312，拆 `plan.rs`(335) / `block.rs`(236)；新类型全 `pub(crate)` |
| shard | 809 → 849 | 55 → 57 | 3 → 3 | `Remote` 占位后端 + 守护测试 |
| storage | 497 → 497 | 40 → 40 | 2 → 2 | 零改动 |
| tensor | 641 → 641 | 49 → 49 | 3 → 3 | 未改 |
| testkit | 449 → 449 | 36 → 36 | 6 → 6 | 未改 |
| tokenizer | 1,930 → 1,930 | 156 → 156 | 14 → 14 | 未改 |
| tool | 2,127 → 2,126 | 140 → 141 | 7 → 7 | 括号判定等私有助手合一；行 −1 |
| train | 3,276 → 3,275 | 245 → 247 | 14 → 14 | `decay_rate` / `reconcile` 助手；豁免只从角色派生 |
| weights | 5,926 → 5,933 | 462 → 465 | 40 → 40 | `read_wire` / `to_segment` / `nonnegative` / `pad`+`fold` 合一 |
| wire | 445 → 445 | 35 → 35 | 2 → 2 | 未改 |

> 生产路径未新增逐元素循环；函数 / 循环口径为仓库内正则扫描，测试件内循环计入上表。
>
> 快照口径：上表是优化轮收口当时的前后对比。其后入口编排迁移（根 bin 并入 `llm-cli`，`llm-bin` 的行分布归 cli 包）与 engine `View`/`ViewMut` 泛化适配使行 / 函数分布再次移动，未重起一轮同基线前后对比；生产路径循环数未因此增加。

### 保留循环及理由（均属串行依赖或长度 / 位布局数据相关，不可整块批处理）

- **weights / storage**：rANS 逐符号推进频率 / 区间（编解码串行依赖）；SHA-256 消息调度与 Merkle 层级折叠（哈希 / 位运算串行）；量化块反量化逐 lane 解包位掩码 × 标度乘法（位布局数据相关）；描述符 / 条目逐条解析（长度数据相关）；`BTreeSet` 名字去重 + 相邻区间检查；storage `remap` 按两套 strides 逐元素重排（`T: Copy` 泛型，布局不同无法整块 memcpy）；坐标轴分解链式串行。
- **train**：优化器矩状态 / 参数写回（逐元素依赖现值与自槽梯度，长度随槽位数据）；快照 f32 表编解码（无 unsafe 前提下 LE 字节转换不能整块重解释）；autograd DAG 扇出 / 深度 / 可达遍历；account 扇出计数。
- **serve / engine / op / tensor / wire**：KV 逐行循环（行序即页行顺序，串行）；executor 借出-回滚事务；排队保图序；逐页 share/alloc/release 可失败回滚；序列步进失败回滚顺序确定；strides 步长累积；CRC 跨块状态串行链（块内已查表展开，尾字节串行）；op 归约 / softmax / 注意力参考 / 解包 / nibble 数学串行依赖。
- **annotation / tool / cutile**：解析游标；词法 token 组递归嵌套；分析器单趟扫描；LRU / 台账等级校验逐条判定；`BTreeSet` 去重逐条落盘。
- **夹具重复**：autograd `plan_tests`/`replay_tests`、train `session_tests` 自重复，按"第二个消费者到位再抽共享 fixture"原则未提前造抽象层。

## 10. 执行过程与瓶颈反馈

- **并行派发**：首轮四条施工轨道（A 加固 / B 装载 / C 训练 / D 推理）真并行派发，四子代理同槽位运行；第二轮仅 M1-C2 单轨（切片无并行争用面）。
- **串行瓶颈**：共享 `target/` 的 cargo 锁——并行轨构建排队互锁，期间各轨先做编辑 / 读码；文档计数与所有权回填在编排侧串行收口。未执行 `cargo clean`，指纹增量命中。
- **增量门耗时**：缓存命中后 `-p <pkg>` 的 check / clippy / fmt 为亚秒级；每轨收口只跑所属 crate 的增量门，未在中间轮次重跑全仓 doctest / criterion target / fuzz harness。
- **合并门耗时**（warm-cache，用户态）：`check` ≈ 0.07s、`clippy --all-features` ≈ 0.17s、`fmt --check` 亚秒；全量 `cargo test --workspace` 1,396 条为合并门主要墙钟耗时；arch/contracts/gate 三条约 0.16–0.44s 各自完成。冷全量重编耗时未单列（CI 环境口径）。
- **本轮收口实测**（warm-cache，2026-11-17 复核刀）：`check --workspace` 0.49s、`fmt --check` 0.35s、`architecture` 0.70s、`contracts` 0.91s、静态 gate 0.49s、nextest 1.92s（1365 条）、全量 clippy 4.73s（含 benches/fuzz 全部 target）。本轮未并行派发施工子代理：活树同时有外部会话在改 cli 入口，编排侧按单轨串行收口；cargo 仍串行化共享 `target/` 锁。交错写入期间出现过未完成编辑被构建误报编译失败，单包复跑即分辨，未回滚外部半成品。
- **P0-6 / M1-A4 / M1-B3 / T1-5 / M2-2 / M2-4 六轨收口实测**（warm-cache，2026-11-17 晚刀）：nextest **1,418/1,418**、libtest **1,454 / 0 / 1**；clippy / fmt / doc / deny / machete / bench `--no-run` / testkit 35/35 全绿；jscpd **1.36% 行**（118 clones / 1,093 行，421 文件）。三族测试夹具抽私有助手去重（serve model_tests、benches regress、train session/snapshot），用例零删除零改名。期间外部并行会话一次回滚曾把 fuzz `p06` 接线连同三条 dev 边从磁盘抹掉，已从提交 `3403d5f4` 恢复 `p06.rs` / `p06_drivers.rs` / 种子块并对齐当前 rustfmt；恢复后 p06 3 条 sweep 用例复跑全绿。注记（2026-11-17 复核）：该 p06 三族 sweep 接线随后按规范裁决移除（范式与 fuzz 入口解析器契约边界错配），上句"恢复 p06 并复跑全绿"为当时历史记录；当前活树 fuzz 仅保留 parsers 解析器目标与 9 个 harness，三族边界由各自仓内单测覆盖。
- **本轮四切片收口实测**（warm-cache，2026-11-17 晚刀二）：nextest **1,429/1,429**、libtest **1,452 / 0 / 1**、doc **43 / 0**；clippy `-D warnings` / fmt / deny / machete / bench `--no-run` / testkit 35/35 全绿；jscpd **1.34% 行**（116 clones / 1,090 行，420 文件，81,262 行）。S1 回载接缝初版两条助手 CCN 超 5，拆私有助手（矩表/累积表槽位对账分派）后静态门转绿；S2 初版回读侧照抄 weights 通用 Value/Reader 骨架造成跨 crate 261 行克隆，改为固定 schema 流式草稿读取器并抽测试夹具助手后，bench_json↔weights 对拍降到 2 对 26 行字节原语，bench_tests 自重复清零。本轮零新增生产 unsafe、零生产外部新依赖（`git diff HEAD -- '*Cargo.toml'` 空），依赖方向未反向；fuzz 三族 dev 边保持移除。活树另有外部并行会话（codex app-server / RustRover ACP）自动提交的 cutile probe / prof select 半成品，本刀未回滚、未纳入切片证据。
- **`.u` 容器 v2 落地刀实测**（warm-cache，2026-11-17 晚刀三）：nextest **1,470/1,470**、libtest（`--exclude llm-fuzz` 口径）**1,493 / 0 / 1**、doc **43 / 0**；clippy `-D warnings` / fmt / deny / machete / bench `--no-run` / 静态 gate 1/1 全绿；jscpd **1.48% 行**（132 clones / 1,242 行，425 文件，83,687 行，行阈 1.5%）。Stage1/2 落地：REF 跨段去重（哈希预检 + 逐位比对，打开期单色 DFS 校验引用图）、外部格式中性条目流导入（safetensors / GGUF 标量 / bin·pt；GGML 块量化显式 Unsupported）、打开后现场 `Metrics`、`import`/`inspect`/`check` CLI（ADR-0045）。CCN 超阈函数按职责轴拆私有助手（archive 引用回填 / writer 封段 / seg 描述符元数据与形态分派 / cli 命令与报告装配 / import 按通路分派）；外部会话新增 `rmsnorm_batch_segments` 复合长度校验 CCN=6，拆 `check_batch_rows` 后门转绿；serve 两层归档装载两条用例抽 `two_layer_archive_fixture` 夹具后重复块回落；diagnostic buildgen 两条临时目录用例在全仓并行下偶发同纳秒撞名串扰，`temp_dir()` 加固为 pid + 原子序号 + 纳秒后 libtest 口径复跑 1,493 / 0 / 1。口径注记：libtest 必须 `--exclude llm-fuzz`——`engine` feature 开启时 9 个 libFuzzer harness 是 `harness=false` 独立二进制，被普通 `cargo test` 直接拉起会进入无限 fuzz 循环（98.7% CPU），归 CI 独立 fuzz job 引导。本轮零新增生产 unsafe、零生产外部新依赖；Stage3 边界（码本/格码、分片索引、词表段解析、真实生成/HTTP/基准、稀疏布局）按 ADR-0045 未决节登记，不回填历史证人数字。活树外部会话的 op 批入口 / serve K·V 重排 / ADR-0046 半成品不属于本刀，未回滚、未纳入本刀证据。
- **批量投影接线复核刀实测**（warm-cache，2026-11-17 晚刀四）：nextest **1,474/1,474**、libtest（`--exclude llm-fuzz` 口径）**1,497 / 0 / 1**、doc **43 / 0**；clippy `-D warnings` / fmt / deny / machete / bench `--no-run` / 静态 gate 1/1 全绿；jscpd **1.47% 行**（132 clones / 1,231 重复行，425 文件，83,960 行，9,593 重复 token 1.75%，行阈 1.5%）。外部会话落 serve 行批投影：`gemv_f32` 单向量 / 行批分派，`project_rows` 按 `m/n` 切段逐行共享驻留权重；`gemv_inputs` 三路视图助手拆出后静态门 CCN 回落 ≤5；batch=2 端到端回归基线夹具（`decode_batch_two` + `logits_batch` 两序列对拍与 KV 归属）就位，`logits_batch` 当前仍走默认逐条回退，engine 整批几何接线未接。复核侧把单序列 / 双序列两条 decode 用例重复的装配块抽成 `block_fixture`（`Projections` 类型别名消解六元组 type-complexity），jscpd 从 1.52% 超阈回落 1.47%；diagnostic buildgen 临时目录加固沿用上刀，未回退。serve lib 130 + protocol 8 + doc 3。本轮零新增生产 unsafe、零生产外部新依赖；engine 图批量几何、缓冲扩批与整批前向仍为 M2-1/M2-2 剩余阻塞，不在本刀硬做。
- **所有权表与静态门复核刀实测**（warm-cache，2026-11-17 晚刀五）：nextest **1,480/1,480**、libtest（`--exclude llm-fuzz` 口径）**1,503 / 0 / 1**、doc **43 / 0**；clippy `-D warnings` / fmt / deny / machete / bench `--no-run` / testkit 36/36（architecture 8/8、contracts 6/6、cross_layer 4/4、gate 1/1）全绿；jscpd **1.49% 行**（135 clones / 1,258 重复行，425 文件，84,252 行，9,879 重复 token 1.79%，行阈 1.5%，exit 0）。修两处红灯：架构所有权门报新公开类型 `prof:KernelKey`（ADR-0027 §1 七元组登记键，`atom.rs`）既不在表也未基线——按"新类型进表、存量缺口才基线"口径补进 `docs/ARCHITECTURE.md` §3 prof 行；静态门报 `op::headed_batch_var` CCN=7（非零守卫三 `||` + 两 `?`），抽 `batch_geometry_nonzero` 私有助手后主函数 CCN 回落 ≤5。外部会话同刀落 prof KernelKey 形状用例与 op 变长历史批段入口（`decode_batch_var` 朴素/flash 两入口），prof lib 56、op lib 216；`tests/atom_tests.rs` 经 `#[path]` 编进 lib 子模块，原台账"prof atom 1"口径更正为 prof doc 1。§0 分布矩阵按逐包实测回填（含 tokenizer doc 1、testkit 合计 36/36）。本轮零新增生产 unsafe、零生产外部新依赖（`git diff '*Cargo.toml'` 空）；engine 图整批几何、缓冲扩批、含注意力块链整批 K/V 搬运与 `step_batch` 整批提交仍为 M2 剩余阻塞，不在本刀硬做。
- **M3-3 量化 lane 收口刀实测**（warm-cache，2026-11-18）：触及仓 libtest 口径 op **246 + 2**、serve **148 + 8 + 3**、engine **77 + 9 + 3**、weights **129 + 3 + 12 + 3**、nn **79 + 2**，全绿 0 失败；`cargo clippy -p llm-op -p llm-serve -p llm-engine --all-targets` 无 warning / error；本刀触及的 op / serve / engine / weights 文件 `fmt --check` 干净。落地三条：①gemm 三档 `(1,1)/(5,1)/(1,7)/(16,12)` 形状矩阵对拍，测试侧独立手写 i32 闭式累加重算（不经过 `G::dot`/`matmul`，避免参考与内核自拍）；②`dequant::check_affine` 单点收口零标度、非有限标度/零点、零点越码域（i2 0..3、i4 0..15；i8 样本码域 0..15 按 4 位传宽），`gemv::finite_scales` 收口非有限标度；③serve 桥接按 `Gemm` trait 泛型化，gemv/dequant/gemm 各一条配置化宏展开，补齐原漏接的 `gemv_i2`/`dequant_i2`/`gemm_i2/i4/i8` 注册 Program，执行器路径对拍全绿。静态门：初版 `gemv_i8::forward` 三条 `?` 守卫使 Ccn=6，把守卫与行切片下沉到骨架 `packed::forward_plain`（逐行点积只留闭包）后该函数从超标清单消失；serve 两条复杂函数指针抽 `GemvQuantFn` / `GemmQuantFn` 类型别名消 type-complexity。口径注记：静态门剩余 4 条红灯（`train/src/session.rs:optimize`、`train/src/backprop.rs:reverse_step`、`train/src/dist.rs:sync_grads` 与 Call=1）与 fmt 残留 diff（`rng/src/dist.rs`、`train/src/backprop.rs`、`train/src/session.rs`）均为基线他人分布式梯度/`gradient_mut` 未完成重构，`git stash` 撤掉本刀改动后仍复现，不属于本刀范围、未纳入本刀证据；设备后端内核与跨后端对拍、FP8/AWQ/GPTQ、文件侧零点/半精度解码仍为 M3-1/M3-2 与后续格式 ADR 边界。本轮零新增生产 unsafe、零生产外部新依赖。
