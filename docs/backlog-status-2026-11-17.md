# Backlog 收口矩阵（2026-11-17 活树复核）

> 口径：状态分 **已落地 / 进行中 / 阻塞**；测试证据按 `crate + 测试数或用例名` 给出，计数为
> `cargo test --workspace` 收口实测。门禁不削弱、负例不回写；历史快照刀次记录不回填。
> 计划编号与验收门见 [`implementation-task-plan.md`](implementation-task-plan.md) §3；
> 不退化清单见 [`llmu-parity-optimization-design.md`](llmu-parity-optimization-design.md) §1.5。

## 0. 收口门禁实测（合并门，全仓口径）

| 门 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | 干净，0 error / 0 warning |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 零告警 |
| `cargo fmt --all --check` | 0 diff |
| `cargo test --workspace` | **1,396 passed / 0 failed / 1 ignored** |
| `cargo nextest run --workspace --all-features` | **1,360/1,360 passed** |
| architecture | **8/8**；contracts **6/6**；静态 gate **1/1** |
| `cargo deny check` | advisories / bans / licenses / sources 四项 ok |
| `cargo machete` | 无未用依赖 |
| jscpd | 1.44% 行 / 1.73% token，111 clones，exit 0（行阈 1.5%） |
| `cargo bench --workspace --all-features --locked --no-run` | exit 0，全部基准目标编译通过 |
| `unsafe` | 生产唯一岛 `crates/storage/src/mmap.rs`；本轮新增 unsafe 为零；backend/metal/cuda 零 unsafe |

各成员 lib 测试数（`cargo test -p <pkg> --lib` 实测）：annotation 34、autograd 65、backend 3、cli 27、
cutile 167、data 16、diagnostic 12、dtype 18、engine 71、fuzz 17、nn 79、op 174、prim 19、prof 45、
serve 113、shard 27、storage 10、tensor 25、testkit 17、tokenizer 36、tool 51、train 75、weights 113、wire 13。

---

## 1. P0 架构 / 质量加固

| ID | 状态 | 测试证据（crate + 用例 / 计数） | 未完成原因 / 暂缓 |
|---|---|---|---|
| P0-1 | 已落地 | 文档计数与指针回填：design §1.2/§1.5、plan D-1…D-7、roadmap §3、code-audit；gate 1/1、architecture 8/8 守住文档链接门 | 历史快照刀次记录不回写：`backend-runtime-design.md:472` 仍记该刀 1325 / contracts 5，属当时快照 |
| P0-2 | 已落地 | annotation：`OP_OVERRIDES` 一行三列 `(名, 参考覆盖, 模块覆盖)`；annotation lib 34 + audit 集成 2；无第二份按名查找循环 | — |
| P0-3 | 已落地 | 大文件按职责轴拆分：weights `bin.rs` → `bin_zip.rs` / `bin_pickle.rs`；cutile `main.rs` → `report.rs`；train `snapshot.rs` → `cursor/entry/state`；根 `main.rs` 837 → 78，拆 `run.rs` / `status.rs` / `train.rs`；gate 行数 ≤899、ccn/nloc 过门 | 其余大文件随后续阶段内拆；根入口四条债务条目登记在 `.config/lints/vocabulary.toml`，待排版下沉 tool 后删除 |
| P0-4 | 已落地 | 报告 / 排版收敛 `tool::Report` 单一报告器；tool lib 51（含 ccn 1、gate 1） | — |
| P0-5 | 已落地 | 手写解析收敛 diagnostic 单源；ADR 0039 草案记依赖策略；diagnostic lib 12 + 集成 1，解析负例全绿 | toml+serde 唯一解析依赖未拍板前不接（D5） |
| P0-6 | 进行中 | 切片内纯函数属性用例 + fuzz sweep 骨架在位：fuzz lib 17 + parsers 集成 2；`check/sweep/sweep_all` 契约在 | tokenizer / op / prof 的属性边界用例在切片外，尚未逐族补齐；libfuzzer harness 仅 feature 门控 |
| P0-7 | 已落地 | 注解 / 声明 / 模板三方参数类型签名交叉校验；注册表 24 条全过；codegen 检查模块 `the_annotation_still_matches_the_declared_operators` | 跨设备文本同构校验随后端阶段 |
| P0-8 | 已落地 | 已具备能力常驻回归：weights `tests/round_trip.rs` 12 条多格式往返；prof Grade L0–L3 闭集往返（prof lib 45 + 1） | — |

## 2. M1-A 装载与驻留权重

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-A1 | 已落地 | weights lib 113；`Reader`/`Loader` 收口，边界检查单点；装载负例（缺条目 / 非 F32 / 尾半字 / 长度不符）全绿 | 远端分块源未接 |
| M1-A2 | 已落地 | `weights::MmapSource`（mmap feature 双门控）复用 `storage::Mmap` 唯一 unsafe 岛；`mmap_source_tests.rs`；mmap 口径 weights 116 lib / storage 13 lib；压缩段保持 `Owned` 断言 | 设备权重驻留、热更新暂缓 |
| M1-A3 | 已落地 | `load_f32` / `decode_entry_f32` 驻留解码；`load_f32_decodes_an_entry_into_the_resident_buffer`、`a_non_f32_entry_is_not_unpacked_into_the_resident_buffer`、`decode_entry_f32_passes_through_the_partial_tail_word_rejection`；`StubLoader` 测试件 | 真实权重端到端装载属 M4-1 |
| M1-A4 | 进行中 | `benches/criterion/weights_load.rs`：File vs Mmap 首条 / 全量、冷热 `*_reopen` / `*_open`，出 ns/iter 与字节吞吐 | 未登记回归基线（跨机绝对值不可比）；RSS 进程读数无公开接口，未做 RSS 采样；冷热缓存对照入 regress 基线待后续刀 |

## 3. M1-B CPU 单序列前向闭环

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-B1 | 已落地 | serve `KERNELS` const 子集表 + 注册表子集断言；未覆盖名查表返回 `Unsupported`，无静默空入口；serve lib 113 | 标量槽通道归 B6 |
| M1-B2 | 已落地 | op lib 174 条标量参考对拍 + 2 集成；model_tests：`gemv_runs_through_the_registered_kernel`、`rmsnorm_runs_...`、`binary_elementwise_runs_...`、`unary_elementwise_runs_...`、`softmax_and_reduce_run_...`、`attention_runs_...`；`testkit::close` 双算预算 | rope 主链、注意力主链、量化 i4/i8 lane 暂缓（见 M2-1 / M3-3） |
| M1-B3 | 进行中 | `plan.rs` 族别计划 + 宽度链 / 权重校验：Projection、Embedding、Unary（gelu/scale/silu/softmax）、Norm（rmsnorm）已接线；`block.rs` 注意力块 q/k/v/o + 块内残差接线；宽度断裂 / 权重缺失装配期拒绝 | standalone 二元（`add_f32`/`mul_f32`/`residual_add_f32`）缺旁路第二输入通道，装配链当前单输入；归约、rope 位置编码未进装配链；两层合成模型端到端对拍未闭环；多 token 矩阵主链暂缓 |
| M1-B4 | 已落地 | `testkit::close` 绝对 + 相对双预算；model_tests `assert_close` 双算；testkit lib 17 + 1 | 第二份独立闭式参考暂缓 |
| M1-B5 | 已落地 | `Classical` 经典采样（温度缩放 → top-k → top-p 核采样 → argmax，不接 RNG）接入 `Completion`；坏温度 / 空候选 / top-p 越界负例；`nucleus_keeps_the_probability_prefix` | top-p 校准表细化暂缓 |
| M1-B6 | 已落地 | engine `ScalarSlots` 定长类型化闭集（首批 `rope_position: Option<u32>`），无字符串键 / 无 HashMap；ADR 0043；`scalar_slots_start_empty`、`a_rope_position_round_trips_through_the_node` | heads / head_dim 槽位随 M2-1 多头通路补；多 token 矩阵几何暂缓 |
| M1-B7 | 已落地 | `BindingDesc{id, entry, shape}` 留 serve 装配侧；engine `Bindings` 纯 `BTreeMap<u32, Vec<f32>>`；ADR 0044；`descriptors_map_ids_back_to_entries_and_shapes` | 设备侧形状元数据暂缓；只读形状访问器待后端实现证明需要再补 |

## 4. M1-C 基准 harness 骨架

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-C1 | 已落地 | `scripts/mechanism_lines.sh` 可复跑；`mechanism-lines-baseline.md` 分账表 233 文件 / 50,713 机制行（L1–L7 分层）；annotation / fuzz / benches / tests / build.rs / vendor 排除口径入档 | 严格口径（逐行扣除内联 `#[cfg(test)]` 体）尚未接 `strip_tests` 进脚本；对标框架同口径分母未采集 |
| M1-C2 | 已落地 | `tool::bench`：`EntryKind` / `Harness` / `Geometry` / `Roofline` / `BenchRecord`，构造自检；tool bench 结构用例 6 条；regress 算子报告与上界报告元数据已迁入统一记录，`weights::json` 自检保留；benches regress 4 用例 | `model_decode` 的 `EntryKind::Model` 条目无归档落地点（基准不进 regress 基线表）；整机峰值 BW / GFLOPS 未测得，Roofline 留 `None`；批次 TPOT / 吞吐 / 内存曲线入表属 M2-4；跨机绝对值结论不出 |

## 5. T1 训练最小闭环

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| T1-1 | 已落地 | `Optimizer` 对象安全 trait + SGD / Adam（矩估计、偏差修正、AdamW 解耦衰减、全局裁剪）；train lib 75；`sgd_takes_one_direct_step`、`adam_moves_the_param_with_momentum`、`adam_moments_round_trip_through_restore`、`restore_rejects_mismatched_moment_tables`、`global_norm_clipping_scales_every_slot`、`a_length_mismatch_is_rejected_before_update`、`weight_decay_applies_only_outside_exempt_groups`、`optimizer_is_object_safe` | 分布式优化器暂缓 |
| T1-2 | 已落地 | `ContractSpec.role: ParamRole` 闭集；`group_mapping_follows_the_role`；wd 豁免 / 可训练判定只从角色 `wd_exempt()` 派生，优化器不猜字段名 | — |
| T1-3 | 已落地 | tape → 反传计划 → `Requirement::Grad` 终点 → 裁剪 → 优化器步；梯度缓冲归 train `Buffers` 自持，autograd 只出 tape/plan/Replay；`a_fixture_dataset_runs_the_declared_steps`、`a_mismatched_batch_shape_is_rejected`、`stepping_without_a_gradient_is_rejected`；autograd 65 lib + gradcheck 3 + 2 集成；data 16 + 4 | 真实大语料通路暂缓 |
| T1-4 | 已落地 | CLI `train` 命令；快照 v4：Facts 五字段（step/world/layers/batch_rows/batch_cols）+ 矩状态段 + 累积梯度段，29 字节 golden 钉版；`a_golden_vector_pins_the_version_four_layout`、`another_version_says_which_side_is_old`、`optimizer_moments_round_trip_bit_for_bit`、`accumulated_gradients_round_trip_bit_for_bit`、`a_slot_with_uneven_moments_is_rejected_before_encoding`；cli 27 lib + 2 集成 | offload / tie 回填随 T1-5 接缝留口 |
| T1-5 | 进行中 | 梯度累积：`accumulate_gradient` / `accumulated_microbatches`；`gradients_accumulate_over_the_window_before_one_step`；累积窗口梯度入快照同格式往返；`register_tie` / `backfill_ties` 接缝 + `tied_parameter_copies_source_on_backfill` | tie 在模型侧的真实装配尚无绑定夹具调用方；offload 真实换入换出通路未接；ZeRO / 分布式并行暂缓 |

## 6. M2 推理运行时（KV 与连续批）

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M2-1 | 阻塞 | 守护用例在案：`a_multi_head_block_is_not_assembled_yet`；B6 标量槽、B7 描述表已就位 | executor 图仍是单序列几何：多层层偏移行切片、多头展平头内切片、注意力 Program 接 KV 行映射均未接；前置 M2-2 批量几何 + B6 槽位；头内并行特化更后 |
| M2-2 | 进行中 | 调度批守回归：`Batch::step` 遍历序列出 `Progress`；`two_admitted_sequences_step_together_and_keep_their_own_history`、`an_admitted_sequence_is_owned_by_the_batch`；engine 71 lib + schedule 9 | 模型内核级批量几何未接：executor 图未扩到批内多序列一步，`Predict` 保持单序列视图；前置 M1-B5；序列并行更后 |
| M2-3 | 已落地 | `Batch::prefix_share` + `Scheduler::admit_shared`，经既有 `Pager::share/release` 引用计数；`identical_prefix_is_adopted_as_shared_pages`、`shared_prefix_row_survives_source_finish_until_last_holder_releases`、`diverging_prefix_takes_fresh_pages`、`prefix_shorter_than_one_page_is_not_split` | 跨请求前缀复用预热暂缓 |
| M2-4 | 进行中 | `model_decode.rs` 单序列 / 四条序列同批变体 criterion 算术保留：ns/iter、tokens/s、页池 `in_use()` 可观测账目；mem_stream 三档 roofline 分母在位 | 未接 `EntryKind::Model` 归档；批次 TPOT / 吞吐 / 内存曲线未入表；无同机 llama.cpp 对照；前置 M2-2 |

---

## 7. M3 后端边界 / 量化 / 性能 —— 阻塞边界与前置条件

**总前置（未满足前 M3 不动工）**：
1. **ADR-0035 仍为 Proposed（待评审）**：本稿只定 unsafe/FFI 边界规则，**不是 M3 实现放行**；不得标 Accepted，不得据此展开后端真实 FFI / 设备内核。评审通过后作为 M3-1 前置约束。
2. **本机无 GPU**：MBP（macOS/aarch64）无 CUDA；云主机 linux 无 Rust 工具链、无 NVIDIA。无 GPU 环境只做编译 + 契约 / 形状对拍，不出设备吞吐数字。
3. **D6 形状约束**：第二后端出现前不预埋设备泛型参数；四 trait（`Executor`/`Allocator`/`Collective`/`Program`）已定形。

| ID | 状态 | 现状（已落地骨架） | 阻塞边界 / 前置条件 |
|---|---|---|---|
| M3-1 | 阻塞 | `backend` 占位适配：`UnsupportedBackend` + `MetalBackend`/`CudaBackend` 能力候选表（`MOUNTED` 名字均在 op 注册表内、不重复）；五个运行入口统一 `Code::Unsupported`，`free` 留空；`vendor_labels_round_trip_per_backend`、`mounted_candidates_belong_to_the_registry`、`placeholders_share_the_same_unsupported_contract`；零 FFI crate、零 unsafe | 前置 0035 评审通过；新建边界成员的依赖方向守门；边界 crate 之外零新增 unsafe；第三加速链路暂缓 |
| M3-2 | 阻塞 | 同一 `Executor`/`Allocator` 端口契约在位（占位三件套同入口行为对拍） | 前置 M3-1；Metal 本机跑通 M2 同一前向、CUDA 在 Linux+NVIDIA 验；同输入同形状对拍一致；无 GPU 不出吞吐数字；图 capture 特化暂缓 |
| M3-3 | 阻塞 | `gemv_i4`/`gemv_i8`/`gemm_i4`/`gemm_i8`/`dequant_i4`/`dequant_i8` 骨架在 `gemv.rs`/`gemm.rs` 泛型内；注册表 24 条含 6 条量化 lane | 前置 M3-1；i4/i8 lane Program 适配 + 反量化内核挂载；量化权重对拍、零点 / 分组负例；FP8 / AWQ / GPTQ 新格式在 M3-3 之后按格式 ADR 接入 |
| M3-4 | 阻塞 | `shard::Remote` 占位后端与 `Local` 同一 `Collective` 端口；`all_reduce` 返回 `Unsupported` 且拒绝时不改调用方缓冲；`all_reduce_is_unsupported_and_leaves_the_buffer_untouched`、`local_and_remote_share_the_collective_port` | 前置 M3-1；真实 NCCL / MPI / Gloo / Metal 对等：组内握手、通信句柄池、拓扑绑定、心跳；集合对拍；拓扑簿记不改；大规模分布式暂缓 |
| M3-5 | 阻塞 | mem_stream 三档（l1/l2/dram）读写实测速率作 roofline 分母；regress 达成率 / 瓶颈分类分析件在位 | 前置 M3-2；按 decode 瓶颈序逐算子优化；同后端对比 burn/candle 需同机对照数字；wmma/wgmma 手写张量核暂缓 |

## 8. M4 模型与服务化 —— 阻塞边界与前置条件

| ID | 状态 | 现状 | 阻塞边界 / 前置条件 |
|---|---|---|---|
| M4-1 | 阻塞 | M1-A 源抽象 / Reader / Loader 在位；中性 `VocabSource`、`Vocab::try_new`、`Fingerprint` 在位；Tokenizer 现由外部注入 | 前置 M1-A + M2 通路；容器字节 → `VocabSource` 解析未接；给定真实权重跑通生成、词表指纹对拍；vision / 多模态暂缓 |
| M4-2 | 阻塞 | `Router` / `Endpoint` / `Request` / `Response` 契约与桩路由在位；serve lib 113 + protocol 8 | 前置 D8 端点契约稳定；真实 HTTP 后端替换桩路由；并发 / 流式通路；gRPC 在 HTTP 通路稳定后 |
| M4-3 | 阻塞 | 基准记录结构（M1-C2）与 mem_stream 分母在位 | 前置 M4-2；并发与 P50/P99、TTFT/TPOT、prefix 命中入表；同负载对照 vLLM / TRT-LLM 口径登记；通用 GPU 峰值竞争暂缓 |

**§3.8 排除边界（本阶段不做）**：投机解码 / beam / MoE / vision / RL / LoRA / FP8 / AWQ / GPTQ / KV 量化 / chunked prefill / 第三后端（MUSA/OpenCL/SYCL/ANE）/ gRPC；无同机对照不写性能倍数。

---

## 9. 优化轮度量对比（同口径脚本：`fn`、`for x in` 标量循环；基线 `baseline_metrics.json`）

| crate | 行 前→后 | 函数 前→后 | 标量循环 前→后 | 备注 |
|---|---|---|---|---|
| annotation | 1,641 → 1,641 | 98 → 98 | 3 → 3 | 解析游标串行依赖保留 |
| autograd | 2,094 → 2,094 | 154 → 154 | 23 → 23 | 未改算术 |
| backend | 256 → 313 | 27 → 31 | 0 → 2 | 占位骨架 + 守护测试；2 条循环在 `#[cfg(test)]` 内（BTreeSet 去重判定），生产路径零循环 |
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
