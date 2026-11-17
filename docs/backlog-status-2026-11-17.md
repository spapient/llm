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
| `cargo nextest run --workspace --all-features --locked` | **1,418/1,418 passed，0 skipped**（nextest 口径，9 个 libfuzzer harness + gen_corpus 二进制被 `.config/nextest.toml` 排除） |
| `cargo test --workspace`（libtest 口径；fuzz libFuzzer harness 由 CI fuzz job 单独引导，不进 libtest） | 收口实测 **1,454 passed / 0 failed / 1 ignored**；与 nextest 是两种口径，计数不可混用 |
| architecture | **8/8**；contracts **6/6**；静态 gate **1/1** |
| `cargo deny check` | advisories / bans / licenses / sources 四项 ok（许可白名单含 NCSA、Unicode-3.0） |
| `cargo machete` | 无未用依赖 |
| jscpd | 1.36% 行 / 1.63% token，118 clones，421 文件，80,553 行，exit 0（行阈 1.5%；token 未设硬阈） |
| `cargo bench --workspace --all-features --locked --no-run` | exit 0，全部基准目标编译通过 |
| `unsafe` | 生产唯一岛 `crates/storage/src/mmap.rs`；本轮新增 unsafe 为零；Metal/CUDA 后端未落地，设备侧零 unsafe |

各成员 lib 测试数（`cargo test -p <pkg> --lib` 实测）：annotation 34、autograd 65、cli 25、
cutile 167、data 16、diagnostic 12、dtype 18、engine 77、fuzz 17、nn 79、op 198、prim 19、prof 48、
serve 124、shard 29、storage 13、tensor 25、testkit 17、tokenizer 44、tool 51、train 80、weights 116、wire 13。
集成 / 文档补充：fuzz parsers 2；tokenizer tokens 16；op 集成 2；prof atom 1；benches regress 8；
serve protocol 8 + doc 3；engine schedule 9 + doc 3；nn 集成 2；train doc 3。

---

## 1. P0 架构 / 质量加固

| ID | 状态 | 测试证据（crate + 用例 / 计数） | 未完成原因 / 暂缓 |
|---|---|---|---|
| P0-1 | 已落地 | 文档计数与指针回填：design §1.2/§1.5、plan D-1…D-7、roadmap §3、code-audit；gate 1/1、architecture 8/8 守住文档链接门 | 历史快照刀次记录不回写：`backend-runtime-design.md:472` 仍记该刀 1325 / contracts 5，属当时快照 |
| P0-2 | 已落地 | annotation：`OP_OVERRIDES` 一行三列 `(名, 参考覆盖, 模块覆盖)`；annotation lib 34 + audit 集成 2；无第二份按名查找循环 | — |
| P0-3 | 已落地 | 大文件按职责轴拆分：weights `bin.rs` → `bin_zip.rs` / `bin_pickle.rs`；cutile `main.rs` → `report.rs`；train `snapshot.rs` → `cursor/entry/state`；入口编排拆 `cmd/run.rs` / `cmd/status.rs` / `cmd/train.rs`；gate 行数 ≤899、ccn/nloc 过门 | 其余大文件随后续阶段内拆；入口编排函数已重构回 ccn/nloc 阈值内，`[gate.debt]` 存量为零，无待删基线条目 |
| P0-4 | 已落地 | 报告 / 排版收敛 `tool::Report` 单一报告器；tool lib 51（含 ccn 1、gate 1） | — |
| P0-5 | 已落地 | 手写解析收敛 diagnostic 单源；ADR 0039 草案记依赖策略；diagnostic lib 12 + 集成 1，解析负例全绿 | toml+serde 唯一解析依赖未拍板前不接（D5） |
| P0-6 | 部分落地 | 三族纯函数表驱动边界用例落在各自仓内单测：tokenizer 44 lib + tokens 16；op 198 lib + 2 集成；prof 48 lib + atom 1；op 边界新增 mul / scale / embed / rope / dequant_i4 / dequant_i8 六条空值/非对齐/越界/饱和/重复对拍。fuzz 侧只保留不可信字节入口解析器目标（parsers 2 + 9 个 libfuzzer harness，`harness=false` + `required-features=["engine"]` 门控）；三族内部纯函数 / 结构化张量接进 sweep 的 p06 接线范式与 fuzz 契约边界错配，已按规范移除，`tokenizer`/`op`/`prof` 不进 fuzz `[dev-dependencies]`，零新生产依赖 | prof `eval_expression` 为 `pub(super)` 未接 fuzz（仓内单测覆盖）；`DType` 名字往返由 `dtype/tests/properties.rs` 覆盖；`--all-features` libfuzzer 链接编译本轮未跑 |
| P0-7 | 已落地 | 注解 / 声明 / 模板三方参数类型签名交叉校验；注册表 24 条全过；codegen 检查模块 `the_annotation_still_matches_the_declared_operators` | 跨设备文本同构校验随后端阶段 |
| P0-8 | 已落地 | 已具备能力常驻回归：weights `tests/round_trip.rs` 12 条多格式往返；prof Grade L0–L3 闭集往返（prof lib 48 + 1） | — |

## 2. M1-A 装载与驻留权重

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M1-A1 | 已落地 | weights lib 116；`Reader`/`Loader` 收口，边界检查单点；装载负例（缺条目 / 非 F32 / 尾半字 / 长度不符）全绿 | 远端分块源未接 |
| M1-A2 | 已落地 | `weights::MmapSource`（mmap feature 双门控）复用 `storage::Mmap` 唯一 unsafe 岛；`mmap_source_tests.rs`；mmap 口径 weights 116 lib / storage 13 lib；压缩段保持 `Owned` 断言 | 设备权重驻留、热更新暂缓 |
| M1-A3 | 已落地 | `load_f32` / `decode_entry_f32` 驻留解码；`load_f32_decodes_an_entry_into_the_resident_buffer`、`a_non_f32_entry_is_not_unpacked_into_the_resident_buffer`、`decode_entry_f32_passes_through_the_partial_tail_word_rejection`；`StubLoader` 测试件 | 真实权重端到端装载属 M4-1 |
| M1-A4 | 已落地 | `regress.rs` 装载 8 条 Op 记录归档（File 4 + mmap 4，`#[cfg(unix)]`）；`registered_load_records_archive_when_measured`、`load_records_carry_every_field_a_reader_needs_to_judge_it`；Harness / Geometry / Roofline / observed 全字段入台账；Roofline 双 None，不采 RSS；criterion `weights_load.rs` 测量循环保留 | 未采 RSS 进程读数；无跨机基线结论；tree 内无 criterion `estimates.json`，归档测试走 skip 路径；冷热缓存对照入 regress 基线待后续刀 |

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
| M1-C1 | 已落地 | `scripts/mechanism_lines.sh` 可复跑；`mechanism-lines-baseline.md` 分账表 233 文件 / 50,713 机制行（L1–L7 分层）；annotation / fuzz / benches / tests / build.rs / vendor 排除口径入档 | 严格口径（逐行扣除内联 `#[cfg(test)]` 体）尚未接 `strip_tests` 进脚本；对标框架同口径分母未采集 |
| M1-C2 | 已落地 | `tool::bench`：`EntryKind` / `Harness` / `Geometry` / `Roofline` / `BenchRecord`，构造自检；tool bench 结构用例 6 条；regress 算子报告与上界报告元数据已迁入统一记录，`weights::json` 自检保留；benches regress 8 用例（装载 4 Op + 模型 2 Model + 既有 4 条基线 / 上界 / 主机 / 字段） | 整机峰值 BW / GFLOPS 未测得，Roofline 留 `None`；批次 TPOT / 吞吐 / 内存曲线入表属 M2-4 后续；跨机绝对值结论不出 |

## 5. T1 训练最小闭环

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| T1-1 | 已落地 | `Optimizer` 对象安全 trait + SGD / Adam（矩估计、偏差修正、AdamW 解耦衰减、全局裁剪）；train lib 80；`sgd_takes_one_direct_step`、`adam_moves_the_param_with_momentum`、`adam_moments_round_trip_through_restore`、`restore_rejects_mismatched_moment_tables`、`global_norm_clipping_scales_every_slot`、`a_length_mismatch_is_rejected_before_update`、`weight_decay_applies_only_outside_exempt_groups`、`optimizer_is_object_safe` | 分布式优化器暂缓 |
| T1-2 | 已落地 | `ContractSpec.role: ParamRole` 闭集；`group_mapping_follows_the_role`；wd 豁免 / 可训练判定只从角色 `wd_exempt()` 派生，优化器不猜字段名 | — |
| T1-3 | 已落地 | tape → 反传计划 → `Requirement::Grad` 终点 → 裁剪 → 优化器步；梯度缓冲归 train `Buffers` 自持，autograd 只出 tape/plan/Replay；`a_fixture_dataset_runs_the_declared_steps`、`a_mismatched_batch_shape_is_rejected`、`stepping_without_a_gradient_is_rejected`；autograd 65 lib + gradcheck 3 + 2 集成；data 16 + 4 | 真实大语料通路暂缓 |
| T1-4 | 已落地 | CLI `train` 命令；快照 v5：Facts 五字段（step/world/layers/batch_rows/batch_cols）+ 矩状态段 + 累积梯度段 + ties 段，30 字节 golden 钉版；`a_golden_vector_pins_the_version_five_layout`、`another_version_says_which_side_is_old`、`optimizer_moments_round_trip_bit_for_bit`、`accumulated_gradients_round_trip_bit_for_bit`、`a_slot_with_uneven_moments_is_rejected_before_encoding`；cli 25 lib + 集成 | offload 真实换入换出通路未接（接缝见 T1-5） |
| T1-5 | 已落地 | 快照格式 v4→v5，新增 ties 段；`accum_snapshot()` / `ties()` / `ties_snapshot()` 导出接缝；offload 只留 `WeightOffload` trait + `DeferredOffload` 占位（一律 `Code::Unsupported`）；两层共享权重夹具回填对拍。train 80 lib + doc 3；用例 `session_window_state_round_trips_through_snapshot`、`tied_parameter_copies_source_across_a_two_layer_fixture`、`accumulated_window_sums_count_and_ties_round_trip_bit_for_bit`、`tie_bindings_round_trip_pair_for_pair`、`an_empty_accumulation_window_round_trips_without_state` | 快照回载到会话的装配通路（ties / accum restore 接线）未建；`cli --load` 汇报未加 accum / ties 行；offload 仅接口接缝；ZeRO / 分布式优化器 / 序列并行暂缓 |

## 6. M2 推理运行时（KV 与连续批）

| ID | 状态 | 测试证据 | 未完成原因 / 暂缓 |
|---|---|---|---|
| M2-1 | 阻塞 | 守护用例在案：`a_multi_head_block_is_not_assembled_yet`；B6 标量槽、B7 描述表已就位；M2-2 批量结果视图与页表守护已落地 | executor 图仍是单序列几何：多层层偏移行切片、多头展平头内切片、注意力 Program 接 KV 行映射均未接；前置 B6 槽位 + M2-2 视图已就位，剩内核级多序列矩阵主链；头内并行特化更后 |
| M2-2 | 已落地 | `Program::run` 已是通用视图通道，无单序列硬编码；补 `SequenceOutcome<'a>` 只读结果视图、`Batch::sequence_at` / `outcome` / `outcomes`；页表引用计数守护。engine 77 lib + schedule 9 + doc 3；用例 `multi_sequence_batch_runs_their_own_history_and_progress`、`batch_outcomes_pair_each_progress_with_its_own_sequence`、`admission_full_handles_back_without_touching_the_page_accounts`、`finishing_a_sequence_releases_all_of_its_pages`、`shared_prefix_reference_counting_is_correct_across_finish_order`、`shared_prefix_refcount_supports_three_holders` | `Predict`（serve 侧）仍是单序列几何；engine 只提供批量结果视图；内核级多序列矩阵主链暂缓 |
| M2-3 | 已落地 | `Batch::prefix_share` + `Scheduler::admit_shared`，经既有 `Pager::share/release` 引用计数；`identical_prefix_is_adopted_as_shared_pages`、`shared_prefix_row_survives_source_finish_until_last_holder_releases`、`diverging_prefix_takes_fresh_pages`、`prefix_shorter_than_one_page_is_not_split` | 跨请求前缀复用预热暂缓 |
| M2-4 | 已落地 | `regress.rs` 模型 2 条 `EntryKind::Model` 记录（batch=1 / batch=4），Geometry 批次维入台账；`registered_model_records_archive_when_measured`、`model_records_carry_every_field_a_reader_needs_to_judge_it`；criterion `model_decode.rs` 单序列 / 四条序列同批变体测量循环保留；mem_stream 三档 roofline 分母在位；Roofline 双 None | 归档 JSON 未生成；不接 llama.cpp / vLLM 同机对照、不出倍数；批次 TPOT / 吞吐 / 内存曲线入表暂缓；设备侧数字待 M3 |

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
| M3-3 | 阻塞 | `gemv_i4`/`gemv_i8`/`gemm_i4`/`gemm_i8`/`dequant_i4`/`dequant_i8` 骨架在 `gemv.rs`/`gemm.rs` 泛型内；注册表 24 条含 6 条量化 lane | 前置 M3-1；i4/i8 lane Program 适配 + 反量化内核挂载；量化权重对拍、零点 / 分组负例；FP8 / AWQ / GPTQ 新格式在 M3-3 之后按格式 ADR 接入 |
| M3-4 | 阻塞 | `shard::Remote` 占位后端与 `Local` 同一 `Collective` 端口；`all_reduce` 返回 `Unsupported` 且拒绝时不改调用方缓冲；`all_reduce_is_unsupported_and_leaves_the_buffer_untouched`、`local_and_remote_share_the_collective_port` | 前置 M3-1；真实 NCCL / MPI / Gloo / Metal 对等：组内握手、通信句柄池、拓扑绑定、心跳；集合对拍；拓扑簿记不改；大规模分布式暂缓 |
| M3-5 | 阻塞 | mem_stream 三档（l1/l2/dram）读写实测速率作 roofline 分母；regress 达成率 / 瓶颈分类分析件在位 | 前置 M3-2；按 decode 瓶颈序逐算子优化；同后端对比 burn/candle 需同机对照数字；wmma/wgmma 手写张量核暂缓 |

## 8. M4 模型与服务化 —— 阻塞边界与前置条件

| ID | 状态 | 现状 | 阻塞边界 / 前置条件 |
|---|---|---|---|
| M4-1 | 阻塞 | M1-A 源抽象 / Reader / Loader 在位；中性 `VocabSource`、`Vocab::try_new`、`Fingerprint` 在位；Tokenizer 现由外部注入 | 前置 M1-A + M2 通路；容器字节 → `VocabSource` 解析未接；给定真实权重跑通生成、词表指纹对拍；vision / 多模态暂缓 |
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
- **P0-6 / M1-A4 / M1-B3 / T1-5 / M2-2 / M2-4 六轨收口实测**（warm-cache，2026-11-17 晚刀）：nextest **1,418/1,418**、libtest **1,454 / 0 / 1**；clippy / fmt / doc / deny / machete / bench `--no-run` / testkit 35/35 全绿；jscpd **1.36% 行**（118 clones / 1,093 行，421 文件）。三族测试夹具抽私有助手去重（serve model_tests、benches regress、train session/snapshot），用例零删除零改名。期间外部并行会话一次回滚曾把 fuzz `p06` 接线连同三条 dev 边从磁盘抹掉，已从提交 `3403d5f4` 恢复 `p06.rs` / `p06_drivers.rs` / 种子块并对齐当前 rustfmt；恢复后 p06 3 条 sweep 用例复跑全绿。
