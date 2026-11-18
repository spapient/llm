# 整批几何 · 设备后端 · 生成服务化设计（M2 / M3 / M4）

- 版本：v1.0
- 日期：2026-11-18
- 状态：设计方案；**本稿不落任何代码**。三块只定契约、几何、接缝、分刀与出口门；代码侧不预埋新类型、空函数、空模块或空编号。
- 适用范围：全 workspace（24 个 `crates/*` 成员；根目录为纯 workspace 根，`llm` / `cutile` 两个 bin 在 `llm-cli` 内）。
- 取证口径：当前工作树源码（HEAD `dcc86a52`）逐件回链 `文件:类型/方法`；未运行、未实测的性能数字一律不预填。
- 对应台账：[`backlog-status-2026-11-17.md`](backlog-status-2026-11-17.md) §6 M2-1 / M2-2、§7 M3-1…5、§8 M4-1…3。

## 0. 结论先行

1. **M2 整批几何**：在已落地的批量结果视图、op 批量段入口与无块链整批前向之上，把含注意力块链的整批通路补齐——逐序列历史长度表、ragged 紧排 K/V 装载与写回、rope / rmsnorm 批段 serve 适配、一轮整批一次提交。几何口径全部沿用 ADR-0043 §5 与 ADR-0046 §7，引擎 `View` 仍是整块切片、不引入 offset / stride。
2. **M3 设备后端**：**不越过 ADR-0035（仍 Proposed）**。本稿只在四 trait 端口（`Executor` / `Allocator` / `Collective` / `Program`）周边预留边界 crate 的形状、依赖方向、白名单收敛点与对拍口径；不写设备 FFI、设备句柄、命令队列、kernel launch，不引 FFI crate，不把 ADR-0035 标成 Accepted。
3. **M4 真实生成与服务化**：打通 `.u` 容器（含预留 `llm.vocab` 段）→ 中性词表 → 注入分词器 → 准入 → 整批前向 → 采样 → 写回页账 → 解码回文本的闭环；服务化只定路由 / 传输接缝、并发与流式通路的契约，真实 HTTP 后端、并发调度与流式在后续刀落地。
4. **口径红线**：不回填历史证人倍数与比率；无同机对照不出吞吐、时延、领先数字；不预埋第三后端、不预埋设备泛型参数；排除清单见 §7。

## 1. 共用基线：既有骨架与裁决回链

### 1.1 运行时四 trait 与 CPU 在位实现

| 端口 | 定义落点 | 现状 | CPU 在位实现 |
|---|---|---|---|
| `Executor` | `crates/engine/src/exec.rs`：`device() -> &Device`、`submit(graph, stream)`、`sync(stream)`（默认 `Ok`） | 图提交到流、同步排水；不解析操作数引用 | `CpuExecutor`（programs / bindings / queue，`register` 重名拒登）；`LocalExecutor` 持 `Box<dyn Allocator>` 与队列、只入队不算 |
| `Program` | `exec.rs`：`name()`、`run(launch, inputs: &[View], outputs: &mut [ViewMut])` | 通用视图通道，无单序列硬编码（M2-2 已落地） | serve 侧 CPU 内核经 `kernels::register_cpu` 挂载 |
| `Allocator` | `crates/storage/src/lib.rs`：`alloc(bytes) -> Result<Buffer>`、`free(buffer)`、`copy_into(src, dst)` | 分配 / 释放 / 显式跨缓冲拷贝；缓冲句柄边界内不透明 | `CpuAllocator`（安全 `Vec<u8>`，无 `unsafe`） |
| `Collective` | `crates/shard/src/collective.rs`：`all_reduce(group, reduce, buffer: &mut [f32])` | 通信后端端口；拓扑簿记（group / partition / reduce / tiling / topology）与实现分离 | `shard::Local`；`shard::Remote` 占位，返回 `Unsupported` 且不动调用方缓冲 |

其余既有事实：

- `Device{vendor, arch}` 是登记标签（字符串），不持私有句柄；`Stream` 由 `id: u32` 标识；`Launch{grid, block, scalars}`；`Operand{id, role: Input / Output}`。
- `Bindings` 是 `BTreeMap<u32, Buffer>`；`Buffer` 枚举 `F32(Vec<f32>)` / `U8(Vec<u8>)`；`View` / `ViewMut` 是整块切片枚举，**不带 offset / stride**。
- `ScalarSlots` 冻结当前仅 `rope_position: Option<u32>`（ADR-0043）。
- `Batch{scheduler, sequences}`：`sequences` 与调度器 `running` 一一对应且顺序相同；`Admission::{Admitted, Full(Sequence)}`（`Full` 是可重试 503 语义，不是 `Err`）。
- 批量结果视图：`SequenceOutcome<'a>{sequence, progress}` 只读借片；`Batch::sequence_at(index)` / `outcome(index, progress)` / `outcomes(step)` / `sequences()` / `ids()`；`step()` 逐条 advance 返回 `Vec<Progress>`，单条 `Blocked` 不中断整批。
- K/V 页账经 `Batch::write_kv(id, token, values: &[f32])` 写、`Batch::kv(id, token)` 读；前缀整页共享 `prefix_share`（页边界整页，不满一页不拆页）。

### 1.2 已落地的批几何前置（不重复设计）

- op 段入口：`decode_forward_batch` / `decode_flash_forward_batch`（显式 heads / batch 守卫，段内复用多头内核）、`decode_forward_batch_var` / `decode_flash_forward_batch_var`（ragged 段游标）；rope `forward_batch`；rmsnorm `forward_batch`（按序列段分组归一化，gamma 按行宽复用）；embed 多行查表；gemm 行批可复用。
- serve 批量接口：`Predict::logits_batch(sequences, batch) -> Vec<Vec<f32>>`，行序对齐准入序，默认逐条回退 `logits`；embed → 投影无块链整批前向已落地（投影行循环 gemv 方案 A，输入 / 激活缓冲按批长重绑，输出按行宽拆回）。
- 基准台账：regress `BenchRecord` 含 Geometry 批次维；手写 JSON 归档与受限回读在位；mem_stream 三档（l1 / l2 / dram）作 roofline 分母。

### 1.3 不可重开的裁决

- **ADR-0025**：trait 定义归运行时一侧；后端实现放独立边界 crate，依赖方向只能从边界 crate 指向运行时 trait，禁止反向（ADR-0001）；调度与准入只认 trait 与 KV 簿记，不引用具体后端类型。
- **ADR-0035（Proposed）**：unsafe / FFI 白名单四类（裸指针解引用、设备缓冲 / 队列 / 事件句柄、FFI 调用、mmap 映射）收敛边界模块；`#[allow(unsafe_code)]` 只在边界模块 / 边界函数，头注释写明越界理由与安全不变量；核心继续 `unsafe_code = "deny"`。评审通过前 M3 不动工，本稿不替评审放行。
- **ADR-0043 §5**：批内各序列位置不同，是长度 = batch 的运行期表，**不进定长 Copy 标量槽**；复用既有 f32 缓冲 lane（整数折 f32，`2^24` 内无损），独立输入操作数，不新增整数 Buffer / View 变体；单序列 `rope_position` 标量槽保留。
- **ADR-0044**：编号 ↔ 形状 / 条目名描述表留装配侧，引擎 `Bindings` 保持纯编号 → 缓冲，引擎不解释编号含义。
- **ADR-0046 §7**：ragged 历史 K/V 扁平一块 `[batch, heads, hist_b, d]` 段间紧排；第 `b` 段起点累计游标 `Σ_{j<b} heads·hist_j·d`，段长 `heads·hist_b·d`，与 op `decode_batch_var` 的 `kv_cursor` 累计口径一致；段内仍头优先块 `[head, token, d]` 复用 `reorder_history_row`；段偏移算术与子区间切片归 serve 适配层；写回只取本段当前行拼页行，编号不翻倍（仍 `hist_k` / `hist_v` 两个编号）。
- **ADR-0045**：`.u` 容器 v2（引用去重、中性条目流导入、现场度量）已采纳。
- **ADR-0036**：Reader / Loader 双契约与 mmap 源抽象在位。

### 1.4 口径红线（三块通用）

- 设计稿引用的代码类型 / 方法名以落地刀当时重读的源码为准；本稿回链的行号可能随外部提交漂移，动工前逐件复核。
- 热路径优先整块 slice、单次 `put`、向量化 / 查表批量处理；只有串行依赖（公共前缀比较、累计游标、top-k / top-p 排序）才保留循环，并在注释写明保留理由（沿用既有偏好口径）。
- 不在代码侧预埋：新设备泛型参数、第二套段游标、空节点 / 空编号、空函数、第二份"一行多宽"常数。
- 架构所有权门只校验 `docs/ARCHITECTURE.md` §3 表与代码产物双向一致；纯设计文档不进所有权表，但任何代码落地刀都要同步该表。

---

## 2. M2 整批几何设计

### 2.1 现状取证与缺口

**在位（取证回链）**

- 调度侧：`Batch` 准入 / 推进 / 结束 / 页账 / 前缀共享齐备；`SequenceOutcome` 只读结果视图与序号对齐 API 就位。
- 模型宿主 `crates/serve/src/model_host.rs`：
  - `predict_input_batch`：仅查表层首层接受 token id 折成 f32 索引缓冲；非查表层首层返回 `Unsupported`（向量主链未接）。
  - `resize_for_batch(batch)`：重绑输入与逐层激活为 `段长 × batch`；驻留权重不乘批。
  - `batch_forward`：无块链整批一次图提交，输出整块按 `output_len` 拆行，行序对齐准入序。
  - `load_history`：逐条序列、逐块、逐 token 经 `batch.kv` 读页行，`reorder_history_row` 头优先块写进 `hist_k` / `hist_v`；`write_current_kv` 取回 K / V 拼 `[k|v]` 页行经 `write_kv` 写回。
  - `logits_batch`：无块链走 `batch_forward`；**含注意力块链时回退逐条 `logits`**（整批 K/V 搬运未接线）。
- 端点 `crates/serve/src/completion.rs`：`State{model, sampling, batch}` 经 `Mutex` 持有；`step_batch` 当前按 `batch.ids()` 逐条 `advance_one`（单条前向），跑完一条无条件 `finish` 归还页。

**缺口（对应 backlog M2-1 残余）**

1. 含注意力块链的整批历史 K/V 装填 / 写回（ragged 紧排）未接；当前历史缓冲按单序列段长预留。
2. rope / rmsnorm 批段的 serve 适配未接：op 段入口就位，装配绑定与内核读取未随块链整批接线。
3. `step_batch` 仍是逐条前向；一轮整批一次提交（`logits_batch` 覆写块链通路）未落地。
4. 多头展平头内切片、多层层偏移行切片、注意力 Program 接 K/V 行映射的内核级多序列矩阵主链仍暂缓（见 §2.9）。

### 2.2 批几何契约

设本轮批长 `B = batch.len()`，按准入序编号 `b = 0..B-1`；第 `b` 条序列历史长度 `hist_b = sequence.len()`，输出行宽 `width = output_len`，块内 `kv_width = heads · head_dim`。

| 缓冲 | 几何 | 归属 | 备注 |
|---|---|---|---|
| 输入索引 | `[B]`（embed 行 gather） | serve 适配层 | token id 折 f32；非查表层首层仍拒绝 |
| 逐层激活 | `[B, width_l]` 整行紧排 | serve 适配层 | 每步按本轮批长重绑；驻留权重不乘批 |
| 逐序列位置表 | `[B]` f32 独立输入操作数 | serve 适配层 → rope 内核 | ADR-0043 §5：运行期表，不进标量槽；整数折 f32 |
| 历史 K/V（块链） | ragged 扁平一块：总长 `Σ_b heads·hist_b·d`，段间紧排 | serve 适配层切段；引擎只给整块切片 | ADR-0046 §7；`hist_k` / `hist_v` 两个编号 |
| 输出 logits | `[B, width]`，行序对齐准入序 | serve 适配层拆行 | 与 `logits_batch` 返回行序同一口径 |

不变量：

- 行序即准入序：`predict_input_batch`、位置表、历史段、输出行四处都按 `sequences()` 顺序排列，对齐只靠序号，不靠 id 重排。
- 缓冲长度校验在装配 / 适配层：整批输出长度必须等于 `B × width`，段总长必须等于各 `heads·hist_b·d` 之和；不符报 `Invalid` / `Internal`，不截断、不补零。
- 引擎 `View` / `ViewMut` 不解释段布局：段起点、段长、子区间切片全部在 serve 适配层算术；引擎侧仍是整块 `&[f32]` / `&mut [f32]`。

### 2.3 整批前向通路逐层接线

本轮整批一次图提交内，逐层按下列顺序接批量段入口：

1. **Embedding 行 gather**：`[B]` 索引 → `[B, hidden]`；复用既有多行查表，不写逐元素收集。
2. **投影 / 点wise / 归一化**：投影行批 gemv 方案 A 现状沿用；rmsnorm `forward_batch` 按序列段分组归一化，gamma 按行宽复用——serve 适配层把 `[B, width]` 激活按段长表切成分组区间交给内核。
3. **Rope 段内旋转**：`forward_batch(positions, x, out, head_dim)`；位置表 `[B]` 经独立输入操作数进内核（ADR-0043 §5）；段内逐序列独立位置，不借用单序列 `rope_position` 标量槽。
4. **注意力块链**：见 §2.4 的 ragged 历史映射；块内四投影（Q/K/V/O）行批接线，当前步 K/V 与历史段同批进 decode 段入口。
5. **输出拆行**：整批 `[B, width]` 按 `width` 拆成 `Vec<Vec<f32>>`，行序对齐准入序。

接线刀落地时再核 op 段入口的实际签名；本稿不假定参数顺序。装配侧描述表（ADR-0044）登记各编号的批量几何事实，供装配自检与测试读取，生产前向不读表。

### 2.4 历史 K/V 整批装填与写回（ragged 紧排）

**装填（每步重绑前）**

1. 按本轮各 `hist_b` 预留历史缓冲总长 `Σ_b heads·hist_b·d`；第 `b` 段起点 `cursor_b = Σ_{j<b} heads·hist_j·d`，段长 `heads·hist_b·d`；游标口径与 op `decode_batch_var` 的 `kv_cursor` 一致，两侧不各写一份段算术。
2. 段内第 `token` 行经 `batch.kv(id, token)` 取 `[k|v]` 页行；映射到该段偏移写进 `hist_k` / `hist_v`；段内沿用 `reorder_history_row` 头优先块 `[head, token, d]`，头分段整块 `copy_from_slice`。
3. 历史单调变长：上一轮同编号缓冲经 `bind` 返回后接住复用容量，多数步次不再堆分配；首次没有旧缓冲时新建。

**写回（当前步）**

1. 取回各块当前步 K / V 投影，只取本段当前行，拼 `[k|v]` 页行；`batch.write_kv(id, position, row)` 写回各自页账。
2. 不跨段读写；页行归属仍按序列号；历史编号仍是 `hist_k` / `hist_v`，不为 ragged 多开编号。
3. `Blocked` 序列本轮没有页：装填 / 写回跳过该段，段游标仍按它的 `hist_b` 计入总长；重试时重新采样，上一步 token 不提前留在历史里（沿用 `emit` 既有口径）。

**串行依赖保留点**：段游标累计、公共前缀比较这类有前后依赖的算术保留循环；其余搬运整块切片处理。

### 2.5 提交、调度与结果对齐

- `step_batch` 整批通路：取本轮在跑序列快照 → 一次 `logits_batch` 整批前向 → 逐行采样 → `emit` 记账推进 → 结束序列 `finish` 归还页。
- 单条 `Blocked` 不中断整批：其余序列照常推进；`Blocked` 序列历史不变，留批内下一轮原样重试。
- 结果对齐：`Batch::outcomes(step)` 把每条 `Progress` 与各自序列事实成对持有；端点按序号取 `SequenceOutcome`，不按 id 重排。
- 失败原子性：任一序列模型 / 采样报错，先 `finish` 归还该序列页，再上抛错误；归还失败作为同级诊断附注保留，不覆盖原始故障（沿用 `step_batch` 既有清理口径）。
- 调度策略（准入、抢占、页池水位）不在本刀改；整批前向不推进计数，计数与页账仍由调度器推进。

### 2.6 关键决策与备选

| 决策 | 选择 | 被否备选 | 理由 |
|---|---|---|---|
| 批量几何承载 | 复用既有 f32 缓冲 lane，整数折 f32 | 新增整数 Buffer / View 变体 | ADR-0043 §5 已定；避免引擎视图枚举分叉 |
| ragged 段算术归属 | serve 适配层算段偏移与子区间 | 引擎 View 带 offset / stride | ADR-0046 §7 已定；引擎保持整块切片 |
| 历史段布局 | 段间紧排、段内头优先块 | 按最大历史长度定长填充 | 紧排无空位；游标与 op `kv_cursor` 同口径；定长填充浪费页宽且拉长内核扫描 |
| 整批提交粒度 | 一轮整批一次图提交 | 逐条前向叠批量视图 | 逐条回退仅作默认兜底；整批通路落地后覆写 `logits_batch` |
| 无块链通路 | 沿用 `batch_forward` | 块链 / 无块链两套装填 | 无块链没有历史段，复用同一行批几何，仅跳过 K/V 装填写回 |
| 向量首层 | 非查表层首层仍 `Unsupported` | 在 serve 层造隐向量占位 | 序列只持 token id，给不出向量；向量主链随整批几何刀后续接入 |

### 2.7 分刀与出口门

每刀出口门（沿用既有合并门口径）：`cargo fmt --all --check`；`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`；`cargo nextest run --workspace --all-features --locked`；libtest 必须 `--exclude llm-fuzz`；`cargo test --doc`；`cargo deny check`；`cargo machete`；jscpd ≤ 1.5% 行；`cargo bench --no-run`；静态门 ccn ≤ 5、nloc ≤ 60、文件 ≤ 899 行。

- **刀 2A：rope / rmsnorm 批段 serve 适配**。位置表通道与段分组区间接到装配绑定；op 段入口复用。出口：batch=2 对拍两行结果与逐条回退一致；位置表越界 / 非有限值报 `Invalid`。
- **刀 2B：块链整批历史装填 / 写回 + `logits_batch` 覆写**。ragged 段游标、段内重排、写回拆行；块链通路不再逐条回退。出口：batch=2 / batch=4 端到端对拍沿用既有夹具（`logits_batch_runs_two_sequences_in_input_order_and_keeps_their_own_kv` 同口径），两行 logits 与各自 KV 归属正确；段总长 / 段长不符报错。
- **刀 2C：`step_batch` 一轮整批提交接线**。端点批步进从逐条 `advance_one` 切到整批前向；`Blocked` 与结束态映射不变。出口：多序列一轮对齐结局；页归还无泄漏对拍；单条失败路径清理附注保留。

### 2.8 故障模式与对拍口径

| 故障 | 表现 | 归属侧 | 处置 |
|---|---|---|---|
| 段总长 / 段长不符 | 历史缓冲越界或残留 | serve 适配层 | 装填前按 `hist_b` 表核算，不符报 `Internal` / `Invalid`，不切片硬取 |
| 输出行宽不符 | 拆行越界 | serve 适配层 | `flat.len() != B × width` 报 `Internal` |
| 位置表越界 / 非有限 | 内核读到脏位置 | serve 适配层 | 校验后再进输入缓冲；非有限按 `Invalid` 拒绝 |
| 某条 `Blocked` | 本轮无页 | 调度器 | 跳过该段推进，历史不变，下轮重试 |
| 整批中一条报错 | 页占用泄漏风险 | 端点 | 先 `finish` 再上抛；归还失败附注保留 |
| 行序错位 | 结果串到别的序列 | serve 适配层 | 序号对齐守护用例：两行结果与各自历史 / KV 归属对拍 |

对拍口径：整批通路结果与逐条回退在同输入、同形状下逐元素一致（f32 同口径）；不引入同机对照倍数，不写吞吐数字。

### 2.9 暂缓与排除

- 内核级多序列矩阵主链、头内并行特化（wmma / wgmma 手写张量核）更后；本刀只接段入口与几何接线。
- chunked prefill、多 token 预填充：`plan` 当前 `tokens != 1` 是层形状契约，prefill 多 token 层不在本阶段放开。
- 投机解码、beam、MoE、vision、KV 量化、前缀复用预热跨请求——见 §7。

---

## 3. M3 设备后端边界设计（不越 ADR-0035）

### 3.1 前置状态与动工门槛

backlog §7 总前置，未满足前 M3 不动工：

1. **ADR-0035 仍 Proposed**：本稿只定边界形状与预留接缝，不是 M3 实现放行；不得标 Accepted，不得据此展开真实设备 FFI / 设备内核。
2. **硬件两侧实测在位**：本机 MBP Apple Silicon（Metal 路线）；远端 Windows T600 sm_75 / 4GB / CUDA 13.3（CUDA 路线）。本稿不写死 SM 版本，内核与批量尺寸按目标能力约束。
3. **剩余真实门槛**：① 0035 评审通过；② D5 生产零外部依赖——不引 cuda-sys / metal-rs 之类 FFI crate，要么手写 FFI 绑定，要么走 ADR 评审放宽策略；③ 源码同步与跨机构建未通（win 上尚无 git 工作树、未跑 `cargo check`）；④ 量化 lane / 通信 / 内核管线未建。
4. **D6 形状约束**：第二后端出现前不预埋设备泛型参数；四 trait 已定形，不再分叉第二套运行时抽象。

### 3.2 边界 crate 形态与依赖方向

- 评审通过、且 M3-1 动工前，才新建后端边界成员；当前 24 成员中无后端 crate，本稿不在工作树建空骨架。
- 依赖方向（ADR-0025 / ADR-0001）：边界 crate → `engine` / `storage` / `shard` 的 trait；**不反向**依赖 `serve` / `nn` / `op` 的机制代码。模型、调度、算子代码只认 trait，拿不到指针与厂商句柄。
- 层位归属：`Device` 标签与 `Executor` 在 `engine`（L6）；`Buffer` / `Allocator` 在 `storage`（L4）；`Collective` 在 `shard`（L5）；边界 crate 是挂载实现的组装侧，不新增运行时领域语义。
- 边界内部再分模块：设备适配模块（句柄 / 队列 / 事件收敛于此）、Program 适配模块（把 op / cutile 产物挂进 `Program` 端口）、对拍与测试模块；越界对象只出现在设备适配模块。

### 3.3 端口形状与不透明句柄

- **trait 签名不出现裸指针与设备句柄类型**：方法参数仍是 `&[View]` / `&mut [ViewMut]`、`&[f32]` / `&mut [f32]` 这类安全切片；设备私有句柄（缓冲、命令队列、事件）藏在边界 crate 内部类型里，经 `Allocator` 返回的不透明缓冲句柄向上传递。
- **`Allocator`**：`alloc` 返回边界内不透明拥有缓冲；`free` 归还；`copy_into` 是显式跨缓冲 / 跨设备拷贝入口，框架层不发生隐式拷贝。缓冲状态（拥有 / 借用 / 静态借用）的判别在边界侧，trait 只给拥有型入口。
- **`Executor`**：`device()` 给标签；`submit(graph, stream)` 入队；`sync(stream)` 排水与事件同步——CPU 默认 `Ok` 立即返回，设备后端在此等事件。执行器持 `Box<dyn Allocator>`，不持缓冲字节。
- **`Program`**：内核经注册表按名挂载（`register` 重名拒登沿用）；`run` 拿到的是拆好角色的只读 / 可写视图，与 `Node::inputs` / `outputs` 一一对应。边界适配只做句柄映射与调用，不解释编号含义。
- **`Collective`**：`Group` / `Topology` 纯拓扑簿记与实现分离；边界实现接真实通信后端时不改拓扑层。

### 3.4 后端装配与执行生命周期

评审通过后的目标形态（仅设计，不落地）：

1. **装配根**（cli / 编排层）按设备档案选边界实现，构造设备 `Device` 标签、`Allocator` 实现与 `Executor` 实现；trait 对象注入执行器。
2. **缓冲映射**：权重驻留经 `Allocator` 分配 / 映射；激活与 K/V 缓冲在边界侧拥有，视图借片给内核。跨设备拷贝只经 `copy_into`。
3. **提交与同步**：图节点按图序提交到流；`sync` 排指定流，跨流先后不属于执行器层；一条目成功后才出队，失败时仍在队列、"还没跑完"查得到（沿用 `CpuExecutor` 既有口径）。
4. **内核产物**：复用 cutile 协商 → 拼接 → emit 路线；边界 crate 不重新发明 IR；设备内核编译产物挂载为 `Program`。
5. **对拍**：同输入、同形状、同权重，CPU 实现与设备实现逐元素一致；两侧未建设备内核前不出吞吐 / 时延数字。
6. **图 capture 特化**暂缓：先逐节点提交通路稳定，再谈捕获与重放。

### 3.5 量化 lane 挂载

- 现状：`gemv_i2` / `gemv_i4` / `gemv_i8`、`gemm_i2` / `gemm_i4` / `gemm_i8`、`dequant_i2` / `dequant_i4` / `dequant_i8` 骨架在 op 泛型内；注册表 27 条含 9 条量化 lane。
- 接缝：CPU 侧 lane 已按 `Gemm` trait 泛型挂成 `Program`（serve 桥接宏展开，位宽 / 收缩轴取标记类型常量）；权重对拍矩阵、零点 / 分组负例在 M3-3 刀内收口；设备后端 `Program` 挂载仍随 M3-1/M3-2 边界 crate 落地。
- 暂缓：FP8 / AWQ / GPTQ 新格式在 M3-3 之后按格式 ADR 接入；KV 量化不在本阶段。

### 3.6 集合通信

- 现状：`Local` 在位；`Remote` 返回 `Unsupported` 且拒绝时不改调用方缓冲（用例 `all_reduce_is_unsupported_and_leaves_the_buffer_untouched`）。
- 接缝：真实后端（NCCL / MPI / Gloo / Metal 对等）实现 `Collective`：组内握手、通信句柄池、拓扑绑定、心跳；集合对拍；大规模分布式暂缓。

### 3.7 能力台账与 roofline

- mem_stream 三档读写实测速率作 roofline 分母；regress 达成率 / 瓶颈分类分析件在位。
- 设备侧数字（带宽、吞吐、TPOT）待 M3-2 设备通路跑通后回填；按 decode 瓶颈序逐算子优化；同后端对比 burn / candle 需同机对照数字，无对照不写倍数。

### 3.8 关键决策与备选

| 决策 | 选择 | 被否备选 | 理由 |
|---|---|---|---|
| 抽象归属 | 四 trait 在运行时一侧，边界 crate 只填实现 | 边界 crate 定义 trait、运行时反向依赖 | ADR-0025 已定；依赖方向门禁 |
| 句柄可见性 | 设备句柄边界内不透明，trait 签名只出安全切片 | trait 暴露指针 / 句柄类型 | ADR-0035 §1；模型与调度层拿不到厂商对象 |
| 泛型参数 | 第二后端出现前不预埋设备泛型 | 现在就把 Executor / Buffer 泛型化 | D6；过早泛型会把单实现的猜测固化成契约 |
| FFI 来源 | 手写 FFI 绑定（或评审放宽后引绑定 crate） | 直引 cuda-sys / metal-rs | D5 生产零外部依赖；放宽需走 ADR 评审 |
| 通信抽象 | 复用 `Collective` 端口 | 后端私造通信 trait | ADR-0025；拓扑与实现分离 |
| 执行语义 | 提交入队、同步排水，图序不重排不融合 | submit 即同步算完 | 沿用 `CpuExecutor` 口径；上层依赖正确依赖顺序 |

### 3.9 分刀与出口门（待 0035 评审通过后启动）

- **刀 3-1（M3-1）**：边界成员落地、依赖方向守门；设备适配模块白名单四类越界对象收敛；CPU 实现作参照保留。出口前置：0035 经评审转为 Accepted 后才动工；边界 crate 之外零新增 `allow(unsafe_code)`；架构所有权表同步登记。
- **刀 3-2（M3-2）**：Metal 本机跑通 M2 同一前向；CUDA 在 win（T600）验；同输入同形状对拍一致。出口：逐元素对拍夹具；设备事件同步路径；无对照不填性能数字。
- **刀 3-3（M3-3）**：i4 / i8 lane Program 适配 + 反量化内核挂载；量化权重对拍、零点 / 分组负例。
- **刀 3-4（M3-4）**：真实 `Collective` 后端；握手 / 句柄池 / 拓扑 / 心跳；集合对拍。
- **刀 3-5（M3-5）**：按 decode 瓶颈序逐算子优化；设备侧 roofline 分母与达成率回填。

### 3.10 阻塞边界与暂缓

- 0035 未评审前：不写 FFI、不引绑定 crate、不建设备句柄 / 队列 / 事件 / launch 代码。
- 第三加速链路（MUSA / OpenCL / SYCL / ANE）暂缓；图 capture 特化暂缓；wmma / wgmma 手写张量核暂缓；大规模分布式暂缓。

---

## 4. M4 真实生成与服务化设计

### 4.1 现状取证与缺口

**在位**

- 权重装载：M1-A 源抽象 / Reader / Loader 在位；`.u` v2 读写 / REF 去重 / 外部格式导入 / 现场度量在位；`Archive::get(name)` 按名取段、`read(seg)` / `load(seg)` 取字节。
- 词表中性值对象：`VocabSource{family, pieces, special, scores, merges}`；`VocabFamily`（Bpe / Unigram / WordPiece / WordLevel / Byte）；`Merge{left, right}`；`Vocab::try_new` 构造期闭合双向查表、编号范围、族附属表不变量；`Fingerprint{family, vocab_size, digest: u64}`（FNV-1a 非加密摘要，可 `fold_extra` 并入合并秩序 / 分数等附属内容）；Tokenizer 由外部注入。
- 服务契约：`Router` / `Endpoint` / `Request` / `Response` 与桩路由在位；`Request` 行文法 `id=` / `prompt=` / `max_new=`，未知 / 重复 / 缺字段 / 解析失败 `Invalid` → 400；运行期不带 TOML / JSON 解析器。
- 生成闭环：`Completion<M, S>` 泛型端点；`Predict` 两方法；`Sampling` / `Greedy` / `Classical`（温度 → top-k → top-p → 确定性 argmax，不接 RNG）；`Generation{tokens, ending}`，`frame()` 每 token 4 字节 LE 一次性预分配，`text(tokenizer)` 越界编号由分词器报错；`Ending::{Limit, Stop}`；跑完无条件 `finish` 归还页。

**缺口（对应 backlog M4-1 / M4-2 / M4-3）**

1. `.u` 内预留 `llm.vocab` 段字节 → `VocabSource` 解析未接；词表指纹与权重条目对拍未接。
2. 给定真实权重跑通 CPU 端到端生成；真实 HTTP 后端未替换桩路由；并发 / 流式通路未接。
3. 并发与 P50 / P99、TTFT / TPOT、prefix 命中入表；对照口径登记未跑分。

### 4.2 容器词表段解析接缝

- **段落点**：保留名 `VOCAB_NAME = "llm.vocab"`（`weights/src/archive.rs`）；与权重条目同一 Reader / Loader 通路按名取段字节。
- **解析适配**：段字节 → `VocabSource` 的映射放在装配点（编排 / cli 组合根或 serve 装配侧），weights 层不接 tokenizer 类型、不出现格式键名；适配器把段内族标识、词片、特殊句柄、合并对 / 逐片分数映射成中性 `VocabSource`。
- **构造与指纹**：`Vocab::try_new(source)` 闭合不变量；产出 `Fingerprint`（族、词表大小、FNV-1a 摘要）。若容器条目声明了期望指纹（族 / 词表大小 / 摘要），装配期逐条对拍：对不上报 `Invalid`，不静默降级；不在容器缺声明时伪造一份摘要写回。
- **注入端点**：构造好的 `Tokenizer` 实现作为 `Box<dyn Tokenizer>` 注入 `Completion`；端点只持有分词器端口，不持有词表字节副本。
- **格式边界**：段内编码格式由后续刀定（族标识 + 载荷的中性自描述布局）；本稿只定"字节进、`VocabSource` 出、指纹对拍"的接缝，不预埋第二种词表解析状态。

### 4.3 真实权重 → 生成闭环

CPU 端到端通路（M2 整批几何落地后）：

1. **装载**：`.u` 条目经 Reader / Loader 读出 → 驻留参数表；形状 / 宽度 / 权重缺失在构造期报出（沿用 `CpuModel::new` 既有校验）。
2. **编词**：请求 prompt 经注入的 Tokenizer `encode` 成 token id 序列；端点一行收集成 `u32`，不留第二份 token 中间向量。
3. **准入**：构造 `Sequence`（id、prompt、max_new、页宽取自页池）；`admit_batch` 准入；`Full` 翻译为可重试响应，`fits` 区分"现在满"与"永远装不下"。
4. **前向**：整批 `logits_batch`（M2 通路）；逐条 `logits` 仅作回退兜底。
5. **采样**：`Sampling::sample` 选 token 或 `None`（停止符）；`emit` 记账推进；`Limit` / `Stop` / `Blocked` 三态映射到 `Ending` 与响应。
6. **写回与归还**：当前步 K/V 写回页账；结束序列无条件 `finish` 归还页。
7. **解码**：`Generation::text(tokenizer)` 把 token 编号还原文本；`frame()` 保留 4 字节 LE 线格式。

错误分类（不混类）：请求侧（非 UTF-8、未知 / 重复 / 缺字段、编号越界、词表指纹不符）→ 400 类；未命中路由 → 404；页不够可重试 → 503 类；权重缺失、形状不符、页账读不出历史 → 服务端 `internal` / `io`。

### 4.4 服务化拓扑：路由、传输、并发、流式

**路由契约（不变）**

- `Router::mount` 重复方法 + 路径拒登；`Router::call(method, path, body) -> Result<Response>`；未命中 404。端点只实现 `Endpoint::route()` 与 `Endpoint::call(body)`。
- 传输层替换桩时，只把方法、路径、请求体交给 `Router::call`；路由 / 端点契约不改。

**HTTP 后端（M4-2 刀落地）**

- 接入方式：HTTP 服务作为组合根装配侧，监听连接、解析请求行与主体，映射到 `Request` 行文法；响应经 `Response::frame()` 回写。
- 依赖口径：运行期仍不引入通用 JSON 解析器；请求 / 响应线格式沿用既有闭集（行文法 + LE 字节帧）。若后续刀需要 JSON，走格式 ADR 评审，不在本稿预埋。
- 并发现状：`State` 经 `Mutex` 持有 model / sampling / batch；并发准入、批调度、连接生命周期在 HTTP 后端刀设计。请求 id 去重在准入侧拒绝重复编号。
- 流式通路：增量 token 帧（SSE 类）作为接缝登记；本轮不落地流式协议。生成循环先按整段 `Generation` 返回，流式化再把 `frame()` 拆成增量帧。
- gRPC 在 HTTP 通路稳定后；本阶段排除。

### 4.5 可观测与台账口径

- 指标入表：并发度、P50 / P99 时延、TTFT（首 token）、TPOT（每 token）、prefix 命中；批次几何沿用 M2-4 既有 `BenchRecord` 字段。
- 分母：mem_stream 三档实测速率；roofline 双可选口径沿用现状。
- 对照：vLLM / TRT-LLM 同负载口径登记；无同机对照不写吞吐倍数、不写领先结论。
- 观测不进热路径：指标采集在装配 / 端点侧包装，内核与算子不持统计依赖。

### 4.6 关键决策与备选

| 决策 | 选择 | 被否备选 | 理由 |
|---|---|---|---|
| 词表解析归属 | 装配点适配器把段字节映射成 `VocabSource` | weights 层直接依赖 tokenizer | 分层依赖方向；weights 不接文本边界语义 |
| 指纹校验 | 构造期对拍，不符 `Invalid` | 运行期容错降级 | 词表与权重不匹配会产出乱码且难定位；构造期闭合 |
| 分词器注入 | 端点持 `Box<dyn Tokenizer>` | 端点内置词表 | 现状端口；词表来源（容器段 / 外部）可替换 |
| 请求文法 | 沿用行文法闭集 | 引 JSON 解析器 | ADR-0022 后运行期零解析器依赖 |
| 流式 | 接缝登记，本轮整段返回 | 现在就实现增量帧协议 | 生成闭环与 HTTP 后端先稳定；流式不阻塞主链路 |
| 并发 | Mutex 现状保留，并发调度随 HTTP 刀 | 现在就设计多队列批调度 | 无传输层前并发策略无法定形；不预埋猜测 |

### 4.7 分刀与出口门

- **刀 4-1（M4-1）**：`llm.vocab` 段读取 → `VocabSource` 适配器 → `Vocab::try_new` → 指纹对拍；Tokenizer 装配注入。出口：容器内词表段往返对拍；指纹不符 / 编号越界负例；无外部词表依赖。
- **刀 4-2（M4-2）**：真实权重 CPU 端到端生成闭环；prompt → 准入 → 整批前向 → 采样 → 解码；结束三态与页归还对拍。出口：给定 `.u` 权重跑通生成；错误分类守卫；页无泄漏。
- **刀 4-3（M4-2 / M4-3）**：HTTP 后端替换桩路由；并发准入与连接生命周期；指标入表。出口：桩路由契约对拍；并发下批次与页账一致；指标字段齐备，无对照倍数结论。

### 4.8 阻塞边界与排除

- 前置：M2 整批通路（块链）未通前，真实权重端到端只能走逐条回退；M4-2 整批生成依赖 M2-2 刀 2B / 2C。
- 排除：vision / 多模态、投机解码、beam、MoE、RL、LoRA、FP8、AWQ、GPTQ、KV 量化、chunked prefill、gRPC、第三后端。

---

## 5. 风险与故障模式汇总

| 风险 | 影响 | 缓解 | 归属刀 |
|---|---|---|---|
| ragged 段游标两侧各写一份 | 历史错位、串条 | 游标口径唯一对齐 op `kv_cursor`；适配层单点算术 | 2B |
| 位置表误进标量槽 | 定长 Copy 槽容纳不下变长批 | 走独立 f32 输入操作数（ADR-0043 §5） | 2A |
| 设备句柄泄漏到模型层 | 安全边界枚举面失控 | 句柄边界内不透明；trait 签名只出安全切片 | 3-1 |
| 0035 未评审先写 FFI | 违反 deny 门禁与裁决 | M3 动工门：0035 经评审 Accepted 后才放行；边界之外零新增 unsafe | 3-1 |
| 词表指纹不符仍加载 | 输出乱码、难定位 | 构造期对拍拒绝；不静默降级 | 4-1 |
| 失败路径页泄漏 | 长跑后页池耗尽 | 结束 / 报错无条件 `finish`；归还失败附注保留 | 2C / 4-2 |
| 无对照填性能倍数 | 结论不可证伪 | 指标只登记实测口径；倍数留待同机对照 | 3-5 / 4-3 |
| 行序与准入序错位 | 结果串到别的序列 | 序号对齐守护；整批输入 / 位置 / 段 / 输出四处同序 | 2B |

## 6. 回链表

**Backlog（`docs/backlog-status-2026-11-17.md`）**

- §6：M2-1（阻塞：内核级多序列矩阵主链）、M2-2（已落地：批量视图与 op 段入口）、M2-3（前缀共享）、M2-4（基准台账）。
- §7：M3-1 边界 crate、M3-2 双后端对拍、M3-3 量化 lane、M3-4 集合通信、M3-5 优化与 roofline。
- §8：M4-1 词表与真实权重、M4-2 服务化、M4-3 并发与指标。

**ADR（`docs/adr/`）**

- 0001 分层依赖方向；0003 内核操作数视图与运行时张量不混用；0025 运行时后端抽象；0028 设备数据与机制边界；0035 unsafe / FFI 边界（Proposed）；0036 Reader / Loader / mmap 源；0043 Program 标量槽（§5 位置表通道）；0044 绑定编号元数据；0045 `.u` 容器 v2；0046 批量扁平缓冲几何（§7 ragged 段映射）。

**源码落点（动工前复核行号）**

- engine：`crates/engine/src/exec.rs`（Executor / Program / View / Buffer / Bindings / CpuExecutor / LocalExecutor / Device / Launch / Operand）、`batch.rs`（Batch / SequenceOutcome / Admission / step / outcomes / write_kv / kv）、`kv.rs`（Pager / Table / Slot）、`lib.rs`（Sequence / Progress / Scheduler / Stream）。
- serve：`model_host.rs`（CpuModel / predict_input_batch / resize_for_batch / batch_forward / load_history / write_current_kv / logits_batch）、`generate.rs`（Predict / Sampling / Greedy / Classical / SamplingParams / Generation / Ending）、`completion.rs`（Completion / State / step_batch）、`router.rs`、`request.rs`、`lib.rs`（Response / Status / Endpoint / Route / admit_batch）、`protocol.rs` + `frame.rs`（Version / Kind / Envelope / Frame / Decoder 线帧）、`assembly.rs`、`block.rs`、`kernels.rs`、`params.rs`、`plan.rs`、`descriptors.rs`。
- storage：`lib.rs`（Allocator / CpuAllocator / Buffer / Layout / remap）、`mmap.rs`（既有 unsafe 岛）。
- shard：`collective.rs`（Collective）、`local.rs`、`remote.rs`。
- weights：`archive.rs`（`VOCAB_NAME` / Archive::get / read / load）、`lib.rs`（Entry / Header / Reader / Loader）。
- tokenizer：`vocab.rs`（VocabSource / VocabFamily / Merge / Vocab）、`fingerprint.rs`。
- op：`attention.rs`（decode batch / batch_var）、`rope_f32.rs`、`rmsnorm_f32.rs`、`embed_f32.rs`、`gemv_f32.rs`。

## 7. 不做与暂缓总表

投机解码、beam search、MoE、vision / 多模态、RL、LoRA、FP8、AWQ、GPTQ、KV 量化、chunked prefill、prefill 多 token 层放开、第三后端（MUSA / OpenCL / SYCL / ANE）、gRPC、图 capture 特化、wmma / wgmma 手写张量核、大规模分布式、跨请求前缀复用预热、随机采样 RNG（截断后仍确定性 argmax）。无同机对照不写性能倍数；历史证人数字（旧 v4 快照）不回填进本稿结论。
