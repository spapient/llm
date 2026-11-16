# 推理后端整仓分析与目标设计（P1 收尾 → P2 → P3）

- 版本：v1.1（设计草稿，待评审；落点裁决：**复用现有成员，不新建 crate**）
- 日期：2026-11-16
- 范围：全 workspace（24 个 `crates/*` 成员 + 根包 `llm`）。回答两个问题：**从整仓看，真实后端落在哪、按什么顺序落**；以及**现有契约面哪些可直接复用、哪些缺口必须先补 trait**。
- 取证口径：静态只读。所有断言回链 `文件:行`；性能数字一律不预填，M1/M2/M3 实测后回填 [`engine-benchmark-roadmap.md`](engine-benchmark-roadmap.md) §1.1。工作树混有并行会话未提交改动，本文以磁盘现状为准，不 stage、不回退。
- 立场：本文是**整体分析 + 目标设计**，不是实现代码。落地步骤见 §7；trait 形状的最终变更须在动工前以 ADR 回填（§9）。
- 配套：[`ARCHITECTURE.md`](ARCHITECTURE.md)（分层/边/所有权门禁）、[`ADR 0025`](adr/0025-runtime-backend-abstraction.md)（后端抽象三件套）、[`ADR 0014`](adr/0014-real-weights-execution.md)（真实权重执行）、[`ADR 0028`](adr/0028-device-data-vs-mechanism-boundary.md)（设备数据与机制边界）、[`cutile-analysis.md`](cutile-analysis.md)（L3 机制层取证）。

---

## 0. 复核命令

| 用途 | 命令（工作目录 `/Users/fei/code/llm`） |
|---|---|
| 成员清单 | `grep -A30 'members' Cargo.toml` → `crates/*` |
| 算子注册表 | `grep -oE '"[a-z0-9_]+_(f32\|i4\|i8)"' crates/op/src/lib.rs \| sort -u` → 22 条 |
| 谁依赖 `engine` | `grep -l 'llm-engine\|../engine' crates/*/Cargo.toml` → 仅 `engine` 自身与 `serve` |
| engine 执行契约 | `sed -n '257,425p' crates/engine/src/exec.rs` |
| engine KV 数据 | `sed -n '204,393p' crates/engine/src/kv.rs` |
| 服务 Predict 契约 | `sed -n '18,46p' crates/serve/src/generate.rs` |
| train 内核表先例 | `sed -n '168,215p' crates/train/src/kernel.rs` |
| 架构守门项 | `grep -nE 'fn ' tests/architecture.rs` |
| 阶段出口门 | `cargo test -p llm-engine`、`cargo clippy -p llm-engine`、`cargo fmt -p llm-engine -- --check`、`cargo test -p llm --test architecture --test contracts` |

---

## 1. 整仓现状盘点

### 1.1 分层与唯一依赖规则

分层表（`docs/ARCHITECTURE.md:14-20`）：

| 层 | 成员 | 运行时相关职责 |
|---|---|---|
| L1 | `diagnostic`、`testkit`、`dtype`、`wire`、`tool` | 唯一错误类型；浮点对拍/随机数/设备跳过夹具；精度闭集 |
| L2 | `prof`、`op`、`prim` | 设备能力闭集；算子声明/签名/样本/标量参考；原语片段 |
| L3 | `cutile` | 协商 → 拼接 → emit → 编译执行；产物按内容寻址缓存 |
| L4 | `tensor`、`storage`、`tokenizer` | 张量元数据与双形态；分配器 trait 与字节缓冲；文本边界 |
| L5 | `weights`、`autograd`、`data`、`nn`、`shard` | 权重容器；反向图；样本；层定义（Contract/Stack/Linear/Attention）；切分拓扑与集合通信 trait |
| L6 | `train`、`engine`、`serve`、`cli` | 训练循环与参考内核；连续批/序列/KV/提交执行；端点契约与线格式；命令行解析 |
| L7 | `llm`（bin） | 组装各层，**不定义新类型**（门禁 `tests/architecture.rs:217` `the_entry_point_defines_no_types`） |

唯一规则：**依赖只能指向更低层；同层只能依赖更靠左成员**。方向、成员集合、边集合、所有权表由 `tests/architecture.rs` 双向解析守门（成员 `:29`、边 `:57`、方向 `:80`、所有权公开定义 `:125`、pub 类型入表或基线 `:152`）。

### 1.2 可直接复用的公开面（回链）

**提交与执行（`engine::exec`）**

- `Executor` trait：`device() -> &Device`、`submit(&mut self, &Graph, Stream)`、`sync(&mut self, Stream)`（`exec.rs:261-275`）。`sync` 有空默认实现，CPU 立即返回，设备后端在此做事件同步。
- `Device`：厂商/架构轻量字符串标签，不持私有句柄（`exec.rs:232-255`）。
- `Graph` / `Node` / `Launch` / `Operand` / `OperandRole`：图是提交单元，不做融合重排（`exec.rs:122-209`、`:84-118`、`:42-65`）。`Node` 持算子名、几何、按声明顺序的输入/输出编号。
- `LocalExecutor`：只入队排水，持注入的 `Box<dyn Allocator>`（`exec.rs:304-354`）。
- `CpuExecutor`：真跑登记过的 `Program`。`register` 重名拒绝（`exec.rs:501-514`）；`bind`/`get` 操作数表（`exec.rs:517-529`）；`run` 查内核、借出操作数、执行、放回，失败时缓冲先放回再报错（`exec.rs:536-549`）；`sync` 只排指定流，成功一条才出队一条（`exec.rs:632-644`）。
- `Program` trait：`name() -> &str`；`run(launch: Launch, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()>`（`exec.rs:383-389`）。
- `Bindings`：`BTreeMap<u32, Vec<f32>>`，编号由编排层分配，本层不解释编号含义（`exec.rs:396-425`）。

**KV 与连续批（`engine::kv`、`engine::batch`）**

- `Pager` 已持**真实** `Vec<f32>` 页数据：整池构造时一次分配；`row` / `row_mut` / `write` 只经 `Slot` 寻址并先验页活着、偏移在页内；`release` 是唯一清零点，归还即清零（`kv.rs:217-393`）。行宽由装载模型的一侧给，页池不认识层与头。
- `Table::slot(index, page_size) -> Slot`：token → 页号 + 页内偏移的唯一算术（`kv.rs:145-163`）。
- `Batch`：`Scheduler` + `sequences` 唯一来源；`admit` / `fits` / `advance` / `emit` / `write_kv` / `kv` / `step`（整批逐序列推进）/ `finish`（页与序列一起释放）（`batch.rs:33-227`）。
- `Sequence`：id、prompt_len、produced、max_new、tokens、状态、流、页宽、所需页数、KV 形状（`src/lib.rs:105-290`）。
- `Scheduler`：准入、适配判定、推进、阻塞集、结束、页表、页池访问、slot、写/读 KV（`src/lib.rs:309-573`）。

**内存与张量（L4）**

- `storage::Allocator`：`alloc(bytes) -> Result<Buffer>`、`free(Buffer)`、`copy_into(src, dst)`；跨设备拷贝显式，框架层不发生隐式拷贝（`storage/src/lib.rs:171-190`）。`CpuAllocator` 是无状态安全 `Vec<u8>`，零 `unsafe`。
- `tensor::OwnedTensor`：元数据 + 持有字节向量；`zeros` / `from_bytes` / `dtype` / `shape` / `strides` / `numel` / `data` / `data_mut` / `view` / `is_contiguous`（`owned.rs:17-96`）。`View<'a, E>` 元素级零拷贝（`view.rs:17-201`）。`SUPPORTED = F32/F16/BF16/I8/U4`。

**层定义（`nn`）**

- `Contract` trait：`name` / `output(&Shape)` / `spca(&ContractSpec)` / `bounds`（默认 `Ok(None)`）/ `linear`（默认 `None`）（`nn/src/lib.rs:166-196`）。
- `Linear`：in/out 宽度 + `WeightFormat`；编码到算子名的映射归 `WeightFormat::operator`（`nn/src/lib.rs:198-270`）。
- `Attention`：seq / head_dim / 算法（朴素 `attention_f32` / Flash `flash_attention_f32`）（`attention.rs:33-94`）。
- `Stack`：`new` / `of(Vec<Box<dyn Contract>>)` / `from_header` / `layers` / `uncovered` / `floor`（`stack.rs:107-197`）。
- `ContractSpec`：层名、输出形状、算子名列表（`nn/src/lib.rs:101-159`）。

**算子声明与参考（`op`）**

- `REGISTRY`：22 条编译期闭集，名/声明/模板/语义表同属一条（`op/src/lib.rs:122-164`）。
- `Model`：cost/sample/reference 三个 fn 指针、dtype、轴角色、形状粒度；`model::bind(model, m, n, x, w)` 是真实数据进样本的唯一入口，量化权重明确拒绝注入（`model.rs:181-200`）。
- 签名形参闭集：`F32` / `U8` / `Int`（标量）；标量不占缓冲（`sig.rs:18-39`）。
- f32 标量参考 `forward` 已是全函数签名：一元 `softmax/silu/gelu/scale/reduce_max/reduce_sum`；二元 `add/mul/residual_add/rmsnorm/gemv`；特殊 `rope_f32(pos, x, out)`、`embed_f32(idx, table, out)`（行向量 gather）、`attention_f32(q,k,v,out,heads,d)`；量化族 `gemv_i4/i8`、`gemm_i4/i8`、`dequant_i4/i8` 走 u8 lane。

**权重（`weights`）**

- `safetensors::File`：打开只读头部，长度校验（`safetensors.rs:145-180`）。`Archive::get(name) -> Option<&Stored>` / `read` / `load`（`archive.rs:172-182`）；`Stored` 持 name/codec/dtype/shape/quant/start/len/logical_len（`archive.rs:31-82`）。

**通信（`shard`）**

- `Collective::all_reduce(&mut self, group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()>`；`Local` 进程内镜像已存在（`collective.rs:11-14`）。

**服务端点（`serve`）**

- `Predict::logits(&mut self, &engine::Sequence) -> Result<Vec<f32>>`；`Sampling::sample(&[f32]) -> Result<Option<u32>>`（`generate.rs:18-46`）。
- `Completion<M: Predict, S: Sampling>` 注入式；推进循环 parse → 编词 → 准入 → 逐步前向+采样 → 归还页 → 帧；失败路径同样归还页（`completion.rs:121-156`、`:225-245`）。生产 `Predict` 实现**不存在**，只有测试替身 `Fixed(Vec<f32>)`（`tests/completion_tests.rs:5-12`）。
- 现状依赖：`diagnostic`、`engine`、`tokenizer`（`crates/serve/Cargo.toml`）；`generate.rs:3` 注释明示"服务层不实现模型"，这条边界随 §4 落点裁决修订。

**训练侧可借鉴机制（`train`，模式先例，不作代码复用件）**

- `Kernel` trait：`compute(op, inputs: &[View], output: &mut ViewMut)`；`Reference` 算术复用 `op` 参考实现（`kernel.rs:17-36`）。
- `KERNELS` 表长度与顺序同 `REGISTRY`，编译期 `const` 断言锁死（`kernel.rs:168-197`）；unary/binary/gemm 宏按 lane 拆形参（`kernel.rs:60-100`）。全 22 算子覆盖，含 `attention_f32` / `flash_attention_f32` / `rope_f32` / 量化族（`kernel.rs:102-166`）。
- 形状推导在 `shape.rs`（`Kind` / `Lane` sealed / `Data` / `View` / `Inputs`）。`train` 不依赖 `engine`（`crates/train/Cargo.toml`），其 `View`/`Lane` 缓冲形态与 engine `Vec<f32>` 不同形。

### 1.3 当前唯一跑通的“前向”路径

根包 `llm run`（`src/main.rs:330-360`）逐层：权重条目字节 → `decode_f32` → `op::model::bind` 造样本 → `cutile::execution::execute` **子进程**编译片段并回读输出与 `max_rel_err` → `x = executed.output`；不支持的层停下，已算结果保留。`Stack` 由头部装配（`main.rs:266`、`:455-469`）。

这条路径是**验证/对拍 harness**，不是生产运行时：每算子一次子进程编译，不经过 `engine::Executor`，没有驻留权重、没有 KV、没有 `Predict` 装配。

---

## 2. 目标前向通路

### 2.1 P1 收尾（CPU 单层/单序列前向，M1）

```
weights::File ──加载条目──▶ 参数表（serve::model 持有，f32 驻留缓冲）
                                   │
nn::Stack/Contract ──装配──▶ engine::Graph（Node + 操作数编号）
                                   │
CpuExecutor::register(Program 表) ── submit(Graph, Stream) ──▶ sync 真跑
                                   │
                          Bindings 读出 logits/激活
                                   │
                    独立重算 op 标量参考 ──对拍 max_rel_err（testkit::close）
```

- 权重在装载期一次性解码进驻留缓冲；生成循环内不再读文件。
- 图由层定义装配：算子名取 `ContractSpec::ops` / `WeightFormat::operator`，激活编号在层间接线，权重编号绑参数表。
- `Program` 算术 P1 直接调 `op` 标量参考（正确性基线）；hot 路径优化留待实测 roofline 后逐算子替换。

### 2.2 P2（推理运行时，M2）

```
多序列 Batch ──▶ 每条序列一步前向
                     ├─ Q/K/V 投影 → 写 KV 行（Batch::write_kv，Pager 真实缓冲）
                     ├─ 注意力从 Pager 读该序列已有 K/V 行（按 Slot 映射进 Bindings）
                     ├─ 残差/RMSNorm/激活/输出投影 → logits
                     └─ Predict::logits 返回；Sampling 采 token；emit 记账推进
prefix cache：共享前缀的页被多条序列共引用，复用而非重算
```

- KV 行宽（`2 · kv_heads · head_dim`）与层偏移表归模型装配侧；`Pager` 只按行寻址。
- 调度与准入只依赖 `Executor` / `Allocator` / KV 簿记，不引用具体后端类型（ADR 0025 §7）。
- 基准 harness 统一记录硬件/形状/批次/后端标识/roofline 上界；decode 时延与吞吐入回归基线。

### 2.3 P3（后端边界与量化/通信原型，M3）

- 独立边界 crate 实现 `Executor` / `Allocator` / `Collective`；`unsafe` 与 FFI 只收在边界内，核心继续 `unsafe deny`（ADR 0025 §9；动工前出单独 ADR）。
- 同一 `Executor` trait 下挂 Metal（macOS 本机验证）与 CUDA（Linux+NVIDIA；无 GPU 只出编译与契约对拍，不出吞吐数字）。
- 量化 u8 lane 内核、反量化；集合通信原型经 `shard::Collective` 挂载。
- 算子选择只走能力闭集（`prof` 档案见证过的供给），模型与调度代码不出现 `if Metal {} else if CUDA {}`（ADR 0025 §4）。

---

## 3. 缺口清单（按阶段，证据 → 缺口 → 归属）

### 3.1 P1 残留

| # | 现状证据 | 缺口 | 归属 |
|---|---|---|---|
| G1 | 生产路径无 `Program` 装配；`CpuExecutor::programs` 只被测试填（`exec.rs:457`） | 没有注册表驱动的 CPU 内核桥；算子名 → `Program` 的映射表未建 | `serve::model`（§4） |
| G2 | `Predict` 只有测试替身（`generate.rs:18-24`、`completion_tests.rs:5-12`） | 没有驻留权重 + 图装配 + 执行的生产 `Predict` 实现 | `serve::model` |
| G3 | `Program::run` 只有 f32 切片，无形状/标量上下文（`exec.rs:388`） | `rope_f32(pos, x, out)` 的位置标量、`attention` 的 heads/head_dim 没有通道；形状只能从切片长度猜。P1 只接纯切片算子，标量通道 P2 再议 | `serve::model`；engine trait 预留扩展（§5.1） |
| G4 | `Bindings` 只有 `u32 → Vec<f32>`，无 dtype/形状元数据（`exec.rs:396-425`） | 装配侧需要一处记录编号 ↔ 形状/条目名；engine 侧是否扩元数据待定（§5.2） | `serve::model` 为主，engine 视需要补只读访问器 |
| G5 | 形状/下界推导只在 `train::shape.rs`；训练 `View` lane 与 engine `Vec<f32>` 不是同一缓冲形态 | 装配侧的形状推导不能直接复用 train 缓冲视图；需要一份面向 `Vec<f32>` 的窄形状适配 | `serve::model` |
| G6 | 对拍只在 `cutile::execution` 子进程回路；无进程内单层前向对拍 harness | M1 需要进程内：同一输入两份计算（Program 输出 vs 独立 `op::forward` 参考），`testkit::close` 判 `max_rel_err` | `serve::model` + dev testkit |

### 3.2 P2 缺口

| # | 现状证据 | 缺口 | 归属 |
|---|---|---|---|
| G7 | 两条 prefill 注意力（`attention_f32` / `flash_attention_f32`）已登记进 serve CPU 内核表（三路 `q/k/v`，头宽 `d` 取静态语义表、`s` 由切片长推）；decode 变长历史变体未登记；`Pager` 行仍只能经 `Batch::kv`/`write_kv` 按 id+token 访问（`batch.rs`） | 注意力尚未接 KV 页：装配侧要把序列已有 K/V 行映射成操作数；Q/K/V 投影后按层偏移写回 KV 行；残差旁路与图扇出通道待扩 | `serve::model` 装配 + attention Program |
| G8 | `Table` 独占 `Page`，无共享引用（`kv.rs:59-183`） | prefix cache：前缀页共引用、引用计数、回收判定；页表结构要留共享句柄 | engine `kv` + `serve::model` |
| G9 | `Batch::step` 逐序列推进页账；`CpuModel` 已接单序列 `Predict`，尚无模型侧批量前向 | 多序列端到端：批内每序列一步前向 → logits → 采样；当前主链只接单 token decode | `serve::model` `Predict` |
| G10 | 基准只有 7 条算子 criterion benches；无 decode 整步/整模型 harness | M2 需要单序列与批次的 TPOT/吞吐/内存对照口径 | benches（工具成员）+ `serve::model` facade |
| G11 | `Predict::logits(&mut self, &Sequence)` 单序列签名已由 `CpuModel` 实现；不带批/KV 视图（`generate.rs:24`） | 模型读写 KV 需要访问页池；trait 路线 B 只读视图传参随 G7 接注意力时定形（§5.4） | serve trait 评审 |

### 3.3 P3 缺口

| # | 现状证据 | 缺口 | 归属 |
|---|---|---|---|
| G12 | 设备后端无实现；`LocalExecutor` 只记账（`exec.rs:304-375`） | Metal/CUDA 边界 crate：`Executor` + `Allocator` 实现，事件同步、设备缓冲 | 新边界成员（P3） |
| G13 | 无 unsafe/FFI 边界 ADR；核心 `unsafe_code = "deny"`（`Cargo.toml` workspace lints） | P3 动工前必须出 ADR：哪些函数/句柄可越安全边界，局部 `allow` 收敛到边界模块 | 文档 + 边界成员 |
| G14 | 量化族走 u8 lane，`Program` 现签名不可表达 | u8 lane 的 Program 适配、反量化内核挂载 | 边界/`serve::model`，P3 |
| G15 | `Collective` 只有 trait 与 `Local` 镜像（`collective.rs:11-14`） | 通信后端实现（NCCL/Metal 对等），拓扑簿记不改 | 边界成员 |
| G16 | 本机 macOS/Apple Silicon；CUDA 链路在本机无环境 | CUDA 只保留编译与契约对拍；Metal 本机验证；Linux 云主机验 CUDA | 环境约束 |

---

## 4. 落点裁决：复用 `serve`，不新建 crate

### 4.1 合法落点穷举（依赖方向是硬约束）

实现 `engine::Program`、持有 `CpuExecutor` 的类型，其所属成员**必须依赖 `engine`**。全仓成员依赖事实：

- `grep -l 'llm-engine' crates/*/Cargo.toml` → 只有 `engine` 自身与 `serve`。
- L7 根包 `llm` 虽依赖 `engine`，但门禁禁止定义任何类型（`architecture.rs:217`）。

| 候选 | 判决 | 证据与理由 |
|---|---|---|
| 适配类型放 `engine` | 排除 | 既定裁决：engine 不加 `op` 边，只认不透明编号与 `Executor`/`Program` 契约，不引用算子闭集（`exec.rs:9-10`，ADR 0025 §4） |
| 装配/`Predict` 放根 bin `llm` | 排除 | L7 禁止定义新类型（`architecture.rs:217`）；bin 只做注入式组装 |
| 复用 `train` 承载运行时 Program | 排除 | ① `train` 不依赖 `engine`（`crates/train/Cargo.toml`），而 `train → engine` 是同层向右，方向非法；② 语义是训练循环，`Kernel`/`View`/`Lane` sealed 缓冲形态与 engine `Vec<f32>` 不同形；③ `serve → train` 同样是同层向右，非法 |
| **`serve` 内新增模型模块** | **采纳** | 唯一同时满足：已依赖 `engine`；可合法向下补 `op`/`nn`/`weights`/`tensor` 边（L6→L2/L5/L4）；`Predict`/`Sampling`/`Completion` trait 本就归 `serve` |

### 4.2 落点：`crates/serve/src/model.rs`（模块名待定，下称 `serve::model`）

L6 顺序不变：`train` → `engine` → `serve` → `cli`；不新增成员、不调整层内位置。

**模块内拥有**：

- 算子 `Program` 适配表（const 锁定，复刻 `train::KERNELS` 模式）；
- 层定义 → `Graph`/`Bindings` 的装配器；
- 驻留参数表；
- 生产 `Predict` 实现（CPU 模型）；
- 进程内对拍 harness 的装配侧。

**不拥有**：执行/调度/KV 簿记（engine）；算子声明与参考算术（op）；层形状语义（nn）；权重字节读取（weights）；错误类型（diagnostic）。

**依赖变化**（`crates/serve/Cargo.toml` + `ARCHITECTURE.md` 边表同步）：

- 已有：`diagnostic`、`engine`、`tokenizer`；`+dev: testkit`。
- 新增向下边：`op`、`nn`、`weights`；`tensor` 仅在装配侧做形状/视图校验时引入，能不引则不引。
- `serve` 不依赖 `train`、`cutile`、`shard`；设备后端代码不进本模块。

**边界修订**：`generate.rs:3` 与 serve 模块头注释里"服务层不实现模型"改为分层表述——端点契约、路由、线格式保持模型无关；CPU 模型实现是注入到 `Completion` 的 `Predict` 提供者，收在 `model` 模块内；Metal/CUDA 真设备后端仍按 ADR 0025 收在独立边界 crate，不回流 serve。

### 4.3 复用清单（直接用 vs 复刻模式）

| 件 | 来源 | 复用方式 |
|---|---|---|
| `Predict` / `Sampling` / `Completion` 注入式装配 | `serve::generate.rs`、`completion.rs` | 直接复用；生产模型实现照搬测试替身 `Fixed` 的 trait 形状 |
| `CpuExecutor` / `Graph` / `Node` / `Operand` / `Bindings` / `Program` | `engine::exec` | 直接复用 |
| `Batch` / `Sequence` / `Pager` / `write_kv` / `kv` | `engine::batch`、`engine::kv` | 直接复用（P2 接 KV 时） |
| `Stack` / `Contract` / `Linear` / `Attention` / `WeightFormat` | `nn` | 直接复用装配输入与算子名选择 |
| `File` / `Header` / `Stored` 权重读取 | `weights` | 直接复用；条目字节解码 f32 沿用 `main.rs:decode_f32` 的 `as_chunks` 整块口径 |
| 标量参考 `op::<op>::forward` | `op` | 直接调用：P1 既是生产 CPU 算术，也是对拍参考的独立重算源 |
| 内核表 const 锁定 + unary/binary/gemm 宏 + 同名同序断言 | `train::kernel.rs:60-197` | **复刻模式**，不复刻代码：train 的 `View`/`Lane` 适配层不带过来，适配函数直接收 `&[&[f32]]` / `&mut [&mut [f32]]` |
| 对拍口径 `close` / `approx_eq` / `all_finite` | `testkit` | 直接复用（dev 依赖） |

### 4.4 门禁同步清单（不新增成员，只补边与所有权）

1. `crates/serve/Cargo.toml`：补 `op`、`nn`、`weights`（按需 `tensor`）依赖，`.workspace = true` 口径沿用既有成员写法。
2. `docs/ARCHITECTURE.md` 边表：`serve → op, nn, weights`（`serve → tensor` 视需要）；层表行职责描述补"注入式 CPU 模型提供者"。
3. 所有权表：`serve::model` 每个新 `pub` 类型登记主人 `serve`；不进表的私有类型不受影响；存量基线 `lints/vocabulary.toml` 仅在必要时补登。
4. `tests/architecture.rs` 自动核对：边集合、方向、所有权公开定义；`tests/contracts.rs` 注解/注册表镜像不受影响。
5. 内核表仿 `train::KERNELS` 写编译期同名同序 `const` 断言。

---

## 5. trait 形状裁决与待定问题

### 5.1 Program 的标量/形状通道（G3，P2 接 rope/attention 前定）

现状 `run(launch, inputs, outputs)` 只传 f32 切片。需要标量的算子：

| 算子 | 额外形参 | 来源 |
|---|---|---|
| `rope_f32` | `pos: usize`（解码位置） | `Sequence::produced + prompt_len`，随步变化 |
| `attention_f32` / `flash_attention_f32` | 序列长 `s`、头宽 `d` | 两条 prefill 已进 serve CPU 内核桥：`d` 取静态 `op::attention::MODEL.shape.1`，`s = q.len()/d`；decode 变体 `q_rows = q.len()/d`、`hist = k.len()/d` 同法切片长推，均不需运行时标量槽 |
| `embed_f32` | 运行时无额外形参；行宽 `D` 由 `out.len()/idx.len()` 推导，表长锁 `V·D` | 切片几何自洽 |
| 量化族 | u8 lane，P3 再议 | — |

**首选（零 trait 变更，复用现有签名）**：P1 只接纯切片算子——`gemv_f32`、`softmax_f32`、`rmsnorm_f32`、`silu/gelu`、`add/mul/residual/scale`、两条归约；`rope_f32` 与 `attention_*` 随 P2 KV 工作一起落地。现有 `Program::run(launch, inputs, outputs)` 不动，engine 单测不变。代价：roadmap P1 清单里的 `rope_f32` 后移到 P2，M1 对拍范围相应收窄。

**预留扩展（P2 接 rope/attention 时再定）**：若届时切片通道仍不够用，再扩 engine 侧节点携带标量槽——`Node` 增加 `params: Vec<ParamSlot>`，`ParamSlot { name: String, value: i64 }`，engine 不认识名字语义只透传；`Program::run` 改收 `node: &Node`。此扩展属 ADR 0025 §后果允许的情形："trait 形状在 P1 暴露不足时，先扩 trait 定义，不在上层加后端特判分支"；落地前回填 ADR 0025 修订节。

不推荐：把标量塞进 `Launch` 三轴——几何与语义参数混在一起，设备后端映射网格时会误读；或把位置编进 1 元素 f32 缓冲——类型不安全，是隐藏约定。

### 5.2 Bindings 元数据（G4）

P1 推荐：编号 ↔ 形状/权重条目名的对照留在 `serve::model` 装配侧的参数表；`Bindings` 维持纯 `u32 → Vec<f32>`。只有当设备后端需要在 engine 侧按编号查 dtype/形状时，再给 `Bindings` 补只读元数据访问器——不为假想需求提前扩面。

### 5.3 u8 lane 延后（G14）

P1/P2 的 `Program` 保持 f32 切片闭集。u8 适配随 P3 量化内核一起进；届时再把 `Program`/`Bindings` 泛型化到 lane，不在 P1 预埋会被第二套实现推翻的抽象。

### 5.4 KV 与 Predict 签名（G11）

`Predict::logits(&mut self, &Sequence)` 当前拿不到页池。两条路线：

- **路线 A**：`Predict` 实现内部持 KV 缓存（按 sequence id 索引），trait 不变；serve 完成页账回收后通知模型淘汰条目。
- **路线 B**：扩 trait，把批/KV 视图作为参数传入；模型直接读 `Batch` 页行、写回当前步 K/V。

**裁决（P2-4 主体动工前定形）**：走路线 B，弃用 A。模型不另存一份 KV 账——路线 A 的内部账（`SequenceKv`/`KvCache`）已从 src 移除（见 P2-4a 记录），机制待按 Batch 页行视图重落。

- **trait 形态**：`logits(&mut self, sequence: &Sequence, batch: &mut Batch)`。读历史经 `batch.kv(id, token)`（`&self` 方法在 `&mut Batch` 上自动只读重借）；写回当前步 K/V 经 `batch.write_kv(id, token, values)`；序列上下文仍由 `&Sequence` 形参给。一步内只持一个 `&mut Batch`，不并存 `&Batch` 与 `&mut Batch`。
- **借用拆分**：`completion.rs::logits` 已示范字段级拆借——`let State { model, batch, .. } = state;`，`model` 与 `batch` 是不重叠字段，两个 `&mut` 并存成立；trait 调用点沿用这套拆法，不在 `&mut State` 上整体借。
- **图层能力**：`Node` 已支持多输入、多输出，同一编号可被多个下游节点读；Q/K/V 扇出（一个 hidden 喂三投影）与残差旁路（Add 引用旁路 hidden）在装配 plan 层补编号接线即可，engine `exec` 不改、不新增标量槽。
- **装配缺口（主体刀）**：plan 需记录旁路编号与注意力三路输入编号；nn 侧 Q/K/V 投影组织约定（三个 `Linear` 同名前缀分组，还是一个组合块）在主体刀按 `nn::Contract` 视图定形；历史 K/V 页行在前向一步前整块读入绑定缓冲（slice 批量拷贝，不逐元素循环）。

### 5.5 对拍口径

- 进程内对拍：Program 输出缓冲与独立重算的 `op::forward` 参考缓冲逐元素比较，口径用 `testkit::close(abs, rel)` / `approx_eq`；`all_finite` 先排非有限值。
- 参考与生产算术在 P1 是同一份 `op` 语义的两次调用；第二份独立参考（如手写闭式）随算子覆盖度补，不在 M1 强求。
- 跨后端对拍（CPU vs Metal/CUDA）是 M3 门：同输入同形状，输出 `max_rel_err` 在 dtype 容忍阈值内。

---

## 6. CPU 内核桥设计（代码约束落地）

### 6.1 注册表锁定，不写特判分支

- 内核表是编译期定长数组，覆盖到的 f32 算子子集；表内名字顺序对 `op::REGISTRY` 做 `const` 子集断言（复刻 `train::kernel.rs:168-197` 先例）；未覆盖的算子在查表时返回 `Unsupported`，而不是留一个静默的空入口。
- 适配函数按形参 arity 分宏生成：unary `[f32]→[f32]`、binary `[f32,f32]→[f32]`、gemv `[x, w]→[out]`；每个适配只做拆切片 + 调 `op::<op>::forward`，算术不重写。
- 算子名选择来自 `WeightFormat::operator` / `ContractSpec::ops`，装配代码不出现算子名字面量的 `match` 特判；后端选择来自能力闭集，不出现厂商分支。

### 6.2 整块 slice，批量处理

- 缓冲搬运一律 `copy_from_slice` / `fill` / `as_chunks`，整块处理；不写逐元素手写循环。
- 保留循环的唯一正当理由：串行依赖（如 softmax 归约两轮、rope 逐位置旋转）或长度数据相关；循环处注释写明保留理由（对齐编码偏好）。
- 权重一次解码进驻留缓冲；激活缓冲按图复用编号，不每步重分配。

### 6.3 声明驱动

- 层 → 节点的接线由 `Contract` 的形状与算子列表驱动；形状不符在装配期报 `Invalid`，不带进执行期。
- 新算子接入：补 `op` 声明 + 表里一行适配；不改 engine、不改调度、不改端点。

### 6.4 失败原子性

- 装配期：形状/长度/dtype 不符整体失败，不留半绑状态。
- 执行期：沿用 `CpuExecutor` 既有不变量——操作数先放回再报错，失败条目留在队列可查（`exec.rs:545-548`、`:632-641`）。

---

## 7. 分阶段实施步骤与出口门

### P1 收尾（M1：CPU 前向可重复、对拍达标、首批对照数字、零新增 unsafe）

1. **serve 模块落地**：`src/model.rs`；补 `op`/`nn`/`weights` 向下边；边表与所有权表同步（§4.4）；模块头注释按 §4.2 修订边界表述。P1 不改 engine trait。
2. **内核桥**：const 锁定的 Program 表，先覆盖纯切片主路径 `gemv_f32`、`softmax_f32`、`rmsnorm_f32`、`silu_f32`、`gelu_f32`、`add/mul/residual/scale`、两条归约；`rope`/`attention` 随 P2 接入。
3. **装配器**：`Stack`/`Linear`/`Attention` → `Graph` + 权重 Bindings；驻留参数表。
4. **单层前向对拍 harness**：进程内双算 + `testkit` 判误差；登记首批基准数字到 roadmap §1.1。
5. **出口门**：`cargo test -p llm-engine` 全绿；`cargo clippy -p llm-engine -D warnings` 零告警；`cargo fmt --check`；architecture/contracts 门通过；新增 `unsafe` 为零；用例数不跌。

#### 落地记录（第一刀：内核桥 + 进程内对拍）

- 模块 `serve::model` 已落地（`pub mod model`）；本轮向下边只补 `op`（边表 96 → 97，dev 仍 32）。`nn`/`weights` 边随步骤 3 装配器与驻留参数表一起补，未引用不提前扩面。
- 内核表 `KERNELS` 覆盖 11 个 f32 纯切片算子：`add/gelu/gemv/mul/reduce_max/reduce_sum/residual_add/rmsnorm/scale/silu/softmax`；`const` 子集断言锁定每条名字都在 `op::REGISTRY` 内；未覆盖算子不登记，执行时由 `CpuExecutor` 按名拒绝。
- 适配宏 `unary`/`binary`/`gemv` 只拆切片 + 调 `op::<op>::forward`；算术不重写；无 `unsafe`。
- 对拍 harness（`tests/model_tests.rs`）：经执行器跑注册内核，与独立 `op::forward` 参考双算，`all_finite` 排非有限后 `close(1e-6, 1e-6)` 判误差；含一条图内编号串联（add → silu）与一条未覆盖算子拒绝用例。
- **下一刀（步骤 3）**：补 `nn`/`weights` 边；`Stack`/`Linear` → `Graph` 装配器 + 驻留参数表；真实权重解码进驻留缓冲；生产 `Predict` 宿主。
- 本刀门禁：engine/serve 测试全绿（serve lib 72、protocol 8、doctest 3）；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5 全过。

#### 落地记录（第二刀：装配器 + 驻留参数表 + CPU 宿主）

- 补 `serve → nn` 边（边表 97 → 98，dev 仍 32）；`weights` 边本轮仍不引：`Params` 是内存驻留表（条目名 → f32 整块），真实权重文件装载（`Stack::from_header` + `File::open` + `decode_f32`）留独立一刀，届时再补 `serve → weights` 边并同步边表总数。
- `Params`：条目名 → f32 整块；`CpuModel` 把 `Stack` 的 F32 线性层装配成 `engine::Graph`，权重在构造期驻留一次，`forward` 每步只覆写输入与逐层激活缓冲，不重新分配。
- 装配期校验：空栈 `Invalid`；非投影层、非 F32、无算子名 `Unsupported`；缺权重 `Missing`；权重长度不符、层间宽度链断裂 `Invalid`。算子名只取 `WeightFormat::operator()`，不写名字面量 `match`。
- 编号隔离：激活从 0 起（输入 0、逐层输出 1..），权重从 `WEIGHT_BASE = 10_000` 起；编号按层索引推导，无显式计数器自增。
- 测试（`tests/model_tests.rs`）：两层 F32 链 `forward` 与串行 `op::gemv_f32` 参考对拍；权重长度不符、缺权重、非 F32、宽度链断裂四条装配期错误用例。
- **未做（切分）**：`Predict` trait 实现留 P2 嵌入/词表通路落地后——`Sequence` 只有 token、尚无嵌入查表通路，本轮不硬造把 token 当向量的实现；宿主先暴露张量级 `forward(&[f32]) -> Vec<f32>`。KV 账由调度器持有，模型不另存。
- 本刀门禁：serve lib 77、protocol 8、doctest 3 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5 全过。

#### 落地记录（第三刀：真实权重条目装载进驻留参数表）

- 补 `serve → weights` 生产边（边表 98 → 99，dev 仍 32）；f32 小端解包下沉 weights 为公共件 `decode_f32`（`as_chunks::<4>` 解包，余字节非零报 `Invalid`），serve 不另写 `from_le_bytes` 循环。
- `Params::from_stack(stack, header, loader)`：逐层按 `{层名}.weight` 查头部条目；`entry_for` 缺条目报 `Missing`，`load_f32` 判 dtype——非 F32 返回 `None` 不解包（留装配期按格式拒绝），F32 走 `Loader::load` + `decode_f32` 解码进驻留表。serve 只依赖 `Loader` 契约，不直接 `File::open`；路径打开与 bin 串联留调用方。
- 测试：内存 safetensors 归档（`Archive::parse`）两层 F32 装载后装配 `forward`，与串行 `op::gemv_f32` 参考对拍；头部缺条目 `Missing`；非 F32 条目装载期跳过、装配期 `Unsupported`。weights 侧 `decode_f32` 两条单测：完整字解包对拍、尾部半字 `Invalid`。
- **未做（切分）**：`Predict` trait 实现仍待 P2 嵌入/词表通路；`File::open` 路径装载与命令行串联留调用方/后续刀。
- 本刀门禁：serve lib 80、protocol 8、doctest 3 全绿；weights 测试全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5 全过。

#### 落地记录（P2-1：装配器接入一元点wise与归一化族）

- 装配器从仅认 `Linear` 扩到按算子族别接线：`Projection`（输入向量 + 驻留权重 `[out,in]`）、`Unary`（单路输入等长输出、无权重）、`Norm`（输入 + 驻留 gamma `[width]`）。族别由常量表 `UNARY_OPS` / `NORM_OPS` 检索，算子名取 `spca().ops()`，不写层类型 match，也不写后端厂商分支。
- 几何：投影宽度取 `linear()` 视图；一元/归一化宽度取 `output()` 单轴形状 `[width]`，标量（归约）与多轴形状在装配期报 `Invalid`。宽度链：投影输入对上一层输出，等长族输出对上一层输出；权重期望长度按族别 `weight_len` 派生（投影 `in×out`、归一化 `width`、一元 `None`）。
- 本刀不接：二元残差（需旁路 DAG 接线）、归约、rope、注意力、查表层，均按 `Unsupported` 拒绝；残差旁路与 KV 通路随注意力刀落地。
- 测试：投影→silu、投影→rmsnorm 两条混合链 forward 与串行 `op::gemv_f32 → silu/rmsnorm` 对拍；一元宽度链断裂 `Invalid`、缺 gamma `Missing`、二元残差 `Unsupported` 三条拒绝用例。
- 本刀门禁：serve lib 85、protocol 8、doctest 3 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；静态复杂度门通过；architecture 8、contracts 5 全过。无新增 pub 类型与依赖边，边表/所有权表不变。

#### 落地记录（P2-2a：`embed_f32` 改造为行向量 gather）

- 不新增算子：现有 `embed_f32` 从标量 gather（`y[n]=table[idx[n]]`）改为行向量 gather（`y[n,:]=table[idx[n],:]`）。表 `[V,D]`、索引 `[n]`、输出 `[n,D]` 均按行展平；`V=64` 仅为样本档，运行时行数由表长/行宽推出，行宽 `D` 由 `out.len()/idx.len()` 推导，表长须为 `D` 的整数倍。
- 三件套同步改写：`embed_f32.rs`（两轴 `n/d`、`project_shape`、`cost` 按访问行 `n·D` 计价、`sample(tokens,dim)`、行拷贝参考与内核；空索引、不可整除、表长不符、下标越界报 `Invalid`）；`.toml` 布局补 `y=row`、summary 改行公式；`.k` 模板内层按 `D` 逐元素 `@store(y, n*D+d, row[d])`，标量 `N/V/D`。
- arity 不变：`forward(idx, table, out)` 三参，train `binary!` 桥与注册表行不动；复用 `embed` 骨架（闭集 12 个不扩），注解 shape `[8,64]`。
- 测试：行 gather 参考对拍；下标越界、表长不符两条拒绝用例。
- 本刀门禁：op 167、nn 65 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5、gate 1 全过；无新增 pub 类型与依赖边。
- **未做（切分）**：`Sequence` 补完整 prompt token 归 P2-2c 落地。

#### 落地记录（P2-2b：`Embedding` 接入 serve 装配族）

- `Contract` 对称新增类型视图访问器 `embedding() -> Option<&Embedding>`（与 `linear()` 同模式）；装配侧据此取 `vocab/dim/tokens`，不写层类型 match 特判。族别分流：类型视图（投影/查表）优先，其余按 `UNARY_OPS` / `NORM_OPS` 常量表归类。
- 新增 `LayerKind::Embedding`：索引序列 + 驻留权重表 `[vocab, dim]` → 行向量；权重期望长度 `vocab·dim`，链宽度输入取 `tokens`、输出取 `dim`。`LayerPlan` 把权重期望长度收为显式 `weight_len` 字段（投影 `in·out`、归一化 `width`、查表 `vocab·dim`、一元 `None`），不再从族别反推。
- 几何切口：当前向量主链只接单 token decode 步（`tokens == 1`）；多 token 矩阵主链随连续批刀落地，装配期报 `Unsupported`。`embed_f32` 复用二路适配宏（激活/索引 + 驻留权重 → 输出），三参 arity 不变。
- 运行时词表行数不再锁死样本档 `V=64`：表长按行宽整除推行数，下标越界对照实际行数；`VOCAB` 仅用于样本生成。
- 测试：单步查表对拍、查表→投影链串行 `embed → gemv` 对拍；缺词表 `Missing`、表长不符 `Invalid`、多 token `Unsupported` 共五条。
- 本刀门禁：serve 90、nn 65、op 167 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5、gate 1 全过；无新增 pub 类型与依赖边。
- **未做（切分）**：多 token 矩阵主链归 P2-5 连续批；归档头部推层目前只认秩 2 `Linear`，`Embedding` 经 `Stack::of` 直构，头部推查表层留装载串联刀。

#### 落地记录（P2-2c：`Sequence` 留存完整 prompt token）

- 单一 `all: Vec<u32>` 存完整上下文（提示段在前、生成段在后），`prompt_len` 标记分界；`tokens()` 仍切出生成段，客户端历史口径不变；新增 `context()` 返回整段序列，供嵌入查表与注意力回看历史。
- 构造改为 `Sequence::new(id, prompt: &[u32], max_new, page_size, stream)`，`prompt_len` 由切片长推出（避免长度两处漂移）；`prompt_len + max_new` 溢出校验沿用 `checked_add`；`push_token` 上限按生成段长度 `all.len() - prompt_len` 判定，追加进 `all`。
- 生产链路（`completion`）：`parse` 已得到编码后的 `Token` 序列，`.id()` 转 `u32` 整段传入序列；`history()` 仍只取 `tokens()`，响应不含提示段。
- 测试：`context()` 含提示+生成、`tokens()` 只剩生成段一条内联对拍；各测试 helper 内部造占位切片，保住页数/长度断言语义。
- 附带修正：`attention::check_decode_len` 条件叠加到 CCN=8，拆出 `check_rows` / `check_pair` 两个低复杂度小校验，主函数只串 `?`；测试仍只断言 `is_err`，错误消息泛化但语义不变。
- 本刀门禁：engine 62、serve 90、op 171 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5、gate 1 全过；无新增 pub 类型与依赖边。

#### 落地记录（P2-3：`CpuModel` 实现单序列 `Predict`）

- `CpuModel` 接 `crate::Predict`：`logits` 先经 `predict_input` 把序列当前位置折成一步输入，再走既有张量级 `forward`；模型不另存 KV 账，页账仍归调度器。
- `Assembly` 记首层族别 `input_kind`（装配期 `chain` 取出首层 `LayerPlan`）。查表层首层收一个索引：decode 一步取 `Sequence::context()` 末位 token，`u32 as f32` 喂入（索引 f32 承载是既有 `embed_f32` 契约）；投影/点wise/归一化首层需要隐向量，而序列只持有 token id，给不出向量，按 `Unsupported` 拒绝——向量主链随连续批刀落地。
- **trait 签名本刀未扩**：路线 B 的 Batch/KV 只读视图传参留 P2-4 接注意力 KV 时定形（§5.4）；当前单序列前向只需序列上下文，不为假想的 KV 读写提前扩面。借用拆分沿用 `completion.rs` 示范的 model/batch 分字段借用法。
- 测试：查表→投影链序列 `logits` 对拍同栈张量级 `forward`；投影首层 `logits` 报 `Unsupported`；空序列（无 token）报 `Invalid`。
- 本刀门禁：serve 92、engine 62、op 171 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5、gate 1 全过；无新增 pub 类型与依赖边。

#### 落地记录（P2-4a：注意力 prefill 两条进 CPU 内核表 + 模块拆分）

- `attention_f32` / `flash_attention_f32` 登记进 serve CPU 内核表：三路 `q/k/v → out`，头宽 `d` 取静态 `op::attention::MODEL.shape.1`，序列长 `s` 由 `q.len()/d` 推出，形状不符交算子参考校验；朴素与 Flash 同几何只换内核算式。几何口径复刻 `train` 桥（`d` 取静态语义表、`s` 由切片长推），本刀不引入 Program 运行时标量槽。
- 模块拆分：原 `model.rs` 超 899 行行数上限，按职责拆成 `model/mod.rs`（编译期锁定的算子表与适配函数）与 `model/assemble.rs`（层栈装配、驻留参数表、前向宿主 `CpuModel`）；`mod.rs` 经 `pub use` 把 `CpuModel` / `Params` 放回 `serve::model` 路径，外部引用不变。
- 清理过渡死账：路线 A 模型内部 KV 账（`SequenceKv` / `KvCache`）无持有者，且将被 §5.4 路线 B 的 Batch 只读视图取代，删除其 src 副本与平行测试副本；行追加、历史拼接与淘汰机制待 P2-4 主体按 Batch 页行视图重新落地，不在无视图通道时预留死代码。
- **未做（切分主体）**：Q/K/V 三路投影写回页行、历史 K/V 读入拼块、注意力族接装配链、残差旁路、decode 变长历史变体登记；Predict 路线 B 视图形态与图扇出/旁路通道在主体刀定形（§5.1、§5.4）。
- 本刀门禁：serve 93 全绿；`cargo clippy --workspace --all-targets -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5、gate 1 全过；无新增成员与依赖边。

#### 落地记录（P2-4b：nn `AttentionBlock` 注意力组合块）

- nn 侧新增 `AttentionBlock`：内聚 Q/K/V 三个投影（`hidden → kv_width`）+ `Attention` + 输出投影（`kv_width → hidden`），作为单个 `Contract` 层；子层按 `{块名}.q` / `.k` / `.v` / `.attn` / `.o` 前缀对齐权重条目。`Contract` 新增 `attention_block()` 类型化视图（默认 `None`），装配侧经此取回子层与块几何，不在外层按层名前缀分组扇出。
- 几何与下界：`hidden` / `kv_width` / `head_dim` / `seq` 非零，`kv_width` 整除 `head_dim`（头数为整数）；块下界逐子层求和，任一子层无可计价形状则整块上报 `None`；块级 ops 列去重后的投影算子 + 注意力算子。
- 边界：组合块不内聚残差旁路与层归一化——二者引用块外主干隐藏向量，归装配侧接线；多头展平与头内切片仍由 `Attention` 既有几何口径承担。`AttentionAlgo::operator()` 升为公开算子名访问器（取代私有 `op()`），供组合块拼块级 ops。
- **未做（装配主体）**：把组合块拆成 q/k/v/attn/o 子节点接进图、三路扇出与 KV 页行读写、残差旁路编号接线；Predict 路线 B 视图形态与图扇出/旁路通道在 P2-4 主体刀定形（§5.1、§5.4）。
- 本刀门禁：nn 全绿；workspace 0 失败；`clippy --workspace --all-targets --all-features -- -D warnings` 零告警；`fmt --check` 干净；architecture 8、contracts 5、gate 1 全过；无新增成员与依赖边。

### P2（M2：多序列端到端、CPU 同档 decode、KV 对拍与边界测试）

1. KV 行映射进注意力 Program；Q/K/V 写回 Pager；层偏移表归装配侧。
2. `Predict` 多序列前向；与 `Batch` 推进循环对接；prefix cache 共享页（引用计数 + 回收不变量）。
3. decode 基准 harness：单序列/批次 TPOT、吞吐、内存；对照 llama.cpp 同硬件曲线入表。
4. 出口门：KV 分配/释放有对拍与边界测试；调度不违反页表不变量；连续批端到端跑通；架构门通过。

### P3（M3：双后端挂载 + 量化 + 通信原型）

1. 前置：unsafe/FFI 边界 ADR；边界 crate 骨架；`Executor`/`Allocator` 实现。
2. Metal 本机跑通 P2 同一前向；CUDA 在 Linux+NVIDIA 验，无 GPU 环境只编译 + 契约对拍。
3. 量化 u8 lane 与反量化内核；`Collective` 通信原型。
4. 出口门：同算子 CPU/加速后端对拍一致；同后端对比 burn/candle；边界 crate 之外零新增 `unsafe`；roofline 达成率可追。

---

## 8. 不做与暂缓

- 不造通用 IR、不做算子融合重排（ADR 0017 无 IR 冻结）；图只是提交单元。
- P1/P2 不接 GPU、不写量化内核、不做分布式；u8 lane 抽象不预埋。
- 不新增 workspace 成员（P3 边界 crate 除外，届时按 ADR 0025 §1 单独裁决）。
- 不引入模型动物园、vision、RL；产品面不在本路线。
- 不在核心仓开第二套设备能力闭集；设备能力继续以 `prof` 档案与 L0–L3 证据分级为准。
- 不预填未实测的性能数字；跨机绝对值不可比，CI 只守不回归。

## 9. 动工前待补记录

| 项 | 落点 | 触发时点 |
|---|---|---|
| Program 标量槽与签名扩展裁决（P1 纯切片先行；rope/attention 落地时再判是否扩 Node 参数槽） | 回填 ADR 0025 修订节 | P2 接 rope/attention 前 |
| `serve::model` 模块名与最终向下边表（是否引 `tensor`） | ARCHITECTURE 边表 + 所有权表 | 已定模块名 `model`；已补 `op`、`nn`、`weights`；`tensor` 暂不引 |
| Predict/KV 传参路线（A/B） | 新 ADR 或 ADR 0025 补充 | P2 动工前 |
| 嵌入算子形态：改造 `embed_f32` 为行向量 gather（表 `[V,D]`，按 idx 取整行到 `[n,D]`），不新增算子 | op `embed_f32.rs` + `.toml` + `.k` + 自测；复用 `embed` 骨架 | 已落地（#58）：三参 arity 不变，train 桥与注册表行未动；骨架闭集 12 个不扩 |
| unsafe/FFI 边界规则（函数/句柄越界白名单、局部 allow 收敛点） | 新 ADR（ADR 0029） | P3 动工前 |
