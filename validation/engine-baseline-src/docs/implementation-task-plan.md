# 落地编码规范与任务分解（v1.2）

- 日期：2026-11-17
- 状态：决策看板 D1–D13 已定形（§4）；动工前按编号补 ADR 正文，结论以看板为准
- 配套：[`llmu-parity-optimization-design.md`](llmu-parity-optimization-design.md)（能力补齐与不退化清单）、[`backend-runtime-design.md`](backend-runtime-design.md)（缺口 G1–G16 与落点裁决）、[`engine-benchmark-roadmap.md`](engine-benchmark-roadmap.md)（对标指标与里程碑）、[`code-audit-2026-11-17.md`](code-audit-2026-11-17.md)（基线实测与本轮修复）
- 口径：本文是**编码方式规范 + 任务 backlog**，不是实现代码；性能/倍数在可运行后端同机实测前不预填。

## 0. 本轮评审结论（对照代码与标杆，v1.0 → v1.2）

评审方法：backlog 逐条回读工作树现状（trait/表/模块/既有实现），对照 roadmap §4 对标矩阵、backend G1–G16 缺口、design §1.5 不退化清单。结论分四类：**订正**（与现状不符）、**补任务**（遗漏）、**补强规范**（约定不够可执行）、**显式暂缓**（防范围蔓延）。

### 0.1 与代码现状不符，已订正

| # | v1.0 表述 | 取证现状 | 处置 |
|---|---|---|---|
| R-1 | M1-A1：`Source` 抽象定形，下分 `Reader`/`Loader`，`FileSource` seek+read | `weights::Reader`（`lib.rs:268`）/`Loader`（:277）**已定形**；`Archive` 双实现（`safetensors.rs:103/127`）、`File` Loader（:191）已存在 | 任务改为沿用并收口既有 trait，只新增 `MmapSource`（Reader 零拷贝）；不另立平行 `Source` 父 trait（D1 同步订正） |
| R-2 | M2-2：`Predict` 批量前向，接 `Batch::step` 循环 | 调度层批步进已落地：`Batch::step` 遍历序列出 `Progress`，`emit`/`finish` 摘出归还页（G9）；未接的是**模型内核级批量几何**（executor 图仍单序列几何，Predict 保持单序列视图） | M2-2 拆成"调度批（已具备，守回归）"与"模型批形状（未接）"两层 |
| R-3 | M3-3：量化 u8 lane | 注册表 lane 为 `i4`/`i8`；backend G14/§5.3 称 u8 是通道形态描述 | 任务名对齐注册表：`i4/i8 lane` 适配 + 反量化内核；u8 仅作字节通道描述 |
| R-4 | §1.3 第 4 条把注解双表合一指向 P0-4 | 双表合一任务是 P0-2；P0-4 是报告器合一 | 交叉引用订正 |
| R-5 | P0-3 大文件路径写 `cutile/main.rs` | 实际为 `src/bin/cutile/main.rs`；清单以 design §5.4 为准 | 路径订正 |
| R-6 | M1-B1：算子名 → `Program` const 表待建 | `serve::KERNELS` 已是 const 切片子集表并带注册表子集断言（`kernels.rs:160/179`） | 任务改为补缺口算子 + 断言加固，不新建表 |
| R-7 | 已具备能力（P9 多格式往返、P13 设备档案闭集）无回归守护 | design §1.5 标"已具备"，但 backlog 无守回归条目 | 新增 P0-8 守护任务，防后续重构退化 |

### 0.2 backlog 遗漏，已补入

| # | 遗漏点 | 依据 | 落点 |
|---|---|---|---|
| R-8 | 关键纯函数属性/模糊加固 | roadmap P0"关键纯函数补属性测试，退化情形进 fuzz"；`fuzz` crate 已有 contract/finding/seed/sweep 骨架 | P0-6 |
| R-9 | 片段参数类型签名交叉校验 | ADR-0027 §6 注：缓存身份字段已落地，片段签名交叉校验随首条真实内核落地，尚未做 | P0-7 |
| R-10 | Program 标量/形状通道（rope 位置标量、attention heads/head_dim） | backend G3、§5.1；D9 已定：定长类型化闭集槽，不用查表 | M1-B6 按 D9 落地 |
| R-11 | Bindings 元数据（编号 ↔ 形状/条目名） | backend G4、§5.2；D10 已定：描述表留装配侧，engine 纯缓冲 | M1-B7 按 D10 落地 |
| R-12 | 词表（vocab）从权重容器装载 | `Tokenizer` 现由 `Completion` 外部注入（`completion.rs:29`）；weights 无 vocab 读取通路 | M4-1 补条目 |
| R-13 | 训练样本夹具装配 | 样本三件套 `Dataset` trait/`Chunks`/`Batch` 在 `train::samples`（原 `data` 成员，ADR 0038 §修订并入），T1-3 未列装配交付物 | T1-3 补条目 |
| R-14 | 机制代码分账规则落档 | design §5.3 排除规则与 L1–L7 分账口径，M1-C1 只说"出基线" | M1-C1 补"规则可复跑、分账表入档" |

### 0.3 规范薄弱，已补强（§2）

- 复用优先级：既有 trait/闭集/常量表 > 泛型抽取 > 新类型；禁止绕过既有抽象新写平行实现。
- 配置派生：形状/宽度/参数组归属由层声明与档案推出；模型与调度代码出现字面量形状宽度须有架构门或测试拦截。
- 向量化手段：`copy_from_slice`/`fill`/`as_chunks`/整块查表优先；标量循环只在串行依赖或长度数据相关时保留并注释。
- 验收阈值引用既有工具：`testkit::close(abs, rel)`、dtype 容忍表、gradcheck 阈值、bench 不回归口径；不在任务里新造阈值。
- 文档漂移守护：代码指针/计数/能力状态回填进同一闭环（呼应 D-1…D-7 教训）。

### 0.4 显式暂缓（不进本路线，防止范围蔓延）

见 §3.8：投机解码、beam search、MoE、vision/多模态、RL、LoRA/适配器、FP8/AWQ/GPTQ 新格式、KV 量化、chunked prefill、第三及以上后端（MUSA/OpenCL/SYCL/ANE/第三加速链路）、gRPC。这些只在对应阶段预留接口形状，不删除既有模块；动工新增须先经决策看板落 ADR。

### 0.5 第二轮代码取证（任务前提逐条核对）

| # | 任务前提 | 取证结果 | 对 backlog 的订正 |
|---|---|---|---|
| R-15 | 权重借/有二分 | `Archive::load` 已返回 `Cow`：`Codec::Store → Borrowed`、`Rans → Owned`（`archive.rs:182-186`） | M1-A2 不再做二分设计，只补 `MmapSource` 让 Store 段经映射借用；二分测试断言保留 |
| R-16 | 参数组元数据落点 | `ContractSpec` 现只有 `name/output/ops`（`nn/src/lib.rs:105`），**无角色/可训练/wd 豁免字段** | T1-2 含 `ContractSpec` 扩展：新增 `ParamRole` 闭集字段；优化器不猜字段名（见 D2/D11） |
| R-17 | KV 前缀共享引用计数 | `Cache` 已有 `refs` 引用计数：`alloc` 置 1、`share` +1、`release` 归 0 才退回并清零；`Pager::share/release` 已接线（`kv.rs:419-555`） | M2-3 不再重做引用计数；任务改为前缀匹配、准入时 `share` 采纳、回收判定策略（见 D12） |
| R-18 | 词表装配点 | `VocabSource` 中性数据结构已存在（`vocab.rs:61`），`Vocab::try_new` 闭合不变量；缺的是容器字节 → `VocabSource` 解析 | M4-1 限定为该解析通路，词表指纹对拍沿用既有 `Fingerprint` |
| R-19 | 优化器是否已存在 | `train` 只有 Schedule/Checkpoint/Buffers/Pool/Store；无 Adam/SGD/矩状态/梯度更新代码 | T1-1 从零建 `Optimizer` trait 与两实现，前提成立 |
| R-20 | 注解宏消费者规模 | 27 个算子模块挂 `#[annotation::model]`，宏已承担声明展开主路径 | D7 定保留宏；双表合一（P0-2）不改宏对外形状 |
| R-21 | 生产外部依赖 | 生产路径仅 `diagnostic` 的 optional `toml`（build feature 门控）、`dtype` optional `bytemuck`；无 serde/正则/命令行库 | D5 定维持零生产依赖；解析收敛到模块内单源 |
| R-22 | 后端 trait 现状 | `Executor`/`Allocator`/`Collective`/`Program` 均已在 `engine`/`storage`/`shard` 定形；`LocalExecutor` 为契约后端，`CpuExecutor` 持 `Program` 注册表 | D6 定沿用既有 trait，第二后端不回改上层 |
| R-23 | 检查点状态覆盖度 | `Checkpoint` 只记 step/tape_nodes/batch_rows/layers/weight_entries/world；`resume` 对账拓扑/层数/批行；`Snapshot` 格式含版本与 Facts，**无优化器矩状态、无累积梯度** | T1-4 扩快照格式落矩状态；T1-5 累积状态同格式往返 |

## 1. 文档评审：已发现的问题

### 1.1 事实漂移（须回填到对应文档）

| # | 位置 | 旧写法 | 实测现状 | 修正动作 |
|---|---|---|---|---|
| D-1 | 设计 §1.4、backend §0/§1.2、roadmap §3/§4 | 注册表 22 条 | **24 条**（含 `decode_attention_f32`、`decode_flash_attention_f32`） | 三处计数与证据指针回填 24 |
| D-2 | 设计 §1.2、roadmap §3 | 测试 1,305 / 1,102 通过 | **1,396 通过 / 0 失败**（`cargo test --workspace`，1 ignored） | 基线计数回填；口径注明"测试结果"与"测试函数数"不可混 |
| D-3 | 设计 §1.2 | Rust 总行数 70,577 | 粗口径 **76,839**（含测试，不含 `target/`） | 回填；机制代码分母见 §5.3 分账表 |
| D-4 | roadmap §3、iteration §2 P2 | KV"页内不持数据，只有页表/位图簿记" | `engine::kv::Cache` 页池已持真实 `Vec<f32>`（`kv.rs:220`，`row/row_mut/write` 经 Slot 寻址）；模型侧历史读/当前步写回已接路线 B | 改为"页池数据已落地；前缀页共享引用、多序列批量循环未接" |
| D-5 | 设计附录 A | Predict 首层限制指针 `assembly.rs:585-603` | 宿主段已拆到 `serve/src/model_host.rs::predict_input`（:63） | 指针改指 `model_host.rs` |
| D-6 | backend §9 末行 | unsafe 边界 ADR 标注"ADR 0029" | 0029 是 admitter 端口；边界裁决文档为 `0035-unsafe-ffi-boundary.md`（并发产物，待评审） | 编号订正；P3 动工前按 0035 复核 |
| D-7 | 设计 §1.3 KV 行 | "serve Predict 仅接 Embedding 首层" | 路线 B 签名已定形 `logits(&Sequence, &mut Batch)`；非查表层首层仍 `Unsupported`，状态描述补全 | 措辞区分"trait 签名已接 Batch"与"向量首层通路未接" |

### 1.2 口径矛盾（须统一，不允许两套说法并行）

- **里程碑标签两套**：roadmap/iteration 用 `P0–P4`，设计/backend 用 `M1-A / M1-B / M1-C / T1 / M2 / M3 / M4`。映射见 §5，任务 backlog 一律用设计侧编号，roadmap 标签作对照列。
- **KV 成熟度**：backend §1.2 已记"Pager 持真实 `Vec<f32>`"，roadmap/iteration 仍记"簿记"。以 backend 为准，§1.1 D-4 回填其余两处。
- **内核表口径**：train 24/24 全量定长表；serve 16/24 纯切片子集。两种口径并存是刻意的（train 参考全覆盖、serve 只登记 f32 子集），须在 `kernels.rs` 注释写清边界，避免后人当不一致缺陷补齐。

### 1.3 文档缺口（本轮补齐）

1. **缺分层编码规范**：设计 §3.2 只有挂载点表，没有 trait 形状选择、注解边界、单态化/dyn 边界、批量性能规则的成文约定 → 本文 §2。
2. **缺任务级 backlog**：阶段目标到了"做什么"，但没到"谁做、依赖谁、交付什么、门是什么、暂缓什么" → 本文 §3。
3. **决策看板**：v1.0 的 D1–D10 只列倾向与触发条件；v1.2 已逐条定形（选定项/被否备选/取证依据/ADR 落点），并补 D11–D13 → 本文 §4。
4. **注解双表合一未进任务**：`annotation::codegen.rs` 的 `REFERENCE_OVERRIDES` / `MODULE_OVERRIDES` 同以注册名为键，两张表可合一为一行三列（审计 §4.2），未列入减量 backlog → §3 P0-2。
5. **密度基线**：粗口径 76,839 含测试；机制代码分母已按 §5.3 分账（M1-C1，见 `mechanism-lines-baseline.md` 目录级 50,713），严格口径（逐行扣除内联 `#[cfg(test)]` 体）仍待 `strip_tests` 接入脚本。

## 2. 编码方式规范（泛型 / 注解 / 配置化优先）

### 2.1 分层与依赖不变量（沿用 `ARCHITECTURE.md`，不得破）

- 依赖只能向下或同层向左；新 `pub` 类型必须进 `ARCHITECTURE.md` 所有权表或 `vocabulary.toml` 豁免基线。
- 运行时新增件只落在 L1–L7 既有层内；后端 FFI/`unsafe` 收在 L3 之下的独立边界成员，不回流核心。
- L7 根包只组装，不定义新类型。

### 2.2 trait 与泛型形状

| 扩展点 | trait/抽象 | 归属层 | 形状约定 |
|---|---|---|---|
| 后端执行 | `Executor` / `Allocator` / `Collective`（ADR-0025） | L6 engine + 边界 crate | CPU 首实现定形；设备句柄不进模型/调度代码；trait 在第二后端出现前不预埋多余泛型参数 |
| 权重源 | 既有 `Reader`（借用）/ `Loader`（拷贝）trait（`weights::lib.rs:268/277`） | L5 weights | 沿用既有 trait，不另立平行 `Source` 父 trait；一条源一份实现；缺失类型编译期报错，不静默降级为整文件读入 |
| 优化器 | `Optimizer` | L5/L6 train | `step(params, grads, meta: &ParamRole)`；矩状态优化器自持；角色元数据来自 `nn::ContractSpec` 声明（D2/D11） |
| 采样 | `Sampling` | L6 serve | 温度/top-k/top-p 参数进闭集档案；实现经注入挂载到 `Completion` |
| 算子内核 | `Program`（engine 既有）+ 注册表 | L6 serve 适配 | 算子名 → `Program` const 锁定表；适配只拆切片调 `nn` 标量参考内核，不重写算术 |

选择规则：

- **复用优先级**：既有 trait / 闭集 / 常量表 > 泛型抽取 > 新类型；先查上游是否已有同构抽象，禁止绕过既有实现新写平行 trait 或第二份映射。
- **关联类型优先于泛型参数**：实现一次定形的能力（后端、源、优化器）用 `&dyn` 注入边界 + 关联类型；同一数据形态多版本特化才用泛型参数。
- **热路径零动态分发**：前向/解码内核保持编译期单态化；`dyn` 只允许出现在注册表登记点、`Completion` 注入边界这类一次装配处，不出现在逐元素/逐 token 循环内。
- **不为假想第二实现预埋抽象**：trait 形状在第一个实现落地时定形；预留扩展点记录在案，不提前加被推翻的参数。
- **格式差异收进 trait 关联项/闭集**：不在算子里写 `if Metal {} else if CUDA {}`，不写按厂商名字的 `match`。

### 2.3 注解边界

- proc-macro 只做**声明展开**：`spec` 产出 `ContractSpec`，`model` 产出 `Model` 条目与常量表、骨架名投影；不持有运行期注册表，不做代码生成以外的魔法。
- 新算子接入路径固定四件：`prof` 声明一行 + `.toml`/`.k` 片段 + 语义模块（cost/sample/reference）+ 注解 `#[annotation::model]`；train/serve 内核表按口径补适配。新层经 `#[spec]` 声明 `ContractSpec`。
- 新层接入：`nn::Contract` 类型视图优先；族别归属走常量表检索，外层不按层名前缀分组扇出。
- 注解例外表合一后，注册名 → (骨架, 参考入口, 模块) 一行三列；缺省取骨架默认。

### 2.4 配置与表驱动

- 闭集、族别、字段、内核表、参数组归属各有**唯一常量表**；表内名字由声明派生（沿用 ADR-0031/0032/0034 模式），不手抄第二份映射。
- 形状、算子选择、权重格式、参数组由层声明与档案推出；模型与调度代码不出现层名宽度、厂商特判、第二份手写常数；字面量形状宽度进代码须有架构门或测试拦截。
- 运行时配置（lr/warmup/步数/裁剪/累积、采样档）进命令参数与参数组档案；种子、量化常数、零点不进配置（iteration §4 既有取舍）。
- 参数组归属（可训练 / wd / bias / norm 豁免）来自 `nn::ContractSpec` 声明，优化器不按字段名猜测。

### 2.5 性能规则（编码偏好，对齐既有偏好记录）

- 缓冲搬运一律 `copy_from_slice` / `fill` / `as_chunks` / 整块查表，整块处理；禁止随意逐元素遍历循环。
- 向量化优先利用编译器自动单态化与整数列展开；需要显式 SIMD 时走标准库/平台 intrinsic，不在框架层自造 IR。
- 保留循环的唯一正当理由：串行依赖（softmax 两轮归约、rope 逐位置旋转）或长度数据相关；循环处注释写明保留理由。
- 权重装载期一次性解码进驻留缓冲；生成循环只覆写输入与逐层激活，不重新分配元素缓冲。
- 读路径贯穿零拷贝视图；压缩段显式解压并标 `Owned`，raw 段 `Borrowed`，`Cow` 二分测试断言。
- 热点不引入 `dyn`、不做 IR 往返；算子表达直接落 einsum 求和式与片段模板。

### 2.6 质量、错误与安全边界

- 错误类型唯一 `diagnostic::{Code, Error, Result}`；不另立错误类型。
- 装配期拒绝形状/长度/dtype/格式不符，不留半绑状态；执行期沿用操作数先放回再报错的原子性。
- 每条生产路径保留独立标量参考对拍；`max_rel_err` 在 dtype 容忍阈值内；空序列/非对齐尾段/越界/非有限值都有负例。
- 核心 `unsafe_code = "deny"`；FFI/mmap 是唯一 unsafe 岛，按 ADR-0035 收敛到边界文件与 feature 后。
- 生产依赖继续趋零；任何破零依赖取舍必须 ADR 显式拍板。

### 2.7 减量规则

- 减仪式、重复闭集、手写解析、装饰性台账与无构造点变体；运行时能力代码不按行数砍。
- 每个模块动工前给行数棘轮预算（L1–L7 分账，沿用 10k-analysis §6.1）；超预算红灯，先说明超支来源，功能排队，不靠压注释/砍常数凑数。
- 对拍、负例、契约测试只允许重构合并，不允许削弱断言。
- 死代码与无构造点变体：出现即删；泛型/抽象抽取前先确认有第二个消费者，单实现不提前参数化。

### 2.8 验收阈值引用（不新造口径）

- 数值对拍：`testkit::close(left, right, abs, rel)`，`abs/rel` 取 dtype 容忍表；`all_finite` 先排非有限值。
- 训练梯度：gradcheck 阈值取既有 autograd 校验口径；checkpoint 往返张量逐位一致。
- 基准：criterion 入 `tools::regress` 基线，CI 只守不回归；跨机绝对值不进结论（roadmap §7）。
- 每条任务出口同时过 design §8.3 四维门：能力（清单逐条回归）、性能（同硬件对照或 roofline 上界）、质量（对拍/负例/门禁）、密度（同口径机制行数）。

### 2.9 文档与代码指针同步

- 代码指针（文件:行）、计数（算子/测试/行数）、能力状态三类事实，改动实现时同步回填对应文档；评审漂移按本文 §1.1 清单格式登记。
- 历史快照文档（`audit.md`、`REPO-review`、既存 ADR 原文）不回写；活文档（design/roadmap/backend/iteration/ARCHITECTURE）必须与工作树一致。

## 3. 任务分解 backlog

状态标记：`待动工` / `在途` / `已落地`。每条出口门统一含：测试不跌、clippy `-D warnings` 零告警、fmt 干净、architecture/contracts/gate 门通过、新增 `unsafe` 为零（除非边界任务）。

### 3.0 P0 加固（不引后端、不扩运行时面）

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| P0-1 | 回填 §1.1 事实漂移：注册表 24、测试 1,391、行数粗口径待优化后重测、KV 数据状态、指针与 ADR 编号（本轮活文档已回填，留一次一致性复核） | 全仓文档 | 无 | design/roadmap/backend/iteration/ARCHITECTURE 计数与指针一致 | 文档链接门通过；无陈旧计数；历史快照不回写 | — |
| P0-2 | 注解双表合一：`REFERENCE_OVERRIDES`+`MODULE_OVERRIDES` 合并为一行三列表 | `annotation` | D7 | 单表 `(名, 参考覆盖, 模块覆盖)`；查找仍闭集 | 注解生成测试全绿；无第二份按名查找循环 | 宏去留裁决留 D7 |
| P0-3 | 大文件按职责轴拆分（design §5.4 表）：先 `weights/bin.rs`、`src/bin/cutile/main.rs`、`train/snapshot.rs` | 对应 crate | 行数预算 | 解析/排版/台账移出机制文件；单文件 ≤899。**已落地**：`crates/device` 三巨石按职责轴拆成目录子模块——`cuda/` 7 文件（shared/grid/gemm/chunk/scan/segments/mod）、`backends/` 7 文件（common/ascend/harness/cuda/metal/parallel/mod）、`metal/` 8 文件（host/device/chunk/stage/program/runtime/witness/mod）；对外 `llm_device::{cuda,backends,metal}::*` 路径不变，父 `mod.rs` 只做模块声明与重导出，子模块私有 helper 不外泄。拆分前 3 文件 3,832 行，拆分后 22 文件最大 360 行（gate 生产单文件 600 上限内），lib/test 编译、clippy `-D warnings` 零告警、fmt 干净、gate 行数/CCN/NLOC 无 device 红项 | gate 行数与 ccn 过门；测试断言不削 | 其余大文件（weights/train/annotation/prof/fuzz/tensor/cutile）随阶段内拆 |
| P0-4 | 报告/排版合一：audit/tools/witness 三处报告器收敛为一个 | `tools` / `diagnostic` | 10k-analysis §6.2 | 单一报告器，三处调用 | 输出条目不丢；测试不跌 | — |
| P0-5 | 手写解析收敛到一份；是否引入唯一解析依赖按 D5 实测阈值裁决 | `diagnostic` | D5 | 解析单源；依赖取舍入 ADR | 解析负例全绿 | toml+serde 引入未拍板前不接 |
| P0-6 | 关键纯函数属性/模糊加固：解析、归约、归一化、词表查找等补表驱动边界用例；退化情形进 `fuzz` sweep | `fuzz` / 对应 crate | 无 | 纯函数边界用例集；fuzz 种子与 sweep 接入 | 负例覆盖空/非对齐/越界/饱和；不新增生产依赖 | libfuzzer harness 仅 feature 门控。现状注记：三族内部纯函数 / 结构化张量的 fuzz sweep 接线经裁决移除（与 fuzz 入口解析器契约边界错配），边界用例归各自仓内单测；fuzz 只保留不可信字节入口解析器目标 |
| P0-7 | 片段参数类型签名交叉校验：注解/声明/模板三者参数类型一致，不一致编译期或测试期拒绝 | `annotation` / `prof` | P0-2 | 交叉校验条目 + 守护测试 | 错配片段被拒；既有 24 条全过 | 跨设备文本同构校验随后端阶段 |
| P0-8 | 已具备能力回归守护：P9 多格式往返 round_trip、P13 设备档案 L0–L3 闭集，纳入常驻回归 | `weights` / `prof` / `tools` | 无 | 守护测试入 workspace 回归 | 格式往返全绿；档案闭集不被绕过 | — |

### 3.1 M1-A 装载与驻留权重

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| M1-A1 | 收口既有 `Reader`/`Loader`：复核 `Archive`/`File` 实现与边界检查单点；非 Unix 缺失类型编译报错，不提供假装映射的空实现 | `weights` | D1 | trait 形状文档化；既有实现负例补齐 | 装载负例（缺条目/非 F32/尾半字/长度不符）全绿 | 远端分块源 |
| M1-A2 | `MmapSource` 实现 `Reader`（Unix feature 门控，复用 `storage::Mmap` 唯一 unsafe 岛）；`Archive::load` 既有 Cow 二分（Store→Borrowed / Rans→Owned，`archive.rs:182-186`）保持，mmap 只让 Store 段经映射借用 | `weights`/`storage` | M1-A1、D1 | 映射借用生命周期断言；压缩段 `Owned` 断言 | raw 零拷贝有测试证据；压缩段必须 `Owned`；新增 unsafe 为零 | 设备权重驻留、热更新 |
| M1-A3 | 驻留缓冲接线：装配期按描述符一次解码进驻留 `Buffer`；循环内不二次分配 | `weights`/`serve` | M1-A2 | 参数表驻留路径；激活编号复用 | 装载基准 RSS/时延可重复；`CpuModel::forward` 形态保留 | — |
| M1-A4 | 装载基准 harness：mmap vs read 首条/全量、冷热缓存、RSS 与时延 | `benchmarks`/`tools::regress` | M1-A3 | 对照条目入回归基线 | 数字可重复；跨机绝对值不进结论 | — |

### 3.2 M1-B CPU 单序列前向闭环

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| M1-B1 | CPU 内核桥补缺口：既有 `KERNELS` const 子集表加固；未覆盖名查表返回 `Unsupported`，不留静默空入口 | `serve` | 既有 `kernels.rs` | 子集断言锁注册表；适配按 unary/binary/gemv 分宏 | 表内每条名字在 REGISTRY；重名拒绝 | 标量槽（rope/attention 通道，见 M1-B6） |
| M1-B2 | 首批纯切片主路径算子内核落地：gemv/softmax/rmsnorm/silu/gelu/add/mul/residual/scale/两条归约 | `serve`/`nn` | M1-B1 | Program 适配 + 进程内双算对拍 | `testkit::close` 达标；all_finite 先排 | rope、attention、量化 i4/i8 lane |
| M1-B3 | 装配器补全族首层通路：embedding 查表已注册；投影/点wise/归一化首层由装配侧提供隐向量输入，不再直接 `Unsupported` | `serve`/`nn` | M1-B1 | 全族 `LayerPlan` 接线；宽度链校验 | 两层合成模型端到端对拍；宽度断裂负例 | 多 token 矩阵主链 |
| M1-B4 | 进程内对拍 harness：同输入 Program 输出 vs 独立 `nn` 标量参考 | `serve`/`testkit`(dev) | M1-B2 | 双算 + `max_rel_err` 判定 | 一层/两层模型对拍达标；负例齐全 | 第二份独立闭式参考 |
| M1-B5 | `Sampling` 接端点步：logits → 采样 → emit 记账；停止符/上限/被挡三态保持 | `serve` | M1-B3 | 注入式采样进 `Completion` | 坏温度/空候选/top-p 越界负例 | top-p 校准表细化 |
| M1-B6 | 按 D9 落地标量/形状通道：`Node`/`Launch` 加定长类型化闭集槽（rope 位置、heads、head_dim），不引入字符串键查表；静态头宽仍取 `prof::model::Model.shape`；`Program::run` 缓冲切片签名不变 | `engine`/`serve` | D9 | 标量槽类型 + 最小 CPU 实现 | 槽位形状进 ADR 0043；未接算子仍按名拒绝；热路径无 map 查找 | 多 token 矩阵几何 |
| M1-B7 | 按 D10 落元数据：编号 ↔ 形状/条目名描述表留 `serve` 装配侧；engine `Bindings` 保持纯缓冲；只读形状访问器仅当后端实现证明需要时再补 | `serve`/`engine` | D10 | 装配侧描述符表 + 宽度链校验 | 形状断裂装配期拒绝；负例齐全；engine 不解释编号 | 设备侧形状元数据 |

### 3.3 M1-C 基准 harness 骨架

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| M1-C1 | 机制代码口径分账：排除 tests/内联测试/fuzz/benchmarks/build.rs/生成片段/vendor，按 L1–L7 出本仓基线；排除规则与分账表入档可复跑 | `tools` | 10k-analysis §6.1、design §5.3 | 同口径机制行数表 + 分账规则文档 | 分账规则可复跑；不与含测试口径混 | 对标分母同口径采集 |
| M1-C2 | 统一基准记录：硬件/驱动/形状/批次/精度/后端标识/roofline 上界；无基线登记跳过不伪造 | `benchmarks`/`testkit` | M1-C1 | 对照条目结构 | 每条算子/层/模型有条目；CI 守不回归 | 跨机绝对值结论 |

### 3.4 T1 训练最小闭环（不阻塞 M1 推理出口）

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| T1-1 | `Optimizer` trait + SGD/Adam（矩估计、偏差修正、AdamW 解耦衰减、梯度裁剪） | `train` | D2 | 两实现；矩状态按参数组自持 | 小模型一步更新权重变化可断言；gradcheck 阈值内 | 分布式优化器 |
| T1-2 | 扩展 `ContractSpec` 新增 `ParamRole` 闭集（Trainable/Bias/Norm/Embedding，D11），由层构造器填值；wd 豁免、可训练判定从角色派生；优化器按角色取组，不猜字段名 | `nn`/`train` | T1-1 | 角色字段 + 层默认角色；参数组视图由角色派生 | 组归属对拍；错组/缺角色负例 | — |
| T1-3 | 接 autograd：tape → 反传计划 → 梯度需求分类（`Requirement::Grad` 终点）→ 裁剪 → 优化器步；梯度缓冲由 train `Buffers` 扩展自持（autograd 只出 tape/plan/Replay，不持有梯度存储）；`train::samples` 的 `Dataset`/`Chunks`/`Batch` 夹具装配进训练循环 | `train`/`autograd` | T1-1 | 前向缓存→反向→更新闭环；梯度缓冲复用；夹具批次通路 | 夹具语料跑通指定步数；loss 趋势可记录；批次形状往返一致 | 真实大语料通路 |
| T1-4 | CLI `train` 命令：配置 → 会话 → 周期 checkpoint → `--load` 恢复；快照格式扩优化器矩状态（当前 `Checkpoint` 仅元数据、`resume` 只对账拓扑/层数/批行，R-23） | `cli`/`train` | T1-3 | 训练命令 + 装载恢复 + 矩状态落盘 | 权重与矩状态存载往返逐位一致；步数/图规模/批形状可恢复；旧快照版本拒绝负例 | offload/tie 回填 |
| T1-5 | 梯度累积落地；累积窗口梯度汇总入快照（与 T1-4 同格式）；offload/tie 权重回填留接口 | `train` | T1-3 | 累积窗口梯度汇总；acc 状态可往返 | 累积汇总对拍；tie 回填有对拍 | ZeRO/分布式并行 |

### 3.5 M2 推理运行时（KV 与连续批）

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| M2-1 | 多层层偏移行切片、多头展平头内切片；注意力 Program 接 KV 行映射 | `serve`/`engine`/`nn` | M1-B2、M1-B6 | 层偏移表归装配侧；KV 行进绑定缓冲 | 多步 decode 与参考序列对拍 | 头内并行特化 |
| M2-2 | 模型内核级批量几何：调度批步进 `Batch::step` 已落地（守回归）；本任务把 executor 图从单序列几何扩到批内多序列一步，`Predict` 视需要补批量视图（G9/G11） | `serve`/`engine` | M1-B5 | 模型侧批前向通路；批量视图接口 | 多序列跑通；调度不违反页表不变量；批量对拍达标 | 序列并行 |
| M2-3 | prefix cache 接线：准入时前缀匹配 → `Pager::share` 采纳共享页；淘汰/回收判定走既有 `release` 计数归 0 路径；引用计数原语已在 `Cache`/`Pager` 在位，不重做 | `engine::kv`/`serve` | M2-2、D12 | 前缀匹配表；共享句柄；回收不变量 | 共享/回收对拍；前缀一致/不一致负例；重复归还不越界 | 跨请求前缀复用预热 |
| M2-4 | decode 基准 harness：单序列/批次 TPOT、吞吐、内存；对照 llama.cpp 同硬件曲线 | `benchmarks`/`serve` | M2-2 | 对照曲线入表 | KV 分配/释放有对拍；内存占用入表 | — |

### 3.6 M3 后端边界、量化与性能

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| M3-1 | 边界 crate 骨架 + ADR-0035 复核：`Executor`/`Allocator` 实现；FFI/`unsafe` 局部 allow 收敛边界内 | 新边界成员 | D6、ADR-0035 | 边界依赖方向守门 | 边界 crate 之外零新增 unsafe；架构门通过 | 第三加速链路 |
| M3-2 | Metal 本机跑通 M2 同一前向；CUDA 在 win 远端（T600 sm_75 / CUDA 13.3）验；设备内核管线未建前只编译+契约对拍 | 边界成员 | M3-1 | 两条后端同一 trait 挂载 | 同输入同形状对拍一致；未建设备内核前不出吞吐数字 | 图 capture 特化 |
| M3-3 | 量化 `i4`/`i8` lane：Program 适配 + 反量化内核挂载（字节通道形态见 backend §5.3） | `nn`/`serve`/边界 | M3-1 | i4/i8 lane Program 通路 | 量化权重对拍；零点/分组负例 | FP8/AWQ/GPTQ 新格式 |
| M3-4 | `Collective` 通信原型（NCCL/Metal 对等经 trait 挂） | `shard`/边界 | M3-1 | 通信后端实现 | 集合对拍；拓扑簿记不改 | 大规模分布式 |
| M3-5 | roofline 达成率报告；按 decode 瓶颈序逐算子优化 | `benchmarks`/`nn` | M3-2 | 达成率可追 | 同后端对比 burn/candle 有对照数字 | wmma/wgmma 手写张量核 |

### 3.7 M4 模型与服务化

| ID | 任务 | 归属 crate | 依赖 | 交付物 | 验收门 | 暂缓 |
|---|---|---|---|---|---|---|
| M4-1 | 主流权重格式端到端装载与前向（接 M1-A 源抽象）；容器字节解析为中性 `VocabSource` → `Vocab::try_new`（D13），挂到 `Tokenizer`，不再仅靠外部注入 | `weights`/`serve`/`tokenizer` | M1-A、M2 | 真实权重 + vocab 生成通路 | 给定真实权重跑通生成；词表指纹沿用既有 `Fingerprint` 对拍 | vision/多模态 |
| M4-2 | `Endpoint` 的真实 HTTP 实现；传输层与端点模型分离 | `serve` | D8 | HTTP 后端替换桩路由 | 契约测试全绿；并发/流式通路 | gRPC |
| M4-3 | 服务基准：并发与 P50/P99、TTFT/TPOT/prefix 命中 | `benchmarks`/`serve` | M4-2 | 服务对照入表 | 同负载对照 vLLM/TRT-LLM 口径登记 | 通用 GPU 峰值竞争 |

### 3.8 显式暂缓清单（本路线不做；接口形状预留，不删既有模块）

| 项 | 暂缓理由 | 若未来动工的前置 |
|---|---|---|
| 投机解码 / beam search | 路线定位 decode 主路径与连续批；正确性对拍成本高 | M2 decode 基准稳定后另立 ADR |
| MoE / 专家并行 | 拓扑与通信后端未落地 | M3 通信原型之后 |
| vision / 多模态 / RL | roadmap §2.2 明确不含产品面广度 | 独立路线评审 |
| LoRA / 适配器 | 参数组元数据先服务全量训练 | T1 参数组稳定后评估 |
| FP8 / AWQ / GPTQ 新量化格式 | 精度事实源与 i4/i8 lane 先行 | M3-3 之后按格式 ADR 接入 |
| KV 缓存量化 | 依赖 KV 缓冲与后端通道形态 | M2 KV 数据通路稳定后 |
| chunked prefill | 调度几何先通单批与 decode | M2 连续批之后 |
| 第三及以上后端（MUSA/OpenCL/SYCL/ANE/第三加速链路） | 双后端 trait 未稳定前预埋会分叉 | M3-2 双后端对照之后 |
| gRPC 服务 | 传输先 HTTP 一条通路 | M4-2 端点契约稳定后 |

> 上表条目在代码里已有的模块/契约保留为占位，按 L0/L1 证据等级标注，不得被误读为已落地能力（iteration §3 骨架审查）。

## 4. 决策看板（本轮提前定形；ADR 正文动工前按编号落档，0033 空缺）

状态口径：**已定** = 方案、备选与理由在本文锁定，实现不得另起平行抽象；**已定-触发** = 方向锁定，仅在触发条件出现时复核。每条决策给出：选定方案、被否备选、取证依据。

| # | 议题 | 已定方案 | 被否备选 | 取证依据 | ADR 落点 |
|---|---|---|---|---|---|
| D1 | 权重源抽象形状 | 沿用既有 `Reader`（借用）/`Loader`（拷贝），**不新增平行 `Source` 父 trait**；`MmapSource` 实现 `Reader`（Unix feature 门控），`File` 保持 `Loader` | 另立 `Source` 父 trait 统一两入口；全文件强制 mmap；只留文件流 | `weights::Reader/Loader` 已定形（`lib.rs:268/277`），`Archive` 双实现、`File` Loader 已在跑（`safetensors.rs:103/127/191`）；父 trait 无第二消费者 | 0036 |
| D2 | 优化器 trait 归属与形状 | 归 `train`；`Optimizer` 对象安全 trait，`step(params, grads, meta: &ParamRole, lr)`；矩状态按参数编号由优化器自持；参数角色元数据来自 `nn` 声明，优化器不按字段名猜测 | 优化器下沉 `autograd`；按参数类型泛型特化；优化器直接改私有字段 | `autograd` 只出 tape/plan/Replay/Store，无更新策略（`autograd/src/lib.rs` 导出清单）；`train` 现无任何优化/梯度更新代码；`Role::is_parameter` 已可派生 | 0037 |
| D3 | 训练闭环排期 | 单列 T1，并行推进，**不阻塞 M1 推理出口**；资源紧张时先 M1-B | 训练与推理同阶段串行；训练优先 | 设计 §4.4；T1 依赖的 `train::samples`/autograd/nn 接口均已存在 | 记录在设计 §4.4，不单列 ADR |
| D4 | crate 合并节奏 | 维持 24 个成员到 M2；M2 后只合并"单一消费者 + 同层"的叶子成员，合并以边数/行数/编译时间实证为据 | 立即 31→8 模块；按目录猜合并 | 当前 24 成员；编译期并行与架构门边界是既有长板。**2026-11-19 文件级合并实证（ADR 0049）**：crate 边界不动，src 255→195、测试文件 49→38，测试数 1,779 不跌、门禁全绿——文件级内联+re-export 即可达成 ~25% 文件削减，无需动 crate 边界 | 0038（已落档）+ 0049 |
| D5 | 解析器依赖策略 | 维持生产零外部依赖；手写解析按格式收敛到模块内单源；`diagnostic` 的 optional `toml` 保持 build feature 门控。**2026-11-19 修订（ADR 0048）**：L1–L6 生产路径零外部依赖不变；L7 组合根（`cli`）允许唯一例外 `toml`（配置文件层解析），版本由 workspace 锁定 | 引入 toml+serde 单依赖换数千行手写解析 | 生产路径仅 optional `toml`/`bytemuck`（R-21）；解析分散在 weights 的 json/gguf/safetensors/rans | 0039（已落档）+ 0048（L7 例外修订） |
| D6 | 后端 trait 形状 | 沿用 ADR-0025 的 `Executor`/`Allocator`/`Collective`/`Program`；CPU 首实现验证形状后再挂 Metal/CUDA；第二后端出现前不预埋多余泛型参数 | 为双后端现在预埋设备泛型；后端私有假设上浮到模型/调度代码 | 四 trait 已定形（`exec.rs:261/383`、`storage.rs:171`、`shard/collective.rs:11`）；`LocalExecutor` 契约后端 + `CpuExecutor` 注册表在位 | 修订 0025 或 0040（M3-1 前） |
| D7 | 注解宏去留 | 保留两个既有注解宏：`#[annotation::model]`（算子侧产出 `Model` 条目）、`#[annotation::spec]`（`nn` 内重导出为 `#[spec]`，产出 `ContractSpec`）；宏只做声明展开，不持运行期注册表 | 普通构造函数替代宏 | 27 个算子模块挂 `model`（R-20）；`spec` 覆盖 nn 层；去掉宏会把声明负担抬回每个模块 | 0041（已落档） |
| D8 | 服务传输层 | M4 再接 HTTP；当前端点契约（`Endpoint::route/call`）保持零依赖；传输实现与端点模型分离 | M2 提前接 axum/tokio | `Endpoint` trait 已定形（`serve/src/lib.rs:182`），无 socket | 0042 |
| D9 | Program 标量/形状通道 | 在 `Node`/`Launch` 上扩**定长类型化标量槽**；首批槽位仅 rope 解码位置；heads/head_dim 等静态几何仍取 `prof::model::Model.shape`，不进运行时槽；槽位集合闭集冻结，新增走 ADR 修订；不引入字符串键查表；`Program::run` 缓冲切片签名不变 | 把位置标量塞成 1 元素缓冲；节点上挂 `HashMap` 标量上下文；静态几何也压成运行时槽 | `Launch` 现仅 grid/block（`exec.rs:42-47`）；rope 需位置标量，attention 头数/头宽可由静态形状推；热路径不能加 map 查找 | 0043 |
| D10 | Bindings 元数据落点 | 编号 ↔ 形状/条目名描述表留在 `serve` 装配侧；engine `Bindings` 保持纯缓冲 `BTreeMap<u32, Vec<f32>>`；不先给 engine 加形状访问器，后端实现证明需要时再补只读视图 | 形状元数据下沉 engine Bindings；装配侧按层名前缀猜形状 | `Bindings` 现无形状字段（`exec.rs:396-425`）；ADR-0025 §2 规定 engine 不解释编号含义；宽度链校验属装配职责 | 0044 |
| D11 | 参数角色闭集落点 | `ContractSpec` 扩展 `ParamRole` 闭集（Trainable / Bias / Norm / Embedding 等），由层构造器填值；wd 豁免、可训练判定从角色派生 | 角色写进优化器配置；按权重条目名字符串猜角色 | 现 `ContractSpec` 无角色字段（R-16）；`nn` 层是构造期唯一知道自身角色的地方 | 0037（随 D2 同档落） |
| D12 | KV 前缀共享接线 | 引用计数/共享/回收原语已在 `engine::kv` 落地；本路线只做前缀匹配、准入时 `share` 采纳共享页、淘汰与回收判定；不改页池数据结构 | 在 serve 侧另造一套共享账；重写 Cache 引用计数 | `Cache::share/release/reference_count` 在位（`kv.rs:470-555`），`Pager::share/release` 已接线 | 记录在 M2-3，不单列 ADR |
| D13 | 词表装配通路 | 容器字节在 `weights` 侧解析为中性 `VocabSource`，交 `tokenizer::Vocab::try_new` 闭合；engine/serve 不持有词表格式解析 | 词表解析写进 engine 或服务状态机 | `VocabSource`/`Vocab`/`Fingerprint` 已存在（`vocab.rs:61/88/163`）；缺的只是容器 → Source 解析 | 记录在 M4-1，不单列 ADR |

> 0035（unsafe/FFI 边界）是并发独立裁决，评审通过后作为 M3-1 的前置约束；**未评审前 M3 不动工**。0036–0044 编号已预留，ADR 正文在对应任务动工前补齐，结论以本表为准。

## 5. 里程碑口径对齐

| roadmap/iteration 标签 | 设计/backend 编号 | 范围 | 出口判据 |
|---|---|---|---|
| P0 | P0 加固 + M1-C1 | 文档回填、结构减量、属性/fuzz 加固、片段签名校验、已具备能力守回归、基准口径分账 | 门禁全绿；机制基线可复跑；守护测试常驻 |
| P1 | M1-A + M1-B + M1-C2 | 装载驻留、CPU 单序列前向、首批对照 harness | CPU 前向对拍达标；首批同硬件对照数字入表 |
| —（并行） | T1 | 训练最小闭环（含样本夹具装配） | 反传可更新参数；checkpoint 往返一致；批次形状往返一致 |
| P2 | M2 | KV 真实数据、连续批（调度层已落地）、模型内核批形状、prefix cache、decode 基准 | 多序列端到端；KV 边界测试；对照曲线入表 |
| P3 | M3 | 双后端边界、量化 lane、通信原型、roofline | 同后端对拍一致；边界外零 unsafe |
| P4 | M4 | 真实权重端到端、HTTP 服务、服务基准 | 真实权重生成；端点被真实服务器替换 |

## 6. 护栏与不可回填内容

- 性能/吞吐/时延/倍数：可运行后端落地、同硬件同模型对照前不写成结论；静态 flops/带宽只作 roofline 上界。
- 设备内核管线未建前，CUDA/Metal 条目只算编译与契约对拍。
- "少 15 倍 / 强 20 倍"只作带判定门的目标：密度按 §3.3 M1-C1 同口径测量；能力对照设计 §1.5 不退化清单逐条回归；四维门见 design §8.3。
- 每条任务出口都必须守住：测试数不跌、clippy 零告警、fmt 干净、架构/契约/静态门通过；减量不得删除正确性证据。
