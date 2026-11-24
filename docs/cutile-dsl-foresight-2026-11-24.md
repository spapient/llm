# cutile DSL 长周期前瞻分析：40–50 年视角下的护城河、负债与风险（2026-11-24）

> 取证口径：**confirmed** = 本轮回读到源码/文档的事实；**inferred** = 由结构与调研推出的判断；**unknown** = 仓内未检索到证据或受环境阻塞。本报告不改仓库代码。前瞻部分基于 2026-11-24 的联网调研快照，趋势判断属于推测，已在 §8 标注边界。配套现状分析见 `docs/cutile-dsl-architecture-2026-11-24.md`。

## 0. 结论先行

**"超越竞品 40–50 年"不是一个性能或语法命题，而是一个资产命题。** 软件史上能跨过 30–50 年存活的，从来不是某种 kernel 语法或调度模型，而是三类与硬件代际解耦的东西：**可机器检查的规范、可复验的证据、可迁移的中间表示**。cutile 现在的设计公理（闭集、参数化能力契约、双源对拍、证据分级、可解释选择链）方向上正落在这三类资产上，这是它真正的长周期赌注；但证据层级目前停在**测试/执行级**，IR 语义没有形式化模型，诊断产物面向人读而非机器消费，这三件事是 40 年尺度上最致命的负债。

十个前瞻问题按风险排序（详见 §4）：

1. 证据层级：测试对拍 vs 形式化证明（**最高风险**）；
2. 规范层缺失：当前 DSL 在让人写实现，不是让人写规范；
3. 语义未编码进 IR：验证工具只能手写，无独立语义参照；
4. IR 绑宿主语法，缺中立可存档的长期形态；
5. 闭集与硬件代际断裂之间的张力；
6. 面向 agent 的自描述与机器可读诊断接口；
7. 并发安全证明尚未落地，且证明强度停留在代码层约束；
8. 性能模型是单算子静态上界，缺图级代价、能耗与遥测回流；
9. 生态与采纳风险：单仓自有、无开放词汇表标准；
10. 闭集覆盖率无台账，可能从可判定优势退化为表达力枷锁。

**一句话判断**：cutile 若要兑现长周期领先，必须把交付物从"生成的内核源码"升级为"**带规范引用、证据等级、变换保位记录与成本上界的可审计包**"；语法和发射层注定会被换代，证据包才是要留 40 年的东西。

## 1. 四十年时间尺度的判定基准

### 1.1 语言语法层在长周期里几乎必然换代

- 计算语言史：Fortran(1957) → C(1972) → Matlab/Mathematica 领域语言 → CUDA(2007) → Triton(2019)。单一 DSL 的有效半衰期通常以 10–15 年计；消亡的 DSL 多死于语法绑死、生态封闭与维护者单点，而不是算法表达力不足<RichMediaReference>["https://ceur-ws.org/Vol-2707/oopslepaper5.pdf"]</RichMediaReference>。
- 长寿语言靠的不是语法，而是存量代码、生态、向后兼容与运行时载体；新语言在衰退期多缩窄为特化内部层或被并入更大栈<RichMediaReference>["https://koder.ai/fr/blog/pourquoi-langages-programmation-survivent-niches"]</RichMediaReference>。

### 1.2 长周期存活的三类资产

| 资产类型 | 为什么跨代存活 | 今天在 cutile 中的对应物 |
|---|---|---|
| 可机器检查的规范 | 规范不依赖指令集；任何新后端都要重新满足同一份规范 | nn 标量参考实现（公理 5，`README.md:119`）；能力签名 `(原语,形状区间,dtype,空间)`（公理 2，`README.md:116`） |
| 可复验的证据 | 证据记录的是"某次实现曾被验证"，随代码一起存档可重放 | L0–L3 证据台账（`README.md:860-863`）；逐分量缓存指纹 diff（`cache.rs:343-363`） |
| 可迁移的中间表示 | IR 形态中立于宿主语言与后端，可在新栈上重新生成 | typed IR 的 `Kernel/Expr/Write` 数据结构；但目前仍绑 Rust 函数语法入口（见问题 4） |

### 1.3 40 年后确定会变的东西

执行模型（SIMT→tile→张量核→下一代异步执行）、指令集、内存层级命名、调度模板数量、前端宿主语言；**这些在 cutile 里都应是可替换的下游，而不是内核。**

## 2. 长周期趋势调研（2026-11 联网快照）

### 2.1 规范语言正在取代编程语言：人写规范，机器实现，verifier 兜底

Djinnlang（arXiv 2609.23954）提出：程序员只写形式规范，LLM 填充实现与证明，verifier 校验"实现满足规范"且"满足规范的任意其它实现同输入同输出"（确定性）；不可信模型写代码，可信检查器收紧语义，生成代码永不被人阅读、可随时从规范重生成。在这个图景里规范语言与编程语言的界限消失，LLM 成为编译器工具链的一部分<RichMediaReference>["https://arxiv.org/html/2609.23954"]</RichMediaReference>。

**对 cutile 的含义**：DSL 的长期价值不在"再提供一种写 kernel 的语法"，而在规范条目能否独立存在、被多个实现变体复用与对拍。

### 2.2 GPU kernel 形式化验证已是活跃前沿，信任边界上移

- **LARK**（arXiv 2611.02028）：Lean 4 一次建模 GPU 机器模型与算子语义，agent 把大核分段、分段证明与 PyTorch 参考等价；人工只需预先核验算子级 Lean 语义，不必逐行检查生成代码。12 个 L1 核验 10 个、5 个 L2 全验；10 个生产核验 9 个，并查出一个新 data race<RichMediaReference>["https://arxiv.org/html/2611.02028v1"]</RichMediaReference>。
- **Lean/VerCors FP 验证**（arXiv 2610.24548）：对 LLM 生成 CUDA 核做语义等价与线程安全/无竞争证明，浮点核验经 Lean kernel 检查，不靠编译产物<RichMediaReference>["https://arxiv.org/pdf/2610.24548"]</RichMediaReference>。
- **ProofWright**（arXiv 2511.12294）：SMT + 演绎证明 + agent 反馈循环；KernelBench L1 内存安全/无竞争覆盖 74%，语义等价仅在逐点子类成立（14%）<RichMediaReference>["https://arxiv.org/pdf/2511.12294v2"]</RichMediaReference>。
- **Kuiper**（F★/Pulse）：产物是 proof-carrying GPU program，证明义务随代码交付<RichMediaReference>["https://mtzguido.github.io/pubs/kuiperbench.pdf"]</RichMediaReference>。
- **Shard**（IEEE QRS 2025）：轻量注解（前置/后置条件、循环不变量、状态机）+ Dafny/Verus  discharging，偏连续保障而非全量形式验证<RichMediaReference>["https://xplorestaging.ieee.org/document/11173504/similar"]</RichMediaReference>。

**现状对照（confirmed）**：仓内全仓检索 `lean/coq/smt/z3/verus/invariant/theorem/refinement` 无任何证明设施；`invariant` 仅出现在测试属性名（如 `softmax_is_shift_invariant`），`proven` 指 `ProofState` 的执行见证。cutile 的 L2 证据 = 与参考对拍通过，L3 = 真机达 roofline ≥70%，两者都是**采样测试 + 实测档位**，不是形式化保证。

### 2.3 编译器正确性：定理证明与翻译验证两条路线，语义成为 IR 一等公民

- **CompCert**：全编译器在 Coq 中定理证明，16 pass / 10 中间语言；先进优化可由不可信代码执行、再经"已验证验证器"事后符号解释校验<RichMediaReference>["https://compcert.org/research.html"]</RichMediaReference>。
- **翻译验证（translation validation）**：逐执行/逐变换事后校验源与目标的 refinement 关系；Alive2 是 LLVM IR 专用工具，但管不了无界循环与外部调用<RichMediaReference>["https://arxiv.org/pdf/2401.16797.pdf"]</RichMediaReference>。
- **MLIR 一流验证 dialect**（PACMPL 2025，DOI 10.1145/3729309）：把操作语义编码进 dialect 本身，编译器开发者定义语义 lowering，SMT 专家提供编码，工具建造者写方言无关的验证器；已用它在上游 MLIR 找出 5 个误编译 bug、验证规范 pass 与数据流迁移函数<RichMediaReference>["https://dl.acm.org/doi/10.1145/3729309"]</RichMediaReference>。
- **Trivet**（arXiv 2609.19583）：LLM 生成证明 scaffold + Lean kernel 复核；148 个 LLVM 变换验/驳 147 个，含符号位宽与受限循环类<RichMediaReference>["https://arxiv.org/html/2609.19583v1"]</RichMediaReference>。

**趋势判断（inferred）**：编译器正确性正在从"测试编译器本身"转向"语义显式编码进 IR + 逐变换可校验"。cutile 现有 pass 框架（`Pass` trait、fixpoint、每 pass 复验、定律表）结构上契合翻译验证的落点，但缺一个独立于 pass 实现的语义参照。

### 2.4 AI lowering 与 LLM autotune：传统 lowering 管线可被绕过，但假加速风险同步放大

- **TAIC**（arXiv 2609.36800）：LLM agent 直接 Triton→PTX，0.83–3.34× autotuned Triton；增益来自 Triton lowering 管线不做的目标级变换（ packed 权重直入张量核、warp shuffle 重组、卷积窗口复用）<RichMediaReference>["https://arxiv.org/html/2609.36800"]</RichMediaReference>。
- **Helion LLM 引导 autotune**：LLM 看核/负载/当前最优配置提议新配置，比 LFBO 少约 10× 配置、6.7× 墙钟，几何均值 1.009×；效果模型不敏感<RichMediaReference>["https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/"]</RichMediaReference>。
- **假加速问题**：错误核通过测试仍可跑出最高 50× 的伪提速；正确性与性能必须分别核验<RichMediaReference>["https://arxiv.org/html/2603.12440v2"]</RichMediaReference>。
- **源级兼容 ≠ 部署就绪**（KernelGenBench-MC）：同一 Triton 接口在六平台上掩盖系统行为差异；模型/scaffold/编译器/运行时/后端是耦合生成系统，可移植性必须端到端测；同一模型在 NVIDIA 0.86× 到平台 A 0.18×<RichMediaReference>["https://arxiv.org/pdf/2607.27231v3"]</RichMediaReference>。

### 2.5 硬件代际断裂与碎片化：跨厂是算法重构，跨代是执行模型重写

- **Meta KernelEvolve**（arXiv 2512.23236）：内存层级异构、编程模型碎片化；即便同厂，Ampere→Hopper 引入 TMA、128 线程 warp-group WGMMA、mbarrier 异步流水线，执行模型本质断裂；抽象差异不是语法翻译能抹平的<RichMediaReference>["https://arxiv.org/html/2512.23236"]</RichMediaReference>。
- **CKTI**（ICS'26）：CUDA→Triton-IR 降层，线程模型转 tile 模型，显式控制流转动态掩码；跨 NVIDIA/AMD/MetaX/寒武纪部署<RichMediaReference>["https://dl.acm.org/doi/full/10.1145/3797905.3800551"]</RichMediaReference>。
- **Triton-MTIA**：基线 Triton 保通用可移植，架构专属能力以可选扩展暴露（TLX 策略）；另设低层 DSL 直出架构特性<RichMediaReference>["https://arxiv.org/html/2608.00325v2"]</RichMediaReference>。
- **分层堆叠已成常态**：高层语义 DSL（Helion，Python 嵌入、自动生成 Triton + autotune）→ 中层 tile IR（Triton/TileLang）→ 底层库（CUTLASS/CuTe/PTX）<RichMediaReference>["https://pytorch.org/blog/helion/"]</RichMediaReference>。
- **系统级编排视角**（FORGE-UGC，arXiv 2604.16498）：编译器演化为系统级编排器，理解各加速器代价特征、跨设备划分计算图、管理跨设备数据搬运；机无关 pass 与机相关 lowering 分层复用，新目标只扩后端模块<RichMediaReference>["https://arxiv.org/html/2604.16498"]</RichMediaReference>。
- **开放标准路线**：Google Coral NPU 以 RISC-V 开放架构 + IREE/MLIR 编译器栈，一次优化跨多 SoC/功耗档部署<RichMediaReference>["https://developers.googleblog.com/en/unifying-edge-ai-how-coral-npus-open-standard-brings-portable-energy-viable-intelligence-from-model-to-silicon/"]</RichMediaReference>。

### 2.6 LLM 时代 DSL 的存活条件

- 内部 DSL（如查询构造器）正被"自然语言直生成目标代码"挤压；但 DSL 的 toolchain（类型检查、诊断反馈）能 grounding agent、抓幻觉，这层价值反而上升<RichMediaReference>["http://www.skuunk.com/2026/04/ai-has-killed-dsl-long-live-dsl.html?m=1"]</RichMediaReference>。
- 新 DSL 存活要点：机器可读文档/自描述 introspection 接口供 agent 查询；运行时与语言服务收敛为单一二进制；借宿主语言熟悉度降低学习曲线；但过早追求 LLM 兼容会挤占核心语义与生态成熟度，嵌入式语言也会妨碍独立语义演进<RichMediaReference>["https://fooqux.com/article/5734"]</RichMediaReference>。

## 3. cutile 现状在长周期坐标上的位置（confirmed 矩阵）

| 长周期关注点 | cutile 现状 | 证据位置 | 状态 |
|---|---|---|---|
| 能力契约参数化 | `(原语,形状区间,dtype,空间)` 签名，区间包含协商 | 公理 2 `README.md:116`；`cap.rs` | confirmed，已是长周期资产雏形 |
| 布局作为契约 | 4 构造子闭集，构建期校验 swizzle 粒度 | 公理 3 `README.md:117`；`space.rs` | confirmed |
| 调度作为枚举 | 4 调度模板，用户选模板填槽不写流水线 | 公理 4 `README.md:118` | confirmed；代际弹性见问题 5 |
| 双源正确性 | nn 标量参考 + 片段对拍，非有限值定性 | 公理 5 `README.md:119`；`reference.rs:10-37` | confirmed；证据层级为测试级 |
| 证据分级 | L0 声明 / L1 编译 / L2 对拍 / L3 实测档位 | `README.md:860-863`；`cap.rs:39-119` | confirmed；无形式证明档 |
| 可解释选择链 | NeedReport/ChainStep/VariantCheck/Unlock/NoPath | `cap.rs:123-195` | confirmed，仅结构可达性问答 |
| 缓存逐分量 diff | 12 分量键快照比对，单侧缺失也列出 | `cache.rs:282-331`、`343-363`、`376-383` | confirmed |
| 编译器变换保位 | 定律表 9 条逐位恒等 + 常量折叠 + 每 pass 复验 | `pass.rs:163-173`、`73-117` | confirmed；靠定律表与差分测试，无语义参照 |
| 形式化语义/证明 | 全仓无 Lean/Coq/SMT/Verus 设施 | 本轮全仓 grep 无命中 | confirmed 缺失 |
| 中立 IR 存档形态 | `Kernel/Expr/Write` 是 Rust 值；入口为 `#[dsl]` Rust 函数 | `kernel.rs`；`annotation/src/dsl.rs` | inferred：可序列化性与中立性未保证 |
| 并发安全模型 | 设计保留 `partition()` 不相交分区；代码未落地 | `README.md:389-412` | blocked |
| 图级/能耗代价 | roofline 单算子静态上界，不算流水掩盖/寄存器/icache | `plan.rs:58-82`、`325-330` | confirmed；无图级与能耗维度 |
| 覆盖率台账 | 无"声明算子面 vs IR 可表达族"度量 | 本轮检索未见 | unknown/缺项 |

## 4. 十个前瞻问题（现状 → 趋势 → 风险 → 最小动作）

### 4.1 证据层级：测试对拍 vs 形式化证明（最高风险）

- **现状**：L2 是对拍通过（采样有限样本），L3 是实测性能达上界比例；pass 正确性靠 9 条定律表 + 差分执行证据。仓内无证明基础设施（§2.2 已检索）。
- **趋势**：前沿把 kernel 正确性推到 Lean/SMT 等价证明与 proof-carrying 产物；信任边界上移到算子级语义一次建模、分段证明。
- **风险**：40 年尺度上，"对拍通过"会像今天的"跑过单元测试"一样弱；当生成代码不可信、检查器可信成为常态，只有测试证据的产物无法进入可审计交付链。FP 边界（-0.0/次正规/inf/NaN）cutile 已在测试层钉住，但测试覆盖不到的输入域仍然开放。
- **最小动作**：先不做全量定理证明（当前算子面也不支持）。务实中间层是**翻译验证**：给 IR 操作定义一套显式语义，每个 pass 产出前后 IR 做 refinement 校验（类比 Alive2/MLIR 语义 dialect），把现有"每 pass 复验"从结构类型检查升级为语义保位校验。证明义务分级：逐点恒等先 SMT 可解，归约/循环留不变量注解类轻量方法（Shard 路线）。

### 4.2 规范层缺失：DSL 在让人写实现，不是写规范

- **现状**：`#[dsl]` 入口要求人写一个受限 Rust 函数体（循环 + 一次写回）；prof 侧 toml 是配置（需求/布局/调度），nn 参考实现是对拍源但不是一等规范条目。
- **趋势**：规范独立于实现存在，可被多实现变体复用；LLM 填充实现、verifier 兜底（§2.1）。
- **风险**：规范与实现绑在同一 IR 里，换后端/换生成方式时要重写规范侧；证据附着在片段上而非规范上，无法回答"这份规范历史上被哪些实现满足过"。
- **最小动作**：把 nn 参考实现提升为一等规范条目（语义 + 输入域 + 精度约定），IR 核与设备片段都登记为该规范的实现变体；证据等级与对拍记录附着到规范-实现边上，而不是只挂在片段上。

### 4.3 语义未编码进 IR：验证工具只能手写

- **现状**：IR 操作语义由 Rust 侧实现隐式携带；pass、校验、发射各自重复理解语义。
- **趋势**：MLIR 验证 dialect 把语义作为 IR 一等公民（§2.3）；语义 lowering 与验证工具分离职责。
- **风险**：每加一种验证/分析工具都要手写语义理解；pass 之间语义理解漂移时，只能靠测试兜底。
- **最小动作**：为 `Expr`/`Write`/`Kernel` 各操作写一份机器可读的语义定义（输入域、输出关系、FP 舍入模式、副作用/内存访问模式），作为 pass 与发射的唯一参照；发射偏差与 pass 偏差都对照这份定义校验。

### 4.4 IR 绑宿主语法，缺中立可存档的长期形态

- **现状**：typed IR 是 Rust 值结构，前端是 Rust 过程宏；没有中立序列化形态，也没有"从存档 IR 重生成"的路径。
- **趋势**：跨代资产是中立 IR；宿主语言与前端会换代（§1.1）。
- **风险**：20 年后若 Rust 生态或宏形态变迁，存量核无法从可存档形态重生成；LLM agent 直接消费 IR 也缺稳定的中立数据面。
- **最小动作**：把 `Kernel/Expr/Write` 定义为可序列化数据结构（已有 Rust 值基础，补中立编码与版本化 schema）；`#[dsl]` 降级为众多前端之一，归档与重生成走中立 IR。

### 4.5 闭集与硬件代际断裂之间的张力

- **现状**：公理 1 闭集优先、枚举外显式失败；调度 4 枚举；布局 4 构造子；原语 16 槽位。
- **趋势**：同厂跨代执行模型本质断裂（TMA/WGMMA/mbarrier 类新机制每代出现），跨厂是算法重构（§2.5）。
- **风险**：闭集优先把成本挡在 pass 维护之外，但代际快速更替时，枚举外失败会从"可判定性优势"变成"硬件缺口"；4 调度模板不足以覆盖下一代异步流水/ warp 特化模式。
- **最小动作**：所有闭集（原语/布局/调度/空间）做版本化登记，新代际以**新增枚举与新签名**扩展，不改旧条目语义；schema_ver 机制已在缓存指纹中存在（`cache.rs:284`），把同一版本化口径推广到能力词汇表。闭集扩张节奏与覆盖率台账（问题 10）联动。

### 4.6 面向 agent 的自描述与机器可读诊断接口

- **现状**：诊断产物是人读文本——`Error` = Code 五类 + message + ctx 链 + notes（`diagnostic/src/error.rs:9-14`）；`NoPath`/`NeedReport` 是 Rust 值但未见中立序列化输出；能力词汇表（Prim/Space/Layout/Schedule）在代码闭集内，无机器可读文档面。
- **趋势**：LLM agent 是主要消费者；DSL 需自描述 introspection API、结构化诊断反馈来 grounding agent、抓幻觉（§2.6）。
- **风险**：agent 无法程序化消费"缺哪个原语/最近可行变体/哪条指纹分量变了"，只能解析自然语言文本；新语言在 agent 生态里缺乏可发现性。
- **最小动作**：诊断结构（NeedReport/ChainStep/Unlock/NoPath/Changed）输出稳定结构化 schema；词汇表导出机器可读清单（含签名、定义域、证据等级要求）；保留人读渲染作为派生视图。

### 4.7 并发安全证明尚未落地，证明强度停在代码层约束

- **现状**：`#![deny(unsafe_code)]` 是仓库代码层约束；`partition()` 宿主分区模型设计保留、代码未落地（`README.md:389-412`）；产物 kernel 的无竞争保证目前不存在。
- **趋势**：CuTile Rust 用借用检查器证明无数据竞争；前沿形式化工作把线程安全/竞争自由列为 kernel 一等证明项（§2.2）。
- **风险**：交付包若不含并发安全证据，跨设备复用分区模型时只能运行期检查或留 unsafe 假设。
- **最小动作**：落地分区不相交校验（构造期拒绝重叠），先做静态可验部分；不可验部分显式标注运行期检查，与现有证据分级口径一致。证明强度标注清楚：构造期校验 ≠ 借用检查级证明。

### 4.8 性能模型是单算子静态上界，缺图级代价、能耗与遥测回流

- **现状**：roofline 单算子上界，显式不算流水掩盖/寄存器压力/icache 膨胀（`plan.rs:325-330`）；融合增益由次可加性代数保证；无跨算子图代价、无能耗维度、无运行时遥测回流调优。
- **趋势**：编译器向系统级编排器演化，跨设备划分、能耗/延迟预算、数据搬运成为编译期输入；运行时遥测（热/内存压力/电量）回流引导 autotune（§2.5，FORGE-UGC）；LLM 引导搜配置降 10× 采样（§2.4）。
- **风险**：单算子上界在图级融合与跨设备部署场景不可用；无能耗维度会在 edge/NPU 场景失去决策资格。
- **最小动作**：代价模型预留图级接口（算子链复合代价的输入域）；能耗作为档案能力的可选签名项；autotune 候选以 roofline 上界排序、实测收敛落盘（现有缓存指纹可复用），未测设备回退 `Unscorable` 不猜实测。

### 4.9 生态与采纳风险：单仓自有、无开放词汇表标准

- **现状**：cutile 是 workspace 内自有 crate；词汇表/签名/证据等级定义在仓内；无外部消费者、无开放规范文档面。
- **趋势**：DSL 消亡主因是 lock-in 担忧、生态封闭单公司、维护者单点、4GL 坏历史；反证是抽象层级足够高时，后端替代只是资源问题（§2.6，CEUR OOPSLE）。
- **风险**：闭集与私有词汇表叠加，形成事实上的语言 lock-in；维护者流失后替代成本被高估。
- **最小动作**：把能力签名、证据等级、诊断 schema 定义为与实现解耦的开放规格（纯数据 + 词汇表）；声明/模板/片段数据与机制分离的现有架构是有利基础，保持这一分离不被后端实现污染。

### 4.10 闭集覆盖率无台账：可判定优势可能退化为表达力枷锁

- **现状**：22 算子声明 / 16 原语 / 4 布局 / 4 调度（README 实测口径）；未见"算子需求面 vs IR 可表达族"的覆盖率度量。
- **趋势**：LLM 算子面持续扩张（扫描、混合精度、稀疏、新注意力变体）；自动调度前沿承认无约束穷举不可行，高层优化仍多靠人（上一份报告 §2.1 引 Nautilus）。
- **风险**：闭集增长失控（每个新模式都开枚举）或闭集过窄（新算子落不进既有族，只能破公理 1）；没有覆盖率台账就无法判断闭集在哪一侧失衡。
- **最小动作**：建覆盖率台账：算子声明逐条标注可表达族（逐点/归约/扫描/融合/...）、缺口档位（IR 可表达 / 模板可拼 / 闭集外拒绝）；新算子先入台账再决定扩闭集还是拒绝。

## 5. 问答（诊断）能力前瞻分析

### 5.1 现状（confirmed）

**已能回答的问题**：

| 问题 | 结构 | 位置 |
|---|---|---|
| 这个变体为什么没选中？ | 逐项 mismatch + 等级不足 + 本次未见证分开列 | `NeedReport` `cap.rs:123-145` |
| 降级链走到哪一档、每步成败？ | 每步变体/调度/可行与否/失败需求/MMA 占比 | `ChainStep` `cap.rs:149-160`；`VariantCheck` `cap.rs:164-169` |
| 补哪一个原子就能解锁？ | 最近变体 + 去重缺失原语 + 占比 | `Unlock` `cap.rs:173-180` |
| 无可行变体时的完整说明 | 算子/设备/明细/全链/解锁建议 | `NoPath` `cap.rs:184-195` |
| 缓存为什么失效？ | 12 分量逐项 diff，旧值/新值/单侧缺失 | `CacheKey::changes` `cache.rs:343-363`；`Changed` `cache.rs:376-383` |
| 失败属于哪类、上下文链？ | Code 五类 + ctx 帧链 + 同级 notes | `error.rs:9-14`；`code.rs:7-18` |

**刻意取舍**：错误不挂 `source` 链，因为 `Error` 必须可克隆可比较（退出码契约与诊断对拍依赖 `Clone/PartialEq/Eq`），`Box<dyn Error>` 不满足；底层 io 类别已归进 `Code`（`error.rs:93-101`）。

### 5.2 缺口（对照长周期预期）

1. **结构可达性问答 ≠ 语义/性能问答**：不能回答"这个核慢在哪条依赖链、哪级缓存放不下、精度为什么掉档、哪条规则改变了舍入"。性能侧只有 roofline 上界，无瓶颈归因；精度侧有阈值口径但无逐分量误差归因。
2. **诊断文本面向人读，非机器可读**：§4.6 已列；agent 消费缺结构化协议。
3. **证据回放链不完整**：报告能说出"这次有见证"，但一份历史结论回溯不到"哪次执行、哪个样本、哪位见证、哪份产物"的原始记录链；台账聚合在 `shard/ledger`，与 cutile 前端诊断的接缝未在本轮展开。
4. **变更溯源只到分量 diff**：指纹 diff 能说清哪个分量变了，但不解释"档案/模板/声明改动后旧结论为何失效"的因果链；配置版本与结论失效的映射未登记。
5. **解锁建议无前瞻推演**：`Unlock` 给出缺哪个原语，但不推演"补齐后预期落在哪档、代价/峰值几何、是否触发降级链上的下一档否决"；plan 层的 roofline 未接到解锁建议上。
6. **无自然语言解释层**：公理 7 要求可解释，但解释载体是结构化链 + 人读文本；面向非专家（模型作者）的归因叙述层不存在。

### 5.3 演进方向（inferred）

- 近：诊断结构中立序列化（问题 6）；解锁建议挂 plan 层推演（落档 + 代价区间）。
- 中：证据回放链贯通（执行记录 → 样本 → 产物 → 见证条目可追溯）；精度/性能归因视图（误差分量、上界瓶颈侧）。
- 远：诊断作为规范与实现之间的契约说明，可被 agent 查询、可被审计员复核；解释层从人读文本升级为带证据引用的结构化问答。

## 6. 护城河 / 负债 / 伪护城河 三分

### 6.1 真护城河（应加固）

- 参数化能力契约：能力是类型化签名而非字符串，协商可区间包含；
- 证据分级台账：自述与见证分离，缺设备失败不跳过；
- 双源对拍口径：非有限值定性、f64 误差、按 dtype 分档阈值；
- 闭集可判定：枚举外显式失败，pass 维护成本可控；
- 声明与机制分离：数据事实源在 prof，机制层不持第二份闭集；
- 可解释选择链 + 逐分量缓存 diff；
- schema 版本化（缓存指纹已带 `schema_ver`）。

### 6.2 长周期负债（必须还债，否则贬值）

- 测试级证据，无形式语义与证明设施（问题 1、3）；
- 规范未独立于实现（问题 2）；
- IR 绑宿主语法，缺中立存档形态（问题 4）；
- 诊断非机器可读，无 agent 自描述面（问题 6）；
- 并发安全模型未落地（问题 7）；
- 代价模型单算子静态，缺图级/能耗/遥测（问题 8）；
- 覆盖率无台账（问题 10）。

### 6.3 伪护城河（不要误当优势）

- **闭集窄 ≠ 可判定性强**：窄闭集只是降低了 pass 维护成本，不代表能力覆盖完备；
- **4 调度枚举 ≠ 调度能力强**：枚举是表达边界，不是调度优化能力；
- **L2 对拍通过 ≠ 正确性成立**：采样测试的输入域开放，非有限值已定性但全体输入域未证明；
- **roofline 上界 ≠ 性能预测**：显式排除流水掩盖、寄存器、icache，缺这三项不可对实测提速下结论。

## 7. 分阶段远中近路线（与现状硬边界对齐）

> 既有硬边界不突破：`#![deny(unsafe_code)]`；不接 NCCL/GPU/网络/FFI；vendor 语法不进 cutile；配置是输入不是职责；闭集外显式失败；门禁阈值不放宽。

**近（2026–2028）**

1. IR 类型/结构缺口收口（dtype 档、比较/选择、归约族、多维形状，详见现状报告 §5 阶段 1–2）；
2. 诊断结构中立序列化 + 词汇表机器可读清单；
3. IR 中立编码与 schema 版本化；
4. 覆盖率台账上线；
5. 解锁建议挂 plan 层落档与代价区间推演。

**中（2029–2033）**

1. 算子规范条目化：nn 参考成为一等规范，证据挂规范-实现边；
2. IR 显式语义定义 + 翻译验证级逐变换保位校验（先逐点恒等 SMT 可解类）；
3. 证据回放链贯通：执行记录/样本/产物/见证可追溯；
4. `partition()` 并发安全落地，构造期校验不相交；
5. 图级代价与能耗签名项；autotune 实测收敛落盘。

**远（2034–2046）**

1. 形式语义一等公民：算子级 Lean/SMT 语义模型，proof-carrying 交付包；
2. 规范多实现、多后端、多代际重生成；
3. 系统级编排接入：跨设备划分/能耗预算/数据搬运，机无关 pass 与机相关 lowering 分层；
4. agent 原生消费面：自描述词汇表 + 结构化问答 + 证据引用叙述层。

## 8. 反证、边界与未知

- **形式化验证当前不可全覆盖**：ProofWright 语义等价只在逐点子类（14%）成立；无界循环与外部调用超出 Alive2 类工具能力。cutile 不必一步到定理证明，翻译验证 + 轻量注解是务实中间层；全量证明在可见未来都不是刚需。
- **40 年后硬件形态本身不确定**：存算一体、光计算、神经形态都可能改变执行模型；任何把执行模型写死的路线都会过时，证据/规范/IR 层比执行层长寿——这是本报告路线判断的前提。
- **未闭合项（unknown）**：① 仓内是否在其它 crate 存在形式化校验设施，本轮仅按语言关键字检索，未穷尽语义核验；② 台账（`shard/ledger`）与前端诊断的证据回放接缝，本轮未展开；③ 现有 22 算子与 IR 可表达族的逐条映射，需覆盖率台账落地后才能给出数字。
- **前瞻判断强度**：§2 趋势为 2026-11 调研快照；§4 风险推导为 inferred，不构成对 cutile 必然成败的预测；DSL 实际存活还取决于组织投入、生态与时机，这些不在代码可判定范围内。

## 9. 参考来源（本轮联网检索，2026-11-24）

**规范语言与 AI 编译**
- Djinnlang：规范语言 + LLM 实现 + 确定性证明 https://arxiv.org/html/2609.23954
- TAIC：AI lowering，Triton 直出 PTX https://arxiv.org/html/2609.36800
- Helion：高层 DSL → 自动生成 Triton + autotune https://pytorch.org/blog/helion/
- Helion LLM 引导 autotune https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/

**形式化验证**
- LARK：Lean 4 分段等价证明 GPU 核 https://arxiv.org/html/2611.02028v1
- Lean/VerCors：LLM 生成核 FP 与线程安全验证 https://arxiv.org/pdf/2610.24548
- ProofWright：agentic 形式验证 CUDA https://arxiv.org/pdf/2511.12294v2
- Kuiper：F★/Pulse proof-carrying GPU 程序 https://mtzguido.github.io/pubs/kuiperbench.pdf
- Shard：轻量形式方法保障 CUDA 核 https://xplorestaging.ieee.org/document/11173504/similar
- CompCert：已验证 C 编译器 https://compcert.org/research.html
- MLIR 一流验证 dialect https://dl.acm.org/doi/10.1145/3729309
- Trivet：LLM + Lean 自动翻译验证 https://arxiv.org/html/2609.19583v1
- 翻译验证综述（含 Alive2 局限）https://arxiv.org/pdf/2401.16797.pdf

**硬件多样性与系统编排**
- KernelEvolve：Meta 异构加速器 agentic kernel 编程 https://arxiv.org/html/2512.23236
- CKTI：CUDA 降层到 Triton-IR 跨平台 https://dl.acm.org/doi/full/10.1145/3797905.3800551
- Triton-MTIA：定制 NPU 编程模型桥接 https://arxiv.org/html/2608.00325v2
- FORGE-UGC：系统级编排编译器 https://arxiv.org/html/2604.16498
- Coral NPU：开放标准 + IREE/MLIR https://developers.googleblog.com/en/unifying-edge-ai-how-coral-npus-open-standard-brings-portable-energy-viable-intelligence-from-model-to-silicon/

**可移植性与正确性风险**
- KernelGenBench-MC：跨平台生成系统可移植性 https://arxiv.org/pdf/2607.27231v3
- KernelFoundry：LLM kernel 优化综述（假加速风险）https://arxiv.org/html/2603.12440v2

**DSL 采纳与长寿**
- DSL 采纳不足反思（lock-in/生态/坏历史）https://ceur-ws.org/Vol-2707/oopslepaper5.pdf
- AI killed DSL, long live the DSL http://www.skuunk.com/2026/04/ai-has-killed-dsl-long-live-dsl.html?m=1
- LLM 时代新 DSL 存活条件 https://fooqux.com/article/5734
