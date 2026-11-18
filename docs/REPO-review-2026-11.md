# 全仓现状分析：精简空间 / 质量深度 / 未到位功能

- 日期：2026-11-16
- 取证口径：当前工作树源码 + `ARCHITECTURE.md` / `engine-benchmark-roadmap.md` / `audit.md` / `10k-analysis.md`；标注「已结清」的项已在代码里复核，「残余」项在当前树仍命中，「未实现」项有源码或契约注释佐证。
- 测量基线：24 个 `crates/*` + 根包 `llm`（25 成员）；`src/` 约 4.5 万行（含内联测试）；本仓测得约 1151 条测试通过，另有 1 条在途红灯（见 §3.7）。

---

## 0. 结论先行

1. **架构与治理是本仓最强的资产**：唯一依赖规则被可执行门禁守住、一个类型一个主人、生产路径零外部运行时依赖、`unsafe` 收敛到单 feature 单文件、错误类型唯一。这一层面对标 burn/candle 属于上游水平，不需要"重写"。
2. **代码还能精简，但空间集中在少数几处，不是全局臃肿**：残余双份规则、少量超大单文件、一张装饰性闭集、若干静默降级；全仓重复密度很低（audit §7.1：跨文件真重复约 9 对）。结构性减法（crate 合并、解析器收敛、宏去留）是路线决策，需要 ADR 拍板，不应在重构里顺手做掉。
3. **质量短板不在类型系统，而在门禁覆盖面与负例测试**：门禁扫描面有 5 处盲区（audit P1-7），部分解析文件零本地测试，三套格式词汇没有表驱动校验，文档内容漂移没有门禁。
4. **功能未到位的部分全部集中在"运行时实现"**：后端执行、KV 真实数据、连续批调度、量化内核、通信后端、真实服务端点、模型端到端前向——这些在路线图里分别是 P1–P4，当前以契约/簿记/片段形态占位，不是缺陷，是阶段选择。
5. **"注解 + const 泛型"这条路线的长板已经成立，但决定能否"远超 Burn"的瓶颈不在注解机制，而在后端实现密度与算子矩阵**：机制层再精致，没有可运行 kernel 就拿不出吞吐对照数字（roadmap §2.1）。

---

## 1. 与 Burn 的定位对比：注解配置泛型这条路线

### 1.1 现有机制（当前树证据）

- `#[model]` 标在 `type 名字 = <Format 别名>` 上：生成物是 `MODEL` 常量、`Format::CHECK` 的 `const` 自证、一条对齐测试（`crates/annotation/src/codegen.rs`；契约见 `ARCHITECTURE.md` §三 注解宏行）。
- 格式差异收成 `Format` 关联常量与方法；`Quantized<G>` / `Dense<B>` 用 const 参数承载粒度与累加块宽，`cost`/`sample`/`reference` 按格式单态化（`op/src/format.rs` 族骨架）。
- 注册表 `registry!` 一行一对，语义表唯一来源是注解生成物，门禁 `the_registry_and_the_model_annotations_are_mirrors` 双向锁死（`tests/contracts.rs`）。

### 1.2 长板（相对 Burn 泛型后端塔）

- **编译期闭合**：别名不实现 `Format` 直接编译不过；形状不是粒度整数倍在 `const` 断言失败。
- **语义表零第二来源**：不存在手写 `Model { .. }` 字面量的回退路径。
- **无运行期注册表反射**：注解不持有运行期注册表（ADR 0005），新增算子不改机制层。

### 1.3 残余泄漏（仍在当前树）

- **P1-5 残留**：宏里仍有一行 `REFERENCE_OVERRIDES` 与两行 dequant 粒度分支；新增"需要自管 cost/sample/reference"的算子仍要改宏。
- **P1-6 两套骨架词汇表**：`op/src/decl.rs:22` 的 `enum Skeleton` 与 `annotation/src/syntax.rs:419` 的 `SKELETONS: [&str; 12]` 不是投影关系；`Attention`/`Scan` 变体全仓无构造点，是装饰性闭集，违反本仓自己"闭集必须有唯一声明处"的契约精神。
- **P3-1 措辞**：`annotation` 既是工具成员（不参与方向判定）又是 `op`/`nn` 的编译期硬依赖，架构文档措辞需要更准确的一段说明。

### 1.4 真正差距

Burn 的后端泛型塔覆盖多后端 trait 分发；本仓走"声明 + 档案 + 片段生成"路线，后端实现尚未落地。要在同后端打平到领先 burn/candle，关键路径是 P1 的三个 decode 主路径算子真实 kernel + 对照 harness，而不是继续加厚注解层。

---

## 2. 代码精简空间（按收益 / 风险排序）

### 2.1 超大单文件拆分（audit P2-6 / S3，低风险，公开 API 不变）

当前树最大的非测试源文件：

| 文件 | 行数 | 混合职责 |
|---|---:|---|
| `weights/src/bin.rs` | 900 | pickle 35 操作码字节码分派 + 解析 + 测试辅助 |
| `train/src/snapshot.rs` | 768 | 格式常量、编码 Sink、解码、verify/check_weights |
| `op/src/decl.rs` | 758 | 声明解析结果 + 三张闭集 + 注册表投影，混三职责 |
| `annotation/src/syntax.rs` | 755 | 注解语法词法 + 骨架名表 |
| `prof/src/atom.rs` | 726 | `Closed`/`Set` 契约与位集 |
| `cutile/src/device.rs` | 711 | 设备档案装配 |
| `weights/src/gguf.rs` | 678 | GGUF 格式适配 |
| `autograd/src/plan.rs` | 671 | 显存计划 |
| `weights/src/quant.rs` | 630 | 量化分组推断 |
| `prim/build.rs` | 620 | 目录扫描生成内嵌表 |
| `cutile/src/cap.rs` | 602 | 能力协商 |
| `cutile/src/cache.rs` | 594 | 产物缓存与回收 |

拆法沿用已验证的口径：保持 `pub` 路径不变，只搬模块与再导出（S3）。`bin.rs` 的 pickle 分派器按 audit §8.2 第 14 条**不强行压 Ccn**，目标是分段可读，不是形式主义拆函数。

### 2.2 残余双份规则（真重复，优先收口）

| 项 | 现状证据 | 处置建议 |
|---|---|---|
| 步幅规则双份 | **【2026-11-16 已结清】** `Order` 上移 `tensor`（`tensor/src/order.rs`），`Strides::ordered(shape, Order)` 是两种顺序唯一累积实现，`contiguous` 委托它；`storage::Layout::strides` 只委托 `Strides::ordered`，`Order` 由 storage 再导出保留路径。`remap`/`Buffer`/`Allocator` 仍归 storage | — |
| 元素数双份 | `weights/src/seg.rs:300` `numel(&[u64])` 与 `tensor::Shape::numel`（`usize`） | 根因是 `seg` 用 `u64`、`Shape` 用 `usize`；统一到一条饱和乘积实现，类型换算在边界做 |
| CRC-32 | `wire/src/crc.rs`（L1 主人）；`serve/src/proto/crc.rs` 另有一份 | **已裁决保留**：serve 只依赖 `diagnostic`，不能向上依赖 `wire`，20 行算术拖入整条依赖链不划算（文件头注释已写明理由）。weights 侧 CRC 已收口到 wire，无第三份 |

### 2.3 已结清，不要重复动手

- **P1-4 双渲染路径：已结清**。`cutile/src/substitute.rs:37` 的 `render` 现在只是 `ast::build` + `ast::render` 的薄包装，规则各只有一份；`ast_renders_byte_identical_text` 守的是将来不再长出第二份。
- **P2-5 cutile 反导 op 类型**：当前 `cutile/src/lib.rs` 已无 `pub use op`。
- **P0-1..4 不可信输入崩溃路径**：四条已修（audit §15），回退样板已落在 dsa/rans 校验路径上。
- **P1-3 `pack::*` 过时引用**：当前树 0 命中。
- **步幅规则双份：已结清**（见 §2.2），`Order` 归 tensor，累积规则唯一。
- **P2-2 未知 mode 码静默降级：已结清**。`mode_of` 改返回 `Result<Mode>`，未知码走 `Invalid`；无标记段的旧文件仍默认 `Infer`（兼容口径）。
- **P2-4 构建脚本静默丢目录项：已结清**。`prim/build.rs` 抽 `read_dir_paths`，目录条目读失败显式 `fail`，不再 `filter_map(ok())` 吞掉。复核保留：`prof::parse_version` 的 `.ok()` 是版本串匹配容错（整体返 `None` 由调用方判不匹配），`check.rs` 读见证失败走显式 `WitnessFile` 报告，均非静默丢弃。

### 2.4 weights 五领域混杂（audit P1-1 / S1，结构性，需分两步）

`weights` 当前 13 个源文件，容器/线格式、外部格式适配、rANS 压缩、哈希、ML-DSA 密码学仍在同一 crate（`bin.rs` 900 行里含 pickle 分派；`proof.rs`/`quant.rs` 相邻）。建议先把零行为变化的两组移出去：`crypto`（dsa/ntt/golay/lattice/ledger 证明）与 `codec`（crc/leb/sink/merkle），外部格式适配留本体。这是 audit 第三梯队第 10 条，收益最大但周期 1–2 周。

### 2.5 装饰性闭集与静默降级（小而明确）

- **Skeleton 双表收口**：以 `op::Skeleton` 为唯一声明处，注解侧 `SKELETONS` 投影到它；无构造点的 `Attention`/`Scan` 要么补注册点，要么从闭集删除，不留变体。
- ~~P2-2 静默降级~~、~~P2-4 静默丢目录项~~：均已结清，见 §2.3。

### 2.6 入口与命名摩擦（P3 级）

- **P3-3 下半截仍在**：`src/main.rs` 的 `decode_f32`/`parse_input`/`floor_row`/`read_sample` 是可复用逻辑，却只存在于二进制；L7 组装点不应沉淀算法。
- **P3-4**：`src/bin/cutile.rs` 排版逻辑占大头，排版属 `tool::Report` 职责域。
- **P3-2 同名跨层**：`train::Schedule`（学习率）vs `autograd::Schedule`（反传顺序）；`autograd::Plan` vs `cutile::Plan`（单 kernel 选型）。靠散文区分，建议类型名上带领域后缀。

### 2.7 需要 ADR 拍板的结构性减法（10k-analysis §6.2，不在重构里顺手做）

- 31→8 模块式合并（当前 25 成员，边界是否再收）；
- 手写解析器群收敛到一处，或接受 `toml+serde` 单依赖换掉约千行手写解析；
- annotation 宏去留（组合式风格偏爱函数而非宏）；
- 缓存侧砍 GC ledger / cache-why / 状态台账，只留内容寻址 key/artifact/复用计数；
- `benches` 是工具成员（架构门豁免），不是孤儿，但 `fuzz`/`benches` 是否移出 members 可单独立项。

---

## 3. 质量与规范：还能深度提升的点

### 3.1 门禁覆盖面盲区（P1-7，已逐条复核口径）

**【2026-11-16 复核结论】** 旧清单 1/2 在前人改动中已纳入：`build.rs`、`crates/*/tests/`、根 `src/` 按生产/脚本/集成测试三档进入编排；`docs/**` 递归 walk，含 ADR 与各 crate README。本轮补的残余面：

- `strip`：已识别 inner 属性 `#![cfg(test)]`（`attr_at` 返回 hash/bracket 下标）与复合 `#[cfg(any(test, ..))]`；`not(test)` 仍排除。
- `call`：已补 UFCS 关联函数形态 `Type::unwrap(..)`/`Type::expect(..)`；`unreachable!` 在既有门内。

仍登记为已知限制（注释已写在对应源文件顶部）：

- `ccn::collect` 跳过整个 `macro_rules!`/macro 定义体，宏模板分支密度不在本门口径内；
- `assert!`/`assert_eq!` 不纳入 call 门（测试断言与运行期不可信输入错误不混口径）。

本仓**不设 `examples/`**：根与各 crate 下出现 `examples/` 目录即由 `forbidden_example_dirs` 判失败；可演示用法由测试夹具与集成测试承担。

### 3.2 测试缺口（P2-9）

- `weights` 解析侧 `reader`/`seg`/`codec` 历史上零本地测试，截断/位翻转覆盖全在 `tests/round_trip.rs`；当前组织成 archive/seg/quant 后需重新核对各文件的负例覆盖。**【2026-11-16 已补】** seg 写端四条边界：短缓冲长度不符、空段名拒绝、零元素形状空缓冲合法、`[u64::MAX, 2]` 溢出形状按饱和乘积报错不回绕成 0；archive 补未知 mode 码拒绝用例。
- 待补边界：截断、单比特翻转、超长 rank；P0 校验样板尚未复制到 `zip64_extra` 与 `gguf` 的 `start+len` 路径（audit §15.4 残余）。
- 属性测试与模糊测试：`fuzz` 是契约驱动骨架，关键纯函数（形状乘积、步幅累积、字节数公式、调度排序）属性覆盖偏少；roadmap P0 出口要求"关键纯函数补属性测试，边界与退化情形进模糊测试"。
- 可空转测试：无夹具时直接 `return` 的用例（bin/zip、witness、ccn 旧二进制）应改成显式 skip 门控，与 `testkit::device_enabled` 同一风格。

### 3.3 三套格式词汇无表驱动校验（P2-11 残余）

`dtype::DType` 是主人；`weights/src/gguf.rs:29` 仍有本地 `Kind` 枚举。bin 侧 `Kind` 已随重组消失。剩余映射（如 `I8|U8 → DType::I8`）仍靠注释解释，建议加一条契约测试把外部格式码与 `DType` 闭集逐项对齐，与注解名词汇表那条门禁同口径。

### 3.4 鲁棒性残余

- 生产路径残余的 `expect`/`unwrap` 需按 §3.1 修正后的扫描口径逐处复查：构造期不可信输入（权重文件头、pickle、zip 长度字段）必须落到 `Err`，不允许 `assert!`/`expect` 兜底。
- 审计结论：类型系统守住了"错误怎么表达"，没守住"错误在哪里产生"；下一批应沿不可信输入路径（文件头 → 条目表 → 缓冲分配）逐个补交叉校验。

### 3.5 文档可执行性

- `docs/lld.md`（约 2410 行）与 `10k-analysis.md` 的规模数字已落后当前树，且未纳入任何门禁；P3-5 建议冻结 `lld.md`，新设计进 ADR。
- 当前文档门禁只查链接存在，不查内容是否与代码一致；P1-8 历史上先漂的正是没被门禁覆盖的那几份文档。可补少量"数字/名单派生自源码"的断言（如成员数、算子数、依赖边数已由 architecture 门覆盖，算子数目前仍是散文）。

### 3.6 覆盖率口径

CI 目前只对 `cutile`（行 75% / 函数 70%）与 `weights`（行 70% / 函数 64%）设下限；`dtype`/`tensor`/`autograd`/`train` 等基座没有覆盖率口径。P1 落地真实算子后，应对标量参考与内核路径补覆盖率下限。

### 3.7 棘轮与在途红灯

- 复杂度基线：`lints/vocabulary.toml` 现存 20 条 ccn/nloc 存量登记；棘轮只拦涨不拦跌，改善项需结清（audit §22 已改判据，存量条目当轮下调过一次）。
- **当前在途红灯**：工作区未提交的 `op/src/rmsnorm_f32.rs`、`softmax_f32.rs` 加了空输入校验分支，`reference`/`forward` Ccn=6 超过阈值 5，`tool` 静态门报 1 条失败。这两个文件是他人在途改动，应在合入前降分支或登记基线，不属于本次 tensor 重构。

### 3.8 已经守住、不要回退的质量资产

- 架构双向校验（成员/边/方向/所有权表）、闭集与位集唯一声明处、注解↔注册表镜像门禁、错误类型唯一（禁止 `Result<_, String>`）、`unsafe_code = deny`、clippy `-D warnings`、doc 零警告。这些是评审检查表本身，简化代码时不得放宽。

---

## 4. 未到位功能清单（按路线阶段，证据回链）

成熟度口径：**契约** = 类型/接口已定无实现；**簿记** = 只管理元数据；**片段** = 生成代码文本未接运行时；**实现** = 可在真实缓冲跑出结果。

| 维度 | 当前状态（证据） | 缺口 | 阶段 |
|---|---|---|---|
| 后端执行 | 契约：`engine/src/exec.rs` 的 `Stream`/`Graph`/`Node`/`Launch`/`Executor`；无真实流、设备队列、kernel launch、事件同步 | CPU `Executor` 实现；P3 再挂 Metal/CUDA | P1 CPU / P3 加速 |
| 张量与内存 | 只读 `&[u8]` 视图 + `OwnedTensor`；CPU `Buffer`/`Allocator`/`CpuAllocator` 簿记；跨设备拷贝是 `Allocator` 显式方法、实现未落地 | 可写缓冲贯穿训练/推理通路；地址抽象从 CPU 指针预留到设备句柄 | P1 |
| 算子内核 | 22 条注册（`op::REGISTRY` 投影）+ 模板片段 + 标量参考；无真实后端 kernel | decode 主路径优先：`gemv_f32`、`softmax_f32`、`rope_f32` 真实实现，elementwise 次之；每条保留标量参考对拍 | P1 起逐算子 |
| 注意力 / KV | 页表 + 存活位图簿记，页里不持数据（`engine/src/kv.rs`）；`attention_f32`/`flash_attention_f32` 两条声明 | KV 页背后挂真实张量；注意力内核接入；prefix cache | P2 |
| 批处理调度 | 调度契约（`engine/src/exec.rs`）；`Sequence`/`Scheduler`/`Progress` 状态类型 | continuous batching 准入与推进；池不够返回 `Blocked` 的真实路径 | P2 |
| 量化 | 精度事实源 `dtype` + 权重分组推断（`weights::quant`）；无计算内核 | INT4/INT8/FP8 反量化内核与权重布局；新格式不回改 weights 读取 | P3 |
| 分布式 | TP/PP/DP 拓扑与集合通信 trait 簿记（`shard/src/*`），无通信后端 | NCCL/MPI 类集合通信原型；后端 trait 填实现 | P3 |
| 性能测量 | criterion 结构/编解码基准 + 3 条 decode 行宽基准（gemv/softmax/rope 已入回归基线） | 统一 harness：记录硬件/形状/批次/后端；真实 kernel 吞吐采样；同硬件对照 llama.cpp/burn/candle | P1 骨架，P2/P3 接内核 |
| 模型端到端 | 能读 safetensors，层栈与性能下界可推；无模型前向 | 给定真实权重跑通生成；一层模型前向与标量参考对拍达标 | P4（P1 先一层） |
| 服务化 | `Route`/`Endpoint`/`Response` + `proto` 线格式契约；无 socket | 真实 HTTP/gRPC `Endpoint` 实现；并发、流式输出；P50/P99 服务基准 | P4 |
| 随机数与采样 | 实现（CPU）：`rng` Philox4x32-10 计数器引擎 + `Uniform`/`Normal`/`Bernoulli`，显式种子 / 流号 / 下标派生，不引 `rand`、不持全局状态，CPU 顺序填充与 GPU 按索引寻址走同一块纯函数可逐位对拍；`serve::Multinomial` 与经典采样共用一份 top-k / top-p 截断，保留集上 softmax 逆变换抽样，同种子同 logits 逐位可复现 | 并行后端 Gumbel-max（逐候选 `logit/T + gumbel` 后 argmax，不建累积表）等价口径；随机张量初始化（半精度输出舍入归 `dtype` 转换，中间计算留 f32）；`Normal` 第二值缓存评估；token 侧 Categorical 分布 | P1 CPU / P3 加速 |
| kernel 侧操作数视图 | ADR 0003：DSL 宿主 API（`&mut` 分片即 grid）P0 尚未实现；`cutile` 只收裸指针与标量 | 真要直接调 kernel 时按 cutile README §4.4 重建；不得依赖 `tensor` | P0/P3 衔接 |
| 分词 | `Bytes`/`Bpe`/`Unigram` 主体在；预切分（空白/标点/Unicode 脚本边界）是 D2 待办（`tokenizer/src/bpe.rs:6`） | 预切分层接入；Unigram 前向 Viterbi 已在，边界用例补齐 | P1 后 |
| 训练侧 | 反传图/调度/计划/重放簿记齐（`autograd`）；真实缓冲在 `train` `Buffers`；算术走 `Kernel`（默认 `Reference` 调标量语义） | 激活显存计划落地；重放序列驱动真实缓冲；训练循环端到端 | P1/P2 |
| 产物缓存 | 内容寻址 key/artifact/复用计数结构在 `cutile::cache` | AOT/JIT 产物复用路径接真实编译产物 | P0/P1 |

---

## 5. 建议的推进顺序

**第一梯队（低风险纯收益，不引入后端）**

1. 补门禁扫描盲区 §3.1.1/3.1.5；解析侧负例测试 §3.2；收敛 `Mode::Infer` 静默降级与 `filter_map` 静默丢条目 §2.5。
2. `storage` 步幅规则并入 `tensor`（audit S2，零生产入边，回滚面小）。
3. 删除已无意义的装饰性 `Skeleton` 变体或补注册点，两套骨架表收口到一处。

**第二梯队（结构性，1–2 周量级）**

4. 按 §2.1 清单拆分超大单文件（公开 API 不动）。
5. `weights` 分两步拆出 `crypto` 与 `codec` 组。
6. 三套格式词汇加表驱动校验；覆盖率口径补到基座 crate。

**第三梯队（路线驱动，P1 出口优先）**

7. CPU 最小闭环：可写缓冲通路 + `gemv_f32`/`softmax_f32`/`rope_f32` 真实内核 + 一层模型前向对拍；对照 harness 采第一批同硬件数字回填 roadmap。
8. 之后按 P2（KV/调度/注意力）→ P3（双后端/量化/通信）→ P4（模型/服务）推进。

**明确不做（ADR 已冻结）**

- 跨算子融合 / 通用运行时 IR（ADR 0017；解冻需 L2 证据 + 70% roofline，当前无数据）；
- 模型动物园、vision、RL 等产品广度；
- 第三加速链路、投机解码、大规模分布式（P3 暂缓项）。

---

## 附：本分析未覆盖的取证项

- 未在目标硬件实测任何吞吐数字；roadmap 中所有性能结论仍是静态代价估算。
- `fuzz` 驱动对象的覆盖完备度、`benches` 各条基线的采样稳定性未逐一核验。
- 密码学模块（dsa/ntt/golay/lattice）的算法正确性不在本次静态分析范围内。
