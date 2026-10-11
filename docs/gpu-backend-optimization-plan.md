# CUDA 与 Metal 后端全量审查与优化执行计划

- 状态：**审查计划**；G0 基线、G1 契约一致化与部分 G2 测试补强已实施（见第 10 节），其余条目仍未实施。
- 仓库：`spapient/llm`
- 基线分支：`main`
- 计划日期：2026-10-12
- 覆盖范围：`crates/cuda`、`crates/metal`、`crates/api` 的后端契约、相关集成测试、质量脚本与 CI。
- 目标：先保证语义一致、数值正确、资源边界可靠，再优化吞吐和延迟；所有性能结论必须有可复现的真实设备基准支持。
- 约束：不新增 demo 文件；不以重命名/搬目录代替架构改进；不在未验证的情况下合入 main；不把没有 GPU 的编译通过说成 GPU 验证通过。

## 1. 结论摘要

### 1.1 目录结构决策

当前工作区通过根 Cargo.toml 的 `members = ["crates/*"]` 自动纳入各 crate。CUDA 与 Metal 是独立 crate：`crates/cuda`、`crates/metal`；二者都依赖 `api`、`tensor`、`lib`，不应相互依赖。公共契约位于 `crates/api`。

**当前建议：暂时保留两个并列 crate，不迁移到 `crates/gpu/cuda` 和 `crates/gpu/metal`。** 父目录名称不会自动提供抽象、调度或性能收益；移动后还需同步 workspace 成员规则、脚本、文档、路径依赖与 CI。只有未来出现确切的公共 GPU 运行时职责（设备枚举、能力匹配、后端选择策略、跨后端调度）时，才评估新增薄的 GPU 门面 crate。公共契约仍归 `api`，不能把硬件无关张量或 API 强塞进 GPU 层。

### 1.2 本轮源码检查到的结构事实

- `crates/api/src/lib.rs` 定义硬件无关的 backend / compute / capability / device 契约。
- CUDA：Linux 上启用 `cudarc`，使用 cuBLAS、cuDNN、NVRTC；非 Linux 编译为空库。
- Metal：macOS 上启用 `objc2-metal` 等绑定；非 macOS 编译为空库。
- 两个执行器都做 rank、元素数、字节预算校验，也都单独校验 gather 输出预算，存在可审查的重复逻辑。
- CUDA 有独立的 `matmulshape` 几何计划；Metal 的矩阵乘法形状逻辑仍在 executor 中，两边组织不对称。
- 两边各自拥有 `geometry.rs`、`source.rs`、`ffi.rs`；后端特有的 kernel 源码、设备 API 与 unsafe 边界必须继续独立。
- 两边的算子内核都包含简单矩阵乘法参考实现；CUDA 的生产路径使用 cuBLAS，Metal 目前使用自有 MSL kernel。不能在没有基准的情况下断言哪一方更快。
- `.github/workflows/ci.yml` 已包含通用质量任务、macOS Metal 测试、Linux CUDA 编译/设备无关契约测试，以及手动触发的 self-hosted CUDA GPU 测试。
- `scripts/checkunsafe.sh` 对 GPU FFI unsafe 位置和登记数量实行机械门禁；任何 FFI 改动都必须同步审查该门禁及安全说明。
- 项目采用无分隔符命名约定；本计划不擅自进行全仓公共 API 改名。

以上是已读取源码与配置的审查结果，不代表本轮已经运行这些检查或验证了真实设备执行。

## 2. 目标与不可妥协的正确性标准

1. **契约一致**：相同合法输入在 CPU、CUDA、Metal 上具有一致的 shape、索引、错误分类和 causal 语义；允许浮点运算顺序造成的、被明确容差约束的数值差异。
2. **边界安全**：任何维度乘法、字节计算、grid/threadgroup 计算、索引偏移都不得发生整数溢出；超预算必须在分配/派发前返回错误。
3. **资源生命周期正确**：失败路径不得泄漏或错误复用设备资源；并发请求之间不得共享可变请求状态。
4. **数值稳定**：softmax、RMSNorm、RoPE、Attention 在极端值、非有限输入、空维度与极端合法维度上有明确语义和测试。
5. **跨平台诚实**：Linux 的 CUDA 编译检查、macOS 的 Metal 编译检查、真实 CUDA/Metal 设备测试分别记录；任何一类通过不能冒充另一类通过。
6. **性能可证**：先建立基线，再更改实现；固定设备、驱动、OS、编译参数、形状、warmup、重复次数与统计方式。报告 median、P95（适用时）、吞吐、内存占用；不接受无数据的“性能提升”结论。
7. **回归可追踪**：每个修复对应测试和 CI 证据；出现回归必须阻断合入。

## 3. 整改分级与执行顺序

优先级定义：
- **P0**：可能导致错误结果、越界/溢出、资源超限失控或测试误报，先处理。
- **P1**：契约漂移、错误分类不一致、生命周期/并发问题、关键测试缺口。
- **P2**：可测量的性能优化。
- **P3**：文档、开发体验和后续架构扩展。

### G0 — 建立可审计基线（P0，先做）

**整改项**
- 记录当前 main 的 commit SHA、Rust/Cargo 版本、macOS/Linux 版本、GPU 型号、驱动和 CUDA 运行时（如可用）。
- 运行现有质量脚本及各目标平台测试；保存原始日志、失败项和测试是否因缺少设备而跳过。
- 检查 CI 的 CUDA GPU job 是否已配置有效的 self-hosted runner 标签，并确保强制设备模式 `LLM_CUDA_REQUIRE_DEVICE=1` 生效。
- 建立 CUDA 与 Metal 的算子覆盖表，逐项标明：契约、设备无关测试、真实设备测试、误差阈值、基准状态。

**验收**
- 基线报告能区分 PASS、FAIL、BLOCKED、NOT RUN。
- 未执行项目明确标为 NOT RUN；没有设备的测试不得被记录为 PASS。
- 在未保存基线之前不做大规模重构，避免无法判断回归来源。

### G1 — 统一 shape 与资源预算校验（P0/P1）

**现状/风险**
- CUDA 与 Metal executor 中存在重复的张量 rank、元素计数、字节数、gather 输出预算逻辑。
- 矩阵乘法 shape plan 的结构不对称，可能导致边界修复只覆盖一侧。
- 资源预算的检查时点必须早于输出分配、GPU buffer 分配与 dispatch。

**整改方案**
1. 先把 CPU 参考实现、`api::compute` 契约和两个 GPU executor 的校验顺序列成表：输入 rank、维度、元素数溢出、bytes 溢出、limits、shape mismatch、空维度、输出预算、设备执行失败。
2. 为两后端采用同一套纯主机端几何校验测试向量。优先共享测试数据/测试辅助，而非立即抽出复杂的公共 runtime。
3. 评估把真正硬件无关的预算计算提取到 `lib` 或 `api` 之下；只有确认职责归属和依赖方向后才抽取，避免 `api` 反向依赖具体 GPU crate。
4. 将 Metal 的 matmul shape validation 也整理为独立、无设备副作用的 plan，与 CUDA 结构对齐；如果两边的错误顺序是公共契约，必须显式测试。
5. 所有 `checked_mul`、字节数计算、grid 尺寸计算都保留溢出检查；不允许使用饱和运算悄悄改变语义。
6. 对零元素输入逐个算子定义行为：合法空结果或明确拒绝，不允许把零 block/grid 直接交给驱动后依赖驱动报错。

**必须新增/强化的测试**
- rank 小于/大于上限；空 dims；单元素维度；零维度；最大合法 rank。
- `usize::MAX` 附近的维度乘积与字节乘积溢出。
- 元素预算刚好等于限制、超过一个元素；字节预算刚好等于限制、超过 1 byte。
- gather：空索引、首尾索引、越界索引、rows × width 溢出、输出预算超限。
- 两后端对相同输入返回相同的公共错误类别及算子名称（契约允许差异必须写明）。

**验收**
- 所有预算在任何设备分配或 kernel dispatch 前验证。
- 两个后端的纯 shape/budget contract 在无 GPU 主机上也可运行。
- 错误映射不再把 size overflow、resource exhaustion 与一般 execution failure 混为一谈；若公共错误枚举限制映射精度，应记录为单独 API 设计决策。

### G2 — 核心算子契约与数值正确性（P0）

对每个已实现算子建立同一份测试矩阵，至少覆盖 matmul、add、mul、SiLU、RMSNorm、softmax、embedding/gather、RoPE、Attention。

**矩阵乘法**
- 测试非方阵、不同 m/k/n、最小合法尺寸、batch 维度、输出尺寸溢出、零维度策略。
- 与 CPU 参考实现比较，采用基于 dtype/规模的绝对与相对误差阈值；不要要求不同硬件的浮点归约顺序逐 bit 相同。
- 检查 CUDA 的 cuBLAS 路径与 NVRTC reference 路径；Metal MSL 路径与 CPU 参考路径分别对比。
- 重点评估大 k 累加误差、极端有限值、NaN/Inf 的约定行为。

**Softmax**
- 覆盖大正数/大负数、全相等值、长行、单元素、极端但合法的行宽。
- 明确 NaN、正负 Inf、全 -Inf 行的行为；CPU、cuDNN、MSL 的差异必须被显式接受或修复，不能靠模糊容差掩盖。
- 验证输出概率和、无 NaN 溢出、误差范围；特殊值行为按已定义契约单独断言。

**RMSNorm**
- 覆盖小/大 epsilon、权重 shape 与元素数兼容规则、极大/极小输入、宽度非 2 的幂、宽度超过后端限制。
- 核查 CUDA reduction 对 threadgroup 大小、动态共享内存的假设；核查 Metal reduction 对 threadgroup 大小与线程数的假设。不得只测试 256 宽度。
- 检查平方和溢出/下溢是否符合数值契约；若使用更稳定的算法，需测量误差与成本。

**RoPE**
- 覆盖 heads=0、奇数 head dimension、最大 head dimension、序列位置边界、base 的非法值（如契约禁止）及参数转换精度。
- 明确 position 是行索引还是外部位置；测试分段输入/增量 decode 不应无意重置位置。

**Attention**
- 覆盖 heads 与 kvheads 正倍数、causal/non-causal、不同序列长度策略、空序列、极端 logits、重复与非连续索引（如果 API 允许）。
- 检查 causal mask：位置 s 只能访问 t <= s。
- 当前 API 契约/CPU 实现对 query/key/value 序列长度的要求，需要和真实 prefill 及单 token decode 需求对照。不要简单放宽长度校验；应在设计中区分 prefill 与增量 decode，并定义 KV cache、position、cache capacity、请求隔离和溢出行为。

**验收**
- 每个算子有 CPU reference 对照、正向/反向边界测试（如适用）、数值阈值和设备执行证据。
- 数值容差需按算子和测试尺度记录，不能用一个过宽阈值包打天下。
- 每个 GPU 专项测试明确设备是否真的执行。

### G3 — FFI、安全与资源生命周期（P0/P1）

**整改项**
- 审核 `crates/cuda/src/ffi.rs`、`crates/metal/src/ffi.rs` 每一个 unsafe block 的指针有效期、长度、对齐、ownership、同步、错误路径和释放规则。
- 保留 GPU FFI 岛；不要把 unsafe 扩散到 executor、geometry 或公共 API。
- 每个 unsafe 块的 `SAFETY:` 注释必须说明可验证前提，而不是只重复代码意图。
- 核查设备 buffer、command buffer、pipeline、library、module、function、stream/context 的创建和释放；失败分支也要覆盖。
- 核查 host/device 传输是否在读回前同步，异步操作的 buffer 生命周期是否覆盖实际执行。
- 检查 dispatch 尺寸和传入 kernel 的整数宽度（例如 usize 到 u32/uint 的转换）在转换前验证范围。
- 检查运行时编译失败、函数符号缺失、pipeline 创建失败、设备不可用时，错误是否稳定且不泄漏底层资源。

**门禁**
- 每次 FFI 修改运行 `bash scripts/checkunsafe.sh`。
- 不随意更新 unsafe 登记数量以“让检查通过”；必须先完成逐块审查和针对性测试。
- 为资源释放、编译失败和 dispatch 失败增加可触发的错误路径测试；如无法 mock 原生 API，明确标注需真实设备验证。

**验收**
- 所有 unsafe site 数量和登记精确匹配；无未经审查的新 unsafe。
- 所有失败路径返回 Result，不用 panic/unwrap/expect 处理运行时错误。
- GPU 并发测试验证 executor 可共享的前提；如底层上下文或 command queue 不支持当前并发模型，应明确同步策略。

### G4 — 运行时编译与缓存（P1/P2）

**现状**
- CUDA NVRTC kernel 和 Metal MSL library/pipeline 都在运行时编译/创建。首次调用的成本可能显著影响短请求；是否存在每次调用重复编译/创建，必须通过完整调用路径核实，不能只凭文件结构判断。

**整改方案**
1. 先用计时与调用计数确认 compile/library/pipeline 创建频率，区分 cold start、warm start、每次调用成本。
2. 对不依赖输入数据的 kernel module/library/function/pipeline 做惰性缓存；缓存键必须包含设备、kernel 标识、编译选项及所有影响二进制的配置。
3. 明确线程安全、初始化失败的缓存行为、缓存生命周期和设备丢失后的失效处理。
4. 避免在持锁期间进行长时间编译或 GPU 同步；评估并发首次调用是否造成重复编译。
5. 不缓存请求相关 tensor/buffer，除非有独立的内存池设计和明确的所有权/隔离契约。

**验收**
- 冷启动和热路径分别有基准。
- 基准证明缓存减少编译/创建开销且不改变结果。
- 并发首次调用、初始化失败、重复调用测试通过。

### G5 — GPU 内存与执行开销（P1/P2）

**整改项**
- 对每个算子统计 host→device、kernel dispatch、device→host、分配/释放次数与耗时。
- 确认每次 operator call 创建/释放资源的成本；当前文档描述执行器无请求状态，但这不等于设备 buffer/command resource 已被复用。
- 建立明确的 buffer pool 或 request-scoped execution context 前，先测量是否为瓶颈。
- 评估是否能把相邻操作融合，优先考察 RMSNorm、RoPE、逐元素算子及 Attention 的中间张量。
- 任何融合必须保留 CPU 参考路径与数值对照，测量内存节省、吞吐、延迟与寄存器/共享内存占用。

**验收**
- 不因复用导致跨请求数据污染、use-after-free、同步错误或无限增长。
- 记录峰值显存/统一内存、分配次数、传输字节数。
- 仅当端到端推理数据受益时接受复杂化，不能只优化孤立 kernel。

### G6 — Matmul 性能路线（P2）

**CUDA**
- 生产路径保持 cuBLAS，NVRTC 路径定位为独立 reference/对照，除非基准和正确性证据支持改变。
- 检查 cuBLAS 布局、转置、leading dimension、batch 语义是否与公共 row-major 契约完全一致。
- 对小矩阵、长宽不均衡矩阵、典型 Transformer 投影矩阵分别测量，避免只用方阵 benchmark。

**Metal**
- 当前简单 MSL matmul 作为正确性基线保留。
- 在真实 Apple GPU 上建立基线后，评估 threadgroup tile、SIMD-group、矩阵硬件路径（若目标设备/API 支持）和合适的 dispatch geometry。
- 先确保数值与形状契约，再逐步替换 kernel；保留可验证的 reference 实现或等价测试。

**统一基准**
- 固定尺寸集、warmup、重复轮次、计时方法、同步策略。
- 报告 latency、吞吐和相对 CPU/原后端的结果；避免把传输时间排除后宣称端到端提速。
- 不预设 CUDA 与 Metal 必须使用同一 kernel 算法；共享语义而非强制共享实现。

### G7 — CI、测试矩阵与结果可追溯（P1）

**现状**
当前 workflow 已有：
- Linux 通用格式、check、Clippy、测试、文档和 benchmark 编译。
- macOS Metal workspace 测试。
- Linux CUDA crate 编译及设备无关契约测试。
- 手动触发的 self-hosted CUDA GPU contract job，并用 `LLM_CUDA_REQUIRE_DEVICE=1` 防止无设备时假通过。

**整改方案**
1. 确保 PR required checks 覆盖通用质量、macOS Metal 和 Linux CUDA 编译/契约。
2. CUDA 真机 job 保持手动触发或按 runner 可用性调整；一旦作为发布门禁，必须确保标签、驱动、设备可用且无设备即失败。
3. 为 Metal 真机数值测试和性能基准保存结构化结果（设备型号、OS、commit、测试名称、误差、耗时）。
4. 将设备无关测试和设备必需测试明确分组；跳过必须输出清楚状态。
5. 增加最小 smoke 测试以尽早发现初始化、运行时编译、kernel symbol 和 dispatch 错误；完整数值矩阵可在 nightly 或专用 runner 执行。
6. 保持锁文件与工具链策略一致；如果 CI 要求 `--locked`，所有对应任务必须有提交的 Cargo.lock 且运行路径一致。

**验收**
- PR 页面可清晰区分 compile-only、host-contract、real-device correctness、performance benchmark。
- GPU 设备缺失或测试被跳过不能呈现为数值正确性通过。
- 每个回归能追溯到 commit、设备和日志。

### G8 — API 语义与增量 Attention（P1，涉及范围较大）

这项与目录整理分开，不应作为小型重构顺手完成。

**问题**
公共 `compute::attention` 契约与 CPU 参考实现对 query/key/value 序列长度、heads/kvheads 和 causal mask 有明确限制。LLM 单 token decode 通常需要 query 长度小于缓存中的 key/value 长度。若当前接口不支持此语义，单纯让 CUDA/Metal 放宽 shape 检查会造成跨后端不一致。

**整改步骤**
1. 单独设计 attention API：prefill、单 token decode、multi-token decode 的输入/位置语义。
2. 设计 KV cache 生命周期、容量、追加/读取、position offset、请求隔离、超限错误与清理策略。
3. CPU 参考实现先完成并通过边界测试。
4. 再分别实现 CUDA 和 Metal；用同一测试向量核对结果。
5. 加入长上下文、cache 满、连续请求、取消请求、不同 batch/request 互不污染等测试。

**验收**
- 公共契约明确描述 cache 与位置语义。
- CPU/CUDA/Metal 一致通过 prefill 和 decode 测试。
- 不在本次单纯目录/重复逻辑重构中偷偷改变冻结的公共 API。

### G9 — 文档和可维护性（P2/P3）

- 更新架构文档，说明 API、CPU reference、CUDA、Metal 的依赖方向和执行路径。
- 对每个算子维护一张契约/测试/设备覆盖表，作为代码评审清单。
- 记录每项优化的动机、备选方案、正确性风险、复杂度、基准结果与回滚办法。
- 文档中的 Verified 必须有实际命令、commit 和测试证据；计划或源码推断只能标为 Proposed / Not run。
- 避免复制整段实现说明到多个文档；公共契约只指定一个权威来源，其余引用它。

## 4. 共享与独立的边界

| 内容 | 策略 | 原因 |
|---|---|---|
| 算子语义、shape 与错误契约 | 共享于 `api` | 防止 CPU/CUDA/Metal 语义漂移 |
| 通用 limits/溢出校验 | 先评估共享位置，确认依赖后再抽取 | 减少重复但避免错误分层 |
| 测试向量、数值阈值、contract harness | 共享 | 同一输入对照所有实现 |
| CUDA FFI、NVRTC、cuBLAS/cuDNN 绑定 | 独立 | NVIDIA 专属生命周期和错误模型 |
| Metal FFI、MSL、pipeline/command buffer | 独立 | Apple 平台专属 API 与资源模型 |
| kernel 调优参数和 dispatch geometry | 各后端独立 | 硬件架构差异显著 |
| 设备发现/统一调度 | 当前不新增；需求明确后再设计门面 | 防止过度抽象 |
| 模型权重、请求状态、KV cache | 不放入底层 GPU crate 的隐式全局状态 | 保持请求隔离与层次边界 |

## 5. 建议的实施 PR 拆分

不要把所有改动堆进一个大 PR。按以下顺序，每个 PR 都必须独立可审查、可回滚：

1. **PR-A：基线和契约测试补强**（无行为改动）
   - 算子覆盖矩阵、边界测试、统一数值比较辅助。
   - 验收：现有行为被记录；失败测试能复现。
2. **PR-B：shape/预算校验一致化**（小范围内部重构）
   - 对齐 matmul plan；共享测试向量；减少 executor 重复。
   - 验收：全套 host contract 与所有质量门禁通过。
3. **PR-C：GPU FFI/资源生命周期审计**
   - 逐个 unsafe 和失败路径；修复明确缺陷。
   - 验收：unsafe gate、目标平台编译、设备 smoke 通过。
4. **PR-D：运行时编译与资源创建缓存**
   - 仅在基线证明开销存在时实施。
   - 验收：冷/热基准、并发初始化测试、结果一致性通过。
5. **PR-E：真实设备数值覆盖**
   - Metal 真机和 CUDA self-hosted 设备运行完整 contract。
   - 验收：设备、commit、误差、日志均可追溯。
6. **PR-F：Matmul 与融合性能优化**
   - 一次只改一类 kernel/调度策略。
   - 验收：无数值回归，端到端有可重复收益。
7. **PR-G：Attention/KV cache API 专项设计与实现**
   - 独立 ADR/设计评审，不和目录重构捆绑。
   - 验收：prefill/decode 的 CPU/CUDA/Metal 对照完整通过。

## 6. 每个 PR 的强制门禁

按仓库当前脚本与平台能力执行；具体命令应以实际环境为准：

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps --locked
cargo bench --workspace --no-run
bash scripts/checknaming.sh
bash scripts/checksrctests.sh
bash scripts/checkunsafe.sh
bash scripts/checkthresholds.sh
```

平台专项至少包括：
- Linux CUDA 代码：`cargo check -p cuda --all-targets`。
- CUDA host contract：`cargo test -p llm --test it cudacontract`。
- CUDA 真机：设置 `LLM_CUDA_REQUIRE_DEVICE=1` 后运行 CUDA contract。
- Metal：在 macOS 上运行对应 workspace/Metal contract，并确认不是空库构建。
- 对真实设备任务记录命令、环境、commit、退出码与日志。不要在不支持的平台上将未执行的测试标为通过。

注意：以上是建议门禁列表，不表示这些命令已在本次审查中运行。若当前工作区的锁文件/命令参数与脚本实际状态不一致，应先核实并在单独改动中统一，不能静默删减检查。

## 7. 风险与回滚策略

- **公共 API 改动风险**：优先采用内部重构；改变 `api` 契约必须单独设计、兼容性评审和全后端迁移。
- **数值变化风险**：每次只改一个算子，保留 CPU/reference 对照和原始基准。
- **性能回退风险**：将 benchmark 结果与 commit 绑定；没有收益或收益不稳定则回滚。
- **跨平台遗漏风险**：CI 中的空库可能掩盖平台代码未编译，必须保留 macOS 和 Linux CUDA 专项任务。
- **unsafe 扩散风险**：新增 unsafe 一律阻断，直到审查和登记完成。
- **缓存/并发风险**：先证明生命周期和线程安全；缓存必须有容量/失效策略，不能以无界增长换速度。
- **大 PR 风险**：按阶段拆分，每次只解决一组可验证问题；任何阶段失败都停止推进依赖它的阶段。

## 8. 退出条件（完成定义）

只有同时满足以下条件，才能称 CUDA/Metal 后端整改完成：

- [ ] 所有算子都有契约与测试覆盖矩阵。
- [ ] 公共 shape、错误与资源预算语义一致且边界清晰。
- [ ] 所有溢出、索引、资源预算与零维度行为有测试。
- [ ] 所有 FFI unsafe site 已逐块审查，安全门禁通过。
- [ ] CUDA 与 Metal 各自的真实设备数值测试通过；未测项目明确披露。
- [ ] 性能改动有基线和可复现的正向结果，且没有数值回归。
- [ ] CI required checks 不能把缺少设备或跳过的测试当成通过。
- [ ] 文档中的验证状态与实际证据一致。
- [ ] PR 已通过评审并按仓库策略合入；未合入之前不宣称主线完成。

## 9. 本次审查的边界与后续动作

本文件根据当前可读取的主要源码、crate manifest、公共 API、质量脚本和 CI workflow 制定。它是**面向实施的审查计划，不是已经完成的代码整改，也不是所有 GPU 源码逐行审计的替代品**。尤其是 FFI 生命周期、具体 kernel 的全部边界条件和真实设备性能，仍必须在执行阶段逐项核验。

建议下一步从 PR-A 开始：先建立可重复的测试基线，再处理 G1 的 shape/预算校验一致化。这样能先固定行为和失败证据，避免一边大规模改代码、一边失去判断正确性的参照。

## 10. 执行状态与算子覆盖表（2026-10-12）

本节记录 G0 要求的基线与覆盖表，以及截至该日期已实施的条目。未列出的条目一律视为**未实施**。

### 10.1 基线（G0）

| 项 | 记录 |
| --- | --- |
| 基线 commit / 分支 | `55d8efb` / `docs/gpu-backend-optimization-plan`（PR #1） |
| 工具链 | `rustc 1.99.0 (b940084d7 2026-09-28)`、`cargo 1.99.0`，由 `rust-toolchain.toml` 固定 |
| 宿主 | macOS 27.0.1（build 26A434），arm64，Apple M5 Pro |
| CUDA 设备（宿主 macOS） | 无（`nvidia-smi` 不可用）。`x86_64-unknown-linux-gnu` 目标仅用于编译与 lint |
| CUDA 真机（2026-10-12） | NVIDIA GeForce RTX 2080 Ti（`sm_75`，11264 MiB），驱动 580.105.08，CUDA runtime 13.0（**无 `nvcc`**），Linux 5.4.0-182 x86_64、96 vCPU；`rustc 1.99.0`，见 `docs/design.md` §21.10.3 |
| `bash scripts/check.sh` | PASS（exit 0） |
| `cargo test --workspace` | PASS：758 passed / 0 failed；其中 `metalcontract` 40 例在本机 Apple GPU 上真实运行 |
| CUDA 运行时数值 | **RUN（E1，2026-10-12 真机）**：`LLM_CUDA_REQUIRE_DEVICE=1` 下 `cudacontract` 36 passed / 0 failed、`cargo test --workspace` 758 passed / 0 failed、输出 0 条 `E4 未执行`；首次执行即暴露并修正一个用例缺陷（见 10.6） |
| CUDA 编译与静态门禁 | PASS（E1）：`cargo check`/`cargo clippy` 针对 `x86_64-unknown-linux-gnu`，含 `-D clippy::cognitive_complexity -D clippy::too_many_lines` |

### 10.2 算子覆盖表

| 算子 | 无设备契约向量（`tests/it/gpucontract.rs`） | Metal 真机数值用例 | CUDA 真机数值 | 数值容差 | 基准（同轮 CPU 对照） |
| --- | --- | --- | --- | --- | --- |
| `matmul` | `matmul/rank`、`/contraction`、`/batchdims`、`/outelements`、`/outbytes`、`/k0/{elements-at-bound,bytes-at-bound,elements-over,bytes-over}` | `matmul_matches_the_scalar_reference`、`matmul_handles_a_leading_batch_dimension`、`matmul_with_distinct_extents_matches_the_reference`、`matmul_of_a_single_element_matches_the_reference`、`matmul_rejects_incompatible_shapes`、`matmul_rejects_operands_beyond_the_element_budget` | `cublas_matmul_matches_the_scalar_reference`、`nvrtc_matmul_matches_the_scalar_reference`、`the_two_matmul_paths_agree_on_a_non_square_case`、`matmul_handles_a_leading_batch_dimension`、`matmul_with_a_zero_contraction_yields_the_additive_identity`：**已运行（E1，2026-10-12）** | 1e-4 | `gpu/matmul/{cpu,cuda}256`、`gpu/matmulbatch/{cpu,cuda}8x256` |
| `add` | `add/shape` | `add_matches_the_elementwise_sum`、`elementwise_rejects_mismatched_shapes`、`elementwise_rejects_operands_beyond_the_element_budget`、`empty_elementwise_inputs_produce_empty_output` | `add_matches_the_elementwise_sum`：**已运行（E1）** | 逐位相等 | `gpu/elementwise/{cpu,cuda}add` |
| `mul` | `mul/shape`、`mul/elements` | `mul_matches_the_elementwise_product`、同上三个边界用例 | `mul_matches_the_elementwise_product`：**已运行（E1）** | 逐位相等 | `gpu/elementwise/{cpu,cuda}mul` |
| `silu` | `silu/rank`、`silu/elements`、`silu/bytes` | `silu_matches_the_scalar_reference`、`empty_elementwise_inputs_produce_empty_output` | `silu_matches_the_scalar_reference`：**已运行（E1）** | 1e-6 | `gpu/elementwise/{cpu,cuda}silu` |
| `rmsnorm` | `rmsnorm/emptylastaxis`、`rmsnorm/weightlength` | `rmsnorm_matches_the_scalar_reference_per_row`、`rmsnorm_handles_a_row_wider_than_one_threadgroup`（宽 1000）、`rmsnorm_rejects_an_empty_last_axis`、`rmsnorm_rejects_a_weight_of_the_wrong_length`、`rmsnorm_accepts_a_one_row_weight` | `rmsnorm_matches_the_scalar_reference_per_row`、`rmsnorm_accepts_a_one_row_weight`：**已运行（E1）** | 1e-5 | `gpu/rmsnorm/{cpu,cuda}rows512cols4096` |
| `softmax` | `softmax/emptylastaxis` | `softmax_matches_the_scalar_reference_per_row`、`softmax_handles_a_row_wider_than_one_threadgroup`、`softmax_is_numerically_stable_for_extreme_logits`、`softmax_of_a_negative_infinity_row_is_not_a_number`、`softmax_over_zero_rows_yields_an_empty_output`、`softmax_rejects_an_empty_last_axis` | `softmax_matches_the_scalar_reference_per_row`、`softmax_is_numerically_stable_for_extreme_logits`：**已运行（E1）** | 1e-6 | `gpu/softmax/{cpu,cuda}rows512cols4096` |
| `rope` | `rope/heads0`、`rope/oddhead`、`rope/raggedheads`、`rope/rank1` | `rope_matches_the_scalar_reference_per_position`、`rope_of_a_single_position_is_the_identity_rotation`、`rope_rejects_a_zero_head_count_or_an_odd_head_dimension` | `rope_matches_the_scalar_reference_per_position`：**已运行（E1）** | 1e-5 | `gpu/rope/{cpu,cuda}seq128heads8dim64` |
| `embedding` | `embedding/index`、`embedding/gatherbudget` | `embedding_matches_the_scalar_reference`、`embedding_accepts_the_first_and_last_row`、`embedding_rejects_an_out_of_range_index`、`embedding_rejects_a_gather_beyond_the_element_budget`、`embedding_with_no_indices_yields_an_empty_output` | `embedding_matches_the_scalar_reference`：**已运行（E1）** | 逐位相等 | `gpu/embedding/{cpu,cuda}gather128` |
| `attention` | `attention/rank1`、`/seq`、`/kvshape`、`/headsplit`、`/kvheads0`、`/kvwidth` | `attention_matches_a_scalar_reference`、`attention_non_causal_matches_a_scalar_reference`、`attention_with_equal_head_counts_matches_the_reference`、`attention_single_position_returns_the_value`、`attention_rejects_inconsistent_operands`、`attention_rejects_a_head_dimension_beyond_the_device_limit` | `attention_matches_a_scalar_reference`、`attention_single_position_returns_the_value`：**已运行（E1）** | 1e-5 | `gpu/attention/{cpu,cuda}seq128heads8dim64` |

每头宽度的设备上限（`METALMAXHEADDIM` = 4096、`CUDAMAXHEADDIM` = 8192）是**设备**限制而非契约限制，允许两个后端不同，故不进入共享向量表，各套件单独断言各自的边界。

### 10.3 截至 2026-10-12 已实施的整改

1. **秩与预算谓词单点化（G1）。** 新增 `api::checkbudget`/`api::checkgatherbudget`；`metal` 与 `cuda` 的 `executor.rs` 删除各自那份逐字节重复的实现，改为转发。秩超过声明上限的分类由 `executionfailed` 改为 `resourceexhausted`，与 `cpu` 一致——此前同一操作数在三个后端得到两种不同的公共错误。
2. **`matmul` 结果预算（G1）。** 三个后端此前只校验操作数；`m*k <= max` 且 `k*n <= max` 仍可要求 `m*n` 级别的输出分配。现在结果形状在任何分配之前过同一谓词。`cpu` 为此把 `matmulplan` 与 `cpumatmul` 拆开，`cpuerror` 的公开语义未变。
3. **Metal `matmul` shape plan 抽出（G1 第 4 条）。** `matmulshape` 由 executor 移入 `metal/src/geometry.rs`，与 CUDA 侧同形，两侧各自提供 `outshape()`。
4. **共享的无设备契约向量（G1 第 2 条 / PR-A）。** `tests/it/gpucontract.rs` 定义 30 个向量；`metal` 与 `cuda` 两个套件各自断言「与列表一致」与「与 `cpu` 逐项相同」。
5. **Metal 真机数值扩展（G2 / PR-A）。** `metalcontract` 由 30 例增至 40 例：互异维度的非方阵、单元素 matmul、宽度 1000 的 `rmsnorm`/`softmax`、`heads == kvheads` 的 attention、位置 0 的 `rope`、embedding 首尾行、全 `-inf` 行的 softmax 行为。
6. **Metal 全算子基准（G6 的前置基线 / PR-A）。** `benches/metal.rs` 由只测 `matmul` 扩为覆盖九个算子，与 `benches/cuda.rs` 的工作负载与组名一致，使 AGENTS.md 第 7 节「每个算子至少一个可复现基准」对两个后端成立。
7. **Metal 运行时编译缓存（G4 / PR-D）。** 设备句柄与 MSL pipeline 由 executor 拥有的缓存持有（非全局单例，`Arc` 共享给 clone），每个内核只编译一次：九个算子的 median 提升 15–34%，`unsafe` 站点仍为 4/4。公开 `metalexecutor::getcachedpipelinecount` 使「只编译一次」可从 crate 外机械断言。
8. **dispatch 几何（G5 的一半 / PR-D）。** 逐阶段插桩测出每次调用一次命令缓冲往返约 1.7–2.4 ms，且与负载大小和 dispatch 次数无关；据此把 `matmul` 的全部 batch block 合入一次 dispatch（batch 作为网格第三维）。8×256³ 由 13.969 ms 降至 2.878 ms（−79.4%），成为唯一一处 Metal 快于 CPU 的测量点。
9. **文档更正。** `softmax` 的「全下溢行保持全零」不成立（全 `-inf` 行的 `max - max` 是 NaN），已在 `cpu`/`metal`/`cuda` 的文档与 `docs/design.md` 中更正，并由用例钉住。

**证伪性验证（E1）。** ① 把秩分类改回 `executionfailed` → `metal` 的两个向量用例变红，报文指出具体用例名与两侧值；② 删除 `metal` executor 的结果预算校验 → `matmul/outelements` 变红。两处探针随后还原。

### 10.4 端到端代价与下一步（2026-10-12 第三轮）

新增 `benches/metal.rs` 的 `gpu/modelforward`（同一微型 Llama 前向跑 CPU 与 Metal 两个 provider）：**CPU 约 130 µs，Metal 约 50.5–53.5 ms（约 390–414×）**。归因已由四组对照锁定为「每次 `compute` 调用一次命令缓冲往返」（负载大小、dispatch 次数、等待方式、存储模式、是否虚拟化均不影响），前向约 35 次调用 × 约 1.5 ms 与之一致。

这直接决定 G6 的结论：**在内核层面继续优化无法改变该差距**。唯一的一阶杠杆是把多次 dispatch 合入同一命令缓冲，对应 `docs/design.md` §21.13 的**提议**设计（累加式执行上下文：所有权、请求隔离、预算、错误归属、被否决方案、三阶段实施与验收标准）。该设计**尚未实施**，实施前不得改动冻结的 `compute` 语义。

### 10.5 仍未实施

按第 3 节的编号，下列条目在本轮**未实施**，不得视为已完成：G3 的剩余部分（失败路径与生命周期测试）、G5 中显存与传输次数的系统统计（往返成本已测，但逐算子的分配/传输计数尚未系统化）、G7 的结构化 CI 结果、G8 的 attention/KV cache 契约设计、G9 中除本表以外的可维护性条目；PR-C、PR-E、PR-G 未开始。PR-F（性能优化）的前置设计与端到端基线见 10.4。

**已由测量排除的方向**：继续逐个调优 Metal kernel **不会**缩小它与 CPU 在小/中工作负载上的差距，因为该差距由每次调用一次命令缓冲往返（约 1.7–2.4 ms，与负载大小和 dispatch 次数无关）决定。可用的下一步是把多次 dispatch 合入同一命令缓冲，即一个跨调用的执行上下文——这是对同步契约的改动，必须先出设计（见 `docs/design.md` §21.11）。

### 10.6 CUDA 真机验证（2026-10-12，原 E4 已解除）

在一台 RTX 2080 Ti 主机上（环境见 10.1）以 `LLM_CUDA_REQUIRE_DEVICE=1` 执行 CUDA 侧的全部验证，原始日志与传输校验见 `docs/design.md` §21.10.3：

* **契约（E1）。** `cargo test -p llm --test it cudacontract` → 36 passed / 0 failed（18 例宿主侧定案 + 15 例走设备并对照标量参考 + 3 例与共享向量列表一致）；`cargo test --workspace --locked` → 758 passed / 0 failed，0 条 `E4 未执行`；`bash scripts/check.sh` → exit 0。
* **首次执行暴露的缺陷（E1）。** `the_two_matmul_paths_agree_on_a_non_square_case` 以 shape `[7, 4]` 描述右操作数却只给 20 个元素，报 `lengthmismatch { expected: 28, actual: 20 }`。该套件整体 `#[cfg(target_os = "linux")]` 且无设备时提前返回，故自 67ff611 起从未执行；Linux 目标上的编译绿灯掩盖了它。已修正（28 个元素）。
* **基准（E1，不含任何加速声明）。** `cargo bench -p llm --bench cuda --locked -- --measurement-time 2 --warm-up-time 0.5` 的十个工作负载全部产出数字：CUDA 在当时由 cuBLAS/cuDNN 服务的三项（`matmul` 1.88×、`matmulbatch` 1.81×、`softmax` 1.73×）上快于 CPU，其余由 NVRTC 服务的工作负载仍慢于 CPU（`rope` 37.8×、`embedding` 607×）。
* **每调用固定成本的分解（E1，临时探针后已删除）。** NVRTC 服务的算子每调用约 **12.3 ms** 且与负载无关（2×2 `matmul` 与 2 元素 `add` 同为 12.3 ms；2^20 元素只多 3.4 ms）；其中约 1.5 ms 是设备上下文与库句柄，约 **10.8 ms 是 `compile_ptx` 加模块装载**。故 CUDA 侧的 G4 整改（缓存编译产物与设备上下文）收益大于 Metal 同类改动，其次是 G5 的跨调用 dispatch 批处理。两项均**尚未实施**。
