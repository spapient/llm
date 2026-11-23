# 0054：算子/内核覆盖——骨架族泛型 + 目录行表 + 能力表，覆盖对齐 Burn 而边际实现压到一行

- 状态：提案（待采纳；本文件记录取舍与验收标尺，落地进度以代码与门禁为准）
- 日期：2026-11-24
- 相关：[0050](0050-operator-base.md)（`Operator` 基座六族语义）、[0053](0053-typed-operators.md)（清单驱动注册表与类型化 API 面）、[0018](0018-dtype-single-source.md)（精度单一事实源）、[0052](0052-execution-platform-contract.md)（六平台执行合同）、[0005](0005-annotations-as-contracts.md)（注解即契约）、[0013](0013-dense-projection-operator.md)（加算子只加数据）

## 背景

维度「算子 / 内核覆盖」当前快照：llm 4、llm.u 4、Burn 9（口径见 `docs/COMPARISON-llm-llmu-burn.md` §16.3 第 4 行）。三层事实回链当前源码：

- **装配面已收口**：算子清单只声明一次，`REGISTRY` 与类型化 API 面同源展开（ADR 0053）；`Operator` 契约统一名字/形状/精度/轴/可微/代价/样本/参考（ADR 0050）。这一层不再是缺口。
- **覆盖广度有缺口**：数值面集中在逐元素算术、少量激活与两个归约；比较/逻辑/位运算、结构拷贝族、采样算子、进阶归约（mean/prod/var/前缀累积/topk/sort/one_hot）成族缺位。形状视图族已与 Burn 同宽（`tensor/src/tensor.rs` 的 transpose/movedim/permute/broadcast/reshape/narrow/slice/chunks/split/unbind/select/squeeze/diagonal/unsqueeze/flatten/flip），不计入缺口。
- **骨架绑死 f32**：现有 `Unary`/`Binary`/`Reduce` 骨架的缓冲与运算取 `View<'_, f32>`；元素类型没有进骨架泛型参数，int/bool/低比特档要再开平行族。
- **后端能力未表格化**：不支持的「算子 × 元素档」目前靠注册名在构造期拦；没有一张按 `skeleton × dtype × backend` 组织的能力表，缺项只能给泛化 `Unsupported`。

Burn 的覆盖单位是**后端 trait 方法 + 每后端实现**：ADR 0053 已实测其 `FloatTensorOps` 130 条方法、全部 ops trait 424 条、Tensor 面 296 个 `pub fn`；新增一个算子要同时改后端 trait、各后端实现、Tensor 方法与测试。本仓的等价单位应当是**骨架族 + 目录行表一行**，而不是复制这套架构税。

## 决策

### 1. 骨架族泛型到元素档，不按 dtype 开平行族

现有三族骨架（`Unary`/`Binary`/`Reduce`）的行为接口（shape/cost/sample/reference/forward）泛型到元素档 `K`，`f32` 为首实例；输出档由 L1 `dtype` 的 `Class` 谓词决定（比较 → bool，归约 → `K` 或约定累加档）。承接 ADR 0018 的精度闭集，不在骨架里另造类型枚举。

新增五族骨架，接口与现有三族同构：

| 骨架族 | 输入 → 输出 | 承载算子 |
|---|---|---|
| `Compare` | `K, K → bool` | equal / greater / greater_equal / less / less_equal |
| `Logical` | `bool, bool → bool`（int 侧位运算另立档） | not / and / or / xor / bitwise 族 |
| `Structure` | 视图或拷贝，按秩 `R` 泛型 | concat / stack / gather / scatter / take / pad / arange / eye |
| `ReduceDim` | `K, 轴 → K/标量`，按轴泛型 | mean / prod / var / 前缀累积 / argmin / topk / sort / one_hot |
| `Sample` | `Rng, 分布参数 → K` | random_uniform / random_normal / bernoulli / dropout |

每族默认接线形状/代价/样本/参考，算子实现只填核心规则（`apply`、归约折页、结构索引或采样分布）。族 trait 不删（ADR 0050：族是骨架配置的唯一声明处），`Operator` 仍是其上的统一契约。

### 2. 目录行表是算子覆盖的唯一事实源

算子条目只写一次（`prof/src/ops/<name>.toml` + `for_each_operator!` 一行），字段收 `name / stem / skeleton / axes / needs / dtype / group / arity / differentiable`。注册表、catalogue、编译期名字与顺序断言、API 面全部从这张表派生；新增算子 = 一行目录 + 一个骨架实例，不新增平行注册表分支或平行骨架文件。

轴宽、代价表达式、样本分布进目录配置，不写死在算子代码里；`needs` 仍只声明依赖的原语片段（load/store/dot/reduce/cvt/mma）。

### 3. 后端能力表：`skeleton × dtype × backend` 查表

能力登记为一张行表 `Capability { skeleton, dtype, backend }`，`supports(...)` 查表裁决，不写第二张 `match` 分支。不支持时按三维度定位缺项（缺哪一族、哪一档、哪台后端），不返回泛化 `Unsupported`。能力表与后端真实实现由测试逐条对账；后端不进入 `Operator`（L2 不认识 L3/L6 的设备字面量，守 ADR 0050 边界）。

### 4. 注解扩展骨架标签，不造新宏框架

复用 `#[annotation::model(name, shape, dtype, skeleton, axes)]` 与现有生成宏；骨架标签扩展到 §1 五族。编译期生成标记类型、`Operator` 实现、Contract 条目、样本参考与 edge-case 测试模块。沿用 annotation 的 Streams 依赖倒置与 syntax/decl/render 三层可测结构（ADR 0005）。

### 5. 优先路径

- 新算子优先复用原语片段；底层原语不够时才新增片段。GEMM/attention 等大内核保留特化，不收进通用骨架。
- 结构算子优先零拷贝视图（`tensor::layout` 现有代数）；落拷贝只在视图不可表达时发生。
- 逐元素与归约走整块 slice、单次 put、查表批量；仅串行依赖或长度数据相关时保留循环并注释理由。
- 采样复用 `train::rng` 计数器式 PRNG，不另造随机源。

## 验收标尺（10 分可证伪口径）

每个子项必须有可执行的对拍或断言；计划测试不得在落地前记为已通过。

| # | 子项 | 目标 oracle | 反证测试 |
|---|---|---|---|
| S1 | 算术逐元素 | 全套算术 + scalar 版 + neg/sign/abs/recip/square/pow/remainder/clamp，骨架泛型 `K` | 参考实现逐元素对拍；scalar 广播形状正确 |
| S2 | 数学函数 | 三角/双曲/反三角、sqrt/rsqrt/erf/ceil/floor/round/trunc/log1p/exp2 成族 | 边界值（0/负数/Inf/NaN）与参考库对拍 |
| S3 | 归约 | mean/prod/var/median/argmin/前缀累积/dim 归约/topk/sort/argsort/one_hot | 沿轴归约形状与取值；空切片/单元素边界 |
| S4 | 比较/逻辑/位 | 比较全套 + scalar/elem、bool and/or/not/xor、int bitwise 全套 | 输出 bool 档正确；谓词 NaN 口径明确 |
| S5 | 结构拷贝 | concat/stack/gather/scatter/take/pad/arange/eye/nonzero/one_hot | 拷贝前后形状/步长/数据一致；视图零拷贝不被误拷 |
| S6 | 线性代数 | einsum 执行 + dot/batch matmul；高阶 solve/cholesky/svd/det 列 PARTIAL | einsum 公式枚举与参考对拍；矩阵乘形状对齐 |
| S7 | 采样 | dropout/random_uniform/random_normal/bernoulli/categorical 接张量，复用计数器 Rng | 分布统计矩对拍；种子可复现 |
| S8 | 元素类型通道 | 骨架泛型 `K`，f32/f16/bf16/i8/i4/bool 可表示；不支持组合构造期拒 | 档位不匹配构造期报错并指明规则 |
| S9 | 后端能力表 | `skeleton × 档 × 后端` 登记表；缺项返回具体缺哪一维 | 能力表与后端真实实现一致；无静默回退 |
| S10 | 生成式扩展 | 新算子 = 一行目录 + 一个骨架实例；无平行注册表/骨架文件 | 目录名/顺序/NAME 三处断言一致 |
| S11 | 质量对拍 | 每族全量覆盖参考对拍、有限性/形状/边界 oracle | 任一算子参考偏离即门禁失败 |
| S12 | 反向算子 | **PARTIAL**：仅保留 `DIFFERENTIABLE` 标记位，算子级 bwd 族留自动微分维度另案 | 只核验前向与标记位，不签发自动可微 |

达成 10 分的条件：S1–S11 全部落地且对拍通过；S12 必须显式标注为开放决定，不得以可微标记位冒充反向能力。

## 迁移顺序（每族独立可验证，测试数不跌）

1. **P1 骨架通 `K`**：三族骨架泛型化 + 首批算术/数学/归约缺项 + 目录行表字段 + 能力表骨架。
2. **P2 新族落地**：Compare/Logical/Structure/ReduceDim/Sample 五族 + 首批清单（见下）；einsum 执行落地。
3. **P3 收口**：能力表对账、原语片段组合收口、减码实测与对拍台账。

**首批**（骨架族内可直接生成）：neg、abs、sign、recip、square、pow_scalar、remainder、clamp 族；sqrt、rsqrt、ceil、floor、round、trunc、log1p、exp2、cos、sin；mean、prod、argmin、reduce_mean/prod；equal、greater/equal/less 族、logical_not/and/or；concat、stack、gather、scatter、take、pad、arange、eye；random_uniform/normal/bernoulli、dropout。

**次批**：var、median、cumsum/cumprod/cummin/cummax、topk、sort、argsort、one_hot、nonzero；双曲/反三角全族、erf、atan2、hypot、log2/log10；repeat_dim、unfold、triu/tril/diag_mask；dot、batch matmul 区分。

**PARTIAL（不签发首批）**：高阶 linalg（solve/cholesky/svd/det/LU）；conv1d/conv3d/conv_transpose/unfold 与 3d 池化；算子级 bwd 族；跨后端真内核覆盖。

## 后果

- 加算子流程固定为：骨架实例 → 目录一行；注册表、API 面、能力表、参考对拍同时到位。
- 同等覆盖广度下，边际实现压到一个 `apply`/归约/结构/采样规则；不按 dtype 复制平行族，不按后端复制平行内核。
- 净降行数与文件数目标区间：**待实测**。先以现有 nn/tensor/prof 基线跑 jscpd 重复度扫描后定值，不在落地前写未测降幅。

## 替代方案

- **照抄 Burn 后端 trait 方法族**：每算子一条后端方法 + 各后端实现。否决——ADR 0053 已实测该装配面的架构税（后端原语 130 → 1、手写 Tensor 方法 296 → 0）；本仓生成式路线在同等覆盖下装配面应当更薄，而不是反向补齐方法族。
- **每算子一文件手写内核**：延续平行膨胀。否决——与减码目标冲突；只有 GEMM/attention 等特化大内核保留单独文件。
- **本轮铺开算子级 bwd 族**：与自动微分维度耦合，当前证据不足。否决为独立批次；前向骨架接口为反向预留钩子，但不签发自动可微承诺。
- **编译期钉死后端类型**：否决（ADR 0053 已评估）——本仓要同一二进制在设备间降级，编译期钉死后端与之冲突；能力裁决走运行期查表 + 构造期拒绝。

## 未知与未测

- 后端可跑性取决于 Executor/Backend 重建（维度「后端执行成熟度」另案）；本 ADR 只定算子契约、骨架与能力表，不签发跨后端可跑。
- 高阶线代、3d/转置卷积、采样分布完整度的首批范围，待几何骨架复用度实测后判别。
- Burn `burn-linalg` 内部实现深度未逐行读；对照方法面以 `burn-tensor/src/tensor/api` 与 `burn-backend/src/backend/ops` 清单为准。
- 减码区间、tile 宽度、批次尺寸为待校准参数，需给出测量方式后定值。
