# 模型权重传输容器：最优论证与演进方案

> 版本 v9 · 2026-11-12 · 进度：P1–P6 全部完成；ML-DSA 零依赖自研 D1–D3 已完成并对拍，D4 算法地基（high/low）已完成，三流程进行中。
>
> **归属更新（v5）**：经架构评审，独立 crate `pack` 已撤销，容器实现整体并入 `weights`，各模块
> 直接平铺在 `crates/weights/src/` 根、不设子目录；容器显式仿射量化模块命名为 `affine.rs`
> （避开外部 `quant.rs` 同名，类型名仍 `Quant`）。下文出现的 `pack` 一律指 `weights` 内这个自有
> LLPK 容器；格式名 LLPK、魔数、字节布局与全部能力不变。外部格式适配（safetensors/gguf/legacy）
> 同样以平级文件平铺在根。

## 0. 结论先行

本项目（`/Users/fei/code/llm`）是 llm.u（`/Users/fei/code/llm.u`）的升级版。
旧版 `storage`（10343 行，容器+压缩+IO+账本的大模块）是前身，不是外部参照。
升级版用更细的分层重建同一套能力：**`storage` 只做物理布局，权重容器与权重结构统一收敛到
`weights`（自有 LLPK 容器在其 `llpk` 子模块），检查点归 `checkpoint`**。

"各方面强 10/100 倍、指数级"需要诚实拆分：

- **工程维度可证明数量级领先**：依赖数（0 vs 8）、unsafe 边界、模块可维护性、强类型 API；
- **已触物理下界的维度只承诺"不劣于"**：无损压缩率受香农熵约束、CRC 已是硬件单条指令、
  mmap/原子写是 OS 语义。在这些维度写"指数级"是欺骗，本方案不写。

## 1. 升级版硬约束（一切设计的前提）

| 约束 | 取值 | 后果 |
|---|---|---|
| 第三方依赖 | 全仓 0 个 | 旧版的 rayon/memmap2/zerocopy/serde/faer/fips204/sha2 必须自研或裁剪 |
| unsafe | 全局 `deny`，mmap 隔离到 storage 内 feature-gated 模块 | 主容器内存安全可证明 |
| edition / rust | 2024 / 1.100 | 可用闭集、`#[diagnostic]`、新 trait 语义 |
| 错误类型 | 全仓唯一 `diagnostic::{Error,Code}` | 不允许 `Result<_,String>` |
| 分层 | 依赖只能指向更低层，同层只能依赖更靠左成员 | 环在编译期不可能 |

## 2. 旧版 storage 能力 → 升级版 crate 映射

| 旧版模块 | 能力 | 升级版承接 | 迁移决策 |
|---|---|---|---|
| `format.rs` | v5 字节布局、尾部表区、量化 7 字节 | `pack` | 借鉴尾部表区单遍流式；P1 已实现自有布局 |
| `store.rs` | 单遍流式写、去重、mmap、原子 rename | `pack` + 后续 IO 层 | 吸收原子写/流式；mmap 收敛为 storage 的 feature-gated 模块 |
| `pack.rs` | 块 codec 闭集、指数位聚集 | `pack`（后续阶段） | 压缩阶段吸收，codec 枚举表驱动 |
| `codec.rs` | rANS 8-lane 熵编码 | 待 P2 决策（pack 内或独立 crate） | 自研，零依赖重写 |
| `tans.rs` | FSE 风格 tANS | 同 P2 | 与 rANS 二选一或并存，按实测 |
| `plane.rs` | 字节平面 Kernels（Scalar/Cpu/ISA） | 压缩支撑层 | 标量路径优先，SIMD 隔离 |
| `cert.rs` | 位级无损账本 sidecar | `pack` 或独立账本 | 吸收"落盘原因/逐位一致"记账 |
| `merkle.rs` | 逐张量 Merkle 证明 | 完整性阶段 | 自研 SHA-256，域分离 |
| `sign.rs` | ML-DSA 后量子签名 | 完整性阶段（需自研 fips204 替代） | 工作量大，优先级靠后 |
| `lattice.rs` | D24/E8/3×E8 格量化、rank/unrank 定长索引 | `pack` 量化子模块 | **迁移并增强**：解决旧版 scale 开销（跨块共享/稀疏 scale），并向真 Leech Λ24 推进 |
| `axes.rs` | 五轴自适应无损候选（raw/dict/sparse/quant/rank） | `pack` 编码选择层 | **迁移**：保留逐位验证才采纳的硬约束，作为每张量自动选码框架 |
| `ledger.rs` | 谱隙证书账本、f64 SVD、六项不变量 | `pack` 证书层 + 自研 SVD | **迁移并去依赖**：faer SVD 自研为零依赖数学内核，serde 换强类型编解码 |
| `patch.rs` | 逐张量 XOR 增量补丁 | `checkpoint` | 与 LLMC 增量能力合并评估 |
| `mode.rs` | 训练/发布双模式 | `pack` 元数据键 | 吸收，不新增二进制机制 |
| `gguf.rs` | GGUF v3 只读+反量化 | `weights` | 外部互操作阶段 |
| `safetensors.rs` | HF 导入器 | `weights` | 外部互操作阶段 |
| `ir.rs` | 闭集算子图拓扑（含 SSM 系列）、融合提示 | `pack` 图块（保留名 llm.graph） | **迁移**：复用容器块表/CRC，支撑静态融合 |
| `legacy.rs` | 旧版只读兼容 | 无对应 | 不迁移：新版无历史包袱 |

## 3. 分维度最优论证表

| 维度 | 最强参照 | 升级版目标方案 | 承诺级别 | 可执行验证 |
|---|---|---|---|---|
| 供应链/依赖 | 旧版 8 个第三方 crate | 0 依赖 | **数量级领先** | `cargo tree` 计数；`cargo vendor` 体积对比 |
| 内存安全 | 旧版 storage 内含 mmap unsafe | 主容器 deny unsafe，mmap feature-gated | **数量级领先** | 编译期 lint；unsafe 行数统计隔离 |
| 模块可维护性 | 旧版单文件 700–800 行 | 单文件 ≤250 行、单一所有权、小公开 API | **数量级领先** | 行数/公开项计数；架构测试守门 |
| 类型/编解码 | 旧版 serde 派生 + JSON | 手工强类型编解码，零漂移 | 领先 | 无 serde 事实源；canonical 双射测试 |
| 全精度类型 | 旧版 10 种 Prec | 全精度 wire 21 种（INT1–FP64/F8） | 不劣于且更广 | 全类型逐字节往返 |
| 量化表达 | 旧版 7 种 Pack | 显式 Quant 描述符（axis/block/zero/scale） | 不劣于 | 量化往返 + 布局校验 |
| 对齐/零拷贝 | 旧版 32B 对齐 | 2 的幂对齐（默认 64B，填充必为 0） | 不劣于 | 对齐断言 + 填充校验 |
| 完整性 | 旧版 CRC32C + Merkle + ML-DSA | 两级 CRC（P1）→ Merkle（P3） | 分阶段不劣于 | 损坏注入、比特翻转/截断拦截 |
| 无损压缩率 | 旧版 rANS 逼近熵 | 自研 rANS/tans | 不劣于 | 同数据集对拍，压缩率/熵差 ≤ 参照 |
| 持久化 | 旧版临时文件+fsync+rename | 吸收原子写 | 不劣于 | 崩溃注入，无半截文件 |
| 随机访问/去重 | 旧版块表+全局去重 | 块表 + 预算内去重 | 不劣于 | 按名二分查；重复张量复用对账 |
| 格向量量化 | 旧版 D24/E8/3×E8 | 迁移并解决 scale 开销，向真 Leech 推进 | 不劣于且增强 | E8 壳计数对拍；MSE 对拍；rank/unrank 双射 |
| 自适应选码 | 旧版五轴逐位验证 | 五轴候选框架，逐位验证不可绕过 | 不劣于 | to_bits 全等才采纳；非网格/不精确候选必拒 |
| 谱隙证书 | 旧版 faer SVD + 六项不变量 | 自研 f64 SVD + 强类型账本 | 去依赖且不劣于 | 奇异值对拍；γ_r>2ε 严格；不变量校验 |
| 图拓扑/融合 | 旧版闭集算子图块 | llm.graph 保留块，复用块表/CRC | 不劣于 | 图块往返；闭集未知码拒绝 |

## 4. 分阶段落地与门禁

每阶段门不过不进下一阶段（沿用本项目阶段化交付约定）。

- **P1 容器核心（已完成）**：固定头 + LEB 描述符 + 对齐 + 两级 CRC + 全精度 wire + 显式 Quant。
  门禁：14 单测 + 5 集成 + 1 doc 全过，clippy `--all-targets` 零警告。
- **P2 块压缩（已完成 rANS）**：自研零依赖 rANS 8-lane，codec 闭集表驱动（Store/Rans），
  严格变小才保留否则回退 raw。门禁已过：25 单测 + 5 集成 + 1 doc，clippy 零警告，
  熵对拍证明核心效率 ≥99%、头部开销为有界常数、均匀数据不放大。
  - **tANS 裁决（同维度对比，非因创新而砍）**：tANS 与 rANS 共用同一归一化频率表，
    压缩率无差异；tANS 单状态位级串行、难以 SIMD 向量化（旧版须 rayon 检查点切块，
    ≥256KB 才回本），而 rANS 8-lane 在现代 CPU 自动向量化；表构造更复杂、维护面更大。
    故在有 SIMD 的目标平台 rANS 已更优，tANS 暂不实现。**它何时更优已明确**：
    无 SIMD 的窄目标/嵌入式解码需要单表无乘法路径时，再以"同数据集 rans/tans 解码
    周期对拍、tans 胜出"为可执行门禁启用，而非凭印象加入。
- **P3 完整性与持久化（已完成）**：零依赖 SHA-256（FIPS 180-4 标准向量）、逐张量 Merkle
  证明（域前缀 0x00/0x01、名字绑定、O(log n) 离线验证）、Sink 闭集（Count/Vec/Slice/File/
  Measured/Sealed，算长度与写出同路径零漂移）、原子写（同目录临时文件→fsync→rename，失败清
  临时文件）、只读内存映射收敛为 `llm-storage` 的 `mmap` feature-gated 模块（全仓唯一 unsafe 岛，
  `PROT_READ|MAP_PRIVATE`，Drop 自动 munmap；默认不编译，storage 与 weights 主 crate 保持 deny
  unsafe）。容器随后整体并入 `weights`，模块平铺到根（容器量化模块为 `affine.rs`）。门禁证据：
  weights 85 单测 + 8 集成 + 5 doc 全过，
  storage 含 mmap 模块 8 测试全过，workspace 全量测试零失败，clippy 全 workspace 零警告，
  架构/契约守卫通过。
- **P4 图拓扑与格量化（已完成）**：
  - `ir.rs`：Op 闭集 8 个（码位稳定），LEB128 紧凑编码；每条输入边必须指向更早节点，
    一次性保证无自引用/无环/引用合法；`to_segment()` 经 Writer 落盘。
  - `lattice.rs`：D24、E8（D8∪D8+½ 两陪集，rank/unrank 定长索引覆盖全部格点）、3×E8
    直积；旧版 `assert` 全部改为 `Result`。E8 壳计数对拍 240/2160/6720/17520 全过；
    同 bits 下 3×E8 往返 MSE 严格低于 D24。
  - `scale.rs`：攻克旧版"生产不启用"根因——对数域定点把 scale 区从 `4n` 字节压到 `8+n`，
    端到端 MSE 对拍不劣化超过 5%，体积严格变小。
  - `golay.rs`：扩展二元 Golay G24 `[24,12,8]`，B 对称且 B²=I，26 候选最近译码纠正 ≤3 错误，
    最小距离 8 经全 4096 码字验证。这是向真 Leech Λ24 的码层基础，未宣称达到 Λ24 密度。
  - 门禁证据：weights 124 测试全过（ir 7 + lattice 11 + scale 9 + golay 5 等），clippy 零警告，
    全 workspace 测试零失败，架构/契约守卫各 6 项通过。
- **P5 谱隙证书体系与外部格式（已完成，GGUF/bin 应要求提前并入本阶段）**：
  - `svd.rs`：自研单边 Jacobi SVD，不先形成 AᵀA 以保住小奇异值，重构残差机器精度量级；
  - `ledger.rs`：收敛 60+ serde 占位字段为数学必要闭集，LEB128 强类型编解码，谱隙 γ_r>2ε 严格判定，
    Eckart-Young 秩截断，四项核心约束 `check`；
  - `axes.rs`：五轴 raw/dict/sparse/quant/rank，每候选 decode 后逐位相等才采纳，取字节最小，
    熵编码带 store 回退（严格变小才 rans，修复小数据头部放大）；
  - `gguf.rs`：GGUF v3 只读，六类 GGML 类型（F32/F16/Q8_0/Q4_K/Q5_K/Q6_K）反量化；
  - `zip.rs` + `bin.rs`：零依赖 ZIP_STORED 归档读取 + 栈式 pickle 虚拟机，按后缀定位（顶层目录名随文件名变化），
    拦截 `_rebuild_tensor_v2` 与 BINPERSID，九种 storage 类型，原始字节零拷贝。
  - 门禁证据：weights 152 测试全过、clippy 零警告、全 workspace 测试零失败、架构/契约守卫各 6 项通过；
    **真机对拍**：本机安装 torch 2.9.1，2 个固定文件 + 20 个随机 fuzz 文件共 100 个张量，
    解析字节与 `torch.load` 逐字节一致。
- **P6 双模式与增量补丁（已完成）**：
  - `mode.rs`：不新增二进制机制，模式写在保留名段 `llm.mode`，`opt.` 命名空间为优化器状态，
    `release` 一次过滤剥离训练状态；无标记的旧文件按推理模式，向后兼容。
  - `patch.rs`：逐字节 XOR 自逆，**零额外标记**——base 有同名等长段就 XOR、否则全量，diff/join
    同构共用一个 `rebuild`；XOR 大量零再经 rans 压到很小。
  - 精简与门禁：diff/join 提取公共路径后各为一行调用，新增函数全部满足静态门 ccn≤5；
    集成测试证明 8192 元素仅改 3 值时补丁小于全量一半且 join 逐字节还原。
  - ML-DSA 自研替代仍按工作量单独评估，未在本阶段展开。
  GGUF/safetensors/bin 三类外部格式已在 P5 提前落地。

## 4b. ML-DSA 零依赖自研（FIPS 204，分阶段门不过不进）

旧版 `sign.rs` 直接依赖第三方 `fips204`。自研不手抄常数、不引入依赖，按可对拍的地基逐层推进：

- **D1 Keccak-f[1600] 与 SHAKE（已完成）**：纯安全 u64 实现 24 轮置换与海绵，SHAKE128/256
  可变长输出；空输入标准向量与 Python `hashlib` 逐字节一致，跨块长输出确定。
- **D2 多项式环与 NTT（已完成）**：R_q（n=256, q=8380417, ζ=1753），根表编译期生成；
  关键裁决——NTT 输出是 256 个独立求值点，点积为**逐元素相乘**（非 Montgomery、无 γ 配对），
  经朴素环卷积（含 x^256=-1 回绕）对拍一致，且比配对方案代码更少。
- **D3 采样与位打包（已完成）**：ExpandA/ExpandS/ExpandMask、SampleInBall，通用 bit pack/unpack
  （simple 是 a=0 特例，不平行实现）与 hint pack/unpack；与 Python 严格参考逐值对拍一致。
- **D4 keyGen/sign/verify（已完成）**：三流程编译干净，端到端正确签名通过、篡改拦截。
  排查中取证确认两类真实缺陷并修复：①**系数表示约定**——常规域统一中心表示、NTT 域保持非负，
  转换只在桥梁处（`center`/`inv_center`），Power2Round 要求非负输入故 keyGen 的 t 保持非负；
  ②**verify 漏 inv_ntt**——w_approx 在 NTT 域算完后必须逆变换回常规域才能 UseHint。
  另修复 ExpandMask 位宽（应为 BitLen(a+b)=20、申请 640 字节，原少 1 致末尾系数静默为 0）。
  零依赖纯 Python 完整参考（仅 hashlib）与 Rust 实现 pk/mu/sig **逐字节对拍**通过，等价 KAT。
  公开 hint_pack/unpack 收敛为动态核心的薄封装，消除平行实现。
- **D5 参数集 44/87 与 Envelope（已完成）**：新增 `ML_DSA_44/87` 两常量，复用参数化三流程。
  修复真实缺陷——`w1_encode` 原硬编码 4 位（仅 m=16 的 65/87 成立），44 集高位箱数 m=44 需 6 位，
  改为按 `bit_len(m−1)` 动态宽度，与权威 `W1_LEN` 一致。三参数集与纯 Python 参考逐字节对拍通过。
  新建 `envelope.rs`：魔数 LLEN、参数集标识、公钥/签名 LEB 长度前缀、末尾多余字节即拒；
  被签消息遵循 FIPS 204 外部消息格式（固定 ctx 域分离），`check` 全程 fail-closed，
  真实容器端到端：写端签名、读端重算根验签、篡改（关闭 CRC 后）由签名层拦截。

## 4.5 架构再精简（专家审查）

以 Rust 专家深度做去重与单一来源，只改经论证的真实问题，不为改而改：

- `bin.rs`：NEWTRUE/NEWFALSE 两臂同体，合并为 `|` 模式；
- `writer.rs`：`check_align` 不使用 self，降为关联函数，三处调用同步；
- `dsa.rs`：t1 上界的魔法表达式（出现两次）收敛为 `Params::t1_bound()` 单一来源；
- `dsa.rs`：`bits_vec(.., take_high: bool)` 双职责拆为 `high_bits`/`low_bits` 两个语义函数，
  行数反而更少、调用处自解释；
- cast 专项：逐处确认 42 处 `as` 截断均为有界位运算（取低字节、索引扩展、rem_euclid 后 <2^23），
  无真实丢数据 bug。

经论证**保留**的设计（避免劣化）：const 单态化 + 运行时三维 match 分派是编译期维度检查与
栈分配的合理代价，改运行时维度会损失性能与类型安全；`Entry/Header`（外部格式元数据层）与
`Segment`（LLPK 容器段，可承载 graph/mode 等非张量保留段）语义不同，不合并。

## 5. 明确不做

- 不在压缩率/CRC/mmap/原子写维度承诺"指数级"——物理下界决定，只做对拍不劣于；
- 不因"研究性质"标签放弃任何创新能力：格量化、五轴选码、谱隙账本、图拓扑全部迁移，
  并在零依赖约束下把旧版的第三方依赖（faer/serde）自研替代；
- 论证表未通过、验证手段不可执行前，不写对应实现代码。
