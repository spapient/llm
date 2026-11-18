# llm × Burn 全维度对比分析



* 日期：2026-11-18

* 对比对象：本项目 **llm**（前身 `llm.u`，Rust workspace，根 `/Users/fei/code/llm`，取证提交 `c3844c03`）与 **Burn**（Rust 深度学习框架，\~0.22.x 代际）

* 取证口径：


  * llm：当前工作树源码 + `docs/ARCHITECTURE.md` / `README.md` / `docs/REPO-review-2026-11.md` / `docs/code-audit-2026-11-17.md` / `docs/llmu-parity-optimization-design.md`；结论回链 `文件:行` 或文件清单。

  * Burn：官方文档（burn.dev）、crates.io、官方 release blog；行数为本项目 `docs/llmu-parity-optimization-design.md §1.2.1` 自测快照（11-12）与 tokei 口径。

* 证据分级：**\[确认]** 已对当前源码 / 权威来源核验；**\[推断]** 由设计文档 / 路线图推出，未在源码闭合；**\[未证]** 尚无实测数据，不得当结论。



***

## 0. 结论先行



1. **两条路线，不是同质竞争**。Burn 是「泛型后端塔 + CubeCL 编译器 IR + 全产品面」的生产级深度学习框架；llm 是「声明 / 配置驱动 + 单一事实源 + 极致治理」的、以正确性与密度为第一原则的框架。前者换性能与生态，后者换正确性边界与代码密度。

2. **llm 的真实长板（已核验，非目标）**：可执行架构门禁（唯一依赖方向、类型唯一主人）、三轴精度闭集（`DType` 24 + `QType` 25 + `MxType` 7）、零外部运行时依赖、`unsafe` 收敛到单文件单 feature、24 条算子的声明 / 模板 / 标量参考同源注册、反向图显存规划（tape/plan/account/replay）、自研 `.u` 容器（rANS + Merkle + 量化 + CRC）。这些结构性优势在 REPO-review 与 code-audit 中均被定位为 "对标 burn/candle 上游水平"。

3. **Burn 的现实优势（同样已核验）**：7 个真实后端（CUDA/ROCm/Metal/Vulkan/WebGPU/CubeCL-CPU/Flex）、SOTA 矩阵乘内核（对标 cuBLAS/CUTLASS）、JIT 内核融合与自动调优、INT4/INT8 量化落地、分布式训练（burn-collective、可微集合通信）、ONNX/PyTorch 导入、生产可用的训练与推理闭环。

4. **最大的诚实边界**：llm 目前**没有任何实测吞吐 / 时延数字**（设计文档明文禁止在可运行后端落地前回填性能结论）；其代码密度优势**部分来自尚未重建 Burn 已有的后端 /kernel/ 生态代码**。因此 "远超 Burn" 目前是**可证伪目标，不是现状结论**。

5. **裁决点在未来 P1–P4**：llm 补齐 CPU 闭环、真实 kernel、Metal/CUDA 后端、量化计算核之后，真正要回答的问题只有一个 ——**在能力覆盖不低于 Burn 的前提下，密度优势能否保持，且性能对标、治理不退化**。今天能对照的只有结构指标（密度 / 依赖 /unsafe 边界 / 门禁 / 对拍完备度），这些维度 llm 已领先。



***

## 1. 定位与设计哲学



| 维度    | llm                                 | Burn                                     |
| ----- | ----------------------------------- | ---------------------------------------- |
| 首要目标  | 正确性可证明、代码密度、架构可治理                   | 训练 / 推理统一、跨后端性能、生态可用                     |
| 核心手段  | 声明 + 配置驱动；内核由 `.toml` 声明与 `.k` 模板生成 | 泛型后端 trait + CubeCL 编译器；运行时 JIT 融合       |
| 对标座右铭 | "新增算子只加数据不改机制"                      | "一次编写，CUDA/ROCm/Metal/Vulkan/WebGPU 皆可跑" |
| 取舍    | 宁可慢、宁可少，不牺牲单一事实源                    | 宁可复杂，追求 SOTA 性能与广度                       |

**\[确认]** llm README 首行即声明 "配置驱动算子训练推理框架：kernel 由声明与档案驱动"；Burn crates.io 自述 "unifies training and inference……JIT-compiles streams of tensor operations, performing automatic kernel fusion"。

**判断**：Burn 是 "性能优先、生态优先" 的务实框架；llm 是 "治理优先、密度优先" 的工程实验。二者目标函数不同，单点比较 "谁更好" 没有意义，只能逐维比。



***

## 2. 架构形态与模块治理

### 2.1 llm：严格分层 + 可执行门禁



* **\[确认]** L1–L7 七层 + 4 个工具成员（`annotation`/`fuzz`/`benches`/`testkit`），共 24 crate。

* **\[确认]** 唯一依赖规则 ——「依赖只能指向更低的层，同层只能依赖更靠左的成员」—— 由 `crates/testkit/tests/architecture.rs` 对照 `cargo metadata` 与源码**自动执行**，向上依赖、成环在 CI 即失败（`ARCHITECTURE.md §二、§六`）。

* **\[确认]** 所有权表「一个类型只能有一个主人」双向校验：表里必须能在源码找到 `pub` 定义，源码里每个 `pub` 类型必须入表或登记豁免基线（`ARCHITECTURE.md §三`）。

* **\[确认]** 闭集契约 `Closed` + 位集 `Set<E, WORDS>`：`WORDS` 是 const 参数，装不下第 65 个原语是编译错误；加算子 / 原语会先在编译期被拦住。

### 2.2 Burn：泛型后端塔 + 编译器



* **\[确认]** 后端以 `Backend` trait 抽象，模型代码只依赖类型参数 `<B: Backend>`；`Autodiff`、`Fusion` 是**后端装饰器**，可叠加在任意后端之上（crates.io/burn、burn-common）。

* **\[确认]** CubeCL 是独立的 GPU 计算语言与编译器，kernel 用 Rust 写一次，编译到 CUDA/ROCm/Metal/Vulkan/WebGPU（crates.io/burn-tensor）。

* **\[推断]** Burn 模块边界由 crate 划分（burn-core/tensor/autodiff/train/compute/import/onnx/store/collective…），但没有 llm 那种 "依赖方向 + 所有权表 + 闭集" 三套可执行契约门禁。

**判断**：治理强度 llm 明显更高（**\[确认]** 有可执行门禁；Burn 是常规 CI）。这是 llm 最硬的结构长板，也是其 "机制代码 1/15" 密度目标的来源之一。



***

## 3. 精度与量化体系

### 3.1 身份 / 元数据体系



| 轴    | llm                                                                                          | Burn                                                                     |
| ---- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 标量   | `DType` 24 变体（Bool/U1–U64/I1–I64/F4/F8 两种 / F16/BF16/F32/F64/F8E4M3FNUZ/F8E5M2FNUZ）**\[确认]** | `FloatElem`/`IntElem` 等（f16/f32/f64/bf16、i8/i16/i32/i64、u8/u32）**\[推断]** |
| 块量化  | `QType` 25 格式（GGML Q/K-quant/IQ/TQ 四族，块内 scale 前缀独立成 `Scale`）**\[确认]**                       | 无对应 "身份闭集"；量化走运行时 `quantize_*` 算子                                        |
| 微缩放  | `MxType` 7 格式（OCP MX 六标准 + NVFP4），块共享指数 **\[确认]**                                            | 无对应体系                                                                    |
| 实现约束 | `no_std` + 零依赖 + `forbid(unsafe)` + 全部 accessor `const fn` **\[确认]**                         | 常规实现                                                                     |

**\[确认]** `dtype/src/lib.rs` 明言：三轴身份各在一张行表，反量化算法留在 `weights`（ADR 0018），本 crate 只回答 "什么格式、占几字节"。

### 3.2 计算落地



| 项       | llm                                                                         | Burn                                                       |
| ------- | --------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 量化反量化内核 | 声明存在（`dequant_i2/i4/i8`、`gemv_i2/i4/i8`、`gemm_i2/i4/i8`），**无真实后端计算核**，标量参考有 | **\[确认]** INT4/INT8 量化 + 融合反量化已落地（Burn 0.19），2/4/8-bit PTQ |
| 分组推断    | `weights::quant` 从权重文件头推断分组 / 逻辑位宽，**\[确认]** 存在                             | 量化参数由算子显式携带                                                |

**判断**：**身份体系 llm 压倒性更全更严谨**（GGML/MX 三轴是 Burn 没有的）；**计算落地 Burn 完胜**（llm 的量化还停在声明层，`REPO-review §4` 明列为 P3 缺口）。这再次印证：llm 的 "全面" 集中在类型 / 契约层，Burn 的 "全面" 集中在可运行计算层。



***

## 4. 算子与内核

### 4.1 llm：声明 + 模板 + 标量参考同源



* **\[确认]** `op::REGISTRY` 24 条（GEMV/GEMM 各 f32/i2/i4/i8、逐元素、归约、rmsnorm、rope、attention/decode/flash 变体、dequant i2/i4/i8）—— 唯一注册表，名字 / 模板 / 语义表同属一条 `Registration`（`op/src/lib.rs`）。

* **\[确认]** 代价（flops/bytes）由 `op::einsum` 从求和表达式推出，不手写第二份流量公式。

* **\[确认]** 每条算子有独立标量参考（`Model::reference`），内核与之对拍 `max_rel_err` 达标才算过。

* **\[确认]** CPU 侧：生成 C 由 `cc` 编译执行，与标量参考对拍 `max_rel_err = 0`（README 设备表）。

* **\[确认]** **无真实 GPU kernel**：nvidia/intel/metal 仅档案 / 片段 / 见证，`verify` 会失败（缺设备即失败）。

### 4.2 Burn：SOTA 内核 + JIT 融合



* **\[确认]** CubeCL 矩阵乘引擎对标 cuBLAS/CUTLASS，用双缓冲 + tensor core + 向量化，编译到 CUDA/ROCm/WebGPU/Metal/Vulkan（burn.dev blog）。

* **\[确认]** JIT 内核融合：加速后端默认 `Fusion<Backend>`，融合逐元素 / 归约 /matmul。

* **\[确认]** 自动调优：运行时搜索本机最快执行策略。

* **\[确认]** 0.21 起 GEMV/top-k/FFT 内核提速，框架开销降低至多 8×。

**判断**：算子 /kernel 维度 **Burn 完胜且差距是数量级**。llm 的 24 条声明 + 标量参考在 "正确性双源" 上极强，但**没有任何可运行的 GPU kernel、没有融合、没有 autotune**。这是 llm 与 Burn 之间最硬的一条现实差距，也是 REPO-review §0.5 自己点名的瓶颈。



***

## 5. 后端覆盖



| 后端               | llm（当前树）                           | Burn（0.22.x）                                                                 |
| ---------------- | ---------------------------------- | ---------------------------------------------------------------------------- |
| CPU              | **\[确认]** L2 真实（cc 编译 C + 对拍）      | **\[确认]** CubeCL-CPU（MLIR/LLVM JIT、SIMD、autotune）+ Flex（纯 Rust、no\_std/WASM） |
| CUDA             | **\[确认]** L1 仅档案，verify 失败         | **\[确认]** 真实                                                                 |
| Metal            | **\[确认]** L1 仅档案（可编 metallib 嵌入）   | **\[确认]** 真实（原生 MSL）                                                         |
| ROCm/HIP         | —（`prof` 有 ascend/intel 档案，非 ROCm） | **\[确认]** 真实                                                                 |
| Vulkan/WebGPU    | —                                  | **\[确认]** 真实（wgpu）                                                           |
| LibTorch/NdArray | —                                  | 已弃用                                                                          |

**判断**：后端是 llm **回退最大的维度**。Burn 有 7 个真实后端 + 跨后端一致性（一份 kernel 编译到多后端）；llm 只有 CPU 一个真实后端，其余是 "档案 + 片段 + 见证" 的数据形态。llm 的后端路线（`Executor`/`Allocator`/`Collective` trait + 边界 crate）设计上是正确的，但**实现密度为零**（M3 阶段）。



***

## 6. 自动微分

### 6.1 llm：显式反向图 + 显存规划



* **\[确认]** `autograd` 拥有 `Tape`/`Var`/`Node`（记录）、`Schedule`（可达剪枝 + 深度降序 + 同深度分层）、`Account`（扇出账本，梯度收齐才执行）、`Plan`（预算内保留集 / 释放点 / 重算步数）、`Replay`（重算序列表）、`Summary`（图规模只读快照）。

* **\[确认]** 训练循环三对象复用（`Tape::clear`/`plan_into`/`Account::rebuild`），稳定态零分配。

* **\[确认]** `train::backprop` 在编译期断言 "反向公式目录与可微算子集一一对应"（`op::Registration::differentiable`）。

* **\[未证]** 反向公式接真实数值缓冲、gradcheck 端到端 ——`REPO-review §4` 训练侧列为 P1/P2。

### 6.2 Burn：装饰器式透明反向



* **\[确认]** `Autodiff<Backend>` 后端装饰器，`require_grad()` + `backward()` + `grad()`；不支持的（纯推理）后端上根本调不到 `backward`，类型系统兜底（burn-common）。

* **\[确认]** 可与 `Fusion` 组合；计划实现 "基于计算 / 访存 bound 的自动梯度检查点"（crates.io/burn-tensor）。

**判断**：机制上各有千秋 ——llm 的显存计划（预算 / 重放）与可达剪枝在**记录 / 调度层**更显式、更可审计；Burn 的反向对用户更透明、与真实计算已闭环。成熟度 Burn 领先；显存规划理念 llm 不落下风但**未与数值闭环**。



***

## 7. 训练闭环



| 项       | llm                                                                                                                 | Burn                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 优化器     | **\[确认]** SGD/Adam 真实实现（`train/src/optimizer.rs`：权重衰减、梯度裁剪、矩状态、偏差修正、AdamW 解耦衰减）                                     | **\[确认]** Adam/SGD/AdamW 等优化器族                           |
| 学习率调度   | **\[确认]** `Schedule`（warmup + 线性 / 余弦 / 恒定衰减）纯函数                                                                    | **\[确认]** LR 调度器族                                        |
| 训练会话    | **\[确认]** `Session`/`Learner`/`MetricsBoard` + `train` CLI 命令（`--lr/--warmup/--steps/--clip/--accumulation/--load`） | **\[确认]** `burn-train` Learner、指标、终端看板（Ratatui）、优雅中断     |
| 检查点     | **\[确认]** `snapshot` 字节格式：魔数 + 版本、LEB128 差分、CRC-32、原子写；恢复前逐项校验 `Facts`                                              | **\[确认]** 检查点 + 0.22 Burnpack 训练态持久化                     |
| 分布式     | **\[确认]** TP/PP/DP 拓扑 + `Local` 集合通信（真）+ `Remote`（占位，`all_reduce` 返 Unsupported）                                    | **\[确认]** burn-collective 梯度 all-reduce、DDP、可微集合通信（0.21） |
| RL / 蒸馏 | 接口预留，未落地（`offload`/`DeferredOffload` 仅接缝）                                                                           | **\[确认]** 早期 off-policy RL                               |

**判断**：优化器 / 调度 / 检查点 llm 已从 "无" 补到 "有"（当前树，11-18 提交），且检查点格式（差分 LEB128 + CRC + 原子写）比 Burn 的常规序列化更讲究；但**分布式与训练生态 Burn 领先**（burn-collective 是真实跨机，llm 的 Remote 是占位）。



***

## 8. 推理与服务



| 项     | llm                                                                                        | Burn                            |
| ----- | ------------------------------------------------------------------------------------------ | ------------------------------- |
| 连续批   | **\[确认]** `Sequence`/`Scheduler`/`Batch` 簿记 + KV 页表（`Page/Table/Cache/Pager`，引用计数、最后持有者清零） | 非核心定位（Burn 是训练框架，推理可用但无连续批调度引擎） |
| 采样    | **\[确认]** Greedy/Classical (top-k/top-p)/Multinomial，同种子可复现                                | 推理侧非重点                          |
| 端到端前向 | **\[未证]** CPU 前向未闭环（投影 / 归一化首层部分 `Unsupported`，在途 KV 通路）                                   | **\[确认]** 生产推理可用                |
| 服务    | **\[确认]** 端点契约 + 帧 / CRC 线格式齐，**无 socket**                                                 | 无内置服务引擎（定位不同）                   |

**判断**：推理 "调度 / 账本 / 协议" 层 llm 设计超前（连续批 + 分页 KV + 三种采样都是 Burn 不做的），但**端到端前向未闭环、无 socket**；Burn 胜在 "能跑出结果"。此维度 llm 是 "高设计、低落地"，Burn 是 "低设计（非其定位）、可落地"。



***

## 9. 模型存储与导入



| 项     | llm                                                                                                                 | Burn                                                   |
| ----- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 自有格式  | **\[确认]** `.u` 容器 v2：rANS 压缩、Merkle 证明、显式量化、两级 CRC、原子写、REF 去重段                                                      | **\[确认]** Burnpack `.bpk`（0.22）：快速装载、零拷贝、训练态持久化        |
| 外部读取  | **\[确认]** safetensors 直读、GGUF、JSON、pickle/bin                                                                       | **\[确认]** safetensors、PyTorch 互操作                      |
| 零拷贝装载 | **\[确认]** `MmapSource`（feature 门控，Unix）存在，但**生产路径未接**（`weights` 对 `storage` 是 dev-dep，safetensors 走 seek+read 两次拷贝） | **\[确认]** burn-store mmap 零拷贝                          |
| 导入    | 无（不生成代码；模型由权重档案 + 层声明装配）                                                                                            | **\[确认]** burn-onnx（ONNX→原生 Rust）、burn-import（PyTorch） |

**判断**：容器 "讲究程度" llm 更高（rANS + Merkle 证明 + 差分 CRC 是 Burn 没有的）；**生态与互操作 Burn 完胜**（ONNX/PyTorch 导入 + mmap 生产化是 llm 的 M1-A 缺口）。llm 走 "权重档案 + 层声明装配模型，新模型不改框架代码" 的路线，刻意不做模型动物园，这是减量选择，代价是生态覆盖。



***

## 10. 依赖面与安全边界



| 项           | llm                                                                                                      | Burn                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 生产运行时依赖     | **\[确认]** 趋零（锁文件 122 条；`bytemuck`/`serde` 仅 dtype 可选 feature；`toml` 仅 build 期）                           | **\[确认]** 多（serde、wgpu、cubecl、MLIR/LLVM、tokio、ratatui 等） |
| `unsafe`    | **\[确认]** 全仓 `deny(unsafe_code)`；真实 `unsafe` 仅 `storage/src/mmap.rs`（mmap FFI + Send/Sync），全仓唯一 unsafe 岛 | 边界 crate 含 GPU FFI / 编译运行时                               |
| `no_std` 能力 | **\[确认]** `dtype` 无条件 `no_std`，可嵌裸机 /wasm                                                                | Flex 后端支持 no\_std/WASM                                   |

**判断**：依赖与安全 **llm 压倒性领先**，这是可量化、可验证的结构优势（锁文件 122 vs 基线 220，`unsafe` 1 处 vs 散布）。Burn 因引入 wgpu/LLVM 编译器栈，依赖面与 `unsafe` 边界天然更大 —— 这是其 "全后端性能" 路线的必然成本，不是缺陷。



***

## 11. 工程质量与门禁



| 门禁                            | llm                                                                                               | Burn               |
| ----------------------------- | ------------------------------------------------------------------------------------------------- | ------------------ |
| 架构门（依赖方向 / 成员集 / 所有权表）        | **\[确认]** 有，`cargo test` 自动执行                                                                     | 无同类                |
| 闭集契约（唯一声明处 / 位集容量 / 注解↔注册表镜像） | **\[确认]** 有，`testkit/tests/contracts.rs`                                                          | 无同类                |
| 静态质量门                         | **\[确认]** clippy `-D warnings`、`dbg/todo/unimplemented` deny、doc 零警告、ccn≤5 / 函数≤60 行 / 文件≤899 行棘轮 | 常规 clippy/CI       |
| 覆盖率下限                         | **\[确认]** cutile/weights 设下限                                                                      | 有（CodeCov 徽章）      |
| 回退判定                          | **\[确认]** `tool::regress` 基准棘轮（pass/regressed/improved）                                           | 有 benchmark 但无同款棘轮 |

**判断**：工程门禁 **llm 是行业级甚至超前**—— 架构 / 闭集 / 所有权三套可执行契约在 Burn、Candle 乃至多数开源框架中都罕见。这是 llm 最值得被认真对待的资产，也是 REPO-review 判定 "架构与治理对标 burn/candle 属上游水平" 的依据。



***

## 12. 代码规模与密度



| 口径                        | llm（当前树，tokei 实测）             | Burn（本项目 11-12 自测快照）  |
| ------------------------- | ----------------------------- | --------------------- |
| Rust 文件                   | 464                           | —                     |
| Rust 总行数（含注释 / 空行 / 测试）   | \~91k（`wc`）；tokei 78.3k lines | \~488k（fresh `wc` 复核） |
| Rust Code（tokei，去注释 / 空行） | **69,480**                    | **369,340**           |
| 非测试源码                     | \~62k                         | —                     |



* **\[确认]** 按 tokei Code 口径，Burn ≈ **5.3× llm**；按本项目 parity 文档自己的口径（llm 30.3k /burn 369k，11-12 快照）曾达～12×。密度目标（§5.3）是 "同口径机制代码 1/15"。

* **\[诚实边界，未证]** llm 的密度优势**部分来自未重建 Burn 已有的后端 /kernel/ 生态代码**。Burn 的 37 万行 Code 里，相当大比例是 CubeCL 编译器、多后端 kernel、训练 / 导入 / 服务全产品面；llm 目前只把其中 "契约 + 声明 + 治理" 部分做出来，把 "真 kernel + 真后端 + 生态" 留到了 P1–P4。**因此 "代码少 5–15 倍" 不能直接读作 "更优"，只能读作 "更薄 + 治理更省"**，是否成立要等能力补齐后重测。



***

## 13. 成熟度与可用性



| 维度    | llm                                   | Burn                    |
| ----- | ------------------------------------- | ----------------------- |
| 版本    | 0.1.0（workspace.package）              | \~0.22.x（0.22.1 已发布）    |
| 可运行训练 | 部分（优化器 / 调度 / 命令刚落地，端到端 gradcheck 未证） | **\[确认]** 生产可用          |
| 可运行推理 | CPU 前向未闭环                             | **\[确认]** 生产可用          |
| 生态    | 无模型动物园、无导入、无第三方模型                     | ONNX/PyTorch 导入、社区模型、示例 |
| 文档    | 架构 / ADR / 审计文档极密                     | Burn Book 成书 + API 文档   |

**判断**：成熟度 **Burn 全面领先一个数量级**。llm 是 "0.1 版、契约密度极高、可运行面极薄" 的前沿工程；Burn 是 "0.22 版、可训练可推理可部署、生态完整" 的生产框架。这是本次对比里最不能回避、也最不该用 "路线不同" 来稀释的一条。



***

## 14. 全维度裁决矩阵



| #  | 维度                         | llm             | Burn     | 裁决          |
| -- | -------------------------- | --------------- | -------- | ----------- |
| 1  | 架构治理（分层 / 所有权 / 闭集门禁）      | ★★★★★           | ★★★      | **llm**     |
| 2  | 精度身份体系（DType/QType/MxType） | ★★★★★           | ★★       | **llm**     |
| 3  | 量化计算落地                     | ★★（仅声明）         | ★★★★     | **Burn**    |
| 4  | 算子 /kernel 性能              | ★★              | ★★★★★    | **Burn**    |
| 5  | 后端覆盖                       | ★（仅 CPU 真）      | ★★★★★    | **Burn**    |
| 6  | 自动微分（记录 / 规划 vs 落地）        | ★★★★（机制）        | ★★★★（成熟） | 平           |
| 7  | 训练闭环                       | ★★★             | ★★★★     | Burn（广度）    |
| 8  | 推理调度 / 采样设计                | ★★★★（设计）        | ★★（非定位）  | **llm**（设计） |
| 9  | 模型存储讲究度                    | ★★★★            | ★★★      | **llm**     |
| 10 | 模型导入 / 生态                  | ★               | ★★★★★    | **Burn**    |
| 11 | 依赖面 / 安全边界                 | ★★★★★           | ★★       | **llm**     |
| 12 | 工程门禁                       | ★★★★★           | ★★★      | **llm**     |
| 13 | 代码密度                       | ★★★★（含 "未做" 成分） | ★★★      | llm（有条件）    |
| 14 | 成熟度 / 可用性                  | ★★              | ★★★★★    | **Burn**    |
| 15 | 性能实测                       | 无（未证）           | ★★★★★    | **Burn**    |

汇总：llm 胜出维度集中在**治理、精度身份、密度、安全、存储讲究、推理设计**（即 "正确性与工程结构"）；Burn 胜出维度集中在**后端、kernel、量化落地、生态、成熟度、性能**（即 "可运行与广覆盖"）。



***

## 15. 结论与追赶路径

### 15.1 一句话裁决

**llm 是一个 "架构与治理已到上游甚至超前、但运行时实现密度尚薄" 的前沿框架；Burn 是一个 "全栈可用、性能 SOTA、生态成熟" 的生产框架。二者当前不在同一成熟度档位：Burn 的领先是 "能跑、能训、能部署" 的现实领先，llm 的领先是 "类型系统、正确性边界、依赖安全、代码密度" 的工程领先。**

### 15.2 可证伪的追赶判据（不给确定性 "领先 N 倍" 结论）

llm 若要在运行时维度追上甚至超越 Burn，必须依次满足（全部可验证、可证伪）：



1. **P1 CPU 闭环**：`gemv_f32`/`softmax_f32`/`rmsnorm_f32`/`rope_f32` 真实 kernel + 一层 / 整模型前向对拍达标，采出第一个真实 decode token。

2. **P2 推理运行时**：KV 真实数据 + 连续批 + 注意力 decode 变体，同机对照 llama.cpp/Burn 同档。

3. **P3 多后端 + 量化**：Metal/CUDA 边界 crate 挂 `Executor`/`Allocator`，`i4/i8` 反量化计算核落地，同后端对照 Burn/Candle。

4. **P4 模型 + 服务**：真实权重端到端生成 + HTTP 端点，对照 vLLM/TRT-LLM。

**四个阶段每过一关，才允许在该关的维度上回填实测数字；在此之前，任何 "领先 Burn" 的表述都只允许限定在结构指标（密度 / 依赖 /unsafe/ 门禁 / 对拍完备度）上。** 这正是本项目 `llmu-parity-optimization-design.md §8.3` 已经自我约束的口径。

### 15.3 最需要警惕的一条

llm 最大的风险不是 "比 Burn 慢"，而是**把 "代码更少" 误当成 "更强"**。代码密度是真实优势，但其中混入了 "尚未重建 Burn 已具备的后端 /kernel/ 生态代码" 这个尚未支付的账单。真正的压力测试是：**当 llm 补齐与 Burn 同等的后端矩阵与 kernel 矩阵后，那 1/15 的密度优势还剩多少、性能能否对标、治理是否仍在门禁内。** 今天能说的只有 —— 它的治理、精度、安全、密度这四个结构性长板是**已经兑现**的，而性能与覆盖是**尚未开始计分**的。



***

## 附：证据来源

### llm（本地源码，提交 `c3844c03`）



* `README.md`、`docs/ARCHITECTURE.md`（分层 / 依赖规则 / 所有权表 / 门禁）

* `docs/REPO-review-2026-11.md`（现状与精简空间）、`docs/code-audit-2026-11-17.md`（实测基线与缺口）

* `docs/llmu-parity-optimization-design.md`（对标 Burn 的路线与密度口径）

* `crates/dtype/src/lib.rs`（精度三轴）、`crates/op/src/lib.rs`（24 条注册表）、`crates/autograd/src/lib.rs`（反向图）、`crates/train/src/lib.rs` + `crates/train/src/optimizer.rs`（训练与优化器）、`crates/cli/src/lib.rs`（命令面）

### Burn（官方来源）



* [https://burn.dev/docs/burn/](https://burn.dev/docs/burn/) （后端 / Device / 装饰器）

* [https://crates.io/crates/burn](https://crates.io/crates/burn) 、[https://crates.io/crates/burn-tensor](https://crates.io/crates/burn-tensor) 、[https://crates.io/crates/burn-common](https://crates.io/crates/burn-common) （定位 / 后端矩阵 / 融合 / 自动微分）

* [https://burn.dev/blog/release-0.19.0/](https://burn.dev/blog/release-0.19.0/) （量化、分布式、LLVM CPU 后端）

* [https://burn.dev/blog/](https://burn.dev/blog/) （0.21：8× 开销、burn-dispatch、SOTA matmul）

* [https://burn.dev/books/burn/saving-and-loading.html](https://burn.dev/books/burn/saving-and-loading.html) 、[https://burn.dev/books/burn/print.html](https://burn.dev/books/burn/print.html) （Burnpack .bpk、0.22 迁移）

* [https://docs.rs/burn-store/0.22.1/burn\_store/](https://docs.rs/burn-store/0.22.1/burn_store/) 、[https://rust4ai.com/projects/burn](https://rust4ai.com/projects/burn) （burn-store、ONNX 导入、量化、看板）