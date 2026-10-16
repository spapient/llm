# β 通道落地计划：结合律半群扫描算子（SSM 递推）

> **版本**：v1.3（2026-10-16）　**状态**：现状核对版——scan 算子三后端已实现（见 §0），本计划聚焦剩余缺口
> **v1.3 变更**：**实测核对（2026-10-16，完全访问环境）**——`api::compute` 已有 `scan`/`scan_cached`（`kernels.rs:283/307`，HEAD 已提交）；CPU（`crates/cpu/src/scan.rs`，串行权威 + 并行容差路径）、Metal（executor + MSL，metalcontract 48/48 全绿）、CUDA（executor:378/390）三后端均已实现；测试齐备（`scancontract.rs` 参考 64 点半群前缀/空序列/非 2 幂、`scanunits.rs` 逐位一致 + cached≡全量、apikernelcontract、metalcontract 谱违规拒绝）；design.md §5 契约表已登记（351-352 行）。**剩余缺口 = 基准（benches/ 无 scan 目标）+ M1 判定实测 + GPU 并行扫描核对 + 模型挂载（nn SSM 层 + registry 变体）**。B0–B4 由「从零实现」改为「已交付」；B5/B6 保留。
> **v1.2 变更**：撤回 v1.1 的「拆 `scankernel` trait」——核对 `lib.rs` 与 conformance.md 后，A4 上界 20 的权威口径是 **crate root 导出项数**（当前 18/20），拆 trait 反而 +1 导出项；design.md §5.2 决策 5 的既定路线就是「累加扩展 `compute`」。`scan`/`scan_cached` 作为 `compute` 新方法（11 → 13）。另将谱前置的错误改由 **B0 新增 `backenderror::invalidparameter`**（现有 7 variant 无「参数值非法」类）。其余 v1.1 修正保留：并行容差对齐 §5.10、`A` 限定对角/块对角并补谱校验、离散化边界、SSM 层归属（`nn`）、M1 测量协议（闭集硬件 + 禁 Table 7）、直通项 `skip`、`scan_cached` 输出 `[1,d]`。
> **数学权威**：`docs/lld.md` §5.6（结合律扫描，数学验证 5a/5b）、§1.1 C2、M1/M3/M4；实现契约权威：`docs/design.md` §5（层 0：api，累加式扩展）
> **目标**：让 β 通道（选择性 SSM / 扫描递推）从「算子已实现、验收未闭合」变为「基准齐备、M1 可验收、模型可挂载」。自研 `arch: trap` 底座与 Mamba 类模型由此可挂上 $\mathcal{T}_\theta$。

---

## 0. 现状核对（2026-10-16 实测，完全访问环境）

> 本节为 v1.3 新增，基于本地源码核对 + metalcontract 实测，取代此前"未实现"假设。

| 项 | 实测状态 | 证据 |
| :--- | :--- | :--- |
| `api::compute::scan`/`scan_cached` | ✅ **已提交** | `crates/api/src/kernels.rs:283/307`（签名含 `skip` 直通项） |
| CPU | ✅ 已实现 | `crates/cpu/src/scan.rs`（串行权威 + 并行容差路径）+ executor:369/387 |
| Metal | ✅ 已实现 + 数值通过 | `metalcontract` 48/48 全绿，含 `scan_matches_the_scalar_reference`、`scan_rejects_a_spectral_violation` |
| CUDA | ✅ 已注册 | `crates/cuda/src/executor.rs:378/390`（数值用例在 GPU 真机套件） |
| 测试 | ✅ 齐备 | `tests/it/scancontract.rs`（64 点半群前缀/空序列/非 2 幂）、`scanunits.rs`（逐位一致、cached≡全量）、`apikernelcontract.rs:291/303` |
| design.md §5 契约表 | ✅ 已登记 | `docs/design.md:351-352`（`scan`/`scan_cached` 形状规则） |
| **基准** | ❌ **缺失** | `benches/` 无 scan 目标（违反 AGENTS.md §7「每个算子至少一个可复现基准」） |
| **M1 判定** | ⚠️ **未实测** | lld.md M1 仍 OPEN（理论证明 + 实测 log-log 未出数） |
| 模型挂载 | ❌ 未做 | `registry::architecture` 无 Mamba 变体；`nn` 无 SSM 组合层 |
| GPU 并行扫描 | ⚠️ 待核对 | CPU 有并行路径；Metal/CUDA 内核是否 work-efficient 并行扫描需源码核对 |

**结论**：B0（契约+参考）、B1（CPU 串行）、B2（CPU 并行）、B3（Metal）、B4（CUDA）**已交付**；本计划实际剩余工作 = **B6 基准 + M1 判定 + 并行核对 + B5 模型挂载**。

---

## 1. 背景与动机（为什么现在做）

| 锚点 | 现状 | 结论 |
| :--- | :--- | :--- |
| **M1**（序列计算渐近复杂度） | `未实现并行扫描 ⇒ M1 OPEN，阈值 O(nd) 不降` | β 扫描是 M1 唯一验收路径（π_t=0 域） |
| **§1.1 C2**（长上下文） | 选择性 SSM 状态 $O(dN)$；半群扫描 $W=O(n)$、$T_\infty=O(\log n)$ | 长上下文双轨制的线性通道侧 |
| **§1.2 决策表** | α/β/γ 三通道门控 | β 通道是 $\mathcal{T}_\theta$ 的扫描通道，契约在 §1.5 已冻结 |
| **§1.7 SKU** | 产品默认离散 = **指数梯形半群**（`exponential_trapezoid`） | 自研模型底座本身就是这个半群的实例 |
| **Mamba-3 自验 SKU** | 钉死 `state-spaces/mamba3-mimo-1.5b` | 外部模型挂载的验收目标 |

**一句话**：不是"要实现 Mamba"，而是**把结合律半群扫描做成 $\mathcal{T}_\theta$ 的一等算子**——自研 trap 模型、Mamba 类模型、任何可写为线性递推的模型都经同一个算子挂载。扫描算子落地后，模型层 = `nn` 新增一个 SSM 组合层（$\Delta_t/A/B/C/D$ 投影 + 离散化） + `registry::architecture` 变体 + 参数绑定；scan 算子本身对任何可写为线性递推的模型通用。

## 2. 目标与成功标准

1. **算子落地**：`api::compute` 累加式新增 `scan`（全量）+ `scan_cached`（单步 decode）两个方法，三后端（CPU/Metal/CUDA）数值一致。
2. **并行扫描**：work-efficient 并行前缀（分段三阶段为主、Blelloch 参照，数学验证 5b 已验 64 点），M1 理论 + 实测双门禁可出数。
3. **模型挂载能力**：`registry::architecture` 新增变体的完整路径演示（Mamba 配置 + 自研 trap 配置同算子）。
4. **文档同步**：design.md §5 契约表、lld.md M1 状态、conformance.md 判定行、checkunsafe.sh 登记表。

**成功 = M1 从 OPEN 翻「已验证（β-only 域）」，且 scan 数值三后端一致进 CI。**

## 3. 数学设计（对齐 lld.md §5.6，证据 I/II）

### 3.1 选择性 SSM 递推（离散化，产品默认 = 指数梯形）

$$
h_t = \bar A_t\, h_{t-1} + \bar B_t\, x_t, \qquad
y_t = \bar C_t\, h_t + \bar D_t\, x_t
$$

- 每通道独立：$h_t\in\mathbb R^{d\times N}$（$d$ 通道、$N$ 状态宽），$\bar A,\bar B,\bar C\in\mathbb R^{d\times N}$，$\bar D\in\mathbb R^{d}$（或标量，本算子固定为向量，标量分支收缩）
- $\bar A_t$ 由 $\Delta_t$（时间步长）与连续核 $A$ 经指数梯形离散化得到——**离散是固定语义，`scan_mode` 只换执行路径，不换离散**（§5.6）。**离散化的归属**：$\exp(\Delta_t A)$ 在 `scan` 算子**之外**、由模型层的 SSM 组合层完成；`scan` 只消费已离散的 $\bar A$，不接收 $\Delta_t$ 或连续核 $A$。

### 3.2 半群重写（结合律，数学验证 5a）

定义累积对 $S_t=(\bar A_t,\ \bar B_t x_t)$，半群运算：

$$
S_1 \oplus S_2 = \big(\bar A_2\bar A_1,\ \bar A_2(\bar B_1 x_1) + \bar B_2 x_2\big)
$$

结合律（数学验证 5a）⇒ 前缀扫描可**并行重组**。对角或 2×2 块对角 $\bar A$（Mamba 类复值状态实值扩展）每步 $O(N)$ 次乘加，完整阶 $\mathrm{Time}=O(ndN)$、状态 $O(dN)$。**稠密递推核不在本算子内**——其结合步 $A_2A_1$ 为 $O(N^2)$ 矩阵乘，完整阶退化为 $O(ndN^2)$，须单列复杂度类，不混入 M1 的 $O(ndN)$ 证明。

### 3.3 并行扫描算法

| 算法 | 工作量 | 深度 | 选择 |
| :--- | :--- | :--- | :--- |
| Hillis–Steele | $O(n\log n)$ | $O(\log n)$ | 简单、作交叉对照 |
| Blelloch work-efficient | $O(n)$ | $O(\log n)$ | 数学验证 5b（64 点全一致）；作参照实现 |
| **分段三阶段**（块内串行 → 块间前缀 → 块内应用） | $O(n)$ | $O(\log n + n/P)$ | **CPU 主实现**（L1 尺寸 chunk 内串行避免 2× 访存）+ **Metal/CUDA 主实现**（CUB decoupled look-back 思想） |

**数值语义（Must）**：并行重组改变浮点结合顺序，结果与串行前缀在**§5.10 并行归约容差**（`rtol=1e-3`/`atol=1e-4`）内一致；逐位相同是应然而非必须，容差一致是硬门禁。`数学验证 5b` 的 64 点「全一致」是短序列逐位特例，**不外推**到长序列。F0-2 的 `1e-5/1e-6` 只用于 γ 门控同构，不用于并行扫描。

### 3.4 数值稳定性（对齐 M3/M4）

- **谱条件**：递推一致稳定须 $\prod_t\|\bar A_t\|_2$ 有界/收缩（M3：单步有界不自动给出任意长度界）——对角核取 $|\bar A_{i}|\le 1$，2×2 块对角核按**块特征值谱半径 ≤1**（旋转块恒为 1）作为参数校验的硬前置（拒绝谱半径 >1 的配置，除非另有证书）。
- **量化纪律（M4 教训，数学验证 23）**：$\bar A$ 保持高精度——INT4($\bar A$) 误差是 INT4($\bar B$) 的 **18×**。scan 的 $\bar A$ 输入**永不进低比特路径**；$\bar B,\bar C,\bar D$ 才进入 M4 量化评估域。

## 4. 算子契约（对齐 design.md §5.3 累加式扩展）

### 4.1 签名（草案，f32、显式参数、无逐模型配置藏入 kernel）

`scan`/`scan_cached` 作为 `compute` 的累加式新方法（design.md §5.2 决策 5 的既定路线），与 `attention`/`rope` 等同属「带显式属性的 f32 算子」。直通项参数名为 `skip`（避免与通道维 $d$ 重名）。谱前置校验与 `attention` 的 heads 整除校验同属**算子语义前置**，在各后端 kernel（B1）实现；`invalidparameter` variant 在 B0 就位，供 B1 触发。

```rust
/// 全量半群扫描（prefill 形态）：h_t = A h_{t-1} + B x_t；y_t = C h_t + D x_t。
/// `a` 仅接受对角或 2×2 块对角核（稠密核单列，见 §3.2）。
fn scan(
    &self,
    input: &tensorview<'_, f32>,  // [n, d] 输入序列
    init:  &tensorview<'_, f32>,  // [d, N] 初始状态 h_0
    a:     &tensorview<'_, f32>,  // [d, N] 递推核（对角或 2×2 块对角）
    b:     &tensorview<'_, f32>,  // [d, N] 输入投影
    c:     &tensorview<'_, f32>,  // [d, N] 输出投影
    skip:  &tensorview<'_, f32>,  // [d]    直通项 D（标量分支已收缩）
) -> Result<(tensor<f32>, tensor<f32>), backenderror>; // ([n,d] 输出, [d,N] 末态)

/// 单步增量扫描（decode 形态）：对绝对位置 pos 的一个新 token。
/// 语义 = 把 x 追加到 pos 处续跑递推；与**串行** `scan` 全量逐位一致
/// （M11 流式 ≡ 全量，仅对串行路径成立；并行路径见 §6 门禁）。
fn scan_cached(
    &self,
    input: &tensorview<'_, f32>,  // [1, d] 单 token
    state: &tensorview<'_, f32>,  // [d, N] 当前位置状态（读）
    a: &tensorview<'_, f32>, b: &tensorview<'_, f32>,
    c: &tensorview<'_, f32>, skip: &tensorview<'_, f32>,
) -> Result<(tensor<f32>, tensor<f32>), backenderror>; // ([1,d] 输出, [d,N] 更新态)
```

### 4.2 形状规则与错误分支（全部显式校验，绝不静默广播）

| 规则 | 描述 |
| :--- | :--- |
| rank | `input` rank 2；`init/a/b/c` rank 2；`skip` rank 1 |
| 通道对齐 | `input[1] == a[0] == b[0] == c[0] == skip[0]` |
| 状态宽对齐 | `init[1] == a[1] == b[1] == c[1] == N` |
| 空序列 | `n == 0` ⇒ 输出空张量 + 返回 `init` 原样（显式语义，非错误） |
| 非 2 幂 | `n` 非 2 幂时并行路径按分段/补零语义，结果与串行一致（不要求逐位） |
| 谱前置 | 对角核 $\lvert a_i\rvert\le 1$；2×2 块对角核块特征值谱半径 ≤ 1；否则 `backenderror::invalidparameter`（**B0 新增此 variant**，校验逻辑在 B1 后端 kernel） |
| 预算 | 中间张量（并行扫描块前缀，辅助空间 $O(n)$）计入 executor 预算检查，超限 `resourceexhausted` |

错误枚举：谱半径越界 ⇒ 新增 `backenderror::invalidparameter`（同步 design.md §5.4 的 variant 表 7→8）；通道/状态宽不匹配、rank 不足、`skip` 长度不匹配 ⇒ 复用 `shapemismatch`；预算耗尽 ⇒ 复用 `resourceexhausted`。

### 4.3 接口宽度核算（A4 上界）

A4 上界 20 的权威口径是「crate root 导出的公共项数」（design.md §2.2 DS1）。当前 `api` 导出 **18 项**（17 个 `pub use` + 1 个 `pub mod geometry`，据 `lib.rs` 实点），余量 2。`scan`/`scan_cached` 作为 `compute` 的累加式新方法，`compute` 11 → 13，**不新增 crate root 导出项**（18 不变，仍 < 20）。design.md §5.3 的「trait 方法数 17」是 §21.13 引入 `submission` 之前的过时表述，不是 A4 权威口径。**crate root 导出项已 18/20、余量仅 2，本方案后仍须立项评审「上调 A4 上界或为算子集拆分」**——这是 γ 脉冲等后续算子落地前的既定动作，而非本方案新增的压力。

### 4.4 与既有算子的关系

- `scan_cached` 与 `attention_cached`/`rope_at` 同属 **decode 单步形态**，是 `forwardsubmission_cached` 一次提交链的组成算子（β 侧）
- 不触碰冻结的 `kernel<T>` 单目契约；`compute` 的 `scan`/`scan_cached` 与 `submission` 的 `recordscan`/`recordscan_cached` 是纯累加，三后端同步扩展

## 5. 阶段拆分（AGENTS.md：每个提交一个可测试阶段）

> 阶段编号用 **B0–B6**（Beta 通道），与 lld.md §7.1 的阶段 P0–P4、§1.8 的刀序 ID 严格区分。

> **B1 落地即解锁 decode 侧 β 通道**（SSM 优势在 decode，lld.md §5.6/M11）；B2 的并行红利只在 prefill。二者独立验收，不要求 B2 才「有用」。

| 阶段 | 内容 | 验收标准（独立可测） |
| :--- | :--- | :--- |
| **B0 契约 + 标量参考** | `api::compute` 加 `scan`/`scan_cached`；`api::error` 加 `invalidparameter`；`testkit::reference::scan`（逐步递推标量实现）；形状/边界/错误测试 `tests/it/scancontract.rs` | 契约编译；参考实现 64 点对 5a 半群恒等全一致；错误分支（非 2 幂、空序列）全触发；谱前置校验留待 B1 |
| **B1 CPU 串行** | `cpuscan`/`cpuscancached` 纯净 kernel + `cpuexecutor` 派发 + 预算检查 | 串行结果 vs 参考**逐位一致**（确定性路径）；`tests/it/cpucontract.rs` 或新 `scanunits.rs` |
| **B2 CPU 并行** | 分段三阶段（主）+ Hillis–Steele/裸 Blelloch（对照）；并行 vs 串行容差一致；**M1 理论分析文档**（循环不变量、每步工作量/内存访问计数 → $O(ndN)$）；分段/Blelloch 按 DS3 拆成 ≤80 行、圈复杂度 ≤10 的小函数 | 并行 vs 串行 `rtol=1e-3/atol=1e-4` 全一致；理论证明成文；log-log 实测 $O(nd)$ 斜率 + 常数项核对 |
| **B3 Metal** | `source.rs` MSL kernel（分段三阶段）；`metalexecutor` 注册；数值对照 testkit | `metalcontract` 数值 vs 参考一致；**unsafe 站点登记评审**（期望复用现有缓冲/参数绑定路径，metal=4 不变；若新增交互类须先更新 `docs/design.md` §21.8 登记表） |
| **B4 CUDA** | `source.rs` CUDA C kernel（同一分段三阶段）；`cudaexecutor` 注册 | `cudacontract` 数值 vs 参考一致；走现有 `LaunchArgs::launch` 闭包机制 ⇒ **cuda=3 不变**（逐类交互恰一处原则） |
| **B5 模型挂载演示** | `nn` 新增 SSM 组合层（$\Delta_t/A/B/C/D$ 投影 + 指数梯形离散化，调 `compute::scan`）；`registry::architecture` 加变体（`mamba`，$N{=}64$、复值块旋转经半群实值扩展）；自研 `trap` 配置同算子 | `getarchitecture("mamba")` 解析；最小模型前向 vs 参考逐位；**不替代** Mamba-3 1.5B 端到端（那走 §4.9 协议） |
| **B6 基准 + 文档** | `benches/` 新目标（`scan` vs `attention` 同装置 16K 对照，按 §10.10 M1 卡：闭集硬件、$d{=}4096,N{=}64,n\in\{4K..128K\}$）；design.md §5 契约表 + §21 GPU 清单；lld.md M1 状态翻；conformance.md 判定行 | M1 Should 同装置 16K 相对稠密对照 ≥5×（**禁引用异模型 Table 7 外推**，不达标则如实记 OPEN，不降阈值）；文档与实现一致 |

**依赖**：B0→B1→B2 串行；B3/B4 可并行（依赖 B0/B1）；B5 依赖 B0–B4；B6 依赖全部。每阶段独立提交，不混改。

## 6. 验收门（汇总，Must 不降）

| 门 | 协议 | 判定 |
| :--- | :--- | :--- |
| 数值正确性 | 三后端 vs `testkit::reference`；CPU 串行逐位；并行/GPU 容差（`rtol=1e-3/atol=1e-4`，§5.10） | 全绿才进 CI |
| 流式 ≡ 全量 | `scan_cached` 逐 token 拼接 ≡ **串行** `scan` 全量（M11 语义，参考 `kvcacheforward` 模式） | 逐位一致 |
| 流式 ≡ 全量（并行） | `scan_cached` 拼接 ≡ **并行** `scan` 全量 | §5.10 容差一致 |
| M1 理论 | 循环不变量 + 每步工作量/内存计数 → $O(ndN)$ 完整阶证明（对角/块对角） | 成文评审 |
| M1 实测 | log-log 斜率核对 + 常数项/异常增长检查（§10.10 M1 卡，闭集硬件） | 出数才翻状态 |
| M1 Should | 同装置 16K scan vs 稠密 attention ≥5×（禁引用异模型 Table 7 外推） | 达标记实测，不达标 OPEN |
| 谱前置 | 对角 $\lvert\bar A_i\rvert\le1$；2×2 块对角块特征值谱半径 ≤1（M3 一致性观点） | 拒绝越界配置 |
| 量化纪律 | $\bar A$ 永不进低比特（数学验证 23） | 代码评审门禁 |
| unsafe | 工作站登记表核对（scripts/checkunsafe.sh） | metal=4、cuda=3 期望不变，变则先评审 |
| 文档 | design.md §5 契约表（`compute` 13 方法、`backenderror` 8 variant）/ lld.md M1 / conformance.md 同步 | 与实现一致 |

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| 浮点重组漂移（并行 vs 串行） | 数值不一致 | 容差协议（§5.10 `1e-3/1e-4`）先行冻结；逐位路径（CPU 串行）作权威 |
| $\bar A$ 近 1 时递推放大 | 数值不稳定 | 谱半径前置校验 + M3 乘积界分析 |
| 稠密递推核需求 | $O(ndN^2)$ 污染 M1 证明 | §3.2 限定对角/块对角；稠密需求单列算子另行立项 |
| GPU 分段扫描负载不均（$n$ 小时） | 并行无收益 | $n <$ 阈值走串行路径；基准如实报告，不凑数字 |
| A4 上界余量（crate root 18/20） | 后续算子受限 | 本计划后立项评审「上调上界或为算子集拆分」；scan 塞 `compute` 不新增导出项，余量仍为 2 |
| unsafe 站点意外增加 | 违反登记表 | B3/B4 先评审再实现；期望复用现有交互面 |

## 8. 与 26 维的映射

| 维 | 关系 |
| :--- | :--- |
| **M1** | 核心验收域（β-only $O(nd)$ + $O(ndN)$）——本计划的主目标 |
| **M3** | 谱上界前置校验 + 收敛分析 |
| **M2** | 长程记忆经 β 通道状态 $O(dN)$（与 α 块表互斥分工，规则 0.9） |
| **M4/E8** | $\bar A$ 高精度纪律；β 通道权重压缩进入五轴候选 |
| **E13/E14/E15** | γ 通道独立协议，不因 β 落地改变 |

## 9. 交付物清单

1. `crates/api/src/kernels.rs`：`compute` 加 `scan`/`scan_cached`（含文档注释）；`crates/api/src/error.rs`：新增 `invalidparameter`
2. `crates/testkit/src/reference.rs`：`scan` 标量参考
3. `crates/cpu/src/`：`scan.rs`（串行 + 分段三阶段 + Hillis–Steele/Blelloch 对照）+ executor 派发
4. `crates/metal/src/source.rs` + `executor.rs`：MSL 分段扫描
5. `crates/cuda/src/source.rs` + `executor.rs`：CUDA 分段扫描
6. `crates/nn/src/`：SSM 组合层（投影 + 离散化）；`crates/models/src/registry.rs` + `mamba/`：模型挂载演示
7. `tests/it/scancontract.rs` / `scanunits.rs` / `metalcontract` / `cudacontract` 新增用例
8. `benches/scan.rs`（含 vs attention 对照）
9. 文档：design.md §5/§21、lld.md M1、conformance.md、checkunsafe.sh 登记表

## 10. 建议提交序列（6 个独立提交）

1. `feat(api): compute 加 scan/scan_cached + invalidparameter + testkit 参考实现`（B0）
2. `feat(cpu): 串行扫描算子`（B1）
3. `feat(cpu): 分段并行扫描 + M1 理论分析`（B2）
4. `feat(metal): scan MSL 分段内核`（B3）
5. `feat(cuda): scan CUDA 分段内核`（B4）
6. `feat(nn+models): SSM 层 + mamba 挂载 + scan 基准 + 文档同步`（B5+B6，若需拆则 B5/B6 分提）

---

## 11. M1 理论分析（B2 交付，成文）

**循环不变量**（串行 `cpuscan::apply`，对角 A）：第 t 步内层循环（通道 i）开始前，`state[i·N..(i+1)·N]` 等于 $h_t$（前 t 步对该通道的递推结果）；步内 `state[idx] = a[idx]·state[idx] + b[idx]·x` 维持 $h_t = A_t h_{t-1} + B_t x_t$，累加 `acc = Σ_j c[i,j]·h_t[i,j]` 后 `out[t,i] = acc + skip[i]·x`。归纳基 $t=0$（`state = init`），归纳步由递推式直接给出。

**每步工作量（串行）**：外循环 $n$ 步，内层每通道 $i$ 做 $N$ 次乘加——每通道每步 $O(N)$，共 $O(ndN)$ = **$O(ndN)$**。内存访问：每步读 $a,b,c$ 各 $N$ 个元素、写 `state` $N$ 个元素，$O(N)$ per 通道，总 $O(ndN)$。

**分段三阶段（`apply_chunked`）**：
- 阶段 1（块内累积）：每块串行递推 $O(\text{chunk}\cdot dN)$，$P$ 块共 $O(ndN)$。
- 阶段 2（块间前缀）：$P$ 块前缀 $\oplus$，每块 $O(dN)$，共 $O(P\cdot dN)$。
- 阶段 3（块内应用）：同阶段 1，$O(ndN)$。
- 总工作量 $O(ndN)$（与串行同阶，常数 2× 来自两遍递推），深度 $O(\text{chunk}+P)$（块内串行 + 块间串行前缀）。并行深度 $T_\infty=O(\log n)$ 由块间前缀改用 Blelloch 时取得（§3.3；B2 块间前缀为串行参照）。

**完整阶**：$\mathrm{Time}=O(ndN)$、状态 $O(dN)$，与 lld.md §5.6 一致；分段三阶段不改变完整阶，只改变浮点结合顺序（§5.10 容差门禁）。

---

**状态声明**：本计划为提议，所有算子在实现与三后端实测前不写"已交付"；M1 状态在理论证明 + 实测出数前保持 OPEN，阈值 $O(nd)$ 不降。
