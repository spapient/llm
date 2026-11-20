# 0053：类型化算子入口——一张清单、一个原语、零手写方法

- 状态：已采纳（机制已落地：`op::for_each_operator!` + `train::OperatorSetOps` + 36 个在册算子；
  CPU 运行时对照已实测，见下文）
- 日期：2026-11-20
- 相关：ADR 0002（单一主人）、ADR 0005（注解即契约）、ADR 0013（加算子只加数据）、ADR 0050（算子基座）

## 背景

Burn 的 Tensor API 是"每个算子手写一条方法"：`crates/burn-tensor/src/tensor/api/*.rs` 共 296 个
`pub fn`，后端侧 `FloatTensorOps` 一个 trait 就有 130 条方法（`burn-backend/src/backend/ops/tensor.rs`，
2000 行；全部 ops trait 合计 424 条）。新增一个算子要同时改：后端 trait、每个后端实现、Tensor 方法、
文档与测试。这是它的架构税，不是它的优势。

本仓的差异点在于：算子语义已经收口在 `op::operator::Operator` 与 `REGISTRY`（ADR 0050），执行入口
已经收口在 `autograd::Store::compute` 一处。缺的只是"类型化 API 面"——一个能从同一张表长出来、
且不会与注册表漂移的门面。

## 决策

### 1. 一张清单同时驱动注册表与 API 面

算子清单只声明一次（`op::for_each_operator!`），一行一个算子：

```rust
add_f32: $crate::add_f32::AddF32 => "binary",
gemm_i4: $crate::gemm_i4::GemmI4Model => "gemm_packed",
// …一行一个算子
```

两个消费者都从这张表展开：

1. `REGISTRY`：名字、模板、语义表、可微性（`op` 内展开）；清单的方法名与
   `Operator::NAME` 在编译期逐字节对拍（`same_str` 常量断言）。
2. `train::OperatorSetOps`：唯一原语
   `apply(name: &'static str, inputs: &[Var], output: Var) -> Result<()>`，加上每个算子一条**默认方法**
   `fn add_f32(&mut self, inputs: &[Var; <AddF32 as Operator>::ARITY], output: Var)`；方法名与算子名
   来自同一行，输入个数来自 `Operator::ARITY`。`ARITY` 由 `arity_matches_the_samples` 与算子样本
   （非 `y` 缓冲数）对拍。

后端（本仓是 `Buffers<K: Kernel>`）只实现一次 `apply`，就拿到全部 36 条类型化方法；未来 `serve`
或设备执行器实现同一个原语即可复用整张 API 面。方法调用是单态化的默认方法，无虚表、无运行期注册。
`train` 里没有第二份 36 行装配表——`cargo tree -p llm-train | grep annotation` 为空。

### 2. 不做的事

- **不手写每个算子的方法**：方法体对每个算子都同构，手写只会制造 36 份可漂移的重复。
- **不建运行期注册表**：清单在编译期展开；名字与输入个数都在编译期/单测里对拍。
- **不做第二个列表**：注册表与 API 面共享 `for_each_operator!`；`ARITY` 是 `Operator` 的常量，
  不复制形状/代价/样本。
- **不新增调度层**：`apply` 直接调 `autograd::Store::compute`，执行路径与手写调用完全一致。

## 与 Burn 的对比口径

只比较**同一件事的装配面**，不比较功能广度（Burn 覆盖 f16/bf16/int/bool/量化/autodiff 与多维
张量；本表只统计同一条 36 算子的前向入口）。Burn 数据取 `tracel-ai/burn` 提交
`bb4d9ddd`（2026-11-19）的源码行数统计：

| 指标 | Burn | 本仓 | 说明 |
|---|---|---|---|
| 后端必须实现的原语方法 | 130（`FloatTensorOps`）；全部 ops trait 424 | **1**（`apply`） | 每后端一次 |
| 手写 Tensor 方法 | 296 个 `pub fn` | **0**（36 条由装配表生成） | 生成方法约 6 行/条 |
| 新增一个算子的 **API 面接线** | 后端 trait + 各后端实现 + Tensor 方法 + 测试 | **0 行**（清单一行同时驱动注册表与 API；`Operator` 实现仍在语义层） | 编译期名字断言 + ARITY 测试 |
| 漂移防线 | 后端 trait 缺方法=编译错误；Tensor 面靠人写 | 生成测试逐条对拍名字/顺序/输入个数 | 已实测：删一行、改错个数都会失败 |

按 36 算子切片折算：后端原语面 **130 → 1**（约 130×），手写 API 面 **296 → 0**，单算子装配
**3.6 条 trait 方法 + 8.2 条 Tensor 方法 → 1 条装配行 + 1 条生成方法**。

### 运行时对照（CPU 实测）

同机、同形状、同输入，本仓 `Buffers<Reference>` 与 Burn `0.23.0-pre.2` Flex CPU 后端逐条对照
（独立探针与原始输出：`crates/benchmarks/burn-runtime/`；Apple M5 Pro，release，各跑 3 次取中位数）：

| 用例 | 形状 | 本仓 ns/次 | Burn ns/次 | 比值（本仓/Burn） |
|---|---|---:|---:|---:|
| `add_f32` | 16 | 111 | 179 | 0.62 |
| `add_f32` | 4096 | 340 | 387 | 0.88 |
| `softmax_f32` | 4096 | 10 533 | 6 621 | 1.59 |
| `rmsnorm_f32` | 4096 | 2 646 | 3 227 | 0.82 |
| `matmul_f32` | 64×256×4096 | 4 709 209 | 1 264 833 | 3.72 |

结论分两层：

- **装配面没有可测代价**：36 条名字线性查表 + 池化输出缓冲，`add` 在 16 与 4096 元素上分别比
  Burn 快 1.6× 与 1.1×；`rmsnorm` 快 1.2×。本仓稳态不分配大块缓冲，Burn 每步新建一个 Tensor。
- **内核面有明确差距**：`softmax` 慢 1.6×，`matmul` 慢 3.7×（28.5 vs 106.1 GFLOP/s）。
  这是 `Reference`（单线程分块标量）对 burn-flex SIMD `gemm` 微内核的差距，不能归因于类型化 API；
  设备侧 tile 内核是另一条路径，见下。

同设备 Metal 对照（512×512×4096 f32，本仓 `metal_simdgroup_gemm_512` vs Burn `metal` 后端）：
本仓 simdgroup 4×4 取 6 轮中位 659 646 ns（3 255 GFLOP/s），Burn Metal 取 9 轮中位
1 077 958 ns（1 992 GFLOP/s），本仓约 **1.63×**；Burn 输出与 Flex CPU 参考 `max_abs = 7e-6`。
靶机与并发构建共享，单次中位在本仓 0.52–0.93 ms、Burn 0.79–1.84 ms 间波动，因此只报
区间与多轮中位，不报数量级。

### 注意力与量化 GEMM（同设备实测）

同一台机器、同一输入；本仓注意力走 `op::attention*` 的 tile 路径，Burn 注意力走
`burn::tensor::module::attention`（Flex 在 1M 分数上自动选 flash，Metal 走 cubecl 融合内核）。
量化 GEMM 形状 64×256×4096：本仓 `matmul_i8` 是 `i8 → i32 → f32` 精确整数内核；Burn 用
`Q8S` per-tensor 方案，per-tensor 方案在 CPU 与 Metal 都会先反量化成 f32 再做 f32 matmul
（`burn-backend` 的默认 `q_matmul` 与 `burn-cubecl` 的 `maybe_dequantize`）。

| 用例 | 本仓 | Burn | 比值 |
|---|---:|---:|---:|
| CPU attention 1024×64（本仓 dense tile） | 9 776 000 ns | 4 625 000 ns | 2.11（本仓慢） |
| Metal attention 2048×64（本仓整行分块） | 997 708 ns | 1 081 813 ns | 1.08（本仓快） |
| CPU int8 GEMM 64×256×4096 | 5 050 750 ns | 1 346 917 ns | 3.75（本仓慢） |
| Metal int8 GEMM 64×256×4096 | 302 542 ns | 1 060 688 ns | **3.51（本仓快）** |

设备侧 int8 内核是本轮新增：`engine::tiled::metal_int8_gemm_program`（共享内存分块、
`i8` 载入、`i32` 累加、`f32` 写回），tile=16 时 221.8 G int-op/s，输出与
`op::matmul_i8::forward_tiled` 的 CPU 参考逐元素相等。它目前是引擎级设备内核，
**尚未**被 `cutile` 的 lowering 与档案协商选中，生产接入选型仍是 OPEN。

结论：CPU 的差距在标量内核对 SIMD 微内核（注意力 2.11×、量化 3.75×），不应归因于类型化
API；Metal 上注意力持平、int8 GEMM 领先 3.51×，来源是融合整数内核对反量化 + f32 matmul。

顺带修正一个旧说法：Burn 0.23 的 Tensor 已不是单态化静态派发——`Tensor<D, K>` 持有类型擦除的
`BridgeTensor`，`Device` 也是运行期 `DispatchDevice` 句柄（`burn-tensor-0.23.0-pre.2` 的
`tensor/api/base.rs` 与 `device.rs`）。本仓是单态化 trait 默认方法 + 一次名字查表；实测查表没有
可测惩罚。

**未测**：量化 attention、反向与训练步；CPU 与 GPU 两张表不能互相折算倍数。

## 验证

- `cargo test -p llm-op --lib arity_matches`：36 个算子的 `ARITY` 与样本输入个数逐条相等。
- `cargo test -p llm-train --lib operators`：类型化调用与字符串派发逐元素一致。
- 漂移演练（E1）：把清单里的方法名改成与 `Operator::NAME` 不同的拼写 → 编译期 `const` 断言失败；
  把某个算子的 `ARITY` 改错 → `arity_matches_the_samples` 失败。
- `cargo test -p llm-tool --test gate`：静态门禁通过。
- Burn 运行时对照（E1/E3）：`crates/benchmarks/burn-runtime/` 三套探针各 3 轮，原始输出见
  `crates/benchmarks/burn-runtime/results/2026-11-20-m5pro.txt`；两侧输出校验和一致（matmul 相对差 < 1e-5）。
- Metal 同设备对照（E1/E3）：`metal_simdgroup_gemm_512` 与 `burn-side-metal` 交替 3 轮，
  Burn 输出与 Flex CPU 参考 `max_abs = 7e-6`；区间与中位见上。
- 注意力 / 量化 GEMM（E1/E3）：`metal_flash_attention_long`、`metal_int8_gemm_k4096` 各 3 轮；
  Burn 侧 `burn-side` / `burn-side-metal` 各 3–6 轮；原始输出见
  `crates/benchmarks/burn-runtime/results/2026-11-20-m5pro.txt`。

## 后果

- 加算子的流程固定为：`Operator` 实现 → 清单一行。注册表、API 面与 ARITY 检查同时到位。
- API 面与注册表同源；不存在"注册表有、API 没有"的静默缺口。
- 若将来接入设备执行器，只需为它实现 `apply`；不需要再写 36 条转发方法。
