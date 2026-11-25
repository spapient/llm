# train / serve 算子适配协议统一：复核裁决（2026-11-26）

承接 `validation/cutile-production-assembly-report.md` §5.3：下一批优先消除 `train` 与 `serve` 的适配协议重复，而不是继续拆小文件。本文记录动手前的逐文件对账、下沉候选裁决与实测证据；未落地下沉代码。

## 1. 范围与现状

两侧把 `nn` 算子入口适配到执行缓冲视图：`train` 走 `Kernel` 池化缝（`crates/train/src/kernel.rs`，782 行），`serve` 走 `engine::Program` 执行器缝（`crates/serve/src/kernels.rs` 294 行、`crates/serve/src/dense.rs` 233 行）。底层 `View`/`ViewMut` 同为 `storage::LaneView`/`LaneViewMut`（`engine::exec::View` 是其别名）。适配业务只分布在这三个文件；`train` 其余文件是会话/优化/反向核心，`serve` 其余文件是图装配/HTTP/生成，不构成跨侧适配重复。

## 2. 实测基线（2026-11-26）

| 检查 | 命令 | 结果 |
|---|---|---|
| 全工作区编译 | `cargo check --workspace --locked --all-targets` | Finished，0 error |
| 静态门 | `cargo test -p llm-tools --test gate --locked` | 13 passed / 0 failed |
| 适配文件文本重复 | `npx --yes jscpd` 上述 3 文件，`--min-lines 5 --min-tokens 80 --threshold 0` | 3 文件 1309 行，0 clone，0.00% |

上一轮阻断全量门禁的 `cutile` 在途 IR 重构已在并发会话转绿：lib 编译 0 error，`every_source_passes_the_static_gate` 通过。

## 3. 已是单一事实源（禁止重复造）

- lane 取值：`storage::LaneView::{f32, u8}`、`LaneViewMut::{f32_mut, u8_mut}`。
- 矩阵视图：`tensor::View::{one, matrix, matrix_triple}`。
- 注册表：`nn::registry::REGISTRY` / `contains` / `prof::same_bytes` / `kernel_table!`。
- 两侧表项都经 `kernel_table!` 由函数标识符派生算子名；两侧都有构造期注册表互锁（`train::same_order` 同序同名断言，`serve::all_in_registry` 成员存在性断言）。

## 4. 不可抹平的执行缝边界（必须留在入口层）

- **输出形态**：`train` 单 `&mut ViewMut`；`serve` 多 `&mut [ViewMut]`。
- **执行契约**：`train::Kernel`（`compute_step` / `compute_shaped_step`，带 `Geometry` / `Running`，服务池化与反向）；`serve` 实现 `engine::Program`（`name` / `run`，`Launch` 承载 `block` 几何与标量槽）。
- **几何来源**：`train` 由缓冲长度 + 声明档反推；`serve` 从 `Launch.block` / 标量槽读实例几何。
- **反向能力**：`train` 独有旋转表缓存、`rope_grad`、Newton–Schulz 正交化等反向核心，`serve` 不持有。

这些是两条执行缝的真实边界，不得为减码合并成平行抽象或强行共用函数签名。

## 5. 下沉候选逐项裁决

判据（AGENTS §2.2 / §2.3）：必须是逐字或近逐字真实重复；下沉后不是转发包装；共享层不得认识 `Program` / `Kernel` / `Launch` / `Geometry`；不得在共享核心引入业务分支。

| 候选 | 两侧现状 | 裁决 | 理由 |
|---|---|---|---|
| slot lane 提取 | `train::input_slice` / `output_slice`（泛型 `Lane` sealed 桥，元组解构取槽，错误报"实际 X 要 Y"）；`serve::input_f32` / `input_u8` / `output_f32`（`.get(slot)` 取槽，错误报"不是浮点/字节缓冲"） | **本批不下沉** | 取槽方式、错误文案、泛型机制三套约定不同；底层取值已在 `storage`。下沉前须先对齐取槽语义与错误口径，属入口层协议变更，不是纯逻辑归位。 |
| 注册表互锁遍历 | `train::same_order`（同序同名逐位比对）；`serve::all_in_registry`（`contains` 成员存在性） | **本批不下沉** | 谓词不同（同序 vs 存在性）；const fn 无法注入泛型谓词，强行合并会在共享层引入二选一分支，或被迫把本地 `Entry` 协议泄露进 `nn`。核心谓词与注册表已是事实源，两侧只剩 3 行遍历骨架。 |
| 批次整除几何 | `train::batches`（`k==0` 或不整除报错，返 `len/k`）；`serve::row_batch_rows`（另校验输出长度等于 `rows×n`） | **本批不下沉** | 校验项与公式不同；下沉需先统一几何契约。 |
| 矩阵装配 | 两侧都直接调 `tensor::View::matrix` / `matrix_triple` | **无重复** | 事实源在 `tensor`，未复制第二份。 |
| 适配宏族 | `train::kernel_adapter!`（unary/binary/norm/attention/quant/gemm/dequant…）；`serve::elementwise!` / `gemv_quant_kernel!` / `gemm_quant_kernel!` / `dequant!` | **留入口层** | 签名契约不同（单/多输出、有无 `Launch`）；跨侧共享宏会把两套执行缝差异塞进共享层。 |
| 注意力 prefill 装配 | 两侧 `d = MODEL.shape.1` + `matrix_triple(rows, d, …)` + 调注入内核 | **本批不下沉** | 同构段仅 2–3 行；行数校验口径不同（`train::batches` 显式报错，`serve` 依赖 `matrix_triple` 内部校验）。下沉属增加调用层数的包装，不满足净减门槛。 |
| decode 路由 | `train::decode_paths`（单段）；`serve::decode_paths`（单段 + ragged 批段） | **留入口层** | serve 多批段路由与 `Launch` 段长表，属执行缝差异。 |

## 6. 裁决结论

当前基线上，`train` / `serve` 适配层不存在满足下沉门槛的可提取块：文本级 0 克隆，结构同构部分被第 4 节的真实执行缝边界隔开；底层事实源（lane 取值、矩阵装配、注册表、表项派生）已经统一。本批不新增跨 crate 抽象，避免伪泛型包装与契约泄露。

## 7. 第二批前置条件（满足后再评估下沉）

1. **统一 slot 协议口径**：两侧先对齐取槽语义（索引缺失报错形态）与 lane 错误文案（期望/实际），再评估把单 slot 取值桥归位到 `storage`。
2. **统一几何契约**：先把批次整除与输出长度校验写成同一组纯函数口径，再评估下沉 `tensor` 或 `nn`。
3. 任何下沉新增公开类型须回 `docs/ARCHITECTURE.md` 第三节所有权表登记；自由 `const fn` 不触发 ownership 门（门禁只扫 `pub struct|enum|trait|union`）。

## 8. 其他未完成项（不在本批范围）

- 报告 §5.4：`tensor::View::one` 每次创建维度与步长 `Vec`，RoPE 计时区仍有 2 次分配；需 release 基准证明内联小秩形状或借用形状的收益后另行优化。
- 密度预算全仓 `src` 104,164 / 60,000 行，超 44,164 行（存量结构性债，非本批适配层引入）；`cargo bench --no-run` 与 regress 判定已过，固定硬件 criterion 采样未执行。
- #47 / #50 / #51 / #52 静态化批次的分项门禁（check、nextest、静态门、架构门、jscpd、deny、machete、命名、拼写、expand、actionlint）已在绿基线上复核通过；全量 `ci.sh` 仍被存量密度门阻断。#26 已落地 `llm-dtype`、`llm-diagnostic` 两个 pedantic 样板层，其余大层存量未整改。
- 本批未运行 `scripts/ci.sh` 全量门禁与 release 基准；上述定向 check / 静态门 / jscpd 不替代完整工作流证据。
