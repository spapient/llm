# train / serve 算子适配协议统一设计（2026-11-25）

承接 `validation/cutile-production-assembly-report.md` §5.3：下一批优先消除 `train` 与 `serve` 的适配协议重复，而不是继续拆小文件。本文是动手前的对账与落点裁决；实现与门禁证据另记。

## 1. 现状对账

两侧都把 `nn` 算子入口适配到执行缓冲视图：`train` 走 `Kernel` 池化缝，`serve` 走 `engine::Program` 执行器缝。底层 `View`/`ViewMut` 同为 `storage::LaneView`/`LaneViewMut`（`engine::exec::View` 是其别名）。

| 适配要素 | `train`（`kernel.rs`） | `serve`（`kernels.rs` / `dense.rs`） | 同构度 | 处置 |
|---|---|---|---|---|
| 输入 lane 取值 | `input_slice::<L>(op, idx, view)` | `input_f32` / `input_u8` 按 slot 手写 | 高 | 下沉泛型 slot 提取 |
| 输出 lane 取值 | `output_slice::<L>(op, view)` | `output_f32` 手写 | 高 | 下沉泛型 slot 提取 |
| 表项结构 | `Entry { op, run: fn(&[View], &mut ViewMut) }` | `Entry { op, run: fn(Launch, &[View], &mut [ViewMut]) }` | 中（字段同，签名异） | 结构留各自入口层 |
| 注册表互锁 | 运行期 `find`，未构造期断言 | `all_in_registry` const 断言 | serve 更强 | 下沉 const 断言到 `nn` |
| 元组拆包 | `Inputs` 泛型 1..3 元组 `unpack` | 适配宏手写 0/1/2 槽 | 中 | 第一批保留各自宏，第二批评审 |
| 批次整除几何 | `batches(op, len, k)` | `row_batch_rows` 内联校验 | 中 | 第二批评审，语义需先对齐 |
| 矩阵装配 | `matrix_pair` / `row_matrix` | `View::matrix` 内联 | 高 | 已在 `tensor`，不另造 |
| 逐元素适配 | `kernel_adapter!` unary/binary/norm | `elementwise!` unary/binary | 高 | 宏族留入口层，复用下沉后的 slot 原语 |
| 反量化 | `dequant!` | `dequant!` | 高 | 同上 |
| 量化 GEMV / GEMM | `quant!` / `gemm!` | `gemv_quant_kernel!` / `gemm_quant_kernel!` | 中 | 第二批评审 |
| 注意力 prefill | `attention_paths` | `attention_paths` | 高 | 路由留入口层，几何辅助下沉 |
| decode | `decode_paths`（单段） | `decode_paths`（单段 + ragged 批段） | 中 | 差异留入口层 |

## 2. 不可抹平的差异（必须留在入口层）

- **输出形态**：`train` 单 `&mut ViewMut`；`serve` 多 `&mut [ViewMut]`。
- **执行契约**：`train::Kernel`（`compute_step` / `compute_shaped_step`，带 `Geometry` / `Running`，服务池化与反向）；`serve` 实现 `engine::Program`（`name` / `run`，`Launch` 承载 `block` 几何与标量槽）。
- **几何来源**：`train` 由缓冲长度 + 声明档反推；`serve` 从 `Launch.block` / 标量槽读实例几何。
- **反向能力**：`train` 独有旋转表缓存、`rope_grad` 等反向核心，`serve` 不持有。

这些是两条执行缝的真实边界，不得为减码合并成一个平行抽象或强行共用函数签名。

## 3. 已是单一事实源（禁止重复造）

- lane 取值：`storage::LaneView::{f32, u8}`、`LaneViewMut::{f32_mut, u8_mut}`。
- 矩阵视图：`tensor::View::{one, matrix, matrix_triple}`。
- 注册表：`nn::registry::REGISTRY` / `contains` / `kernel_table!`。

## 4. 第一批可下沉的中性原语

只下沉不认识执行契约的纯逻辑；落点在合法依赖边内（`train` / `serve` 都依赖 `storage` / `tensor` / `nn` / `diagnostic`，反向不可）。

1. **泛型 slot 提取 + 错误封装**：按 lane 取第 `slot` 路输入 / 输出，`op` 名与槽位进 `Invalid` 错误，错误口径统一为"期望形态 vs 实际形态"。落点 `storage`（`Lane` 主人，两侧已依赖）。`train` 的 `Lane` sealed 桥与 `Kind` 别名一并评估是否随之下沉；`Inputs` 元组拆包因带 `DType` 元数对比，第一批留在 `train`。
2. **注册表互锁 const 断言**：`nn` 提供 `assert_all_registered(ops: &[&str])` const fn，替换 `serve` 手写的 `all_in_registry`；`train` 的运行期 `find` 保留兜底错误，但表项在构造期同样过该断言。

第二批再评审：元组拆包、GEMM/GEMV 批次几何纯函数、量化适配宏族——两侧语义对齐后再下沉，避免把业务分支搬进共享层。

## 5. 分层合法性

- 共享原语只依赖 `storage` / `tensor` / `diagnostic` / `nn`，不得引用 `engine::Program`、`train::Kernel`、`Launch`、`Geometry`。
- 依赖边：`serve → train` 合法，`train → serve` 非法；原语不得放 `train` 供下层反向取用。
- 新公开类型按 `docs/ARCHITECTURE.md` 第三节所有权表登记，否则 ownership 门红。

## 6. 扩展方式

新增算子族：入口层加一条表项 + 一个适配函数；新增 lane 或几何变体只扩共享原语的表项，不改执行缝核心。

## 7. 验证计划与当前阻塞

- 定向验收：`cargo test -p llm-storage`、`-p llm-train`、`-p llm-serve`；既有适配测试不退化。
- 全量门禁：统一入口 `scripts/ci.sh`（fmt / clippy `-D warnings` / nextest / gate / testkit / 命名 / 架构门）。
- **阻塞**：工作树当前有一条独立在途的 `cutile` IR 前端重构（`kernel.rs` `Expr` 新变体 + 新 `parser.rs`），`cutile` lib 有编译错误，`parser.rs` 触发静态门复杂度 / 调用 / 命名 / 行数债，导致全量 `ci.sh` 跑不通。第一批切片只能先做定向编译与定向测试，全门禁证据待 `cutile` 转绿后补齐；在绿基线前不宣称闭环。
- 工作树存在多会话并发回写，动手前先 `git status --short` 复核磁盘现状。
