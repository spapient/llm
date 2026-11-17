# 现代码深度审计（2026-11-17）

- 取证方式：工作树源码 + `cargo check` / `cargo clippy` / `cargo test --workspace` 实跑；结论回链 `文件:行`。
- 配套方案：`docs/llmu-parity-optimization-design.md`（能力补齐与减量路线）；本文件只记**本轮代码层面还存在的问题**与分级动作。
- 口径：未在可运行后端上同机实测前，不回填吞吐、时延与"倍数"结论。

## 1. 当前基线实测

| 项 | 实测值 | 备注 |
|---|---:|---|
| workspace 成员 | 24 个 `crates/*` 成员；根目录为纯 workspace 根，无独立 package，`llm` / `cutile` 两个 bin 在 `llm-cli` crate 内 | 工具成员 `annotation`/`fuzz`/`benches` 计入目录数 |
| Rust 总行数（粗口径，含测试） | 76,839 | `find crates src -name '*.rs'`，不含 `target/` |
| 外部测试 | 166 文件 / 21,480 行 | `crates/**/tests/*.rs` |
| 内联 `#[cfg(test)]` 模块 | 164 个 | 测试体行数未单列 |
| 测试结果 | 1,396 通过 / 0 失败（1 ignored 保留） | `cargo test --workspace`（收口实测 2026-11-17） |
| 静态质量门 | 全绿（`llm-tool --test gate` 1/1） | ccn≤5、函数≤60 行、单文件≤899 行 |
| clippy | 全工作区无警告、无错误 | `--all-targets` |
| 真实 `unsafe` | 仅 `storage/src/mmap.rs`（FFI + Send/Sync 实现） | 其余 `unsafe` 字样为 `deny` 注释 |
| 可选外部依赖 | `bytemuck`/`serde` 仅 `dtype` 可选 feature；`libfuzzer-sys` 仅 fuzz；无 GPU FFI 依赖 | 生产运行时依赖面仍趋零 |

## 2. 本轮已修复的编译/门禁问题

### 2.1 train 内核表缺两条 decode 适配器（编译失败）

- **证据**：`crates/train/src/kernel.rs` 的 `KERNELS` 是定长数组 `[Entry; op::REGISTRY.len()]`，编译期与注册表同名同序锁死；注册表已含 `decode_attention_f32` / `decode_flash_attention_f32`，表里缺这两条 → `E0308` 数组长度 22 vs 24。
- **动作**：按既有 `attention_f32` / `flash_attention_f32` 模式补两条 slice 适配器：`q/k/v` 解压三元组，头宽 `d` 取 `op::decode_attention::MODEL.shape.1`，`q_rows`/`hist` 由 q、k 长度按 `d` 推出，分别调 `op::attention::decode_forward` / `decode_flash_forward`；表内按注册表顺序插在 `attention_f32` 之后。
- **状态**：`llm-train` `--all-targets` 通过。

### 2.2 架构门未登记两个新标记类型（测试失败）

- **证据**：`tests/architecture.rs::every_public_type_is_listed_or_baselined` 报 `op:DecodeAttention`、`op:DecodeFlashAttention` 既不在 `docs/ARCHITECTURE.md` 所有权表，也不在 `lints/vocabulary.toml` 豁免基线。
- **动作**：按既有 `FlashAttention` 同类先例，把两个标记类型加入 `lints/vocabulary.toml` 的 `op` 豁免名单；理由：算子模块内部标记类型，公开面是 `REGISTRY`。
- **状态**：架构门 7/7 通过。

### 2.3 在途协作改动（非本轮独立完成，状态以工作树为准）

工作树同时存在半完成的 KV/注意力通路重构：`Predict::logits` 增加 `&mut Batch` 形参、`assembly` 补 `load_history`/`write_current_kv`、serve 内核表补 decode 适配、测试桩与调用点同步、关键字参数改名。这些改动在本轮核查时已落到可编译可测状态；后续实现细节归属该在途工作，提交前需逐条复核。

### 2.4 静态门红线结清（ccn / 行数 / 一处宽度回归）

- **`op/src/decode_attention.rs` 圈复杂度**：`cost`/`geometry`/两条参考实现各带 3 处以上 `?` 与形状分支，ccn 6–7 超阈值 5。抽出 `axis`（形状轴取值）、`scalar_or`（样本标量取值）、`buffers`（几何 + q/k/v 缓冲一并取出）三个辅助函数后四条入口全部过门。
- **`serve/src/assembly.rs` 超 899 行 + 5 个函数 ccn 超限**：在途 KV 通路把装配宿主、块接线、KV 读写压进同一文件。结算方式：宿主与 KV 读写下沉到 `serve/src/model_host.rs`（147 行，`CpuModel`/`load_history`/`write_current_kv`/`bound_slice`）；块校验与接线拆成 `check_head`/`check_block_weights`/`block_operators`/`wire_qkv`/`wire_attention`/`wire_output`；`chain` 的单步拆为 `chain_step` + `advance_block`/`advance_layer`。装配文件回落 887 行，全部函数 ccn ≤5。
- **宽度回归**：单步拆分时把首层入宽 `plan.in_features` 误并入出宽字段，导致投影首层输入长度取成输出宽（`输入 3 维，模型首层要 2 维`），7 条 serve 对拍测试失败。`chain_step` 返回值补回入宽/出宽两列，首层 `input_len` 取入宽、链宽校验与 `prev_out` 取出宽；serve 100 + 8 条测试恢复全绿。

## 3. P1 生产能力缺口（对照不退化下限，见设计文档 §1.5）

| # | 缺口 | 证据指针 | 归属阶段 |
|---|---|---|---|
| G1 | 生产权重路径未接 mmap：`storage` 在 `weights` 是 dev-dependency；生产 `safetensors::File` 按条目 seek+read，随后 `decode_f32` 再收集 `Vec<f32>`，读出→驻留两次拷贝 | `crates/weights/Cargo.toml` dev-dependencies；`crates/weights/src/safetensors.rs`；`crates/storage/src/mmap.rs` feature 门控 | M1-A |
| G2 | 无优化器、无训练命令：autograd 只有 tape/plan/梯度需求分类，接不到参数更新；CLI 无 `train` | `crates/train/src/lib.rs`；`crates/cli/src/lib.rs` 命令集 | T1 |
| G3 | CPU 前向链路未闭环：装配侧投影/点wise/归一化首层仍 `Unsupported`；decode 主链依赖在途 KV 通路 | `crates/serve/src/model_host.rs::predict_input` | M1-B / M2 |
| G4 | 无真实设备后端：执行器只有 CPU 标量参考；无设备队列、kernel launch、Metal/CUDA 实现 | `crates/engine/src/exec.rs`；`prof` 档案为数据 | M3 |
| G5 | 服务层无 socket：端点/线格式/帧解码齐全，HTTP 后端未接 | `crates/serve`（`server.rs` 桩同基线 `llm.u`） | M4 |

## 4. P2 结构与减量问题

### 4.1 大文件按职责轴拆分（不砍正确性证据）

生产源码（静态门 899 行硬上限）：

| 文件 | 行数 | 建议拆分轴 |
|---|---:|---|
| `crates/weights/src/bin.rs` | 900 | ZIP 归档层（条目/CRC/ZIP64）与 pickle 解释器（栈/opcode/可调用拦截）分两文件 |
| `crates/serve/src/assembly.rs` | 887 | 已下沉 `model_host.rs`；剩余计划构造 / 块校验 / 块接线三簇可再分文件 |
| `src/bin/cutile/main.rs` | 883 | CLI 文法 vs 编译/见证执行 |
| `crates/train/src/snapshot.rs` | 768 | 字节布局/编码 vs 校验与原子落盘 |
| `crates/op/src/decl.rs` | 758 | 声明行表 vs 轴/组合推导 |
| `crates/annotation/src/syntax.rs` | 755 | 词法解析 vs 注解展开 |
| `crates/prof/src/atom.rs` | 726 | 档案行表列类型 vs 匹配规则 |
| `crates/cutile/src/device.rs` | 711 | 设备协商 vs 镜像/见证落地 |

测试夹具（不套生产行数门，属密度转向 P2）：`crates/serve/tests/model_tests.rs` 915、`crates/benches/tests/regress.rs` 896、`crates/train/tests/snapshot_tests.rs` 714。

### 4.2 注解侧两张并行例外表可合一

- **现状**：`crates/annotation/src/codegen.rs` 的 `REFERENCE_OVERRIDES` 与 `MODULE_OVERRIDES` 都以注册名为键，各一张 `[(&str, &str)]`。
- **动作**：合并为一行三列的表 `(注册名, 参考入口覆盖, 模块覆盖)`，缺省取骨架默认；消除第二份按名字查找的循环。低风险，动工前随 D7 注解裁决一起落 ADR。

### 4.3 内核表两种覆盖口径并存

- `train` 表：24/24 全量，定长数组 + 同序断言；`serve` 表：16/24 子集，`all_in_registry` 只查存在性。
- **动作**：保留两种口径（train 需要参考全覆盖，serve 只登记纯切片 f32 子集），但在 `kernels.rs` 注释里写清子集边界；新算子进注册表时两处表的职责不要混写。

### 4.4 测试密度转向 golden 对拍

- 外部测试 19,878 行 + 153 个内联模块；按设计 §5.1，单测逐步转向 golden 对拍 + 标量参考 + 少量属性测试；对拍、负例、契约测试只允许合并，不允许削弱断言。

## 5. 边界与安全复核结论

- `unsafe` 收敛在 `storage/src/mmap.rs`：`mmap`/`munmap` FFI 与 `Send`/`Sync` 实现；核心 crate 继续 `deny(unsafe_code)`。
- 解析期依赖（`toml`）只在 `diagnostic` 的 `build` feature 下；运行时不引入。
- 无 memmap2/cudarc/metal/libloading 等生产 FFI 依赖；可选互操作只在 `dtype` feature 与 fuzz harness 内。

## 6. 建议动作顺序

1. **M1-A**：`Source` 抽象接 mmap 后备 + 驻留缓冲 + 装载基准（先消 G1 的二次拷贝）。
2. **M1-B**：CPU 注册表内核桥 + 全族首层接入 + 进程内对拍（消 G3 查表以外的首层拒绝）。
3. **T1**：`Optimizer` trait + SGD/Adam + 训练命令/checkpoint 往返（消 G2）。
4. **M2**：KV 真实数据 + 连续批；收尾在途 `&mut Batch` 通路的历史读写与边界测试。
5. **M3 / M4**：后端边界 crate、量化内核、HTTP 服务化（G4/G5）。
6. 并行 P2：大文件拆分、注解例外表合一、测试密度转向；每步守住测试不回退、clippy/架构门全绿。

## 7. 工作树注意

- `docs/adr/0035-unsafe-ffi-boundary.md` 是并发独立裁决，提交时排除；`docs/llmu-parity-optimization-design.md` 为本方案文档，尚未提交。
- `crates/op/ops/decode_*` 与 `crates/op/src/decode_*.rs` 为在途新增算子片段；提交前核对注册表、注解表、train/serve 三处内核表口径一致。
- `crates/serve/src/model_host.rs` 为装配宿主下沉的新模块（未跟踪）；`assembly.rs` 只留装配与台账，提交前确认 `lib.rs` 兼容导出 `CpuModel`/`Params` 路径不变。
