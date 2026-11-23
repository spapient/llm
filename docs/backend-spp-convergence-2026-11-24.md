# 后端 S++ 收敛方案（广度 · 执行成熟度 · 性能）

- 日期：2026-11-24（R0 基线快照 02:25）
- 范围：维度 2「后端广度」、维度 3「后端执行成熟度 / 性能」（评分基线 llm 5 / 3，llm.u 6 / 6，Burn 8 / 8，见 [COMPARISON-llm-llmu-burn.md](COMPARISON-llm-llmu-burn.md) §16.3）
- 状态：**基线建档与方案**。代码改动前置条件是上游 cutile 中间态闭合、九门回到可测状态；本文不改动任何在途文件（N20）。
- 2026-11-24 收口实测：cutile 中间态已闭合——`Expr::children()` 返回自有 `Vec<&Expr>`（不再构造悬垂临时切片），遍历与按元数重建抽到独立 `expr.rs`，发射侧 `call_name` + `children()` 无逃逸借用；core 链 `cutile / annotation / engine / storage / nn / serve` 全量测试与 `clippy -D warnings`、`fmt --check` 绿。E0515/E0599/E0308 均在清 `target/debug/deps` 陈旧产物后复测排除。
- 2026-11-24 复核实测（03:22，干净重建口径）：core 链门禁复测全绿——engine 100 lib + 3 doc、serve 197 + 3、cutile 139 + 2、annotation 48 + 6 + 2 ignored、storage 40 + 1；`clippy --all-targets -D warnings` 对 engine/serve/cutile/annotation/storage 零告警，`fmt --all --check` 干净。
- 本轮收口裁决（泛型 / 注解 / 配置化优先，减重复代码）：
  - 执行缝 `Executor`：`E: Executor` 泛型基座与 `Trace<E>` 装饰器已成立；绑取表面现状为 f32 / bytes 双路四签名（`bind_f32/bind_bytes/fetch_f32/fetch_bytes`）。合并为单配置路（`bind(id, Buf)` / `fetch(id) -> &Buf`，lane 标签复用 `storage::LaneBuf`）属可减码方向，但牵动 engine trait + cpu 实现 + serve 装配/模型 + 两处测试共 6 文件 32 处调用点，且该文件正是并行工作流在途回退对象（见 §1 R1 落地状态），单写者认领前不重放，避免覆盖并行线未提交改动（N20）。
  - nn 三骨架（`unary / elementwise / reduce`）：算子声明、样本夹具、注解别名已由 `unary_f32 / binary_f32 / reduce_f32` 宏配置展开，新增一族只加一次调用；`View<'_, f32>` 元素档绑定是 ADR-0054 广度批次（比较 / 逻辑 / 位运算、进阶归约、采样、排序）的前置泛型，拆元素 trait 会牵动 57 算子目录、样本与 `#[model]` 契约，不在本轮 core 收口范围。
  - 量化族：`prof::quant::Row{bits,cpb,zero,table,offsets}` 编译期配置表 + `Packed<P: PackedRow, const G>` / `Format` / `Plain` 泛型骨架已就位；i2/i4/i8 三档只差一行 `Row` 常量。
- 依据：[PROMPT-S++-AUDIT.md](PROMPT-S++-AUDIT.md)（N1–N20、P0–P7、密度预算、§7 九门、§8 性能 S++）、ADR 0025 / 0035 / 0050 / 0051 / 0052 / 0054、[backend-runtime-design.md](backend-runtime-design.md)、[parallel-spp-review-2026-11-24.md](parallel-spp-review-2026-11-24.md) §5.4。

## 0. 基线实测（2026-11-24 02:25，只读）

### 0.1 阻塞：上游 cutile 中间态

- rebase 已结束、无冲突标记；当前在途仅 `cutile/src/kernel.rs`、`cutile/tests/reference_tests.rs` 与两份 docs。
- **全链编译被 cutile 中间态卡死**：`Expr::children()` 与 `call_target()` 返回 `&[&Expr]` 临时切片，7 个 `E0515 cannot return value referencing temporary value`（kernel.rs:190 与 608–611）。`reference_tests.rs:85` 另有一处 `collect` 类型推断错误。
- 该阻塞已由并行分账文档取证记载（§5.4）：正确改法是返回 `Vec<&Expr>`（调用点 `.children().iter()` 与 `write_call(.., &[..], ..)` 均兼容），但修复被上游并行流周期回退；本轮按分账约定不擅改。
- 可独立编译、本轮实测绿：`engine`、`prof`（`cargo check -p llm-engine -p llm-prof` Finished）。`device / serve / nn / shard` 依赖 cutile，闭合前无法编译验证。

### 0.2 门禁红账（本轮亲自跑）

| 门 | 命令 | 结果 |
|---|---|---|
| 编译（lib） | `cargo check --workspace --lib` | ✗ cutile kernel.rs 7× E0515（上游中间态） |
| 编译（全目标） | `cargo check --workspace --all-targets` | ✗ 同上 + reference_tests.rs collect 类型错误 |
| 密度 | `bash scripts/loc-budget.sh` | ✗ src 101,983 / 预算 96,100（超 5,883）；388 文件；75 个 >400 行；9 个 >600 行；3 个 crate >8,200 行 |
| 命名 | `bash scripts/name-lint.sh` | ✗ 66 个 `*_*.rs` 文件命中（nn 33、fuzz 16、weights 5、train 3，余者） |
| fmt / testkit / jscpd / bench | §7 其余五门 | 待测（基线编译红，未跑） |

### 0.3 活树取证（纠偏对比文档旧表述）

对比文档 §4.1 / §16.3 写「Metal 未接 Executor 缝」——**该表述已过时**：`device/src/executor.rs:404` 已有 `impl Executor for MetalExecutor`，但只覆盖 `device / submit / sync`，泛型绑定五方法走 trait 默认值。

## 1. 两条后端缝（一缝一角色，不造第二个名字）

| 角色 | 收口 | 职责 | 现状 |
|---|---|---|---|
| 运行时提交基座 | `engine::exec::Executor`（ADR 0051） | 上层泛型 `E: Executor` 驱动；装饰器叠同一契约 | 实现：`LocalExecutor`（queue.rs:144）、`CpuExecutor`（cpu.rs:92）、`Trace<E>`（queue.rs:83）、`MetalExecutor`（executor.rs:279，部分） |
| 机制 / 验证缝 | `shard::backend::{Codegen, Driver, Backend}`（backend.rs:181 / :247 / :279） | 出设备源 + 编译 / 执行 / 回读，服务 verify / bench，不进生产提交 | 注册：Metal、Cuda(nvcc+musa)、Ascend（device/backends/mod.rs:29-37）；ROCm / OpenCL 无 Driver |

> ADR 0051 §4 已预留：Metal / CUDA 的每步表面「在 M3-2 双后端收口时补真实实现，只实现这五个方法、不改 serve」。本方案 R1 即落这条已采纳决策的 deferred 部分，**不是新设计**。

## 2. 审计卡（P0 建档，后端相邻 crate）

公开项数为行首 `pub fn/struct/enum/trait/const/static/type` 粗计（不含 impl 内项），密度 = 行数 / 公开项；P4 精确公开面在绿基线后逐 crate 复核。

### engine（L6 运行时基座）

```text
crate: engine            层: L6            职责句: 提交契约与执行基座
src 行数: 3,017          文件数: 11        最大文件: batch.rs:396
公开项: 39                密度: 77 行/公开项  命名命中: 0（本 crate）
依赖: → diagnostic, tensor, storage
信任清单: 我信任 storage::Allocator 给缓冲字节句柄；我多做了一次操作数冲突检查（输入输出同号拒绝）
替换证明: Executor trait + Local/Cpu/Trace 三实现；Program 缝可换内核体
jscpd: 待测              性能: 无设备内核数字（基座不持内核）
净减: —                  反模式: 无
结论: 结构在位；执行缝泛型表面待设备侧补齐（R1）
P4 盘点: 公开再导出均挂在公开签名上（SequenceOutcome 是 Batch 返回类型），无死导出可删
补丁: Metal 五方法落 trait；绿基线后复核 impl 内公开方法
```

### device（L6 设备源 lowering 与厂商后端）

```text
crate: device            层: L6            职责句: 设备源生成与厂商执行体
src 行数: 7,029          文件数: 35        最大文件: platform.rs:583
公开项: 62                密度: 113 行/公开项  命名命中: 待测
依赖: → diagnostic, cutile, shard(optim), prof, engine
信任清单: 我信任 shard::backend::Backend 出源回读；我多做了一次平台三态证据裁决（Covered 必带证据串）
替换证明: Codegen/Driver/Backend 缝 + 三厂商实例；platform 六行闭集台账
jscpd: 待测              性能: Metal L2 真机 36 算子发射；gemv_i4 带宽达成 57.6%–69.3%（<70% 特化门槛）
净减: —                  反模式: Metal 固有 register/bind 与 Executor trait 默认两套口径（R1 收敛）
结论: 未达 S++：运行时 Executor 只接一半；CUDA/Ascend 无运行时执行体
补丁: R1 Metal 五方法；R3 CUDA/Ascend Executor 位（缺设备登记 OPEN）
```

### prof（L2 算子契约与档案闭集）

```text
crate: prof              层: L2            职责句: 算子目录、模板词法与能力表
src 行数: 9,644          文件数: 29        最大文件: einsum.rs:1,054（>600 否决）
公开项: 168               密度: 57 行/公开项  命名命中: 待测
依赖: → diagnostic, dtype
信任清单: 我信任目录行表是算子身份唯一出处；能力表由目录派生，不另造清单
替换证明: 编译期扫描成表；Capability 两维查表（capability.rs）
jscpd: 待测              性能: 不直接持内核
净减: —                  反模式: einsum.rs 越界大文件；能力表 backend 维预留未接
结论: 未达 S++：单文件否决；能力第三维待 R2 扩列
P4 盘点: einsum.rs 是文档记载的刻意平铺（原五文件共享表达式/轴表合并，§模块头 15–17 行），不反向往回拆
补丁: einsum 拆分需新 ADR；CAPABILITIES 接 backend 维（ADR 0054）
```

### shard（L5 机制缝 / 集合通信）

```text
crate: shard             层: L5            职责句: 后端机制缝与分布式集合通信
src 行数: 6,737          文件数: 23        最大文件: bench.rs:587
公开项: 106               密度: 64 行/公开项  命名命中: 待测
依赖: → cutile, diagnostic, nn, prof
信任清单: 我信任 Backend 闭集注册；Remote 缺实现如实回 Unsupported
替换证明: Codegen/Driver trait + 静态注册表；Collective trait + Local 镜像
jscpd: 待测              性能: 无跨进程传输证据
净减: —                  反模式: Remote 三操作全 Unsupported（collective.rs:112/121/136）
结论: 未达 S++：广度维度的分布式格空缺
补丁: R3 Remote 传输位与拓扑簿记；无真实通道前不签 Covered
```

### cutile（L3 内核 DSL 与代码生成）

```text
crate: cutile            层: L3            职责句: 能力协商、Fragment 拼接与发射缓存
src 行数: 4,814          文件数: 19        最大文件: kernel.rs:658（>600 否决，且在上游中间态）
公开项: 84                密度: 57 行/公开项  命名命中: 待测
依赖: → annotation, diagnostic, prof
信任清单: 我信任 Fragment 编译期扫表；产物身份按 (模板, 档案, flags) 缓存
替换证明: 能力协商 → emit → 指纹缓存链路可离线对拍
jscpd: 待测              性能: 缓存命中口径待登记
净减: —                  反模式: kernel.rs 越界大文件 + 当前 E0515 中间态（上游在途，R0 后认领）
结论: 未达 S++：大文件否决；lower 覆盖面窄于设备大内核需求
补丁: kernel.rs 拆分；Lower 枚举与设备发射表对齐复核
```

### nn / serve / tensor（执行面相邻，附记）

```text
nn      14,955 行 / 56 文件 / pub 254 / 最大 kernel_gemm.rs:595   依赖 annotation, cutile, diagnostic, half
        单 crate 超 8,200 预算；R5 减码重点；内核族表与 prof 目录同序编译期断言钉死。
serve    5,571 行 / 16 文件 / pub  34 / 最大 completion.rs:597  依赖 engine, nn, train, tensor, tokenizer, weights, wire
        CpuModel<E: Executor> 与 assembly 三函数已泛型（model.rs:19/78, assembly.rs:539/561/569）；换后端不改模型代码这条已成立。
tensor   1,990 行 /  7 文件 / pub  13 / 最大 tensor.rs:730        布局算子齐全；后端能力不进类型签名（ADR 0053 刻意分工）。
```

## 3. 差距与补丁（按收益排序）

### R1 执行缝收口（ADR 0051 §4 落地，最高优先）

**证据**：`MetalExecutor`（device/src/executor.rs）只实现 `device/submit/sync`；`register_program` 默认 `Unsupported`，`bind_f32/bind_bytes/fetch_f32/fetch_bytes` 默认 `None`。`serve::kernels::register_cpu::<E: Executor>` 这条通用装配路径接不上 Metal——设备侧用固有 `register(name, source)` / `bind` / `bind_bytes(id, len)`，与 trait 协议是两套口径。

**动作**：
1. `MetalExecutor` 覆盖 trait 五方法：`register_program` 走既有管线登记适配；`bind_f32/bind_bytes` 落共享缓冲（`bind_bytes(id, len)` 现有分配位接进来）；`fetch_f32/fetch_bytes` 从缓冲视图回读。固有 API 保留转发（ADR 0051 §1 口径），不删调用点。
2. 设备缓冲形态在 `Executor` 缝上统一：f32 与打包字节两条 lane（复用 `storage::Lane` / `LaneBuf`，不造第二套 F32/U8 标签）；MetalBuf 补 lane 与元素数。
3. 新增 `CudaExecutor` / `AscendExecutor` 位：先落在同一条 `Executor` 缝，无设备 / 无内核处返回 `Unsupported` 并带 `Code` 与站点；不写第二套运行时 trait。
4. 流级派发：逐节点 `commit + waitUntilCompleted` 改为按流组织命令缓冲；同步仍按流排水，但不在每个节点上阻塞等待（性能见 R4 计时口径复核）。

**本轮落地（R1 第 3 条基座，2026-11-24）**：engine 基座新增 `VacantExecutor`（`engine::exec::VacantExecutor`，queue.rs）——尚未接入运行时的设备后端位，只持 `Device` 厂商标签，不持句柄 / 内核表 / 宿主缓冲；除 `device()` 外，`submit` / `register_program` 回 `Code::Unsupported` 带站点 `engine::VacantExecutor`，`bind_f32` / `bind_bytes` / `fetch_f32` / `fetch_bytes` 回 `None`。CUDA / Ascend 真机管线落地前据此在同一条 `Executor` 缝落位，落地后升级为持句柄的具体执行器，缝消费方不改。契约测试钉住「除标签外全方法拒接、绑/取回不占缓冲」。门禁：engine 100 passed / 0 failed，device 64+14+14，shard 141；clippy 零告警，fmt 干净。

**仍留 R1**：第 1 条 Metal 五方法落缝（在途，device/src/executor.rs 为 macOS-only FFI 岛，待并发线闭合后接）；第 2 条 MetalBuf lane / 元素数统一；第 4 条流级命令缓冲去逐节点阻塞。第 3 条的设备侧厂商构造口暂无运行时装配消费方（serve 仅持 `CpuExecutor`，cli 不构造运行时执行器），按删除优先六步第 1 步不另造无消费者公开面，待真机装配口出现再补。

**出口门**：engine / device 测试绿；`register_cpu::<MetalExecutor>` 类型可装配；serve 不出现厂商字面量；无新依赖边、无新增 unsafe（FFI 岛仍只在 executor.rs，ADR 0035）。

**减码**：Metal 固有方法与 trait 默认去重，不新增第二张绑定表；预计净减而非加表。

**R1 落地状态（2026-11-24 03:00）**：
- Metal 侧 concrete 绑定补丁曾在一个绿周期内落地并真机验证：`llm-device` lib 契约两条往返用例（f32 / bytes，lane 不符取 None，换绑交回旧块）通过；`llm-cli` 16 条 Metal 对拍（pointwise / reduction / scan / two-pass / gemv / i2/i4/i8 / softmax / attention / flash / decode / embed / gemm）全部通过，证明绑定走通泛型缝。
- 同仓另有并行工作流在周期间 `checkout/reset` 工作树，本轮未暂存的五个文件被两次回退，补丁未能留在工作树；落地需等工作树安静后按下方清单一次性重放并立即 `git add`。
- 并行流已在 `engine/src/queue.rs` 落 `VacantExecutor`（R1 动作 3 的 CUDA/Ascend 空位：除 `device()` 标签外全部 `Unsupported` 带站点），与本方案同一条 `Executor` 缝，不另造 trait；R1 动作 3 不再重复做。
- 待重放清单（基期 HEAD，路径相对仓根）：
  1. `crates/device/Cargo.toml`：依赖段加 `storage = { path = "../storage", package = "llm-storage" }`。
  2. `crates/device/src/executor.rs`：
     - 导入 `storage::Lane`；
     - `MetalBuf` 补 `lane: Lane`、`elements: usize`，`new_buf` 构造默认 `Lane::U8 / elements: bytes`；
     - 补 `lane()`、`f32() -> Option<&[f32]>`、`u8() -> Option<&[u8]>`、私有 `into_f32/into_u8`；
     - 固有 `bind_bytes(id, len: usize)` 更名 `reserve_bytes(id, len)`，避免遮蔽 trait `bind_bytes(id, Vec<u8>)`；
     - `impl Executor for MetalExecutor` 补 `bind_f32 / bind_bytes / fetch_f32 / fetch_bytes`：f32 按 `len*size_of::<f32>()` 整块拷位模式、lane=F32、elements=len；字节 lane=U8、空值不拷；取回按 lane 给视图，不符/未绑定为 `None`；旧块经 `and_then(into_f32/into_u8)` 交回；
     - `register_program` 保留 trait 默认 `Unsupported`（尚无 Program→MSL 桥接，诚实未支持态）。
  3. `crates/device/tests/metal/contract.rs`：导入 `engine::exec::Executor`；加 `executor_f32_binding_round_trips_through_trait`、`executor_bytes_binding_round_trips_through_trait` 两条设备门控用例（无 Metal 即 `return`）。
  4. `crates/cli/tests/run.rs`：`bind_f32/bind_u8` 助手改走 trait 缝（`bind_f32(slot, values.to_vec())` / `bind_bytes(slot, values.to_vec())`，忽略首绑 `None` 旧块返回）；输出槽改 `reserve_bytes`。
  5. `crates/cli/tests/device.rs`：三处未初始化输出槽 `bind_bytes(id, byte_len)` 改 `reserve_bytes(id, byte_len)`。
- 注意 trait `bind_*` 返回值语义是「被替换的旧块」，首绑为 `None` 属正常，**不可**把 `None` 当分配失败 `.expect`（首轮踩过）。

### R2 能力与配置化（声明驱动，不写第二张 match）

**证据**：`prof/src/capability.rs` 能力表只有 `skeleton × dtype` 两维，backend 维按 ADR 0050 预留；`Lower` 枚举（shard/backend.rs:92-178）把 15 种骨架形态写死在代码里；几何档位、回落分支散在发射路径。

**动作**：
1. 能力表扩第三维 `backend`（哪台机器能跑），行项从算子目录行表派生，避免与 `REGISTRY` 漂移（capability.rs 既定收敛方向）。
2. 后端选型、网格 / 线程块档位、不满足整除时的回落路径，收进声明表（`ops/*.toml`、`*.k` 档案）；同构 `if/match` 链 ≥3 分支来自数据的一律挪表（删除优先六步第 4 步）。
3. 注解驱动收口：`annotation` crate 的 `#[spec] / #[model] / #[dsl]` 把能力声明、骨架族、元素档挂到编译期校验；行为仍归 trait bound，注解只承载可静态判定的契约（annotation/lib.rs 既定分工）。
4. 泛型化优先：同构两份且在热路径的，抽泛型参数 / trait 常量 / 宏；整块 slice、单次 put、查表批量处理代替逐元素循环（保留循环仅限串行依赖或长度数据相关，注释写明）。

**出口门**：能力查表无运行时字符串选实现（反模式 #3）；注解表与 trait 实现由 bound 绑死；目录改名时能力表编译期跟着变，无第二真相。

### R3 广度补齐（六平台闭集 + 分布式）

**证据**：机制缝注册仅 Metal / Cuda / Ascend；ROCm-HIP、OpenCL/SYCL 只有 Fragment/Emit，无 `Driver`；`shard::Remote` 三操作全 `Unsupported`；ANE/NPU 无位。

**动作**：
1. ROCm / OpenCL 在 `Driver` 缝上补实现位：工具链探测位 + 编译 / 回读通道；本机无设备时 `Standing::Open { why: NoDevice/NoToolchain }`，证据串进台账（ADR 0052 三态）。
2. `Remote` 集合通信：`all_reduce / broadcast / scatter` 补传输抽象位；拓扑簿记与序列化协议先落契约，真实通道（NCCL/对等链路）按设备可用性分批。
3. 六平台台账（`device::platform::audit`）逐格回填机器指纹、工具链版本、证据命令；OPEN 格写明缺的是设备还是编译器。

**红线**：无设备后端只能到 OPEN，禁止用参考实现 / 探测 bool 冒充实测覆盖（反模式 #4）；行数仍是六行闭集，Vulkan/SYCL 吸收进 GPU 行（ADR 0052 §1）。

### R4 性能证据回填（§8 性能 S++）

**证据**：benchmarks 有 kernel / model / attention / peak harness；但三真机矩阵（M5 Pro Metal、T600 CUDA、CPU AVX2）的 roofline 达成率、对 Burn ≥1.2× 对比、回归中位数字均未回填；Metal gemv_i4 带宽 57.6%–69.3%，低于 70% 特化门槛。

**动作**：
1. 计时口径统一：只计派发；3 轮取中位；固定形状与种子；上传 / 下载不计（与 `measure_kernel` 同口径）。
2. 每内核在声明里写上界（compute-bound ≥90%、bandwidth-bound ≥85% 达成率）；实测取中位，禁单次采样、禁理论峰值冒充 roofline。
3. 同形状、同机对 Burn 跑一组 ≥1.2× 对照（CPU / Metal / CUDA 各一组）；回归门每轮劣化 ≤1%，与上一轮产物比。
4. 设备内核管线未建起来的格，不出设备数字（backend-runtime-design G10 既定口径）；CPU 微核路径补齐 NEON/AVX2 对照。

**出口门**：九门第 8 项 `cargo bench -p llm-benchmarks -- --test` 绿；证据含机器指纹、命令、形状、种子、中位值。

### R5 减码与密度（删除优先六步，贯穿 R1–R4）

**基线**：src 101,983 / 96,100（超 5,883）；>600 文件 9 个；>400 文件 75 个；超 8,200 行 crate 3 个（nn / weights / prof）。

1. 删无消费者公开面（调用点 0 且不在架构登记）→ 直接删，进 P4 砍面清单。
2. jscpd 命中（≥5 行 / ≥80 token）抽一份；阈值 0。
3. 同构两份热路径 → 泛型 / 宏；配置化进声明表。
4. 越界大文件拆分：>600 一票否决先清（train/geometry.rs 1,404、train/session.rs 1,209、prof/einsum.rs 1,054〔刻意平铺，需新 ADR〕、train/backprop.rs 916、testkit/contracts.rs 753、tensor/tensor.rs 730、annotation/dsl.rs 643、train/kernel.rs 605、fuzz/drivers.rs 602）。
5. 命名 lint 清零：66 个 `*_*.rs` 按 N5 改名单一概念词；限定词下沉到目录；例外仅 `_new` 构造惯例与 prof/prim 领域 `impl`。

## 4. 泛型 / 注解 / 配置化 / 优先 落点对照

| 诉求 | 落点 | 归属阶段 |
|---|---|---|
| 泛型 | `E: Executor` 基座（已成立）；设备执行器补齐 trait 表面；同构内核抽泛型参数 / trait 常量 | R1、R2 |
| 注解 | `#[spec] / #[model] / #[dsl]` 承载骨架族、元素档、形状契约；编译期校验，行为归 trait | R2 |
| 配置化 | 后端选型、几何档位、回落路径进 toml / `.k` 档案；能力表目录派生 | R2、R3 |
| 优先 | 编译期展开 / 整块 slice / 单次 put / 查表批量；保留循环仅限串行依赖或长度数据相关，注释写明 | R1–R5 编码纪律 |

## 5. 减码账本

| 轮次 | src 行数 | Δ | 文件数 | Δ | jscpd | 主要动作（≤5 条） | 门禁 |
|---|---|---|---|---|---|---|---|
| R0 基线 | 101,983 | — | 388 | — | 待测 | cutile 中间态阻塞，只读建档 | check ✗ / all-targets ✗ / name ✗ / loc ✗；其余待测 |
| R1 | 待测 | | 待测 | | | Metal 五方法落缝；设备缓冲 lane 统一；CUDA/Ascend Executor 位 | 绿后补记 |
| R2 | 待测 | | 待测 | | | 能力表扩 backend 维；几何 / 回落进声明表；注解驱动收口 | |
| R3 | 待测 | | 待测 | | | ROCm/OpenCL Driver 位；Remote 传输契约；台账证据回填 | |
| R4 | 待测 | | 待测 | | | 计时口径统一；三真机矩阵；Burn ≥1.2× 对照 | |
| R5 | ≤96,100 | 净减 ≥5% | — | | 0 | 砍无消费者面；jscpd 抽同构；>600 清零；命名清零 | 九门全绿 |

## 6. 风险与红线

- **N20 避让**：R0 前不写代码；动工前重新 `git status --short` 划干净 / 脏边界，单写者认领，不覆盖并行线未提交改动。
- **真机证据不可替代**：满分 10 必须真机数字；mock / 参考实现 / 理论峰值 / 单次采样一律否决（反模式 #4）。
- **无第二真相**：算子身份以 prof 目录为唯一出处；能力表、注册表、类型化 API 一张清单派生；不引入第二张名字表或运行时字符串分派。
- **无第二套后端名**：运行时基座只认 `Executor`；`shard::backend::Backend` 保留机制缝角色（ADR 0051 §1）。
- **依赖方向**：device → engine 单边；不反向依赖 serve / nn；越界 FFI 仍收敛在 ADR 0035 白名单岛。
- **密度预算只收不放宽**：>600 否决、>400 拆案；阈值漂移按反模式 #1 处理。

## 7. 动工前待办（R0 出口）

1. 上游 cutile 闭合 E0515（`children` / `call_target` 返回 `Vec<&Expr>`）与 reference_tests collect 类型；`cargo check --workspace --all-targets` 回绿。
2. 补跑 fmt / testkit / jscpd / bench 四门，把 R0 账本 jscpd 与基线数字补齐。
3. 复核 P4 公开面：engine / device / prof / shard 逐 crate 列零调用导出项，进 R5 砍面清单。
4. 冻结本轮认领文件清单（引擎缝 + device/executor.rs + prof/capability.rs 等干净文件），在提交记录里留认领依据（N20）。
