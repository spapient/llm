# 0051：执行基座——`Executor` 即运行时 Backend，装饰器叠在同一契约上

- 状态：已采纳（本轮落地基座扩展、`Trace` 装饰器与 `serve` 装配泛化；Metal/CUDA 的每步表面随 M3-2 补）
- 日期：2026-11-20
- 相关：[0025](0025-runtime-backend-abstraction.md)（运行时后端三件套）、[0017](0017-kernel-ast-and-ir-freeze.md)（无 IR 冻结）、[0035](0035-unsafe-ffi-boundary.md)（边界 crate 实现四个 trait）、[0050](0050-operator-base.md)（算子语义基座）、[0002](0002-single-owner-per-type.md)（单一主人）

## 背景

代码里已经有三套「后端」说法，但只有一套是运行时全局基座：

| 角色 | 现有收口 | 缺口 |
|---|---|---|
| 编译 / 驱动（verify / bench） | `cutile::backend::Backend = Codegen + Driver` | 只服务验证流水线，不进生产 |
| 运行时提交 | `engine::exec::Executor`（ADR 0025） | 上层只认具体类型：`serve::CpuModel` 持 `CpuExecutor`，cli 持 `MetalExecutor`；trait 只被提交/同步用到 |
| 算子内核 | `engine::exec::Program` | 只被 CPU 执行器消费；设备后端从各自入口注册 |

Burn 的 `Backend` 之所以能叠 `Autodiff` / `Fusion`，是因为上层与装饰器都只认那一个 trait。本仓的同位角色是 `Executor`：但 `bind` / `fetch` / `register` 仍是各后端的固有方法，trait 覆盖不到，于是上层无法泛化、装饰器也无法替换。**缺的不是新名字，是让已有基座成为唯一入口。**

## 决策

### 1. `Executor` 是运行时 Backend，补齐每步表面

`engine::exec::Executor` 增加五个带默认实现的方法：

```rust
fn register_program(&mut self, _program: Box<dyn Program>) -> Result<()>; // 默认 Unsupported
fn bind_f32(&mut self, _id: u32, _values: Vec<f32>) -> Option<Vec<f32>>;  // 默认 None
fn bind_bytes(&mut self, _id: u32, _values: Vec<u8>) -> Option<Vec<u8>>;  // 默认 None
fn fetch_f32(&self, _id: u32) -> Option<&[f32]>;                          // 默认 None
fn fetch_bytes(&self, _id: u32) -> Option<&[u8]>;                         // 默认 None
```

默认值表达「这个后端不拥有内核表 / 宿主缓冲」，不静默成功。`CpuExecutor` 覆盖全部五个并转发到既有固有方法（`register` / `bind` / `bind_u8` / `get` / `get_u8`），固有 API 不删，调用点按需迁移。

`cutile::backend::Backend` 保留原名与原角色：它是**机制缝**（生成程序 + 编译执行，服务 verify / bench），不是运行时基座。一个名字一个角色，不新增第二个 `Backend`（架构门按全局名字判重）。

### 2. 上层只认 `E: Executor`

- `serve::CpuModel<E: Executor = CpuExecutor>`；`new` 仍构造 CPU 后端，新增 `with_executor` 接受任意已配好的执行器（内核登记仍由后端自己的装配步骤完成）。
- `serve::assembly::{bind_weights, bind_activations, submit_and_fetch}` 与 `serve::kernels::register_cpu` 改为泛型。
- 默认类型参数保持调用点不变：`CpuModel::new(...)` 仍是 `CpuModel<CpuExecutor>`。
- 换后端 / 加装饰器不改模型代码——这是 Burn 那条基座的实质。

### 3. 装饰器实现同一个 trait 并转发

`engine::exec::Trace<E>` 是第一个：记录每次提交的节点名，其余方法原样转发；测试钉住「前向逐位不变 + 提交序可审计」。

- **Autodiff**：本仓的自动微分装饰器已经存在，但基座是 L5 的 `train::Kernel`（`Buffers<K>` 记 tape、`backprop<K>` 反向），不是 `Executor`；模式相同（泛型基座 + 装饰层），不重复造第二套。
- **Fusion**：**不做**。ADR 0017 冻结「无 IR」，融合需要一层可重写的图表示；`Trace` 明确不重排、不融合。若将来解冻 IR，`Fusion<E>` 与 Autodiff 一样实现 `Executor` 即可，基座不用改。

### 4. 后端清单仍是闭集

`MetalExecutor` 目前只覆盖提交 / 同步；每步表面用 trait 默认值（`None` / `Unsupported`），在 M3-2 双后端收口时补真实实现，只实现这五个方法、不改 `serve`。CUDA / MUSA 第三个后端走同一路径。ADR 0025 / 0035 的边界 trait 不变。

## 验收

- `engine` nextest 134 全绿、`serve` 178 全绿（含两条装饰器测试）。
- 无新依赖边、无 unsafe、`serve` 不出现厂商字面量。
- 静态门：新增类型 `Trace` 进所有权表，`Executor` 从存量基线移入表；其余度量不变。

## 影响与回滚

- 公共 API：`Executor` 只增方法（带默认实现，旧实现方零改动）；`CpuModel` 增加默认类型参数与 `with_executor` / `executor()`，旧调用点不变。
- 回滚单位：trait 扩展、serve 泛化、`Trace` 装饰器各为一个 commit，可单独 `git revert`。
