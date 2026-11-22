# 0041：注解宏保留（D7）

- 状态：已采纳（D7 决策；`op::Skeleton::Attention` 死变体同轮删除）
- 日期：2026-11-19
- 相关：[0005](0005-annotations-as-contracts.md)（注解生成契约、不生成运行期注册表）、[0026](0026-gemm-generic-skeleton.md)（GEMM 族泛型骨架）、[0031](0031-kernel-macro-derived-table.md)（宏派生表）；看板见 [`../implementation-task-plan.md`](../implementation-task-plan.md) §4 D7、R-20，分析取证见 [`../10k-analysis.md`](../10k-analysis.md) §6.2 第 3 条、[`../REPO-review-2026-11.md`](../REPO-review-2026-11.md) §1.3

## 背景

`10k-analysis.md` §6.2 提出一个"组合式风格偏爱函数而非宏"的取舍：是否把 `#[annotation::model]` / `#[annotation::spec]` 换成普通构造函数。`REPO-review` §1.3 同时记录了 P1-6 的残留问题——「两套骨架词汇表不是投影关系，`Attention`/`Scan` 变体全仓无构造点，是装饰性闭集」。

两件事必须分开裁决：

1. **注解宏去留**：`#[annotation::model]` 在算子侧产出 `MODEL` 常量与对拍测试，`#[annotation::spec]`（`nn` 内重导出为 `#[spec]`）产出 `ContractSpec` 与校验测试。当前 27 个算子模块挂 `model`，`spec` 覆盖 nn 层。
2. **两套"骨架"词汇表**：`op::Skeleton` 是**声明层执行骨架**（决定 `cutile` 用哪个 lowering 基座），`annotation::SKELETONS` 是 **codegen 模板骨架**（决定生成的 `MODEL` 走哪副操作符骨架）。两者名称有交集但语义不同。

## 决策

### 1. 保留两个既有注解宏

- 保留 `#[annotation::model]`：算子侧声明展开，产出 `Model` 条目（注册名 / 生产形状 / 输出类型 / 骨架 / 常量名），并由生成物在别名上求值 `Format::CHECK`。
- 保留 `#[annotation::spec]`：`nn` 内重导出为 `#[spec]`，声明固定形状层，产出 `ContractSpec` 与 `impl Contract` + 校验测试。
- 两个宏都只做**声明展开**，不持运行期注册表（ADR 0005），不引入 `syn`/`quote` 之外的运行期依赖。

### 2. 两套骨架词汇表是两层词汇表，不做投影合并

- `op::Skeleton`（`crates/op/src/decl.rs`）是声明层执行骨架，闭集 **4 个**：`elementwise` / `reduce` / `gemm` / `scan`。它是 `cutile::execution::lower` 的唯一分派依据，也是声明文件 `[op].skeleton` 的唯一合法取值来源。
- `annotation::SKELETONS`（`crates/annotation/src/syntax.rs`）是 codegen 模板骨架，14 个名字（`gemv` / `gemm` / `elementwise` / `unary` / `reduce` / `rmsnorm` / `softmax` / `rope` / `embed` / `dequant` / `dequant_i4` / `dequant_i2` / `attention` / `scan`），每一个在 `codegen` 骨架表里都有自己的分支，由 `every_skeleton_has_a_branch` 守。
- 二者**不是投影关系**：codegen 需要按算子族选模板（`gemv` 与 `gemm` 分开、`rmsnorm`/`softmax`/`rope`/`embed` 各自一整副），而声明层执行骨架只关心 lowering 的 4 条路。把 codegen 词汇强行投影到 4 个名字会丢掉模板选择，破坏三族并行与格式单态化。

### 3. 真实缺陷只是 `op::Skeleton::Attention` 无构造点

复核当前树：所有真实注意力算子的 `.toml` 都声明 `skeleton = "elementwise"`（注意力语义走逐点 + 归约，不是独立 lowering），`scan_f32.toml` 声明 `skeleton = "scan"`。于是：

- `op::Skeleton::Attention` 在全仓**没有任何构造点**（唯一构造 `Skeleton::Gemm` 出现在 `cap.rs` 的测试里），是装饰性闭集变体，删除；
- `op::Skeleton::Scan` **有**构造点（`scan_f32.toml`），保留；
- `annotation::SKELETONS` 的 `attention` / `scan` 由 `#[model(skeleton = "attention")]` 使用，属 codegen 域，**不动**。

删除 `Attention` 后，`Skeleton::ALL` 是声明层骨架的唯一来源（4 个），`Skeleton::index()`（`self as usize`）仅用于身份，不参与任何 `Set` 位集宽度，因此 `Scan` 的判别值前移不影响任何容量断言。

## 备选与否决

**用普通构造函数替代宏**：否决。理由：

- 会把声明负担抬回每个算子模块（27 个 `model` 挂点 + nn 层 `spec`），每处都要手写注册名、形状、输出类型与骨架，且失去编译期闭集校验（别名不实现 `Format` 直接编译不过）；
- 语义表会重新出现第二来源（手写 `Model { .. }` 字面量的回退路径），与 ADR 0005"注解生成契约、不生成注册表"直接相悖。

**把两套骨架词汇表合并成一套**：否决。二者语义层级不同（执行 lowering 基座 vs codegen 模板选择器），强行投影会同时破坏 `cutile::execution` 的 4 路分派与三族 codegen 的模板分支；P1-6 登记的真实缺陷只是 `Attention` 无构造点，已按 §3 收口。

## 后果

### 正面

- 声明负担留在宏里，新增算子写一行 `#[model(..)]`，单一语义表由注解生成物唯一提供，`the_registry_and_the_model_annotations_are_mirrors` 双向锁死；
- `op::Skeleton` 从装饰性闭集变成"每个变体都有构造点"的真实闭集，符合"闭集必须有唯一声明处"的契约精神。

### 负面（代价）

- 宏仍是编译期硬依赖（`op`/`nn` 依赖 `annotation`），机制层再精致也不能脱离宏单独发布；这属既有事实，不是本轮新增。
- 两套骨架词汇表并存需要文档说明，读者必须知道"声明层 skeleton"与"注解 skeleton"是两回事——本 ADR 即为该说明的唯一出处。

### 重议条件

- 出现普通函数能以更小维护面覆盖宏展开、且不丢编译期闭集校验与单一语义表；
- 出现第三套"骨架"词汇且与前两套都不构成投影，需重新裁决闭集归属。

## 不做

- 不动 `annotation::SKELETONS`（codegen 域 14 名）；
- 不改宏对外形状（P0-2 的"注解双表合一"只合并内部查找表，不改 `#[model]` / `#[spec]` 接口）；
- 不引入运行期注册表反射，不新增 `syn`/`quote` 之外的过程宏依赖。

## 修订

- 2026-11-19：初稿落 D7。保留 `#[annotation::model]` / `#[annotation::spec]`；澄清 `op::Skeleton`（4 名执行骨架）与 `annotation::SKELETONS`（14 名 codegen 骨架）为两层词汇表、非投影；删除无构造点的 `op::Skeleton::Attention`，保留有构造点的 `Scan`。
