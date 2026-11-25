# 0039：解析器依赖策略（D5，P0-5 落档）

- 状态：已采纳（D5 决策；P0-5 已落地）
- 日期：2026-11-17
- 相关：[0021](0021-toml-layer-into-diagnostic.md)（TOML 语法层并入 `diagnostic`）、[0022](../../crates/diagnostic/src/lib.rs)（语料构建期读取，现生成强类型静态视图，运行期无 TOML 解析器）；看板见 [`../implementation-task-plan.md`](../implementation-task-plan.md) §4 D5、R-21，分析取证见 [`../10k-analysis.md`](../10k-analysis.md) §6.2 第 2 条

## 背景

`10k-analysis.md` §6.2 对解析层给过两条互斥路线：

1. **手写解析按格式收敛到模块内单源**——不引入新的外部依赖；
2. **接受唯一外部依赖**（`toml` + `serde`），用约一千行依赖换掉数千行手写解析。这是一条需要显式拍板的取舍，不能在实现里默认走任何一边。

现状（R-21）：生产路径只有 `diagnostic` 的 optional `toml`（`build` feature 门控）与 `dtype` 的 optional `bytemuck`；`serde` / `serde_json` 在 `dtype` 里也都是 feature 或 dev-dependency，默认 feature 纯 std、零依赖。各类解析分散在：

- 编译期语料：`prof` / `tools` 的构建脚本经 `diagnostic::buildgen` 读 `.toml`，算子声明由 `annotation` 在宏展开期读取；
- 运行期容器：`weights` 的 json / gguf / safetensors / rans 解析（属 weights 轨道，本 ADR 不扩面）；
- 声明片段：`prof` 的 `.k` 模板解析；通用动态数据树已经拆除。

## 决策

### 1. 生产运行期维持零外部依赖

- 默认 feature 下全部生产成员只用 `std`；运行期不引入 `toml` / `serde` / `serde_json` / 正则 / 命令行解析库。
- `toml` 只出现在 `diagnostic` 的 `build` feature 下，供各 crate 在编译期扫描、读取语料并生成强类型静态视图；它不进入运行时产物（ADR 0021、ADR 0022）。
- `bytemuck` / `serde` 仅作为 `dtype` 的 optional feature，默认关闭；`serde_json` 留在 dev-dependencies，只喂可选 feature 的互操作测试。

### 2. 每种格式的手写解析收敛到模块内单源

- 同一格式的解析实现只留一份：归属到该格式拥有的模块，不允许第二份平行解析或第二份按名映射。
- 编译期目录与 TOML 读取唯一经 `buildgen` 的 `collect` / `read` / `relative`；档案、算子与门禁字段直接投影到各自的强类型视图，不再生成通用 `Table` / `Value` 树。
- 运行期容器格式（json / gguf / safetensors / rans）各自在 `weights` 拥有模块内单源；跨格式不互相复制解析逻辑。

### 3. 解析错误走统一错误类型

- 所有解析失败以 [`diagnostic::{Code, Error, Result}`](../../crates/diagnostic/src/error.rs) 返回；类别用 `Invalid`（语法/形态不符）、`Missing`（条目缺失）、`Io`（读不出），不在解析器内 `panic!` 或裸 `unwrap`。
- 通用 TOML 读取保持格式原义；非有限浮点、非法日期字段等领域限制由所属生成器在构建期拒绝，非法区间与推导式在强类型视图 owned 化时仍走唯一解析与校验入口。

## 备选与否决

**引入 `toml` + `serde` 单依赖替掉数千行手写解析**：本轮否决。理由：

- 运行期零依赖与可移植性是硬约束（R-21、workspace `unsafe_code = "deny"` 同档基线）；语料 TOML 本来就在编译期读，运行期并不需要 TOML 解析器；
- 运行期容器解析属 weights 轨道，其取舍由装载阶段按实测阈值再议，不在本 ADR 提前拉 serde 进运行时；
- 手写解析当前规模未跨 D5 登记的实测阈值，引入依赖的维护面大于收敛收益。

**触发重审的条件**：手写解析代码量或维护成本跨 D5 实测阈值；或新增一个运行期必须解析的序列化格式，且现有单源实现无法以 std 稳妥承担。重审时再把本 ADR 状态从草案改为接受/修订，并同步 R-21 依赖清单。

## 后果

- 保留现存格式的负例：TOML 语法错误、闭集外取值、非法区间、非法推导式与越界形状都必须失败；已删除的通用树不把旧测试留成空壳。
- 新增解析必须先落在所属模块的既有单源内；第二消费者出现前不提前参数化解析器 trait。
- 本 ADR 不裁决 weights 运行期容器格式的具体实现，也不改变 `ops/*.toml` 等编译期语料的形状。

## 修订

- 2026-11-19：D5 落档，状态由「草案」转「已采纳」。P0-5（手写解析收敛到模块内单源）已按 §2 落地；本轮不引入 `toml` + `serde` 单依赖换手写解析（§备选与否决），生产运行期维持零外部依赖。
- 2026-11-25：档案、算子与门禁配置统一为编译期强类型静态视图；删除 `diagnostic::data`、`buildgen::emit` 及档案 / 声明旧表解析器。`buildgen` 只保留目录扫描、TOML 读取与路径展示，领域校验由生成器和视图 owned 化分担。
