# 0032：装配族别收成描述符表，消除重复 plan 构造

- 状态：已采纳（P1 落地；实现形态相对草案有调整，见决策 §3）
- 日期：2026-11-17
- 相关：[0001](../ARCHITECTURE.md)（装配逻辑归服务层，`nn` 不认识 `Graph`）；设计取证见 [`../serve-deep-optimization-design.md`](../serve-deep-optimization-design.md) D5/C2/G3

## 背景

`model/assemble.rs` 是全 crate 最大文件：投影 / 一元点wise / 归一化 / 查表四族各一个 plan
构造函数，各自填一份 `LayerPlan` 字面量；族别归属用两张裸字符串表（`UNARY_OPS` / `NORM_OPS`）
运行期 `contains` 匹配。新增族别要同时改表、plan 函数与分流臂三处。

## 决策

### 1. 族别闭集 + 描述符表

`LayerKind`（`Projection` / `Unary` / `Norm` / `Embedding`）是可枚举闭集；等长族算子归属
收成 `OPERATOR_FAMILIES` 描述符表：每行 `kind` + 算子名集合，`family_of(operator)` 查表归类，
替代两张裸字符串表。

### 2. 构造路径归一

- `build_plan(...)` 统一填 `LayerPlan` 六个字段；`entry_name(layer)` 统一权重条目名（层名 +
  `WEIGHT_SUFFIX`）；
- 一元点wise 与归一化共用 `width_plan`：宽度都取自输出形状、输入输出等长；只有归一化带 gamma
  `[width]` 权重，一元点wise无权重；
- 宽度链对齐轴（投影/查表对齐输入宽，等长族对齐输出宽）收为 `LayerKind::chain_width` 描述方法，
  `check_chain` 不再散落族别匹配臂；
- 权重条目存在性与长度校验（`check_weight`）、链衔接校验（`check_chain`）行为保持不变。

### 3. 相对草案的形态调整

草案曾设想四个族各实现一个 `Family` trait 薄类型。落地时未采用：投影/查表的几何来自类型化视图
（`linear()` / `embedding()`），一元/归一化来自输出形状，四类取数来源不同；trait 方法签名无法
干净地携带视图，强行抽 trait 只会把下转型与错误透传搬一层。最终形态是**闭集枚举 + 描述符表 +
共享构造函数**，族别策略与接线机制分离，但不引入四个实现类型（也因此没有新 pub 类型进所有权表）。

## 备选方案

- 保持函数式分流，只抽公共构造：族别新增时仍要加函数 + 表 + 臂三处，拒绝。
- 把装配上提到 `nn`：装配是 engine 图接线逻辑，属服务层；`nn` 不认识 `Graph`（分层规则），拒绝。
- 四族各一 `Family` trait 类型：见决策 §3，当前四族规模下抽象偏重且视图取数不自然，暂缓；
  第五族出现且视图形态趋同时再评估。

## 后果

### 正面

- 算子族别归属单点维护；新增等长族 = 描述符表加一行 + 链轴/权重规则在 `LayerKind` 内归位。
- 四族 plan 字面量与条目名重复消除；宽度链与权重校验口径不变。

### 负面

- `LayerKind` 的链轴与权重规则仍以 match 集中在枚举方法内，族别策略没有完全对象化。
- 行数净减不及草案预估：描述符表与共享构造带来前置结构，装配文件体量主要由既有几何校验与
  前向宿主占用。

## 不做

- 不引入运行期插件式后端族；不接线残差旁路、归约、位置编码、注意力（随对应通路落地）。
- 编号基址分区不属族别决策：C4 已在同文件顶部归为 `INPUT_SLOT` / `ACTIVATION_BASE` / `WEIGHT_BASE` 三段常量；不改输入缓冲覆写通路（P4，受 engine `Executor` 入口限制）。

## 修订

- 2026-11-17：初稿并落地为闭集枚举 + 描述符表形态；记录对草案 trait-per-family 的偏离与复核条件。
