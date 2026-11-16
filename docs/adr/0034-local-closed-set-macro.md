# 0034：闭集样板先本地宏复用，不新拉 `prof` 依赖

- 状态：已采纳（P1 落地）
- 日期：2026-11-17
- 相关：[0022](0022-build-time-corpora.md)（依赖表门禁）；设计取证见 [`../serve-deep-optimization-design.md`](../serve-deep-optimization-design.md) D6

## 背景

`Status`（→ HTTP 码）、`Kind`（→ 线上编号 + 短名）、`Ending`（→ 短名）三处闭集同构：枚举、
`ALL` 清单、映射 match 各写一份。`prof::Closed` + `vocabulary!` / `inherent_names!` 是仓库既有
闭集契约，但 serve 当前不依赖 `prof`；新增这条运行时出边要改架构依赖表与门禁测试。

## 决策

第一阶段在 serve 内写等价的本地表驱动宏（`src/macros.rs`），按映射轴分三个臂：

- `code_set!`：只映射 `u16` 编号（`Status`）；
- `name_set!`：只映射稳定短名 + `Display`（`Ending`）；
- `code_name_set!`：编号 + 短名双轴，`from_code` 由 `ALL` 派生（`Kind`）。

变体清单是唯一事实源：`ALL`、`code`、`as_str`、`Display`、解码查找都从表展开，加变体只改表
一行。不新增运行时依赖，`#![deny(unsafe_code)]` 不变。

## 备选方案

- 直接依赖 `prof::Closed`：复用既有机制，但 serve→prof 是新运行时边，需 ADR 级登记；当前三处
  闭集不值得这条边。
- 维持手写：第四处闭集（如未来的服务错误码）到来时样板继续累积。

## 后果

### 正面

- 三处闭集的清单与映射单点维护；解码表不再手写第二份。
- 依赖表保持不变；宏为私有模块，不产生新 pub 类型。

### 负面

- 与 `prof` 闭集机制存在一份小重复；`ALL` 以 `&'static [Self]` 暴露，按值遍历的测试点改为
  `.iter().copied()`。

### 重议条件

闭集数量继续增长，或 `prof` 对本层开放为合法依赖时，再评估切到 `prof::Closed`，届时删除本地宏。

## 不做

- 不为编号闭集额外上提 proc-macro；不把宏导出为 serve 公共 API。

## 修订

- 2026-11-17：初稿并落地三个本地宏；接入 `Status` / `Ending` / `Kind`。
