# 0031：内核适配表由宏单一事实源派生

- 状态：已采纳（P0 落地；Q3 注册表 const 查询复用已落地）
- 日期：2026-11-17
- 相关：[0027](0027-kernel-registry-and-fragment-contract.md)（注册表键与挂载点）；设计取证见 [`../serve-deep-optimization-design.md`](../serve-deep-optimization-design.md) D3/D4

## 背景

`crates/serve/src/model/mod.rs` 原有三套同构适配宏（一元 / 二元 / gemv），各自展开拆切片与
调 `op::<op>::forward`；内核表条目的算子名字符串与适配函数名各写一份，两处可能漂移。算子名是
`op` 词汇表的事实，serve 表不该再手抄第二份字符串。

## 决策

1. 三套适配宏合并为一个 `kernel!`，按 arity 分支：`unary` 一进一出、`binary` 两进一出；
2. `kernel_table!` 表项从适配函数标识符经 `stringify!($run)` 派生算子名，不再手写 `op: "..."`；
3. 表内内核必须在 `op::REGISTRY` 内的子集断言，装配期 const 校验，归属查询复用上游
   `op::contains`（const fn），本层不再自抄比较实现。

## 备选方案

- 上提为 `#[kernel]` proc-macro：暂缓 P2。当前内核族与调用方都只有 serve 一处，proc-macro 是
  第二处消费出现之后的事。
- 保留三套宏：arity 分支只占宏内一个臂，三套拷贝没有必要。

## 后果

### 正面

- 新增内核 = 宏列表加一个标识符；注册名不可能与算子表漂移。
- 适配函数只做拆切片 + 转发，算术不重写。

### 负面

- 宏内 arity 校验的错误信息不如类型系统直观；保留既有"输入路数不符"的诊断文案兜底。

### 遗留（Q3，已解决 2026-11-17）

`op` 侧补出 const `contains`；serve 子集断言改调 `op::contains`，本地三个 const 比较 fn 删除。
本工具链 const 上下文尚未稳定 `str` 相等比较，逐字节比对上移到 `op::contains` 内部（私有 helper），
仅一份；上游 const `str` 比较稳定后可再简化。

## 不做

- 不引入 proc-macro；不改变 `CpuExecutor::register` 契约；不接 u8 lane 量化族（P3）。

## 修订

- 2026-11-17：初稿并落地宏合一与名字派生；同日 Q3 落地——`op::contains` 上线，serve 移除本地 const 字节比较副本。
