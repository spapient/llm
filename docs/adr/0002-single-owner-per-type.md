# 0002：一个概念只有一个主人（所有权表）

- 状态：已采纳
- 日期：2026-11-12

## 背景

仓库里已经出现了三处"同一个概念两个主人"的苗头：

```
crates/cutile/src/host.rs:23   pub struct Tensor<T>       # kernel 侧操作数视图
crates/tensor/src/lib.rs       add()                      # 本该拥有逻辑张量的 crate 是空的
crates/cutile/src/backend.rs   编译 C + 子进程              # 与 exec 的职责重叠
crates/cutile/src/space.rs     GM/FAST/REG                 # 与 storage / mem 的职责重叠
op 的算子声明 / nn 的层定义 / train 的训练图                  # 三套"图"，无人拥有
```

两套张量、两套执行、两套图不会自己消失：它们会各自演进，直到某一天无法互相转换。

## 决策

在 `docs/ARCHITECTURE.md` 第三节维护一张所有权表，**每个类型只写一个 crate 名**。新增类型时先查表：
表里有主人就在主人那里加，没有就把这一行加上——同一行不允许出现第二个名字。

三处重叠的处理方式：

| 重叠 | 处理 |
|---|---|
| `cutile::backend`（编译 C + 子进程） vs `exec`（提交图） | `cutile::backend` 保留"把源文件编译成可执行文件并运行"这一段（它是 DSL 的一部分：片段能否编译决定能力是否成立）；`exec` 只做"把图提交到流"，两者不互相调用 |
| `cutile::space` vs `storage` / `mem` | 地址空间（GM/FAST/REG）是 DSL 的语言概念，留在 `cutile`/`prof`；`storage` 只做布局搬运，`mem` 只做字节分配，都不认识地址空间 |

## 为什么不"先合并再拆"

- 先把 `cutile::host::Tensor` 迁到 `tensor` 会让 **内核层依赖运行时**（L3 → L4），违反 ADR 0001，
  且会让 kernel 的测试夹具被迫拉入整个运行时类型系统。
- 先把 `backend` 与 `exec` 合并会让"编译一次源文件"和"提交一张图"这两件不同频率的事耦合在
  一起：前者按 `(模板, 档案, flags)` 缓存，后者每 step 都发生。

## 后果

- 所有权表是评审时的第一条依据：新 PR 里出现第二个 `Tensor` 就该被拒绝；
- 表是唯一清单：不需要在任何地方再维护一份"谁拥有什么"的记录；
- 存量 `Result<_, String>` 的收敛方向明确：全部指向 `diagnostic::Error`，`Error: From<String>`
  提供逐 crate 迁移的中间态。

## 修订

- 2026-11-12：背景里的 `crates/cutile/src/host.rs` 已删除（零调用，见 ADR 0003 修订）。本 ADR 的
  结论不变——两套张量仍然不是一个东西；但所有权表里不再列那个不存在的主人，待宿主 API 重建时
  再补一行。
- 2026-11-14：`mem` crate 已硬删除（`Arena`/`Region`/`Tensor::from_region` 借视图路径及其测试一并
  移除；该路径生产代码零调用）。字节分配不再有独立成员；地址空间仍归 `cutile`/`prof`，`storage`
  只做布局搬运，二者都不认识地址空间。
