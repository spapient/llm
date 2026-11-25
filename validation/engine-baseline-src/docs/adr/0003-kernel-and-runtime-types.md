# 0003：kernel 侧类型与运行时类型分离

- 状态：已采纳（部分被取代）——本 ADR 的判据「kernel 侧视图不依赖 `tensor`、L3 不得向上依赖 L4」
  仍然成立；但被告 `tensor::Tensor` 本身已按 ADR 0053 删除，`tensor` 只剩 `Shape` / `DType` 元数据描述，
  见 [`../ARCHITECTURE.md`](../ARCHITECTURE.md) 的所有权表。下文保留原貌以记录当时的判断依据。
- 日期：2026-11-12

## 背景

`cutile` 的宿主 API 里有一个 `Tensor<T>`（`crates/cutile/src/host.rs`），而 `crates/tensor`
是空壳。看起来像是"同一个类型写了两次"，最直觉的修法是把 `cutile` 的那个迁到 `tensor`。

但 `cutile` 是 L3，`tensor` 是 L4。迁移会让内核层向上依赖运行时：

```
L3 cutile ──▶ L4 tensor        # 违反 ADR 0001：依赖只能指向更低的层
```

更实质的问题是两者语义本来就不同：

| | `cutile::host::Tensor<T>` | `tensor::Tensor` |
|---|---|---|
| 形态 | 持有 `Vec<T>`，`&mut` 分片 | 只读字节视图 + `DType` / `Shape` / `Strides` |
| 用途 | 分区即网格：分片数推出 `grid.x`，`&mut` 保证同一片不被两个 kernel 写 | 表达逻辑张量的元数据，交给 `storage` / `weights` / `kv` |
| 生命周期 | 一次 kernel 调用 | 一次请求/一次训练步 |
| 能否分区 | 能 | 不能（也不该能：分区是 DSL 的所有权规则） |

## 决策

保留两个类型，并在所有权表里写明它们不是重复：

- `cutile::host::Tensor<T>` 是 **kernel 侧操作数视图**，归 `cutile`；
- `tensor::Tensor` 是 **运行时逻辑张量**，归 `tensor`。

两者之间只允许"由使用者显式转换"：运行时把缓冲区交给 kernel 时构造 `cutile::host::Tensor`，
kernel 不返回任何运行时类型（它只填缓冲区）。

## 后果

- `cutile` 的依赖里永远不会出现 `tensor`：`tensor` 在 L4，而内核层不能向上依赖；
- `tensor::Tensor` 只做元数据与只读视图，不做 `&mut` 分区，避免两套并发规则；
- 若将来需要在 kernel 里读多种 `DType`，走 `prof::atom::DTypeSet`（DSL 的位集），
  而不是把运行时的 `tensor::DType` 拉进内核层。

## 修订

- 2026-11-12：`crates/cutile/src/host.rs` 已删除。它有 152 行与 4 个测试，但全仓零调用——没有任何
  构建路径或对外 API 用它，`tensor` / `exec` / 本 ADR 只是"引用"它。**结论不变**（kernel 侧视图与
  运行时张量不是同一个类型），但 `cutile::host::Tensor<T>` 目前**不存在**：`cutile` 只接收裸指针与
  标量，分片与 `grid` 由调用方给。等 `llm` 真的要直接调用 kernel（DESIGN D5）时按 DESIGN §4.4
  重建，并让它真正进入构建路径，而不是再造一份占位实现。
