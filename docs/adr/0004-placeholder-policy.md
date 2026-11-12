# 0004：占位占接口形状，不占函数

- 状态：已采纳
- 日期：2026-11-12

## 背景

有 16 个成员曾经是同一个占位实现：

```rust
pub fn add(left: u64, right: u64) -> u64 { left + right }

#[cfg(test)]
mod tests {
    fn it_works() { assert_eq!(add(2, 2), 4); }
}
```

后果是可测的：

- `cargo test --workspace` 的用例数从 145 涨到 ~176，**覆盖率一点没动**；
- 每个 crate 看起来"已完成"（有代码、有测试、有文档），实际什么都没承诺；
- 15 个 `publish = None`（= 允许发布）混在里面，`tensor` / `engine` / `kv` 这些名字在
  crates.io 上已被占用，误发布无法撤回；
- 因为 `members = ["crates/*"]`，这些壳还自动成为 `clippy -D warnings` 的失败源（102 条 error）。

## 决策

一个尚未实现的成员，占的是**接口形状**：

1. **类型 / trait / 字段 / 错误语义**先定，函数体后填；
2. 不写"看起来能跑但其实什么都不做"的通用函数；
3. `publish = false`、`description`、`readme.workspace = true`、`[lints] workspace = true` 四件套；
4. 第一行 `//!` 文档写清 **拥有什么 / 不拥有什么 / 依赖谁 / 被谁依赖**；
5. 至少有一条依赖边，并在 `docs/ARCHITECTURE.md` 的层表里有一行。

判定标准仍然是那句：**删掉它会谁编译失败？** 如果谁都不失败，它就该离开 `members`，
而不是在 workspace 里复制占位函数。

## 允许"只定契约、不实现"的地方

| 成员 | 现在有什么 | 缺的是实现，不是接口 |
|---|---|---|
| `weights` | `Header` / `Entry` / `Reader` 与偏移校验 | safetensors / GGUF 的头部解析器 |
| `tokenizer` | `Tokenizer` trait + 字节级参考实现 `Bytes` | BPE / SentencePiece |
| `exec` | `Stream` / `Graph` / `Node` / `Launch` / `Executor` | 真实的流与图提交（CUDA / Metal / CPU 线程池） |
| `train` | `Schedule` / `Step` / `Checkpoint` | 反传循环、优化器、数据加载 |
| `serve` | `Route` / `Endpoint` / `Response` | HTTP/gRPC 服务器 |

这些 crate 的测试只测"契约本身成立"（偏移校验、LR 调度、帧编解码、页复用），
不假装测了实现。

## 后果

- 测试数与"模块数"解耦：新增一个成员不会带来一个 `it_works`；
- `cargo test --workspace` 的每个用例都指向一条真实规则；
- 占位成员被误认为"已完成"的风险降低到"读一眼 `//!` 文档"就能发现。
