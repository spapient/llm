# `llm-gemm`

稠密计算核：矩阵乘族（`gemm` / `gemv` / `matmul`）、激活（`activation`）、归一化（`norm`）、
反量化（`dequant`）、逐元素（`elementwise` / `unary`）、归约（`reduce`），以及 SIMD / SME 通道（`fast`）。

四个 crate 构成一条单向偏序 —— `spec` → `gemm` → `conv` → `op`，箭头指向用户的相反方向，
于是没有哪一层能回读更靠后的一层。

| 层 | crate | 负责 |
| --- | --- | --- |
| 契约 | `llm-spec` | 签名 ABI、布局/调度的闭集、模板词汇、代价表达式、语义模型、确定性样本 |
| 稠密核（本 crate） | `llm-gemm` | gemv/gemm/matmul、量化、逐元素、一元、归约、归一化、激活、SIMD/SME |
| 结构化核 | `llm-conv` | 卷积/反卷积、池化与范数家族、注意力、扫描 |
| 注册表 | `llm-op` | `ops/*.toml` 声明、`REGISTRY`、`builtin()`、`model(name)` |

```rust
// 名字按字节比较：`str` 相等在本书场链上不是 const 的。
assert!(llm_gemm::same_bytes("gemm_i8", "gemm_i8"));
```

## 边界

- **只认识契约，不认识声明。** 本 crate 依赖 `spec`，不依赖 `op`、`conv`，也不读 `ops/*.toml`；
  内核不知道自己是「哪个算子」，只回答「给定这批缓冲，怎么算」。于是「内核能否违反声明」
  这一问在编译期就有答案 —— 它根本没有声明可看。
- **一份骨架，多种精度。** 泛型实现（求和、代价、样本、校验、遍历）只写一次；每个格式只写
  格式契约（权重元素、位宽、粒度、解包），按格式单态化，热路径上没有运行时分发。
- **`unsafe` 被约束在岛上。** `deny(unsafe_code)` 是默认；`fast` 里的 NEON/SME 子模块是显式登记的
  岛（见 `testkit` 的 `UNSAFE_ISLANDS`），岛外不允许出现 `unsafe`。
- **正确性按对拍验收。** 向量化/分块路径与标量路径逐位相等，每条路径都有对拍真值，
  见 `tests/{gemv,gemv_i8,matmul_i8_tests,fast_tests,sme_tests}.rs`。

## 状态

`publish = false`，与工作区其余 crate 一致：工作区内 API 稳定，但不承诺 MSRV 与 semver。
