# `llm-conv`

神经算子：卷积 / 反卷积（`conv` / `deconv`）、注意力（`attention`）、池化与范数（`classic`）、
扫描骨架（`scan`）。几何是自由的，算术不是自己的 —— 位置映射求出之后交给 `gemm` 的稠密路径。

四个 crate 构成一条单向偏序 —— `spec` → `gemm` → `conv` → `op`。

| 层 | crate | 负责 |
| --- | --- | --- |
| 契约 | `llm-spec` | 签名 ABI、布局/调度的闭集、模板词汇、代价表达式、语义模型、确定性样本 |
| 稠密核 | `llm-gemm` | gemv/gemm/matmul、量化、逐元素、一元、归约、归一化、激活、SIMD/SME |
| 结构化核（本 crate） | `llm-conv` | 卷积/反卷积、池化与范数家族、注意力、扫描 |
| 注册表 | `llm-op` | `ops/*.toml` 声明、`REGISTRY`、`builtin()`、`model(name)` |

## 与 `gemm` 的分界

分界不是「精度」而是**形状自由度**：内核算的是「一行乘以一列」，几何固定；本 crate 算的是
「窗口滑过张量」，同一份算术要在 padding / stride / dilation / 分组下都成立。因此每个模块先求出
位置映射（`conv::geom`），再把映射交给内核的稠密路径（`llm_gemm::fast`），**自己不写第二份乘加循环**。

```rust
// 几何归本 crate，算术归 gemm：换 padding/stride 不改内核，换精度不改几何。
assert!(llm_conv::same_bytes("conv2d", "conv2d"));
```

## 边界

- **依赖单向。** 本 crate 依赖 `gemm` 与 `spec`，反之不成立；注册表在 `op`，于是「神经算子是否与
  声明一致」由 `op` 的跨层测试核对，而本 crate 不必知道算子名。
- **无 `unsafe`。** `deny(unsafe_code)` 无条件开启：这里是索引算术与形状推导，不需要裸指针。
- **注意力只有解码路径。** `attention::decode` 是 KV 缓存上的单步解码；预填（prefill）走 `gemm`，
  不在这里复制一份分块矩阵乘。

## 状态

`publish = false`，与工作区其余 crate 一致：工作区内 API 稳定，但不承诺 MSRV 与 semver。
