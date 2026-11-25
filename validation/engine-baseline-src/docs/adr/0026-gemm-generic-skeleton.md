# 0026：GEMM 族泛型骨架边界

- 状态：已接受（f32 / i8 / i4 三格式均已迁入骨架）
- 日期：2026-11-15
- 相关：[0012](0012-quantized-weights.md)（量化权重同源）；对称范式见 `crates/op/src/gemv.rs` 的 `Format` 骨架；路线见 [`../engine-benchmark-roadmap.md`](../engine-benchmark-roadmap.md) §5

## 背景

`crates/op` 的矩阵乘一族有三个格式：`gemm_f32`、`gemm_i8`、`gemm_i4`。重构前三者各自抄了一遍与精度无关的部分（回链源码）：

- 代价三轴取值与 `Cost` 装配（`crates/op/src/gemm_f32.rs:17-26`、`gemm_i8.rs:18-27`、`gemm_i4.rs:24-33`）；
- 样本三缓冲（`a` / `b` / `y`）与标量 `M` / `K` / `N` 装配（`gemm_f32.rs:29-49`、`gemm_i8.rs:30-58`、`gemm_i4.rs:46-75`）；
- `reference` 取缓冲 → 矩阵遍历 → 输出分配（`gemm_f32.rs:52-60`、`gemm_i8.rs:82-90`、`gemm_i4.rs:99-109`）；
- `forward` 尺寸校验 → 矩阵遍历（`gemm_f32.rs:67-79`、`gemm_i8.rs:97-103`、`gemm_i4.rs:116-126`）；
- `matmul` 双重外层循环（`gemm_f32.rs:83-90`、`gemm_i8.rs:72-79`、`gemm_i4.rs:89-96`）。

三者真正的差异只有五点：**元素类型**（f32 / u8）、**输入位宽**（32 / 8 / 4）、**样本取值序列**、**打包长度换算**（i4 每字节两个 nibble）、**点积算术**（浮点乘加 vs. i8→i32 精确累加）。这与 `gemv::Format` 已经收敛的权重格式是同一类问题：遍历、校验、样本外壳与精度无关，算术与打包属于格式契约。

## 决策

### 1. 抽取 `Gemm` trait，与 `gemv::Format` 对称

在 `crates/op/src/gemm.rs` 定义 `pub trait Gemm`，只承载随精度变化的关联项：

- `type Elem: Element`：样本缓冲元素类型（复用 `gemv::Element` 的 f32 / u8 闭集与装箱/借用）；
- `const OP: &'static str`：诊断里点名的算子名；
- `const BITS: Bits`：A / B 输入元素位宽；输出恒为 f32；
- `fn sample_a(m) -> Vec<Elem>`、`fn sample_b(n) -> Vec<Elem>`：确定性样本取值；
- `fn a_len(m) -> usize`、`fn b_len(n) -> usize`：打包后的缓冲长度校验；
- `fn dot(a, b, n, i, j) -> f32`：输出位置 `(i, j)` 的点积。

### 2. 骨架收口与精度无关的流程

`gemm.rs` 提供泛型函数 `cost::<G>`、`sample::<G>`、`matmul::<G>`、`reference::<G>`、`forward::<G>`：

- **代价**：`flops` 固定 `2·m·k·n`；`read` 由 `G::BITS.bytes_for(...)` 按位宽计费（nibble 打包走 `ceil` 口径，与 einsum 层一致）；`write` 固定 `m·n·4`。
- **样本**：装配 `a` / `b`（格式取值）与 f32 零缓冲 `y`，标量 `M` / `K` / `N`。
- **遍历**：双重外层循环只写一份；内层点积落在 `G::dot`。
- **校验**：`a_len` / `b_len` / `m·n` 三项长度判据统一在骨架。

### 3. 固定收缩轴保留为编译期常数

`K = 8` 继续作为 `gemm.rs` 的 `pub const`，不进 trait、不做成运行时参数。它是当前标量原型的成本档；tile / SIMD 化在 P4 后端另立。

### 4. 编译期单态化，不引入运行时分发

trait + 关联常量按格式单态化：热路径不出现 `enum` / `dyn` 分派，也不在遍历里 `match` 元素类型。各格式模块继续导出**具体签名**的 `cost` / `sample` / `reference` / `forward` 薄函数转发到泛型骨架——注解生成器对 `gemm` 骨架走 `generic = false`，按模块名接具体函数（`crates/annotation/src/codegen.rs:212`），这一接线不变。

### 5. i4 打包解包落进 `dot`

i4 的 nibble 解包（低 4 位 / 高 4 位各减零点 8）作为 [`Gemm::dot`] 的内部细节：点积时按字节 / nibble 位置按需解包 A 行与 B 列，不再预先把整张 A、B 解包成临时 `i8` 矩阵。打包后的缓冲长度由 `a_len` / `b_len` 声明（每字节两个 nibble）。位域索引随元素位置变化，这一处循环保留并注释，其余遍历仍由骨架收口。

### 6. 注解生成器与零依赖契约不变

不引入 `syn` / `quote` / `proc-macro2`；本轮不改 `SKELETON_TABLE`，不新增注解代码生成。样本种子、零点、位宽等数值事实留在格式 impl，不进 TOML 配置。

## 后果

- 新增一种精度 / 量化格式：只写一份 `Gemm` impl，不再抄代价、样本装配、遍历与校验。
- 三重循环、长度判据、流量口径各留一份事实源，三格式不会各自漂移一处。
- 三格式行为与标量对拍保持不变；i4 去掉了 reference / forward 各一份整矩阵解包的临时 `Vec` 分配。

## 不做

- 本轮不把 GEMM 迁到 einsum `FormulaCache` 表达式推导；代价先按 `BITS` 常数在骨架内计算，口径与现状逐项对齐。
- 不做 tile 分块、SIMD 或设备指令（dp4a / mma）映射；那是 P4 后端边界内的事（ADR 0025）。
- 不抽取注解生成 `cache()` 静态单元或薄 `forward` 转发；属 P2 评审项。
- 不为减行数删除诊断注释或把样本取值 / 数值常数压进配置。

## 修订

- 2026-11-15：初稿；三格式（f32 / i8 / i4）全部迁入骨架，i4 解包落进 `dot`。
