# 0047：2-bit 量化 lane 的格式口径

- 状态：Proposed（待真实权重样本评审；op 侧参考算术 lane 已按方案 A 落地，设备解包片段与 weights/nn 装载仍待后续）
- 日期：2026-11-18
- 相关：[0012](0012-quantized-weights.md)（量化权重分组与宽度同源）；[0026](0026-gemm-generic-skeleton.md)（GEMM 族泛型骨架）；[0025](0025-runtime-backend-abstraction.md)（后端边界）；`docs/batch-backend-serving-design.md` §3.5「量化 lane 挂载」、刀 3-3

## 背景

### 现状（回链当前 revision）

- 注册表 27 条算子，其中 9 条量化 lane：`gemv_i2` / `gemv_i4` / `gemv_i8`、`gemm_i2` / `gemm_i4` / `gemm_i8`、`dequant_i2` / `dequant_i4` / `dequant_i8`（`../../crates/prof/src/ops`；量化名见各 toml）。
- GEMV 骨架收口在 `crates/op/src/gemv.rs` 的 `Format` trait（`:113`）：随精度变化的关联项是 `Elem` / `EXPR` / `OPERANDS` / `GRAIN` / `DERIVED` / `SCALES` / `STEP` / `cache` / `row` / `weights` / `forward`；格式类型有稠密 `Dense<BLOCK>`（`:199`）、量化 `Quantized<GRAIN>`（`:193`）、对称 8 位 `SymI8<GRAIN>`（`:205`）。
- GEMM 骨架收口在 `crates/op/src/gemm.rs` 的 `Gemm` trait（`:57`），位宽走 `const BITS: Bits`（`:65`），打包长度走 `a_len` / `b_len`，点积落 `dot`。
- i4 实现是新格式的镜像参照（`crates/op/src/gemv_i4.rs`）：4 位打包每字节两个 nibble；`DEQUANT` 16 项对称表 `nibble − 8`；分组必须为偶数；`row = m / 2`；点积按分组推进、整组共用一条标度，低 nibble 在前。
- 反量化 `dequant_i4` / `dequant_i8` 复用 `unary` 单轴骨架（`AXES` / `project_shape`），自带代价、样本、参考与遍历（`crates/op/src/dequant_i4.rs`）。
- 位宽闭集 `Bits`（`crates/op/src/einsum/mod.rs:103`）当前取值 `Two / Four / Eight / Sixteen / ThirtyTwo / SixtyFour`，`Two` 已入闭集；`ALL` / `as_str` / `index`（`Closed` 合同）三处同步。
- weights 推断（`crates/weights/src/quant.rs`）：`bits()`（`:204`）只区分 `I8 = 8` 与其余打包 `4`；`width()`（`:346`）`U4` / `I8` 第二维即元素数，`U8` 一字节两 nibble 乘 2；**不认识 U2 / I2 打包条目**。
- `dtype` 已定义 `U2` / `I2` 枚举（`crates/dtype/src/dtype.rs:73,87`）；`qtype` 的 `Q2_0` / `Q2_K` / `IQ2_XXS` / `IQ2_XS` / `IQ2_S`（`crates/dtype/src/qtype.rs:108,122,138-142`）是 GGML 块量化族，带块内 scale/min 或查表，**不是**当前 lane 的打包方案。

### 缺口

- op 侧 2-bit 参考算术 lane 已落地（`gemv_i2` / `gemm_i2` / `dequant_i2` 注册并对拍），`Bits::Two` 已入闭集；weights 侧仍无 2-bit 打包推断，`dtype` / `qtype` 的 2-bit 定义尚未被装载路径消费。
- [0012](0012-quantized-weights.md)「未决」已注明：只有 4 位打包，2 / 3 位打包的字节数口径不同，需要各自的可测量事实，不能靠「除以 2」外推。

## 要裁决的问题

新扩 2-bit lane 前，必须先定清六件事：打包几何、码值映射、分组与标度、dtype/Wire 承载、骨架复用点、weights 装载推断口径。本 ADR 只定方案契约与接入切片，不写第二套运行时分发。

## 候选方案

### A. 对称均匀 2-bit，镜像 i4 同构骨架

- 打包：每字节 4 个 2-bit 码；行宽按位打包，`m` 为 4 的倍数，余数在解包前报 `Invalid`（对齐 i4 奇数尾守卫，不做运行期补齐）。
- 码值：4 个均匀码，对称映射；候选零点口径见「未决」。
- 标度：per-group 单条 f32 scale，与 i4 同一条派生轴 `g = m / GRAIN`；无零点、无块内 min/偏移。
- 复用：gemv 新增一份 `Format` impl、gemm 新增一份 `Gemm` impl；反量化 `dequant_i2` 复用 `unary` 骨架。
- 覆盖：对齐现有骨架模型（单 scale 派生轴），复用度最高；精度档位最低。

### B. GGML 块格式（`Q2_0` / `Q2_K`）

- 块内除 scale 外还有 min/偏移或 K-quant 混合精度结构；码值非均匀，块元数据随权重条目走。
- 贴近真实部署权重，但现有 `Format` / `Gemm` 骨架只有一条 per-group scale 派生轴，块内 min、偏移表与混合精度没有落点；需要先扩骨架契约，不是加一份 impl 能接住。

### C. IQ2 查表式（`IQ2_XXS/XS/S`）

- 块内 scale 索引一张 4 项码表；解包多一次查表，码值由块内参数决定。
- 精度表现通常优于 A，但查表与块参数同样超出当前单 scale 骨架模型，依赖格式专用元数据。

| 维度 | A 对称均匀 | B GGML 块 | C IQ2 查表 |
|---|---|---|---|
| 现有骨架落点 | 一份 impl 即可 | 需扩骨架（min/偏移/块元数据） | 需扩骨架（块内码表） |
| 标度模型 | per-group 单 scale | 块内 scale + min | 块内 scale + 查表 |
| 真实权重覆盖 | 合成/简单均匀量化 | GGML q2_0/q2_K | GGML iq2 族 |
| 参考算术复杂度 | 镜像 i4，标量可对拍 | 块结构 + 偏移 | 块结构 + 查表 |
| 本阶段可测性 | 内存夹具对拍可行 | 缺真实块格式样本 | 缺真实 iq2 样本 |

## 决策（Proposed，待评审定值）

**建议先采 A 作为 op 侧 2-bit 原型**：理由是当前 gemv/gemm 骨架的形状与代价模型只承认 per-group 单 scale 一条派生轴，B/C 的块内 min、偏移与查表在骨架上没有合法落点；先把打包长度、解包守卫、码表与对拍口径钉住，B/C 等真实块格式权重样本到位后另立格式 ADR。

以下条目在评审通过、且看到可测量的真实 2-bit 导出约定之前，**不签发定值**：

1. **码值映射**：~~对称无零点候选 `{−1.5, −0.5, 0.5, 1.5}`~~；**op 原型定值采用有符号整数码 `code − 2 ∈ {−2, −1, 0, 1}`，零点 2.0**（反量化表 `[-2.0, -1.0, 0.0, 1.0]`）。弃用半整数候选的依据：`gemm::Gemm::dot` 走 `i32` 精确累加整数路径（对应设备 mma 整数路径），半整数会迫使 gemm 改 f32 累加、偏离既有范式，也与 i4 整数码口径一致。最终映射仍以真实权重导出约定为准，评审可推翻。
2. **位序与尾码**：**原型定值低 2 位在前**（对齐 i4 低 nibble 在前）；码位 `c0 = byte & 0x03`、`c1 = (byte>>2) & 0x03`、`c2 = (byte>>4) & 0x03`、`c3 = byte>>6`。行尾不足 4 码时不补零参与点积，前置报 `Invalid`；位序在参考与内核两侧保持同一累加顺序。
3. **分组粒度 `GRAIN`**：**原型定值 32**（与 i4 同档）；`group` 必须为 4 的倍数，不整除或非 4 倍数在解包前拒绝。运行期分组仍由 `gsz` 实参给出（沿用 `Quantized<GRAIN>` 口径），最终档待真实标度条目形状样本。
4. **dtype / Wire 承载**：打包字节走 `u8` 通道（`Values::U8`）；`Bits::Two` 已入闭集。`U2` / `I2` 已在 `dtype` 枚举；**weights 装载段尚未映射到本 lane**，文件侧 2-bit 打包通道（Wire 编码、字节折算）随真实导出约定再接入。
5. **非对称零点**：本刀不带 `zero_points` / 块内 min；非对称量化等内核侧真支持时再进 `Quantized`（沿用 [0012](0012-quantized-weights.md) 未决口径）。

## 接入切片（评审通过后的落地步骤，本轮不执行）

- **op 位宽闭集**：`Bits` 加 `Two`，同步 `Closed::ALL` / `as_str` / `index` 与词汇表白名单；`bytes_for` 按 2 位计费。
- **GEMV**：新增 2-bit 格式类型，`impl Format`；`row = ceil(m / 4)`；解包表与分组守卫镜像 `gemv_i4.rs`；标量参考与内核同一段算术，对拍 `testkit::approx_eq` + 全有限守卫。
- **GEMM**：新增 `impl Gemm`，`BITS = Bits::Two`，`a_len` / `b_len` 按 4 码/字节折算，`dot` 内解包。
- **反量化**：`dequant_i2` 复用 `unary` 单轴骨架；每字节出 4 个 f32，参考逐位解包。
- **注册表**：三条算子 toml + `#[annotation::model]` 装配；测试覆盖打包/解包对拍、分组不整除、奇数尾、空输入与长度不符负例。
- **weights**：`bits()` 识别 2-bit 打包；`width()` 按 4 码/字节折算逻辑输入宽度；命名族与 4 位推断复用还是独立条目，接入时按 [0012](0012-quantized-weights.md) 同源原则定。
- **nn**：`WeightFormat` 映射新 lane，`Bounds` 分组比对沿用既有「分组不同拒绝」口径。

## 后果（若采纳 A）

- 新增一种精度只写一份 `Format` / `Gemm` impl，代价、样本装配、遍历与校验仍由骨架收口，不抄第二份。
- `Bits` 闭集扩一档，流量计费与词汇表合同单点同步。
- 2-bit 原型的参考算术与 i4 同构，可用内存夹具完成标量对拍；不依赖真实设备。

## 不做

- 本轮不写 lane 代码、不扩注册表、不新增 op/weights/nn 生产依赖边。
- 不接 Metal / CUDA 真实设备内核与 FFI（属 P3 边界，ADR-0035 评审通过前不展开）。
- 不引入 GGML 块内 min/偏移、iq2 查表或 K-quant 混合精度结构。
- 不做非对称零点、KV 量化、FP8 / AWQ / GPTQ 新格式。
- 不预埋会被第二套块格式实现推翻的泛型抽象。

## 未决

- 码值映射（原型用有符号 `code − 2`）与 `GRAIN=32`、低 2 位在前：op 参考算术已按此落地，但仍属原型口径；待真实 2-bit 权重导出约定复核，评审可推翻。
- weights / nn 装载映射：`bits()` / `width()` 尚未识别 2-bit 打包，`WeightFormat` 无 Int2 档；文件侧 2-bit 打包的 Wire 编码与字节折算通道未建。依赖真实导出格式，不在本刀预埋第二套抽象。
- `U2` / `I2` 与 Wire、`Values::U8` 通道的映射边界。
- 2 位设备解包片段（cutile prim）随 P3 量化内核补齐；当前模板只定逻辑与签名占位。
- B / C 块格式何时按格式 ADR 接入；本 ADR 不预设其骨架扩展形态。
- 当前缺真实 2-bit 权重样本；无可测量事实前，本 ADR 保持 Proposed，不转 Accepted。

## 修订

- 2026-11-18：初稿 Proposed；仅定方案契约与接入切片，未落地代码。
- 2026-11-18：方案 A 的 op 侧参考算术 lane 落地（`gemv_i2` / `gemm_i2` / `dequant_i2` 三算子注册、对拍与架构/静态门禁通过）；回填原型定值——整数码 `code − 2`（零点 2.0）、每字节四码低 2 位在前、`GRAIN=32`、组须 4 倍数、非 4 倍数尾码前置报 `Invalid`。weights/nn 装载与设备片段未接，保持 Proposed。
- 2026-11-18（M3-3 刀）：三条 i2 lane 接入 serve 注册 Program（泛型桥接 + 配置化宏展开），gemm 形状矩阵对拍与测试侧独立闭式点积重算、零点/分组负例收口骨架；回填注册表 27 条 / 9 条量化 lane 口径。weights/nn 装载映射与设备解包片段仍未接。
