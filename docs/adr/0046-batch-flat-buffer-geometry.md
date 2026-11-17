# 0046：序列批量几何——扁平批缓冲与内核批维，图不复制实例（M2-2）

- 状态：草案（路线与布局合同定型；op 批维地基先行，engine 缓冲扩批与 serve 批量视图分刀落地）
- 日期：2026-11-17
- 相关：[0043](0043-program-scalar-slots.md)（标量槽闭集，语义参数不进几何维）；[0044](0044-bindings-metadata.md)（编号到缓冲，引擎不解释形状）；[0025](0025-runtime-backend-abstraction.md)（后端三 trait 分立）

## 背景

调度层连续批已落地：[`Batch`](../../crates/engine/src/batch.rs) 遍历序列出 `Progress`、`emit`/`finish`
归还页，两条同批序列各自历史、挡页保留、失败隔离、KV 行按序列号归属的对拍已绿。剩余缺口是
**模型一步同时处理多条序列的内核几何**：[`Predict::logits`](../../crates/serve/src/generate.rs) 仍是
单序列签名，[`Bindings`](../../crates/engine/src/exec.rs) 编号到单个 `Buffer`，图节点单实例提交；
`step_batch` 当前逐条串行前向。

序列批不能照搬两条已被否掉的弯路：

1. **装配层为每条序列复制一套图实例与编号**——节点数与编号台账 ×N，与多头阶段否决的
   "展开 H 组节点 + slice 算子"路线同类；权重编号被批数放大，装配期就要知道批大小。
2. **把批数塞进冻结标量槽**——ADR 0043 标量槽只承载 `rope_position` 这类语义参数；批数是几何
   事实，复制一份进标量槽会与缓冲形状漂移。

多头（路线 B）已落地：头优先块 `[heads, q_rows, d]`，图上不展开 H 组节点，头分段编排复用单头内核。
序列批沿用同一原则——几何扩张落在缓冲分段与内核编排，不落在图实例数。

## 决策

### 1. 扁平批缓冲，批维加在最外、段连续

物理缓冲仍是一维 `f32` 整块；逻辑轴序与段 stride 由派生形状算出，不存字符串键：

- 激活 / 投影：`[batch, heads, q_rows, d]`；decode 一步 `q_rows = 1`，第 `b` 段连续
  `heads · q_rows · d`。
- 历史 K/V：`[batch, heads, hist, d]`，第 `b` 段连续 `heads · hist · d`。
- 查表输入：token 索引 `[batch]`，输出 `[batch, d]`；embed 多行查表能力已就位。

### 2. 编号不翻倍，图节点不复制

沿用 [`assembly.rs`](../../crates/serve/src/assembly.rs) 的激活段 / 权重段编号分区：单套激活编号，
一份 `Buffer` 容量 = `batch · 单序列长度`；图节点不为序列复制第二组编号。权重编号与批数无关，
一份驻留权重被所有批内段复用。

### 3. 本刀限等长批，变长历史留后续刀

本阶段批内各序列历史长度一致（统一 `hist`）。continuous batching 的变长 ragged stride、各段独立
`hist_b` 映射不在本刀定；待 engine 缓冲扩批刀再立切片合同。prefill 多 token 主链与 chunked prefill
仍按 §3.8 排除边界暂缓。

### 4. 内核批维编排复用既有单序列 / 多头内核

op 新增 batch 入口：按段 stride 切出每条序列，段内复用现有多头 / 单头内核；段间独立 softmax、
互不串段。对拍 oracle：

- 批内各段结果互不串段；
- `batch = 1` 退化等价多头入口；
- 缓冲长度不是段长整数倍、段宽不符时返回 `Invalid`。

### 5. Predict 批量视图：单条方法保留，批量入口后续接

`Predict::logits` 单序列方法保持原样作为默认与回归基线。批量入口
`logits_batch(&[Sequence], &mut Batch) -> Vec<Vec<f32>>` 的最终形状在 engine 接线刀定稿；op /
engine 阶段不接 serve 批量方法。单条前向可走批量入口 `batch = 1` 退化，但不要求改造既有调用。

## 备选方案

- **路线 A：每序列一套编号实例**——拒绝。编号台账与节点数 ×N，装配期按批展开实例；多头阶段已否
  装配层展开实例路线。页账按序列独立只要求历史读取按归属取行，不要求缓冲编号按序列复制。
- **变长 ragged 批先行**——拒绝。各段独立 `hist_b` 的 stride 与历史映射合同尚未取证落地，先在
  等长批把扁平布局与段编排打通，变长在 engine 缓冲扩批刀单独提案。
- **批数进标量槽**——拒绝。批数是几何事实，标量槽闭集冻结在 ADR 0043；几何通道由 Launch 几何维
  或缓冲形状派生（engine 接线刀最终化），不新增字符串键 / `HashMap`。

## 后果

- op 补 batch 入口与段编排；engine 缓冲容量按批扩；serve 批量视图在后续刀接 `step_batch` 整批前向。
- 不变量沿用调度层对拍：批内段隔离、页表归属不违反、失败序列不回滚好序列。
- 暂缓：序列并行、prefill 多 token 矩阵主链、变长 ragged 历史、chunked prefill、头内并行特化。
- 未定通道（PARTIAL）：批数在 Launch 几何维 / 缓冲形状上的最终承载位置，待 engine 缓冲扩批刀取证
  定型；本刀 op 入口以显式 `batch` 参数与段 stride 承载，不预埋 engine 侧通道。
