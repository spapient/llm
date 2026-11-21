# 0046：序列批量几何——扁平批缓冲与内核批维，图不复制实例（M2-2）

- 状态：路线与布局合同定型；op 批维地基与 Predict 批量视图接口已落地，engine 缓冲扩批与 serve 整批前向分刀落地；2026-11-18 补 §7 ragged 历史 K/V 段映射合同、§8 ragged 段长表通道
- 日期：2026-11-17（2026-11-18 补 §7 ragged 历史段布局、§8 段长表通道）
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

### 3. 等长批先行；变长 ragged 历史段合同见 §7

等长批（统一 `hist`）先把扁平布局与段编排打通。op 变长批段入口 `decode_forward_batch_var` /
`decode_flash_forward_batch_var`（各段独立 `hist_b`）已落地；serve 装填 / 写回侧的 ragged 段映射
合同收口在 §7。prefill 多 token 主链与 chunked prefill 仍按 §3.8 排除边界暂缓。

### 4. 内核批维编排复用既有单序列 / 多头内核

op 新增 batch 入口：按段 stride 切出每条序列，段内复用现有多头 / 单头内核；段间独立 softmax、
互不串段。对拍 oracle：

- 批内各段结果互不串段；
- `batch = 1` 退化等价多头入口；
- 缓冲长度不是段长整数倍、段宽不符时返回 `Invalid`。

### 5. Predict 批量视图：单条方法保留，批量入口已定稿

[`Predict::logits`](../../crates/serve/src/generate.rs) 单序列方法保持原样作为默认与回归基线。批量入口签名定稿为：

```text
logits_batch(&[Sequence], &mut Batch) -> Result<Vec<Vec<f32>>>
```

- 输入是只读序列快照切片；返回每行一条 logits，行序与输入切片一一对应（即 [`Batch::sequences`](../../crates/engine/src/batch.rs) 的准入顺序），调用方按序号对齐，不按 id 重排。
- 默认实现逐条回退 `logits`，与逐串前向行为等价；内核整批通路落地后由模型实现覆写本方法，不改 trait 形状。
- 任一条失败即上抛，已记账序列的页账不在模型层回滚，归调度器统一归还。

### 6. 批数几何通道：长度反推，不进标量槽、不占 grid 维

沿用既有内核几何全部从缓冲长度反推的范式（头数 `q.len()/d`、历史行数 `k.len()/(heads·d)`、查表行宽 `out.len()/tokens`）：

- 批数 `batch = 整块缓冲长 / 装配期单序列段长`；单序列段长是装配台账事实（激活 `activations` 登记宽、注意力块 `kv_width·heads`），不运行期猜测。
- 批数是几何事实，不复制进冻结标量槽（ADR 0043）；[`Launch`](../../crates/engine/src/exec.rs) 的 `grid` 三轴当前不承载批维，`block` 第二维仍只承载头宽。批内各段头数、头宽、段长一致，段内复用多头 / 单头 / 行批内核。
- 缓冲容量在装配期段长已知；运行期按本轮批长绑定整块，一份编号不随批数翻倍。

### 7. 变长 ragged：历史 K/V 段紧排映射与写回拆行

op 变长批段入口已按紧排游标切段；serve 装填 / 写回侧沿用同一份段几何，不在引擎层引入 stride：

- **段布局**：历史 K/V 扁平一块，`[batch, heads, hist_b, d]`，段间紧排。第 `b` 段起点为累计游标
  `Σ_{j<b} heads·hist_j·d`，段长 `heads·hist_b·d`，缓冲总长 `Σ_b heads·hist_b·d`；与 op
  [`decode_batch_var`](../../crates/op-neural/src/attention.rs) 的 `kv_cursor` 累计口径一致，两侧不各写一份
  段算术。`hist_b = sequence.len()`，按准入序对齐。
- **段内复用既有重排**：段内仍是头优先块 `[head, token, d]`，沿用现有 `reorder_history_row` 头分段
  整段 `copy_from_slice`；ragged 只改段间偏移，不改段内布局。
- **装填**：本轮批历史缓冲总长按各段 `hist_b` 之和预留；第 `b` 段内第 `token` 行经
  `batch.kv(id, token)` 取页行，映射到该段偏移写进 `hist_k` / `hist_v`。段偏移算术与子区间切片归
  serve 适配层——engine `View` / `ViewMut` 是整块切片、不带 offset / stride（见取证备注），引擎不
  解释段布局。
- **写回拆行**：当前步 K/V 只取该序列段内当前行，拼成 `[k|v]` 页行经 `batch.write_kv(id, position, row)`
  写回各自页账；不跨段读写，页行归属仍按序列号。
- **编号不翻倍**：历史 K/V 仍是两个编号（`hist_k` / `hist_v`），不为 ragged 多开编号；段长运行期按
  本轮 `hist` 表绑定整块，沿用 §2 编号分区口径。

批装填 / 写回函数与块链整批 `logits_batch` 覆写随整批接线刀落地；本 ADR 只定段映射与写回归属，不
预埋空函数或第二份段游标。

### 8. ragged 段长表：逐序列历史长度走独立输入缓冲

变长批各段 `hist_b` 不同，op 批段入口 `decode_forward_batch_var` /
`decode_flash_forward_batch_var` 收 `hist: &[usize]`。这张运行期表的通道归属同 ADR 0043 §5
位置表口径：

- **段长是逐序列运行期几何，ragged 下不能从缓冲总长反推。** 历史 K/V 总长
  `Σ_b heads·hist_b·d` 是各段长度之和；等长批可由总长 `/(batch·heads·d)` 推统一 `hist`，
  ragged 各段不同，分段无唯一解。`heads` / `d` 是装配期静态几何，走 op `MODEL.shape` 与
  Launch 头宽维（§3 / §6），不进段长表。
- **不进标量槽、不占 grid / block 维。** 标量槽闭集冻结在 ADR 0043（无变长数组字段）；
  批数与段长几何不复制进槽，也不塞进 Launch 三轴。
- **走既有输入缓冲切片通道**：一条**独立输入操作数**，长度 = `batch`，`hist[b]` 对齐准入序
  第 `b` 段；元素以 `f32` 承载、内核侧 `as usize` 取回（历史长度远小于 2²⁴，整数无损，越界
  由装配与算子校验守）。不为这张表新增整数 `Buffer` / `View` 变体（lane 仍仅 F32 / U8）。
- **不与特征缓冲混编**：段长表是与批同宽的独立运行期表，与 q/k/v 特征缓冲分路绑定；
  `View` 整块无 offset / stride，段对齐由装配绑定完成，沿用现有批缓冲口径。
- **等长批不并存两套来源**：整批 `hist` 一致时可走 §6 长度反推；一旦某节点接段长表，就不再
  从缓冲总长反推各段，避免双写漂移。
- **生命周期**：段长表每步随批快照覆写，与位置表、输入缓冲同生命周期；装配期只登记输入编号，
  不预埋表数据或空节点。

装配期给批 attention 节点多绑一条段长表输入编号、内核从 `inputs` 取表、每步覆写，随块链整批
接线刀落地；本 ADR 只定通道归属与形状。

## 备选方案

- **路线 A：每序列一套编号实例**——拒绝。编号台账与节点数 ×N，装配期按批展开实例；多头阶段已否
  装配层展开实例路线。页账按序列独立只要求历史读取按归属取行，不要求缓冲编号按序列复制。
- **合同定型前先铺变长 ragged 接线**——拒绝。段映射合同（§7）未定前不在装配层堆变长历史搬运；
  合同定型后，装填 / 写回函数随块链整批接线刀落地。
- **批数进标量槽**——拒绝。批数是几何事实，标量槽闭集冻结在 ADR 0043；几何通道由 Launch 几何维
  或缓冲形状派生（engine 接线刀最终化），不新增字符串键 / `HashMap`。
- **ragged 段长从历史缓冲总长反推**——拒绝。总长是各段 `heads·hist_b·d` 之和，ragged 下分段
  无唯一解；等长才可反推，变长必须显式段长表（§8）。
- **为段长表新增整数 `Buffer` / `View` lane**——拒绝。为单张运行期表扩 engine 通道面；复用既有
  f32 lane 折整数承载，与位置表、查表 id 先例一致（ADR 0043 §5）。

## 后果

- 已落地：op batch 入口与段编排（#31）；`Predict::logits_batch` 批量视图接口，默认逐条回退（#33）。
- 剩余接线按依赖序分刀：① 激活 / 历史缓冲容量按批长预留；② 查表多行输入放开（`tokens` 批口径）；③ 投影行批（算术路径按下文"接线前置裁决"选定），m=批内段数、共享驻留权重；④ decode 注意力接 op batch 入口；⑤ rope 按段独立位置循环；⑥ K/V 历史段拼接装载、按段写回页行；⑦ `step_batch` 改走整批一次 `logits_batch`。
- 不变量沿用调度层对拍：批内段隔离、页表归属不违反、失败序列不回滚好序列。
- 暂缓：序列并行、prefill 多 token 矩阵主链、chunked prefill、头内并行特化；变长 ragged 历史的
  装填 / 写回接线（段合同见 §7）随块链整批刀落地。
- 已定通道：批数承载位置走 §6 长度反推，不进标量槽 / grid；变长 ragged 各段独立 `hist_b` 的段
  映射与写回拆行见 §7；逐序列段长表走独立输入缓冲切片（§8），不进标量槽、不新增整数 lane。
  剩余未定只在装配 / 接线实现侧，不在通道归属。

## 接线前置裁决（engine / serve 动工前取证）

serve 整批前向动工前，下面三条是已从源码取证的阻塞契约缺口；在裁决落地前不铺图接线：

1. **投影行批算术路径**（已裁决：方案 A）。`gemv_f32` 输入 / 输出宽度任意，但是单向量几何；`gemm_f32` 行批骨架的收缩轴 `K=8` 是固定档（[`gemm::K`](../../crates/op-kernel/src/gemm.rs)），不接任意 `in_features` 的真实驻留权重。两条候选：
   - **A. 行循环 gemv（采用）**：批内每段调一次既有 `gemv_f32`，共享同一份权重；实现直接、不动 op 契约，但内层按行循环。
   - **B. gemm 收缩轴运行期化（搁置）**：整块矩阵乘符合批处理偏好，但要先证明放宽 `K` 不破坏 `gemm_i4` / `gemm_i8` 的分组打包契约与样本 / 校验。
2. **归一化行分组**。`rmsnorm_f32` 行分组 op 入口 `forward_batch` 已就位（各段独立 RMS、`gamma` 按行宽复用）；serve 内核适配接行批分段仍待接线。
3. **缓冲绑定时点**。装配期只存单序列段长（台账 `activations` 登记宽、块 `kv_width`）；整块容量在运行期按本轮批长绑定，编号不随批数翻倍。

### 取证备注（接线刀不要误判）

- [`embedding_plan`](../../crates/serve/src/plan.rs) 的 `tokens != 1` 拒绝的是**层声明**为多 token 的 prefill 矩阵层，不是运行期批大小；decode 模型层声明恒为 `tokens=1`，运行期一次喂多条索引的几何由内核适配从输出长度反推 `tokens = out.len()/dim`，放开多行输入不需要动这条装配期形状检查。
- embed / gemv / rmsnorm 当前都走 serve `binary` 适配宏，几何全部从缓冲长度反推；批维接线时这些适配函数要按段 stride 切片或切到对应 op batch 入口。
- engine [`View`](../../crates/engine/src/exec.rs) / `ViewMut` 是整块 `&[f32]` 切片，**不带 offset 或 stride**；扁平批缓冲的段区间由适配层在拿到整块后用切片子区间（`split_at` / 步长索引）切出，再交给 op batch 入口，不在引擎视图层引入偏移描述。
- 方案 A 行循环 gemv 的适配需要输入宽 `m` 与输出宽 `n`：单向量几何下 `m=x.len()`、`n=out.len()` 可自洽反推；行批整块 `x` 为 `batch·m`、`out` 为 `batch·n`、权重 `n·m`，单从长度拆不出 `batch/m/n`。动工前要把 `in_features` 一并送进节点 Launch 几何（当前投影节点 Launch 第二维只承载头宽，投影只登记了 `out_features`），适配再按 `m` 切段逐行调 `gemv_f32`，行间共享同一份驻留权重。
- 调度侧 [`step_batch`](../../crates/serve/src/completion.rs) 现状逐条 `logits → 采样 → 记账`；整批一次 `logits_batch` 后再逐条采样，失败条目的页账不回滚语义要在模型整批覆写时一并对拍，不能只切调用点。
- batch=2 端到端回归基线夹具已补（默认逐条回退下两行 logits 与各自 KV 归属对拍）；整批几何接线后沿用同一份对拍口径。
