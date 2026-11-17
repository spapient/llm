# 0043：程序标量通道——定长类型化闭集标量槽（D9）

- 状态：草案（M1-B6 落地槽位结构与 Node/Launch 承载；rope 算子接线与标量注入闭包随对应刀落地）
- 日期：2026-11-17
- 相关：[0027](0027-kernel-registry-and-fragment-contract.md)（注册表与算子挂载）；设计取证见 [`../backend-runtime-design.md`](../backend-runtime-design.md) §5.1 D9、[`../llmu-parity-optimization-design.md`](../llmu-parity-optimization-design.md) §1.5 P5/§4.2 M1-B6

## 背景

`engine::exec::Launch` 此前只有 `grid` / `block` 三轴几何；像 `rope_f32` 这类算子还需要
**解码位置**这类语义标量。落地前有三条容易走偏的通道：

1. 把位置编进一个 1 元素 f32 缓冲，走 `Program::run` 的 `inputs` 切片通道——标量被当成张量
   参与 dtype / 形状校验，缓冲通道混入语义参数；
2. 在执行上下文里放字符串键 → 值的 `HashMap`，按算子名查表取标量——热路径出现 map 查找，
   槽位集合没有闭集约束；
3. 把头数 / 头宽这类静态几何也塞进标量槽——几何事实已有 `op::MODEL.shape` 唯一来源，复制
   一份会漂移。

## 决策

### 1. 定长类型化闭集结构挂在 Launch / Node

新增 [`ScalarSlots`](../../crates/engine/src/exec.rs)：定长、类型化的标量槽结构，随 `Launch`
拷贝、由 `Node` 透传。首批槽位**仅** `rope_position: Option<u32>`（`rope_f32` 的 `pos`）。
槽位集合冻结在本 ADR：新增槽位必须修订本 ADR 并在结构上加类型化字段，不开放任意键值对。

### 2. 缓冲切片通道签名不变

[`Program::run`](../../crates/engine/src/exec.rs) 的 `inputs` / `outputs` 切片签名保持原样；
位置这类语义参数不编进 1 元素缓冲。标量槽不是第二份缓冲账，也不是标量上下文 `HashMap`：
无字符串键查表，热路径固定结构直取字段。

### 3. 静态几何仍取算子声明

`heads` / `head_dim` 等静态几何继续以 `op::MODEL.shape` 为唯一事实源；标量槽只承载运行期
逐位置变化的语义标量，不复制几何宽度。

### 4. 未接算子仍按名拒绝

`rope_f32` 尚未接入模型路径：槽位结构就位，但该算子在 serve 内核子集里仍按名拒绝（既有
`an_unregistered_kernel_is_refused_by_name` 口径不变）。标量到内核闭包的注入随 rope 接线刀
一起落地；本刀不预埋第二份算子映射。

## 备选方案

- **位置走 1 元素 f32 缓冲**：拒绝。标量被拉进张量形状/dtype 校验，缓冲通道混入非张量参数，
  后续每颗内核都要分辨"这是缓冲还是标量"。
- **字符串键 `HashMap` 标量上下文**：拒绝。热路径 map 查找；槽位无闭集，拼写错误到运行期才
  暴露，与既有闭集常量表口径不一致。
- **几何复制进标量槽**：拒绝。头数/头宽已有 `op::MODEL.shape` 事实源，双写必然漂移。

## 后果

- 新增标量槽位 = 改 `ScalarSlots` 结构 + 修订本 ADR；不引入运行期键表。
- engine 不解释槽位语义：字段只承载数值，语义归属算子声明与装配侧。
- 已落地回归：默认槽位全空；rope 位置经 Launch → Node 往返一致（`engine` exec 测试）。
