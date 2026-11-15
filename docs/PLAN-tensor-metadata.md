# `tensor` 元数据层收口与薄模块方案

- 版本：v1.0
- 日期：2026-11-16
- 状态：已实施（2026-11-16）
- 适用范围：`crates/tensor`（L4 运行时元数据层）
- 配套契约：ADR 0001（分层依赖）、ADR 0003（kernel/运行时类型分离）、ADR 0004（占位策略）、ADR 0025（后端抽象与张量双形态）、`ARCHITECTURE.md` §1/§3 所有权表、audit.md §6.3 S2（步幅单一主人）

---

## 1. 不可违背的不变量

1. **依赖方向**：`tensor` 只能依赖 `dtype`、`diagnostic`，不引入任何新运行时依赖（ADR 0001）。
2. **零 unsafe**：生产核心 `#![deny(unsafe_code)]` 不放宽；视图零拷贝由生命周期类型保证，不靠裸指针。
3. **一个类型一个主人**：算子、模型语义、布局搬运、设备句柄、反传图都不归 `tensor`（ADR 0002/0003/0025）。
4. **无消费方不预埋**：删掉新增的接口后没有任何编译失败点，就不留在层内（ADR 0004）。
5. **公开面只减不增**：本次不新增 `pub struct|enum|trait`，`ARCHITECTURE.md` 所有权表与 `lints/vocabulary.toml` 基线不需要改动。
6. **既有测试用例数不跌**：`tests/tests.rs` 既有白盒用例全部保留；只给收口后的同一批不变量补对等用例。

## 2. 现状基线（取证事实）

| 项 | 现状 |
|---|---|
| 公开类型 | `Tensor<'a>`（字节只读视图）、`View<'a, E>`（元素零拷贝视图）、`Element`（标量→`DType` 映射）、`Shape`、`Strides`、`OwnedTensor` |
| 公开函数/常量 | `SUPPORTED` 子集表、`supports(dtype)` |
| 外部消费方实际用到的面 | 仅 `Shape::new` / `dims` / `rank` / `numel` / `element_bytes`、`Strides::new` / `contiguous` / `as_slice` / `get`、`Tensor::new` / 访问器、`OwnedTensor` 构造与字节路径（`storage`、`weights`、`nn`、`engine`、`data`、`autograd`、`train` 已核实） |
| 已知重复 | 连续性判定在 `Tensor` / `View` / `OwnedTensor` 三处各自构造一份 `Strides::contiguous` 再逐 Vec 比较；步幅算法本身只在 `Strides` 一处 |
| 模块结构 | 全部实现体堆在 `src/lib.rs`（516 行），与 `dtype`「薄 lib + 分文件」的约定不一致 |
| 路线定位（roadmap P1） | 只读 `&'a [u8]` 视图 + 可写 `OwnedTensor`（元数据 + 拥有字节向量）；布局/stride 与设备解耦；未规划在本层实现形状/视图算子面 |

## 3. 边界裁决：上一轮新增项逐项归属

上一轮按通用框架（Burn）的 Tensor 方法面往本层搬了一批能力。对照所有权表逐项裁决如下：

### 3.1 越界项：从 `tensor` 删除

| 新增成员 | 形态 | 应属层 / 落点 | 理由 |
|---|---|---|---|
| `Dim`（`Fixed`/`Infer`）、`Shape::reshape` | 重排算子 | `op` 声明形态算子 → `nn` 绑定 → `train` Kernel 执行 | 改变张量形状是算子语义，不是元数据描述；本层没有执行上下文 |
| `Shape::flatten` | 形状合并算子 | 同上 | 同上 |
| `Shape::squeeze` / `unsqueeze` / `permute`、`Strides::permute`、`Geometry::permute` / `transpose` / `squeeze` / `unsqueeze`、`View::permute` / `squeeze` / `unsqueeze`、`Tensor::permute` | 轴序/轴数变换算子 | 同上；需要数据重排时搬运归 `storage` | 变换产出的是"另一个张量布局"，属于算子/存储职责 |
| `Geometry::narrow`、`View::narrow`、`Tensor::narrow` | 切片视图算子 | 同上 | 窄切片是算子级视图派生；字节视图下亚字节（`U4`）的位偏移语义本层也不完整 |
| `Strides::broadcast`、`Geometry::broadcast_to`、`View::broadcast`、`Tensor::broadcast_to`、`Shape::broadcast_with` / `compatible` | 广播布局与操作数相容性 | 将来由 `op::einsum` / 算子签名校验消费 | 广播是算子的输入对齐规则；当前零消费方，按 ADR 0004 不预埋 |
| `Geometry::reshaped`、`OwnedTensor::reshaped` | 重排算子 | 同 reshape | 在拥有缓冲上改形状仍是算子语义 |
| `Iter`、`View::iter` / `linear` | 逐元素遍历/线性寻址 | kernel 侧操作数视图（ADR 0003，P0 未实现）或 `train::Reference` 标量语义 | 元素遍历是计算侧行为；元数据层只回答"下标落在哪"，不提供遍历器 |
| `View::from_parts` / `as_contiguous_slice`、`Geometry` 公开构造与访问器 | 新公开布局类型 | — | 无消费方；若校验收口需要，降级为 `pub(crate)`，不进公开面 |
| `Shape::pad_leading` | 形状派生 | 同上暂缓 | 零消费方 |

### 3.2 层内保留：元数据本职

| 能力 | 处置 |
|---|---|
| `Shape` 既有访问器（`new`/`dims`/`rank`/`dim`/`numel`/`element_bytes`） | 原样保留 |
| `Strides::contiguous` 与既有访问器 | 原样保留 |
| `Tensor` / `View` / `OwnedTensor` / `Element` 既有公开 API 与语义 | 原样保留 |
| 连续性判定收口到 `Strides::is_contiguous(&Shape)`（无分配逐轴核对） | 保留并作为唯一实现，三处调用点委托 |
| 秩 / 零元素 / 可达性校验 | 保留在 `View` 内既有口径；不抽新公开类型 |

### 3.3 明确不做（防止再次越界）

- 不在 `tensor` 定义设备、分配器、跨设备拷贝（归 `storage` / `engine` / 后端边界，ADR 0025）。
- 不在 `tensor` 写任何逐元素计算、归约、激活、matmul（归 `op` / `nn` / `train`）。
- 不做 `&mut` 分区视图（ADR 0003：分区是 DSL 所有权规则，P0 未实现）。
- 不预埋广播相容、重排、切片等尚无消费方的算子形状接口。

## 4. 目标结构：薄 lib + 分文件

`lib.rs` 只保留 crate 文档、模块声明、再导出与 `SUPPORTED`/`supports`；所有实现体下沉子模块：

```
crates/tensor/src/
  lib.rs        // 薄入口：文档、mod 声明、pub use、SUPPORTED/supports
  shape.rs      // Shape：轴长度、饱和 numel、字节数公式委托
  strides.rs    // Strides：连续步长、连续性判定唯一实现、访问器
  element.rs    // Element trait 与 f32 / i8 / u8 三个密封实现
  tensor.rs     // Tensor<'a>：字节只读视图
  view.rs       // View<'a, E>：元素零拷贝视图与下标→偏移校验
  owned.rs      // OwnedTensor：元数据 + 拥有字节向量
tests/tests.rs  // 既有白盒测试，#[path = "../tests/tests.rs"] 接入
```

模块间依赖：`tensor.rs` / `view.rs` / `owned.rs` 依赖 `shape.rs` / `strides.rs` / `element.rs`；不互相引用视图类型以外的实现。

## 5. 层内优化点（不含越界功能）

1. **连续性判定单一事实源**
   - 现状：三处 `is_contiguous` 都 `self.strides == Strides::contiguous(&self.shape)`，每次构造一条 `Vec` 再比较。
   - 收口：`Strides::is_contiguous(&Shape)` 反向逐轴核对期望累积值，无分配；`Tensor` / `View` / `OwnedTensor` 委托。
2. **校验口径集中**：`View` 的秩匹配、零元素口径、最高偏移可达性保留现有 `checked_mul` / `checked_add` 饱和路径；错误构造点标记维持 `tensor::View::strided`。
3. **文档对齐**：crate 头注释的「拥有 / 不拥有 / 依赖 / 被依赖」与最终代码逐条对齐，不写未落地的能力。
4. **测试保持白盒**：测试仍经 `#[path]` 以内部模块视角访问既有字段；不为删除的算子面补测试。

## 6. 影响面评估

| 维度 | 结论 |
|---|---|
| 公开 API | 相对 HEAD 只减不增；HEAD 既有项全部保留，外部消费方无需改动 |
| 所有权表 | 不新增 `pub struct|enum|trait`，`ARCHITECTURE.md` 第三节与词汇表基线不动 |
| 依赖边 | 不变：`tensor → diagnostic, dtype`（+dev: testkit） |
| 代码体量 | `lib.rs` 仅留入口；实现按模块均分，单文件不触 899 行硬上限 |
| 风险 | 低：纯内部重构与模块搬迁；无生产消费方依赖本轮新增接口 |

## 7. 验收标准

1. `cargo test -p llm-tensor` 全绿，既有用例不缺失。
2. `cargo clippy -p llm-tensor -- -D warnings` 零告警；`RUSTDOCFLAGS=-D warnings cargo doc -p llm-tensor` 无坏链接/缺文档。
3. `tests/architecture.rs` 所有权双向校验通过；`tests/contracts.rs` 闭集/位集门禁不受影响。
4. 全 workspace 构建通过：`storage` / `weights` / `nn` / `engine` / `data` / `autograd` / `train` 等消费方编译不受影响。
5. `src/` 内无 `unsafe`、无新外部依赖、无算子/遍历/搬运代码。

## 8. 后续衔接（不在本次范围）

- audit.md §6.3 S2 规划的 `storage::Layout` / `Order` / `remap` 并入 `tensor` 布局子模块：是独立迁移步骤，待 `storage` 生产入边明确后单独立 ADR，本次不捆绑。
- 形态算子（reshape / squeeze / permute / flatten / narrow / broadcast）落地时，先在 `op` 注册表声明，再由 `nn` 绑定、`train` Kernel 执行；需要重排数据时经 `storage::remap`，不在 `tensor` 内闭环。
