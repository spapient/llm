# 设计 — 分层 crate 契约与架构决策

状态: 拟定
范围: llm/ 工作区各层 crate 的职责边界、公共接口、不变量、错误路径、复杂度、并发设计与架构决策；含《张量布局契约》《模型支持契约》《检查点格式契约》《训练契约》《推理契约》五份规范
读者: 工程师、评审者、维护者、AI 编码代理

---

本文逐章记录各 crate 的设计决策与阶段性验证。各章「验证」小节是不同阶段的时间快照，「迁移前」记录不代表当前状态；黑盒测试迁移（2026-10-10）后，`crates/*/src` 已无内联单元测试，全部套件位于仓库根 `tests/`。单一测试目标整合（2026-10-10）后，全部套件位于仓库根 `tests/it/`；本文历史记录中形如 `tests/<name>.rs` 的引用，其当前磁盘位置为 `tests/it/<name>.rs`（理由与测量见 docs/engineering.md §8.22）。

## 1. 目的与范围

本文档是对 llm/ 工作区各层 crate 设计契约的完整规范，按层与 crate 组织，覆盖 Layer 0、Layer 1、Layer 2 及以上的全部 crate。

编写原则：保留全部决策、不变量、被否决方案与验证证据。本文件不引用其它文档，全部引用均为本文件内引用。

本文档只描述已实现且经代码核对的契约。来源 ADR 中标为 Accepted 的决策，其实现已存在于 `crates/` 代码与仓库根测试中；标为 Proposed 的契约，其规范内容在本文件内首次成文，依据是对应 crate 源码的核对。

本文件按 crate 组织，而非按决策时间顺序组织。同一份 ADR 的内容可能散布到其 crate 章的相关小节；跨 crate 的决策（例如 `api` 与 `tensor` 的依赖边）归于其契约拥有者所在章，并在受影响章的「决策」中交叉引用。

层级归属以依赖深度为准：`lib` 为 Layer 0 的根；`tensor`、`api` 为 Layer 0；`cpu`、`autograd` 为 Layer 1；`nn`、`models`、`tokenizer`、`data`、`format`、`checkpoint` 为 Layer 2 及以上。其中 `models` 依赖 `nn`，来源 ADR 记为 Layer 3，但仍在本文件中单列成章。`tokenizer`、`data`、`format`、`checkpoint` 是 `lib`（以及 `tensor`）之上的依赖叶，彼此无生产依赖。

## 2. 阅读指南

### 2.1 规范用语与状态词汇

术语 MUST、MUST NOT、SHOULD、SHOULD NOT、MAY 表达要求强度：MUST / MUST NOT 为强制约束，SHOULD / SHOULD NOT 为默认规则（仅在有文档化理由时可偏离），MAY 为允许的设计选择。

每份来源文档带有一个状态。本文档继承来源状态，并在每章「上下文」中标注：

- **Accepted（已接受）**：决策已作出且实现已落地，代码与测试支持其契约。
- **Proposed（拟定）**：要求已规定但尚未与实现对验。本文件中标为 Proposed 的契约内容已按源码成文；其「一致性判定」由 `docs/conformance.md` 记录，本文档不复制判定表。

一致性判定的独立词汇（用于来源文档的验证状态）为：Proposed、Verified、Non-conforming、Blocked。文档或 ADR 的存在不构成一致性的证据；标为 Verified 必须引用已执行的检查或已核对的源码。

### 2.2 设计前置门（DS0/DS5）约定

每个 crate 章统一采用 `rule.md` §6 的设计前置门，条款 ID 保留原文形式：

- **DS0（设计前置准入）**：修改任何模块前，§6 各条必须全部达标，未达标不得进入编码。
- **DS1（边界与接口）**：定义模块边界、职责与公共接口；接口宽度不超过 A4 上界 20。
- **DS2（不变量与错误路径）**：定义算法级与实现级两层不变量；枚举全部错误分支与返回语义；明确资源回滚规则。
- **DS3（复杂度预控）**：单函数圈复杂度 ≤ 10，有效行数 ≤ 80。
- **DS4（并发设计）**：选定同步原语、定义锁序与可见性。
- **DS5（方案评审）**：必须给出至少一个被否决的备选方案，并说明否决理由。

因此本文每个 crate 章的固定小节为：上下文 / 决策 / 接口边界（DS1）/ 不变量与错误路径（DS2）/ 复杂度（DS3）/ 并发设计（DS4）/ 被否决方案（DS5）/ 后果 / 验证 / 复审条件。来源 ADR 的 `Context`、`Decision`、`Design gate`、`Consequences`、`Alternatives`、`Validation`、`Revisit` 逐一映射到上述小节。

### 2.3 命名约定

工作区采用无分隔符约定：公共类型、trait、enum、variant、函数名一律小写拼接（例如 `tensorview`、`getelementcount`）；常量与宏一律大写拼接（例如 `CHECKPOINTVERSION`、`CPUMAXRANK`）。

模块文件名刻意不同于其导出的公共类型名，因为模块与重导出的类型共享同一名称命名空间：`mod manifest` 与 `pub use manifest::manifest;` 会以 `E0255` 冲突。因此模块文件使用无冲突的容器名（例如 `layouts.rs` 导出 `layout`，`arena.rs` 导出 `graph`，`manifests.rs` 导出 `manifest`/`checkpoint`）。常量与宏使用大写紧凑拼写（NM5）。命名总纲为 NM1，动词表为 NM3，梯度命名为 NM4。

## 3. 层 0：lib

### 3.1 上下文

`lib` 是 Layer 0 的根与工作区的基础层。它拥有标识符、精度描述、可复现种子、资源限制与共享的限制校验错误类型。它只依赖 Rust 标准库与 `thiserror`，MUST NOT 依赖任何其它工作区 crate。

来源：`crates/lib/src` 源码核对。`lib` 没有独立 ADR，本章内容由源码推导。

### 3.2 决策

1. **只提供共享基础原语，不提供默认限制。** `lib` 刻意不提供任何默认资源限制；每个边界都由调用方显式提供，因此不会凭空发明未文档化的限制。
2. **`precision` 是计算格式分类法。** `precision` 取值为 `{f32, f16, bf16}`，只命名计算格式，不表示任何后端支持它。后端声明的 `capabilities` 才是执行支持的唯一权威。
3. **标识符为不透明值类型。** `tokenid(u32)`、`requestid(u64)`、`seed(u64)` 都是 `Copy` 的不透明包装；`seed` 只是把原始值传给确定性的生成器所有者，`lib` 不解释其分布。
4. **限制为不可变预算。** `limits` 持有元素计数、字节大小、序列长度、批大小的显式上界，全部为闭区间上界：请求值等于上界被接受，只有严格大于才被拒绝。
5. **错误归属于本层职责。** `limiterror` 描述与 `limits` 校验相关的失败，每个 variant 携带请求值与配置限制。

### 3.3 接口边界（DS1）

`lib` 在 crate root 导出 6 个公共项，远低于 A4 上界 20：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `limiterror` | `error.rs` | 限制校验错误 enum |
| `limits` | `limit.rs` | 不可变资源预算 |
| `precision` | `prec.rs` | 计算格式分类法 |
| `requestid` | `request.rs` | 请求标识 |
| `seed` | `seeds.rs` | 可复现随机种子 |
| `tokenid` | `token.rs` | 词元标识 |

模块文件（`prec`、`seeds`、`limit`、`token`、`request`、`error`）刻意不同于导出的公共类型（`precision`、`seed`、`limits`、`tokenid`、`requestid`、`limiterror`），以满足 E0255 守卫。`lib` 无 workspace 依赖，因此 `tensor` 与 `api` 都可共享它而不形成环。

### 3.4 不变量与错误路径（DS2）

算法级不变量：

- `limits` 的每个边界都是闭区间上界：`checkelements`/`checkbytes`/`checksequencelen`/`checkbatch` 仅在请求值严格大于上界时返回对应错误。
- `precision` 是计算格式集合 `{f32, f16, bf16}`；存储分类法 `dtype` 是它的严格超集（见第 4 章）。
- `seed`、`tokenid`、`requestid` 只是不透明标识，不进行范围或语义校验。

实现级不变量：

- `limits` 不携带默认值；`lib` 不发明任何未文档化限制。
- 所有 `limits` 校验都是 `const fn`，不分配、不阻塞。

错误分支与返回语义（`limiterror`，4 个 variant）：

- `maxelementsexceeded { requested, limit }` —— 请求元素计数超过元素上界。
- `maxbytesexceeded { requested, limit }` —— 请求字节大小超过字节上界。
- `maxsequencelenexceeded { requested, limit }` —— 请求序列长度超过序列上界。
- `maxbatchexceeded { requested, limit }` —— 请求批大小超过批上界。

资源回滚：`lib` 的类型都是值类型，无资源获取，因此无需回滚。

### 3.5 复杂度（DS3）

所有 `lib` 函数都远低于圈复杂度 10 与有效行数 80。最大者为 `limits` 的四个 `check*` 方法，各为单分支判断。（以阅读核对，证据等级 T。）

### 3.6 并发设计（DS4）

`lib` 不引入共享可变状态与同步原语。`limits`、`precision`、`seed`、`tokenid`、`requestid` 都是 `Copy` 的不可变值，可自由跨线程传递或共享，无需加锁。不存在锁序。

### 3.7 被否决方案（DS5）

- **给 `lib` 提供默认限制。** 被否决：会发明未文档化的限制，违反「每个边界由调用方显式提供」的原则。
- **给 `precision` 增加 `precision::f64`。** 被否决（见第 4 章与第 6 章的存储/计算对偶）：`precision` 命名计算格式，LLM 计算不以 `f64` 执行，加入它会暗示一个不存在的计算能力，并使 `From<precision>` 的像包含没有任何后端可接受的格式。
- **把 `seed` 的分布解释放进 `lib`。** 被否决：`lib` 只承载共享类型；生成器算法属于其所有者（`nn`、`data`）。

### 3.8 后果

- `lib` 保持无 workspace 依赖，被 `tensor` 与 `api` 共享而不产生环。
- 不发明默认限制意味着调用方必须显式决定每个资源边界；缺失边界的失败是编译期缺失，而非运行期静默默认。

### 3.9 验证

以源码核对与 `docs/conformance.md` 记录为证：`lib` 实现其文档化契约，`cargo check --workspace --all-targets`、`cargo clippy --workspace --all-targets -- -D warnings`、`RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` 均通过（证据等级 E1）。本记录重测（2026-10-10）：`cargo test --workspace --locked -- --list` 枚举 736 个用例；`bash scripts/check.sh` 报告 "All static quality checks passed"；通过/失败总数未在本轮重跑。相关根套件为 `tests/libcontract.rs`。

### 3.10 复审条件

当某后端声明 `f64` 执行时（则 `precision::f64` 可能变得有意义），或当某个已校验模型要求的存储类型不在现有分类法内时，重新审议本章。

## 4. 层 0：tensor

### 4.1 上下文

`tensor` 拥有张量元数据、存储、形状校验、布局与只读视图。它依赖 `lib`（资源限制与错误辅助），MUST NOT 依赖任何执行后端。来源：本 crate 源码核对；本 ADR 同时冻结了 `dtype` 与 `element` 的覆盖范围。

### 4.2 决策

1. **元数据与存储分离。** `tensor` 只提供元数据与存储，不定义算子与执行逻辑。
2. **形状非空。** `shape` 是有序、非空的维度列表，从最高位到最低位；rank 是维度数，元素计数是维度的受检乘积。
3. **布局显式。** `layout` 由每维一个 stride 加一个基偏移（以元素为单位）描述；`layout::contiguous` 构造规范的行主序布局。
4. **张量持有稠密缓冲。** `tensor` 拥有一个元素缓冲，其长度等于形状的元素计数，布局为该形状的规范连续布局。
5. **视图借用而不拥有。** `tensorview` 只借用拥有者的形状、布局与数据，不分配、不通过 `Arc` 或动态分发共享所有权。
6. **`dtype` 是完整存储分类法，`precision` 是计算分类法。** 二者是不同轴；`From<precision> for dtype` 是把计算格式嵌入存储格式的单射，但非满射。整数存储与 `f64` 只用于存储。
7. **`dtype::f64` 为新增的仅存储类型。** 它没有 `precision` 对应项，也没有任何后端声明 `f64` 执行；它存在是为了让张量能承载双精度参考数据、金标值与高精度累加缓冲。
8. **`element` 覆盖每个有原生 Rust 表示的基元。** `element` 为 `f32`、`f64`、`i8`、`u8`、`i32`、`i64` 实现。`f16` 与 `bf16` 因 Rust 无原生半精度类型而排除；支持它们需要后续层拥有的包装类型。
9. **宽度不变量由测试机械强制。** 每个 `element` 实现 MUST 满足 `DTYPE.sizeinbytes() == size_of::<Self>()`，在 `element` 的单元测试中跨全部六种类型断言。
10. **所有尺寸与偏移运算受检。** 会溢出的计算返回 `tensorerror`，而非回绕或 panic。

### 4.3 接口边界（DS1）

`tensor` 在 crate root 导出 7 个公共项，低于 A4 上界 20：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `dtype` | `dtypes.rs` | 完整存储类型分类法 |
| `element` | `elements.rs` | 可存储元素 trait |
| `tensorerror` | `error.rs` | 元数据校验错误 enum |
| `layout` | `layouts.rs` | stride + 偏移描述 |
| `shape` | `shapes.rs` | 非空维度列表 |
| `tensor` | `tensors.rs` | 拥有的稠密张量 |
| `tensorview` | `views.rs` | 借用的只读视图 |

模块文件（`dtypes`、`elements`、`layouts`、`shapes`、`tensors`、`views`）不同于导出类型，满足 E0255 守卫。依赖边为 `tensor -> lib`，不依赖任何后端。

### 4.4 《张量布局契约》

本节是 `tensor` 布局语义的完整规范。

**形状（shape）。** `shape` 持有 `Vec<usize>` 维度，从最高位到最低位排列。`shape::new` 在维度列表为空时返回 `tensorerror::emptyshape`，因此每个形状 rank ≥ 1。`getrank` 返回维度数。`isempty` 在任一维度为零时为真（注意 `[1]` 持有一个元素，不为空）。`getelementcount` 以受检乘法计算维度乘积，任一乘积溢出 `usize` 时返回 `tensorerror::elementcountoverflow { dims }`。`checklimits` 先算元素计数，再对 `limits::checkelements` 校验，超过时返回 `tensorerror::elementcountlimitexceeded { requested, limit }`。

**步幅（stride）与布局（layout）。** `layout` 持有每维一个 stride 加一个基 `offset`，单位为元素。`layout::contiguous(&shape)` 构造行主序（C 序）布局：最后一维 stride 为 1，更早的每维 stride 是其后续维度之积。stride 计算在溢出时饱和到 `usize::MAX`；需要精确 stride 的调用方 MUST 先校验元素计数（例如 `getelementcount` 或 `checklimits`）。`getstrides` 与 `getoffset` 暴露步幅与基偏移。

- `iscontiguous`：布局不带形状，故只据步幅判定连续性——偏移为 0、最后一维 stride 为 1、且每个外侧 stride 是相邻内侧 stride 的整数倍；当内侧 stride 为 0 时，所有外侧 stride 也必须为 0。
- `isvalidfor(&shape)`：rank 必须匹配；对非空形状，需验证每个维度映射到形状元素计数内的线性下标（受检运算）。元素计数为零的形状接受任何 rank 匹配的布局。任一检查失败返回 `false`。

**张量（tensor）。** `tensor<T: element>` 持有 `shape`、`layout` 与 `Vec<T>` 数据。`tensor::new(shape, data)` 先在形状上计算元素计数（可能返回 `elementcountoverflow`），再要求 `data.len()` 恰等于该计数，否则返回 `tensorerror::lengthmismatch { expected, actual }`；随后以 `layout::contiguous` 建立布局。不变量：元素缓冲长度恒等于形状元素计数，布局恒为该形状的规范连续布局。访问器包括 `getshape`、`getlayout`、`getdtype`（返回 `T::DTYPE`）、`getlen`、`isempty`、`getdata`、`getdatamut`、`getview`。

**视图（tensorview）。** `tensorview<'a, T>` 持有对拥有者 `shape`、`layout`、`data` 的借用，是 `Clone + Copy`，不分配，不共享所有权。`getshape`、`getlayout` 返回借用；`getlen` 返回可见元素数；`getdata` 返回 `&'a [T]`。`toowned` 把可见数据复制为新 `tensor`，可能返回 `tensor::new` 的错误。视图的生命期不得超过被借用数据。

**存储与元素类型。** `element: Copy + 'static` 带有关联常量 `DTYPE: dtype`。每个 `element` 实现的 `DTYPE.sizeinbytes()` 必须等于 `size_of::<Self>()`。已实现：`f32 -> dtype::f32`、`f64 -> dtype::f64`、`i8 -> dtype::i8`、`u8 -> dtype::u8`、`i32 -> dtype::i32`、`i64 -> dtype::i64`。`f16`、`bf16` 有 `dtype` variant 但无原生存储类型，故不实现 `element`。

**dtype 与 precision 的关系。** `dtype` 取值为 `{f32, f64, f16, bf16, i8, u8, i32, i64}`，是完整的存储分类法。`precision` 取值为 `{f32, f16, bf16}`，是计算分类法。`From<precision> for dtype` 把三种计算格式嵌入存储分类法，但整数存储与 `f64` 只用于存储，没有计算精度对应项。`dtype` 只说明值在内存中的布局，不暗示任何后端能执行它；后端声明的 `capabilities.dtypes` 才是执行支持的唯一权威。`dtype::sizeinbytes()` 给出宽度：`f32`=4、`f64`=8、`f16`=2、`bf16`=2、`i8`=1、`u8`=1、`i32`=4、`i64`=8。

**受检算术。** 所有尺寸、下标、偏移、容量运算都以受检形式进行，可能溢出的计算返回 `tensorerror`，不使用 `unsafe`、`unwrap`、`panic`。`layout::contiguous` 是唯一的饱和点，且其结果是布局描述而非可索引边界；索引正确性由元素计数校验与调用方的形状校验共同保证。

**错误路径。** `tensorerror`（6 个 variant）：`emptyshape`（空维度列表）、`elementcountoverflow { dims }`（乘积溢出 `usize`）、`elementcountlimitexceeded { requested, limit }`（超过元素预算）、`lengthmismatch { expected, actual }`（数据长度与元素计数不符）、`invalidstride { dim, stride }`（步幅对维度非法）、`dtypemismatch { dtype }`（元素类型与所需存储类型不符）。

### 4.5 不变量与错误路径（DS2）

算法级不变量：

- 一个 `shape` 一旦存在，其 rank ≥ 1 且元素计数是各维度的受检乘积。
- 一个 `tensor` 一旦存在，其元素缓冲长度等于形状元素计数，布局为该形状的规范连续布局。
- 一个 `element` 实现满足 `DTYPE.sizeinbytes() == size_of::<Self>()`。
- `From<precision> for dtype` 保宽度（`precision::sizeinbytes()` 与 `dtype::sizeinbytes()` 对三种计算格式一致），且 `f64` 不是任何 `precision` 的像。

实现级不变量：

- `layout::isvalidfor` 只在 rank 匹配且每个维度映射到元素计数内的线性下标时为真；元素计数溢出时返回 `false` 而非 panic。
- `tensorview` 只借用，不拥有，不分配；`toowned` 复制为新的拥有张量。
- 索引正确性由形状校验与受检算术共同保证，不使用 `unsafe`。

错误分支枚举：空形状、元素计数溢出、超元素预算、数据长度不符、非法步幅、dtype 不符。

资源回滚：`tensor`/`tensorview`/`shape`/`layout` 都是值类型，构造失败时不返回部分值，拥有的 `Vec` 由作用域释放（OW1、ER4）。

### 4.6 复杂度（DS3）

每个函数都低于圈复杂度 10 与有效行数 80。最大者为 `layout::isvalidfor` 与 `layout::iscontiguous`，都是短循环加少量校验。（以阅读核对，证据等级 T。）

### 4.7 并发设计（DS4）

`tensor` 不引入共享可变状态与同步原语。`shape`、`layout`、`dtype`、`element` 都是不可变值类型；`tensor` 拥有其数据，`tensorview` 是不可变借用。可变共享需由调用方以显式所有权表达；`tensor` 本身无锁序、无可变全局状态（A6）。

### 4.8 被否决方案（DS5）

- **保持 `f64` 不实现元素类型并以「不支持」文档化。** 被否决：这会保留循环论证，且与 `i64` 不一致——`i64` 是同样没有后端支持、但 `dtype` 已承载的原生基元。
- **同时增加 `precision::f64`。** 被否决：`precision` 命名计算格式，没有 LLM 计算在 `f64` 上执行；加入它会暗示不存在的计算能力，并使 `From<precision>` 的像包含没有任何后端可接受的格式。
- **从 `element` 移除 `const DTYPE`，在调用点单独携带 dtype。** 被否决：这会让张量存储类型与元素类型不一致，移除 `T` 与其 `dtype` 的编译期绑定，并为无收益的目的扩大 `tensor` 构造表面。
- **现在就为一个半精度包装类型实现 `element`。** 被否决：它是带自己的舍入、算术与转换契约的新类型，超出本次变更范围，属于后续可累加的阶段。
- **在构造时对 `tensor` 数据做全量校验（例如检查 NaN）。** 被否决：元素语义不是元数据职责；`tensor` 不解释元素值。

### 4.9 后果

- `element` 现在是原生基元到存储类型的全映射，`dtype` 由实际不变量而非遗漏来文档化。
- `f64` 张量可通过 `tensor`、`tensorview` 与 `shape`/`layout` 创建与检视，无需代码改动，因为这些类型对 `T: element` 泛型化；在某个后端声明它们之前，它们不可被任何后端执行。
- 两处描述旧状态的表述得到纠正：`cpu` 章的 Context 现在列出六种已实现类型，`lib::precision` 的文档现在把 `f64` 命名为仅存储。
- 增加 `dtype::f64` 不扩大后端支持：`cpu` 仍只声明 `[dtype::f32]`。依赖图不变（`tensor -> lib`），不引入边、`unsafe` 或动态分发。

### 4.10 验证

本节的验证于 2026-10-10 执行：`cargo fmt --all` 干净；`cargo check --workspace --all-targets` 通过；`cargo clippy --workspace --all-targets -- -D warnings` 通过；`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 通过。（证据等级 E1。）本记录重测（2026-10-10）：`cargo test --workspace --locked -- --list` 枚举 736 个用例；`bash scripts/check.sh` 报告 "All static quality checks passed"；通过/失败总数未在本轮重跑。相关根套件为 `tests/tensorunits.rs` 与 `tests/tensorcontract.rs`，覆盖宽度不变量、`From<precision>` 的宽度保持检查、`f64` 非精度检查与经视图的 `f64` 往返。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）

### 4.11 复审条件

当某后端声明 `f64` 执行时（则 `precision::f64` 可能变得有意义）；当加入原生或包装的半精度类型时（则 `element` 覆盖与宽度不变量扩展到它）；当某个已校验模型要求 `dtype` 未表示的存储类型时，重新审议本章。

## 5. 层 0：api

### 5.1 上下文

Layer 0 定义三个 crate，构成严格的单向依赖图 `lib`、`tensor`、`api`：

```text
lib
tensor -> lib
api    -> lib
api    -> tensor
```

`api` 与 `tensor` 的依赖关系须显式选择：因为内核调用契约直接接受张量，`api -> tensor` 边是可接受的；否则两者必须保持独立可组合。

Layer 1 的差异化（`autograd`）、神经网络（`nn`）与模型（`models`）层需要带第二操作数或逐调用属性的算子，而 Layer 0 冻结的 `kernel<T>` 契约（单目、借用输入、拥有输出）无法承载它们。因此 `api` 作为一次**累加式**扩展，新增带操作数元数的窄 trait 与一个面向工作流的 `compute` trait；冻结的 `kernel<T>` 不被修改。本章同时覆盖内核调用契约与累加式算子扩展两组决策（后者部分取代此前推迟二元/带权算子的决定）。

本章决策保留的依赖图：

```text
lib
tensor -> lib
api    -> lib
api    -> tensor
cpu    -> api
cpu    -> tensor
cpu    -> lib
```

`metal` 与 `cuda` GPU 后端（第 21 章）对同一 `compute` 契约做累加式实现，不改动本章任何决策；`devicekind` 因它们新增两个变体，`api` 的公共接口宽度与依赖边均不变。

### 5.2 决策

1. **采纳 `api -> tensor` 边。** `kernel` trait 的调用契约接受 `tensorview<'_, T>` 作为输入并返回 `tensor<T>`：

   ```rust
   pub trait kernel<T: element> {
       fn getname(&self) -> &'static str;
       fn apply(&self, input: &tensorview<'_, T>) -> Result<tensor<T>, backenderror>;
   }
   ```

   因为内核签名命名张量类型，`api` 直接依赖 `tensor`。这是 §5.3 要求的显式选择。

2. **采纳 `api -> lib` 边，由 `capabilities.limits` 实现。** 后端通过 `lib` 拥有的 `limits` 类型声明其资源预算（经 `capabilities::getlimits` 产生）。这是唯一的 `api -> lib` 依赖，`api` 不引入其它基础耦合。

3. **`api` MUST NOT 引入对 `tensor` 的反向依赖**（即 `tensor` MUST NOT 依赖 `api`），且 `api` MUST NOT 依赖依赖图中位于其上的任何 crate。后端实现（`cpu` 及以后）依赖 `api`，绝不反向。

4. **累加式操作数元数 trait。** `api::kernels` 在冻结的单目 `kernel<T>` 之外新增三个窄尾随 trait：`binarykernel<T>`（两个同类型操作数）、`weightedkernel<T>`（输入加权重）、`gatherkernel<T>`（表加 `&[usize]` 索引）。每个报告 `getname` 与单个 `apply`。单目 trait 不被修改，因此每个既有实现与测试继续原样编译。

5. **一个面向工作流的 `compute` trait。** 差异化与神经网络层必须在不绑定具体后端的前提下组合算子。`api::compute` 以拥有的 `f32` 张量暴露这些层所用的 f32 算子集（`matmul`、`add`、`mul`、`rmsnorm`、`softmax`、`silu`、`embedding`、`rope`、`attention`）。逐调用属性（epsilon、head 数、rotary base、causal 标志）作为显式参数，因此没有逐模型配置藏在 kernel 实例内。后端只实现一次 `compute`，依赖边保持 `cpu -> api`。

6. **内核保持纯净，策略留在 executor。** `cpumatmul`、`cpuadd`、`cpumul`、`cpurmsnorm`、`cpuembedding`、`cpurope`、`cpuattention` 是无状态类型，只校验自身的形状前置条件。`cpuexecutor` 保留唯一的预算检查（rank、元素计数、字节大小）与派发。

7. **形状规则显式且被校验，绝不假设：**
   - `matmul` 要求相等 rank `r >= 2`，相等的前导批维，且 `lhs[.., m, k] * rhs[.., k, n] -> [.., m, n]`；不做广播。
   - `add`/`mul` 要求形状完全相等。
   - `rmsnorm` 归一化最后一轴；权重长度必须等于该轴，且该轴非空。
   - `rope` 要求 rank `>= 2`，最后一轴可被 `heads` 整除，且 `headdim` 为偶数。
   - `embedding` 要求每个索引 `< table` 第一轴。
   - `attention` 要求 rank 2，序列长度相等，`heads` 是 `kvheads` 的正整数倍，且 query 最后一轴可被 `heads` 整除。

8. **朴素、确定性的 attention。** `cpuattention` 实现分组查询缩放的 dot-product attention（`scale = 1/sqrt(headdim)`）与可选 causal masking，作为 O(seq^2 * heads * headdim) 循环。全掩码行保持为零。不引入融合、tiling、SIMD 或线程。

9. **仅 f32，无 `unsafe`。** 后端声明 `[dtype::f32]`。`api` 与 `cpu` 以及全部非 GPU crate 保持 `#![forbid(unsafe_code)]`；内核使用形状校验已界定范围的安全索引。原生 GPU 后端（第 21 章）因平台 FFI 适用一条受限的 `unsafe` 豁免，不改变本层与 `cpu` 的约束。

10. **错误映射显式且在关键处无损。** 内核失败保持 `backenderror`。executor 的 CPU 策略错误保持 `cpuerror`；`compute` 实现把被拒的预算映射到 `backenderror::resourceexhausted`，并原样透传内核错误，因此 `compute` 的调用方只看到抽象层的区分。设备不可达与内核编译失败在边界映射为独立变体（`devicenotavailable`、`kernelcompilefailed`），不塌缩进 `executionfailed`，使"环境无 GPU"与"内核没编译出来"可分别诊断。

11. **`backend` 是窄身份契约。** `backend` 只暴露名称、设备与声明能力；执行、内存与同步职责刻意不并入该 trait，它们属于 `kernel` 实现。

### 5.3 接口边界（DS1）

`api` 在 crate root 导出 10 个公共项：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `backend` | `backends.rs` | 后端身份契约 |
| `kernel` | `backends.rs` | 冻结的泛型内核调用契约 |
| `binarykernel` | `kernels.rs` | 双同类型操作数契约 |
| `weightedkernel` | `kernels.rs` | 输入 + 权重契约 |
| `gatherkernel` | `kernels.rs` | 表 + 索引 gather 契约 |
| `compute` | `kernels.rs` | 工作流 f32 算子集 |
| `capabilities` | `capability.rs` | 声明能力（dtypes、maxrank、limits） |
| `device` | `devices.rs` | 设备句柄 |
| `devicekind` | `devices.rs` | 设备类别 enum |
| `backenderror` | `error.rs` | 内核/派发错误 enum |

`api::kernels` 导出四个 trait（`binarykernel`、`weightedkernel`、`gatherkernel`、`compute`），共 19 个 trait 方法（前三个各 `getname` + `apply`，`compute` 13 个），低于 A4 上界 20。两者都是纯累加接口，无既有符号移动。`api` 依赖 `lib` 与 `tensor`；`lib` 与 `tensor` 都不得依赖 `api`。

`capabilities` 字段公有，便于后端所有者直接组装声明：`dtypes: Vec<dtype>`、`maxrank: usize`、`limits: limits`；访问器 `supports(dtype)`、`getdtypes`、`getmaxrank`、`getlimits`。`device`/`devicekind` 目前支持 `cpu`、`metal`、`cuda` 三类设备，各配 `device::{cpu,metal,cuda}()` 构造与 `device::{iscpu,ismetal,iscuda}()` 判定；`metal` 与 `cuda` 变体是 GPU 后端的累加式扩展（见第 21 章），不改变 `cpu` 语义，也不加 `#[non_exhaustive]`。

`kernels.rs` 四个 trait 的 `apply` 签名：`binarykernel<T>::apply(&self, lhs: &tensorview<'_, T>, rhs: &tensorview<'_, T>) -> Result<tensor<T>, backenderror>`；`weightedkernel<T>::apply(&self, input, weight)`；`gatherkernel<T>::apply(&self, table, indices: &[usize])`；`compute` 的方法均为拥有 `f32` 张量，见下。

`compute` 的方法契约：

| 方法 | 语义 | 形状规则 |
| --- | --- | --- |
| `matmul(lhs, rhs)` | 批矩阵乘 | 相等 rank `r >= 2`，前导批维逐一相等，尾两维 `[m,k]*[k,n] -> [batch...,m,n]`；无广播 |
| `add(lhs, rhs)` | 逐元素加 | 形状完全相等 |
| `mul(lhs, rhs)` | 逐元素乘 | 形状完全相等 |
| `rmsnorm(input, weight, eps)` | 最后一轴 RMS 归一化再加权 | 权重元素数等于输入最后一轴 |
| `softmax(input)` | 最后一轴数值稳定 softmax | 最后一轴非空 |
| `silu(input)` | 逐元素 SiLU | 无 |
| `embedding(table, indices)` | 按索引 gather 行 | 每个索引 `< table` 第一轴 |
| `rope(input, heads, base)` | 旋转位置嵌入 | rank `>= 2`，最后一轴可被 `heads` 整除，`headdim` 偶 |
| `rope_at(input, position, heads, base)` | 单 token 在**绝对** `position` 的旋转位置嵌入（decode 步形态） | rank 1 `[1, heads * headdim]`，最后一轴可被 `heads` 整除，`headdim` 偶 |
| `attention(query, key, value, heads, kvheads, causal)` | 分组查询注意力 | rank 2，序列长度相等，`heads` 是 `kvheads` 正整数倍，query 最后一轴可被 `heads` 整除 |
| `attention_cached(query, key, value, heads, kvheads, causal)` | 单 token 增量注意力（decode 步形态） | query 为 `[1, heads * headdim]`，key/value 各为 `kvpair`（cached `[pos, kvwidth]` + new `[1, kvwidth]`），cached 与 new 宽度相等，`heads` 是 `kvheads` 正整数倍 |
| `scan(input, init, a, b, c, skip)` | 结合律半群扫描（prefill 形态）：`h_t = A h_{t-1} + B x_t`、`y_t = C h_t + D x_t` | `input` `[n, d]`，`init/a/b/c` `[d, N]`，`skip` `[d]`；`a` 对角或 2×2 块对角（谱半径 ≤1）；返回 `([n,d] 输出, [d,N] 末态)` |
| `scan_cached(input, state, a, b, c, skip)` | 单 token 增量扫描（decode 步形态），与**串行** `scan` 逐位一致 | `input` `[1, d]`，`state/a/b/c` `[d, N]`，`skip` `[d]`；返回 `([1,d] 输出, [d,N] 更新态)` |

`kvpair` 是 `api::kernels` 的 `{ cached: tensorview<'_, f32>, new: tensorview<'_, f32> }`。`attention_cached` 的有效操作数是 cached 行后随 new 行（query 在因果掩码下关注 `pos + 1` 个位置），但两部分各居其址、由后端就地拼接语义，**不要求调用方把历史复制进新缓冲**——这是它与 `attention`（物理拼接后的全量形态）的分工，也是 `forwardsubmission_cached` 每次 decode 把该算子与 `rope_at`、`embedding`、`rmsnorm`、`matmul` 组合进**一次提交**、不产生跨步拷贝的组成部分。`rope_at` 独立于 `rope`：`rope` 从输入行索引推导位置（`row % seqaxis`），单行输入行号恒为 0，无法表达 decode 步的绝对上下文位置；`rope_at` 显式接收 `position`，单 token 的 query/key 即可在其绝对位置旋转，无需物化历史行。

### 5.4 不变量与错误路径（DS2）

算法级不变量（来自本章的设计门）：一个经校验的形状计划产生一个输出，其元素计数等于其维度之积。

实现级不变量：内核只在校验之后才分配输出，且用于切分输入的每个索引都 `<` 该输入的元素计数。

错误分支枚举：rank 低于二（`matmul`/`rope`/`attention`）、批/序列轴不等、内维不匹配、权重长度不匹配、被归一化轴为空、`headdim` 为奇、head 数量不匹配、索引越界、元素/字节/rank 预算耗尽、元素计数溢出。

资源回滚：失败的内核在构造输出张量之前返回，不逃逸任何部分分配（所有者：内核；生命期：该次调用）。

`backenderror`（10 个 variant）：`unsupporteddtype { dtype }`、`unsupportedoperation { op }`、`sizeoverflow`、`devicenotavailable`、`kernelcompilefailed { backend, kernel, message }`、`executionfailed { backend, kernel, message }`、`shapemismatch { op }`、`indexoutofrange { index, len }`、`invalidparameter { op }`、`resourceexhausted`。`compute` 的边界映射约定：被拒预算 -> `resourceexhausted`；内核错误 -> 原样透传；设备不可达 -> `devicenotavailable`（让"环境无 GPU"可诊断）；内核编译/模块加载失败 -> `kernelcompilefailed { backend, kernel, message }`（`kernel` 是失败的 `(source, entry)` 对中的 entry，`message` 是驱动诊断——编译失败可定位到具体内核而非笼统的执行失败）；运行期失败 -> `executionfailed { backend, kernel, message }`（`kernel` 是失败阶段或入口：`construct`/`transfer`/`launch`/`blas`/`cudnn`/`alloc`/`run`，`message` 是驱动或宿主诊断——执行失败同样可定位到阶段与根因）。

**错误分类语义（排障契约）。** `compute` 边界把后端失败分为三类，调用方按类采取动作：

| 变体 | 触发 | 排障动作 |
| :--- | :--- | :--- |
| `devicenotavailable` | 后端无法打开设备：无 Metal 设备 / 无可用 CUDA GPU / 驱动缺失 / 沙箱拦截设备系统调用 | 检查 GPU 与驱动、放开沙箱设备权限；**降级 CPU 后端**（`cpu` 永不产生该变体，是最小可用路径） |
| `kernelcompilefailed { backend, kernel, message }` | MSL/NVRTC 编译失败、pipeline 创建失败、模块加载/函数解析失败 | 按 `kernel` 定位 `(source, entry)` 对，按 `message` 核对 MSL/CUDA 版本、语法或资源限制；编译发生在运行期且按对缓存，一次修复全局生效 |
| `executionfailed` | 运行期执行失败：输出构造、缓冲分配、传输、启动、cuBLAS/cuDNN 调用 | 携带结构化诊断 `{ backend, kernel, message }`：`kernel` 标识失败阶段（`construct`/`alloc`/`transfer`/`launch`/`blas`/`cudnn`/`run`），`message` 为驱动或宿主诊断——执行失败可定位到阶段与根因，不再是无上下文的标记；不携带可统一处理的恢复语义 |

前两类是可恢复/可归因的环境与编译问题，调用方应显式处理；`executionfailed` 是后端自持错误，只陈述失败与阶段归属、不承诺恢复。此分类是"错误映射显式且在关键处无损"（§5.2 决策 10）的实例化：分类无塌缩、诊断有归属。

Layer 0 一侧的 `kernel`/`backend` 错误契约：`kernel::apply` 在元素 dtype 不受支持、操作不受支持、元素计数溢出或执行失败时返回 `backenderror`。`backend` 不做校验也不分配，只陈述后端接受什么。

### 5.5 复杂度（DS3）

每个新函数都低于 DS3 上界（圈复杂度 ≤ 10，有效行数 ≤ 80）。最大者为 `cpuattention::apply` 与 `attentionplan`；两者都是循环加短校验。（以阅读核对，证据等级 T。）

### 5.6 并发设计（DS4）

`api` 不引入共享可变状态与同步原语。所有 trait 与类型都是静态、不可变契约；`capabilities` 是后端在构造时组装后被不可变共享的声明。`api` 本身无锁序、无全局可变状态（A6）。`compute` 的调用方以 `&self` 持有 provider，隐含 provider 可在请求间不可变共享。

### 5.7 被否决方案（DS5）

来自内核调用契约：

- **用 trait object 装箱或擦除张量输入以保持 `api` 与 `tensor` 独立。** 被否决：为输入具体为张量视图的契约引入不必要的动态分发与间接层，违反 AGENTS.md §3.3 的静态组合偏好。
- **在 `api` 内复制一份后端本地限制类型。** 被否决：会分叉由 `lib` 拥有的唯一权威限制表示，产生第二个资源上界真值来源。

来自算子扩展契约：

- **就地扩展单目 `kernel<T>` 契约。** 被否决：会迫使每个既有实现者改变，并把不同元数坍缩进一个签名；累加式尾随 trait 让 `kernel<T>` 保持冻结。
- **按键名动态内核注册表。** 被否决：静态分发仍是偏好；注册表增加间接层而无运行期选择需求。
- **复用 `backenderror` 表示预算拒绝。** 被否决：`backenderror` 不建模配置化预算，因此 executor 保留 `cpuerror`，只在 `compute` 边界映射。
- **广播二元算子。** 被否决：范围内没有具体模型需要广播，静默广播会掩盖模型层本应暴露的形状缺陷；`nn` 层改为显式 reshape。
- **把 `rmsnorm`/`rope`/`attention` 也放到新的元数 trait 之后。** 被否决：`rmsnorm` 需要权重加 epsilon，`rope`/`attention` 需要属性；一个带具名参数的 `compute` 方法比三个带不同属性包、更泛化的 trait 更窄。
- **在层 crate 中放一个 `Box<dyn compute>` 字段。** 被否决：静态分发是既定偏好（AGENTS.md §3.3）；层对 `compute` 泛型化，因此推理与训练复用同一实现，无 trait object。
- **现在就以实现融合、带缓存的 attention。** 被否决：正确性优先；朴素路径是 KV-cache 优化必须击败的基线，而 KV cache 尚不存在。
- **现在就为 `matmul` 或 `rmsnorm` 引入 SIMD/线程。** 本次阶段被否决：没有实测瓶颈，也没有正确性计划。

### 5.8 后果

- `autograd`、`nn`、`models` 可通过 `api::compute` 组合算子，且永不依赖 `cpu`，保持后端可替换与无环。
- 冻结的单目契约不变；新 trait 是额外的可实现契约，因此没有既有后端或测试失效。
- `matmul` 由直白三重循环改为经 `fearless_simd` 派发的 8×4 寄存器分块 SIMD 微内核（2026-10-16）；FMA 重结合使结果对标量参照「容差内一致而非逐位」，原三重循环仍是正确性参照（`testkit::reference::matmul`）。
- `embedding` 通过复制 gather 行；每次调用分配 `indices.len() * width` 个元素。那是契约的拥有输出成本，不是隐藏缓冲。
- `cpuexecutor` 现在实现 `api::compute`，因此 executor 既是预算所有者，也是上层消费的后端视图。
- Layer 1 拥有四个单目算子的可用、经测试的 CPU 路径，且不改变 Layer 0；`cpu` 只依赖 `api`、`tensor`、`lib`，图保持无环且每个后端实现保持可替换。
- `kernel` 契约强制每次 `apply` 一次分配（返回的拥有张量）。就地或 arena 执行无法经冻结契约表达，被推迟。
- 直接使用内核类型的调用方绕过 executor 的预算检查；预算强制属于 `cpuexecutor`，内核只校验自身的形状前置条件。

### 5.9 验证

本节的验证于 2026-10-10 执行（仅 scoped 命令，因同批 crate 处于构建中）：

- 仓库根 `tests/cpuunits.rs` 与 `tests/cpucontract.rs`（对应 `cpu`）—— 0 失败。
（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
- `cargo check -p cpu --all-targets` —— 通过，exit 0。
- `cargo clippy -p cpu -p testkit -p bench -p llm --all-targets -- -D warnings` —— 干净，exit 0。
- 仓库根 `tests/testkitunits.rs`（对应 `testkit`）—— 0 失败（含四个新标量参考及其交叉检查）。
- `cargo test -p llm --test cpucontract` —— 15 通过，0 失败（新测试对照 `testkit::reference` 交叉检查 `matmul`、`rmsnorm`、`rope`、`embedding` 与 attention）。
- `cargo check -p llm --all-targets` —— 通过，exit 0（bench 可编译）。

基线：`cargo bench -p llm --bench kernels` 与 `--bench cpuops`，release profile（`lto = "thin"`、`codegen-units = 1`、`debug = 1`），criterion 0.7.0，warm-up 1.0 s，measurement 2.0 s，20 samples，无 SIMD，无线程。环境：`rustc 1.99.0 (b940084d7 2026-09-28)`，macOS 27.0.1，arm64（Apple Silicon）；未记录具体 CPU 型号，故这是同机基线，不可移植。

| Benchmark | Workload | Time (median) | Throughput |
| --- | --- | --- | --- |
| kernel/binary/matmul256 | 256x256x256 f32 | 974.0 µs | 67.3 Melem/s |
| kernel/binary/add | 65,536 f32 | 5.80 µs | 11.29 Gelem/s |
| kernel/binary/mul | 65,536 f32 | 5.95 µs | 11.01 Gelem/s |
| kernel/rmsnorm/rows512cols4096 | 2,097,152 f32 | 1.191 ms | 1.761 Gelem/s |
| kernel/rope/seq128heads8dim64 | 65,536 f32 | 85.59 µs | 765.8 Melem/s |
| kernel/embedding/gather128 | 131,072 f32 | 6.80 µs | 19.27 Gelem/s |
| kernel/attention/seq128heads8dim64 | 65,536 f32 | 1.268 ms | 51.70 Melem/s |
| executor/compute/rope | 65,536 f32 | 80.97 µs | 809.4 Melem/s |
| executor/compute/attention | 65,536 f32 | 1.207 ms | 54.29 Melem/s |

executor 路径与裸内核路径在运行间噪声内一致（噪声带准则）。这些数字是基线；不声明任何改进（VF5）。

**matmul SIMD 前后对比（2026-10-16，同机同工具链，criterion 0.7.0，100 samples）。** 上表 `kernel/binary/matmul256` 的朴素三重循环由 `matmulblock` 替换为经 `fearless_simd`（NEON）派发的 8×4 寄存器分块微内核：端到端 `apply` 由 1.0185 ms → 0.9008 ms（1.13×，含分配与契约开销；受同机热漂移影响）；纯计算内核交错 A/B 为 1.72×（0.615 ms vs 1.059 ms）。正确性经 `cpucontract`/`kvcacheforward`/`modelforward` 全绿（重结合差异 ~1e-8，在既有 1e-5/1e-4 容差内）；`checkunsafe.sh` 零新增 unsafe（`cpu`/`tensor` 保持 `#![forbid(unsafe_code)]`）。

**rmsnorm SIMD 前后对比（2026-10-16，同机同工具链）。** `rmsnormrow` 的平方和改为 `fearless_simd`（NEON）向量化（4 累加器 + fused `mul_add` + `reduce_sum` 水平规约），缩放趟改为向量乘：`kernel/rmsnorm/rows512cols4096` 由 1.1266 ms → 0.7285 ms（1.55×）。平方和重结合使结果对标量参照「容差内一致而非逐位」，归约顺序仅依赖行宽、与 batch 无关（保住逐位确定性）。`weighted_kernel_matches_the_scalar_reference` 全绿。

**新尺度基准与基线（2026-10-16，M5 Pro / Mac17,9，rustc 1.99.0，release `lto=thin` `codegen-units=1`，criterion 0.7.0，matmul prefill sample 10、decode sample 20、attention prefill sample 10，优化前内核）。** 既有 256² 基准的右操作数（256 KiB）整体驻留 L2，无法暴露按行块重扫右操作数的流量放大；新增三组尺度基准：prefill 2048² 方阵、decode 1×4096×4096、attention causal（heads=8、headdim=64）prefill 512/2048 与 decode 4096。基线（中位数）：

| Benchmark | Time (median) |
| --- | --- |
| matmul/scale/prefill2048 | 1.3111 s |
| matmul/scale/decode1x4096x4096 | 14.785 ms |
| attention/scale/prefill512 | 17.473 ms |
| attention/scale/decode4096 | 601.15 µs |
| attention/scale/prefill2048 | 281.24 ms |

这些数字是基线；不声明任何改进（VF5）。

**matmul 分块打包前后对比（2026-10-16，同机同工具链同基准）。** `m >= 32` 改走 k 分块（512）× n 面板（256）打包路径：右操作数面板连续化后驻留打包缓冲、被全部行块复用，消除每 8 行重扫一次右操作数的放大；`m < 32`（decode）保持直连微内核，其单次遍历已最优，打包只会多一次拷贝。prefill2048 由 1.3111 s → 412.48 ms（**3.18×**）；decode 14.785 ms → 14.446 ms（0.98×，设计上不变，噪带内）。每输出元素的 FMA 顺序与直连路径一致（p 升序、分块顺序执行），逐位确定性保持；`cpucontract` 新增 4 个边界测试（跨 chunk/面板与尾部、路径切换、batch、确定性）与 `kvcacheforward`/`modelforward`/`submissioncontract` 全绿；`checkunsafe.sh` 零新增 unsafe。

**attention 分块在线 softmax 前后对比（2026-10-16，同机同工具链同基准）。** 32 行查询块 × 256 key tile、在线 max/sum 重缩放（不物化分数矩阵）、值行按 key 顺序行主累加（向量化 `mul_add`）、GQA 组内 head 背靠背共享 KV tile；每行算术序列只依赖自身参与 key 范围与固定 tile 边界，与查询宽度无关 → `incrementalattention`/`kvcacheforward` 的逐位一致性保持。prefill512 由 17.473 ms → 16.000 ms（1.09×）；decode4096 由 601.15 µs → 520.20 µs（1.16×）；prefill2048 由 281.24 ms → 248.91 ms（1.13×，该尺度 16.7M 次 `expf` 主导，内存收益被摊薄——并行化属后续阶段）。`cpucontract` 新增 3 个参考测试（跨 tile 边界 GQA、非因果、确定性），全量 `tests/it` 976 测试通过。

**softmax SIMD 评估（2026-10-16）：判定不引入。** `softmaxrow` 的 max 与缩放趟经 LLVM 自动向量化已是 4 宽 NEON，而 `exp` 趟占主导且 `fearless_simd` 无向量化 `exp`（Apple Silicon 的系统 libm `expf` 已 NEON 加速，自写多项式难更快）。实测仅向量化 max/缩放反而回退 6.5%（2.4593→2.6182 ms），已回退；softmax 保持标量 `f32::exp`，未引入数值近似。

内核调用契约的验证：`lib`、`tensor`、`api` 以其文档化契约实现，仓库根 `tensorcontract`、`apicontract`、`crosscrate` 套件通过（证据等级 E1）。

### 5.10 复审条件

当剖析显示 `matmul` 或 `attention` 在某个文档化推理负载中占主导时（则评估带正确性计划与前后对比的 tiling/SIMD）；当 KV cache 到来时（则 `attention` 获得带缓存的变体，其收益对照朴素基线测量）；当某模型需要广播或转置操作数约定时（则以累加方式扩展 `compute`，而非重载既有签名），重新审议本章。

## 6. 层 1：cpu

### 6.1 上下文

Layer 0（`lib`、`tensor`、`api`）已实现。其内核调用契约冻结为：`kernel<T>` 单目，接受一个借用输入视图并返回一个新拥有的张量，没有第二操作数、属性或归约轴的参数。

`tensor` 为每个有原生 Rust 表示的基元实现 `element`（`f32`、`f64`、`i8`、`u8`、`i32`、`i64`）；`f16` 与 `bf16` 有 `dtype` variant 但无原生存储类型（测试要求不支持的 dtype 被拒绝，而不只是能编译）。

架构契约要求 `cpu` 实现后端契约、不引入反向依赖、不引入 `unsafe`，并给每个新算子边界测试、数值测试与至少一个基准。

本章覆盖后端身份、首批算子集、错误边界、基线与存储 dtype / 计算精度的区分。推迟的二元/带权算子扩展已在第 5 章吸收。

`cpu` 保留的依赖图：

```text
lib
tensor -> lib
api    -> lib
api    -> tensor
cpu    -> api
cpu    -> tensor
cpu    -> lib
```

### 6.2 决策

1. **静态组合，无注册表。** `cpuexecutor` 持有一个 `cpubackend`，以 `match` 把 `cpuop` 解析为具体内核类型。不引入 trait object、服务定位器、按键名注册表或全局可变状态。内核类型直接实现冻结的 `kernel<f32>`（以及累加式的 `binarykernel`/`weightedkernel`/`gatherkernel`）。

2. **单目优先的算子集，随后累加扩展。** 首批算子集是 `relu`、`silu`、`gelutanh`、`softmax`，因为每个都真正单目。二元与带权算子（`matmul`、`attention`、`embedding`、带权 `rmsnorm`）曾在此前阶段被推迟，后在第 5 章的累加契约扩展中加入。

3. **仅 f32。** 后端在其 capability 中声明恰好 `[dtype::f32]`。不使用 `unsafe` 收窄去触达未实现的 dtype。`i8`/`u8`/`i32`/`i64` 激活未实现并报告为不支持。

4. **crate 自有的策略错误。** `cpu::cpuerror` 包装 `api::backenderror`（内核级失败，经 `#[from]`）并新增 `backenderror` 刻意不建模的区分：被拒的元素或字节预算（`limiterror`）与被拒的 rank（`rankexceeded`）。错误处理准则把 CPU 内核失败归 `cpu`，并允许后端特有内部错误，只要公共抽象只暴露不支持操作、资源耗尽与执行失败三种区分。

5. **策略只强制一次。** `cpuexecutor::apply` 在派发前对后端声明的预算校验 rank、元素计数与字节大小。内核保持纯净、无状态、无资源策略。

6. **先基线后优化。** 算子基准建立基线；不声明性能改进，也不引入 SIMD 或并行。

7. **存储 dtype 与计算精度是不同轴。** `dtype` 是完整存储分类法，`precision` 是计算分类法，`From<precision> for dtype` 是嵌入而非满射。`cpu` 只执行 `f32`，即便张量可承载 `f64` 或整数存储。`element` 的宽度不变量 `DTYPE.sizeinbytes() == size_of::<Self>()` 由测试强制。

### 6.3 接口边界（DS1）

`cpu` 导出 16 个公共符号，低于 A4 上界 20：

| 公共项 | 模块文件 | 类别 |
| --- | --- | --- |
| `CPUMAXRANK` | `backend.rs` | 常量（值为 8） |
| `cpubackend` | `backend.rs` | 后端身份 |
| `cpurelu`、`cpusilu`、`cpugelutanh` | `activation.rs` | 单目内核 |
| `cpusoftmax` | `softmax.rs` | 单目内核 |
| `cpumatmul` | `matmul.rs` | 二元内核 |
| `cpuadd`、`cpumul` | `elementwise.rs` | 二元内核 |
| `cpurmsnorm` | `normalization.rs` | 带权内核 |
| `cpuembedding` | `embedding.rs` | gather 内核 |
| `cpurope` | `rotary.rs` | 属性内核 |
| `cpuattention` | `attention.rs` | 属性内核 |
| `cpuexecutor`、`cpuop` | `executor.rs` | 预算所有者与派发枚举 |
| `cpuerror` | `error.rs` | crate 错误 enum |

`cpubackend::new(lim: limits)` 接受显式 `limits`；本 crate 不发明默认资源上界。`cpubackend` 实现 `backend`：`getname` 返回 `"cpu"`，`getdevice` 返回 `device::cpu()`，`getcapabilities` 返回声明 `dtypes: vec![dtype::f32]`、`maxrank: CPUMAXRANK`、`limits`。

`cpuexecutor` 的派发入口：`apply(op, input)`（单目）、`applybinary(op, lhs, rhs)`、`applyweighted(op, input, weight)`、`applygather(op, table, indices)`，以及 `compute` 实现（`matmul`/`add`/`mul`/`softmax`/`silu`/`embedding` 走上述路径，`rmsnorm`/`rope`/`attention` 走预算检查加具体内核）。`cpuop` 变体为 `relu`、`silu`、`gelutanh`、`softmax`、`matmul`、`add`、`mul`、`rmsnorm`、`embedding`；`rope`/`attention` 带逐调用属性，只经 `compute` 触达。

逻辑/算法内核语义：`matmul` 无广播、左到右 f32 累加、确定性；`add`/`mul` 要求形状完全相等，NaN/无穷按普通浮点传播；`rmsnorm` 对每行 `x` 与权重 `w` 计算 `rms = sqrt(mean(x_i^2) + eps)`、`out_i = (x_i/rms)*w_i`；`softmax` 减行最大值后指数化，和为 0 时保持全零（对有限行不可达：最大项自身的指数为 1），全 `-inf` 行因 `max - max` 为 NaN 而传播 NaN；`relu` 为 `if x < 0.0 { 0.0 } else { x }`；`silu` 为 `x/(1+exp(-x))`；`gelutanh` 为 `0.5*x*(1+tanh(c*(x+0.044715*x^3)))`，`c=sqrt(2/pi)`（tanh 形式，非 erf 形式，两者约差 `1e-3`）；`embedding` 输出 `[indices.len(), table.shape[1..]]`；`rope` 对每段 `headdim` 切片按 `theta_i = base^(-2i/headdim)` 旋转 `(2i, 2i+1)`；`attention` 使用 `scale = 1/sqrt(headdim)`，`heads` 是 `kvheads` 的正整数倍，每对 query head 复用 `heads/kvheads` 个 kv head。

### 6.4 不变量与错误路径（DS2）

算法级不变量：经校验的形状计划产生一个输出，其元素计数等于各维度之积（同第 5 章）。

实现级不变量：内核只在校验之后分配输出；用于切分输入的每个索引 `<` 该输入的元素计数；`cpuexecutor::checkbudget` 对 rank、元素计数（受检乘法）与字节大小（元素数乘 `size_of::<f32>()`，受检）逐一校验；`checkgatherbudget` 对 row 宽度与 gather 输出元素数做受检乘法。

错误分支枚举：rank 超后端最大（`rankexceeded`）、元素/字节预算超出（`limit`）、dtype 不受支持（`unsupporteddtype`）、操作不受支持（`unsupportedoperation`）、操作数形状不匹配（`shapemismatch`）、索引越界（`indexoutofrange`）、元素计数溢出（`sizeoverflow`）、设备不可达（`devicenotavailable`）、内核编译失败（`kernelcompilefailed`）、执行失败（`executionfailed`）。

`cpuerror`（3 个 variant）：`backend(#[from] backenderror)`、`limit(#[from] limiterror)`、`rankexceeded { rank, max }`。`maptobackend` 把 `cpuerror::limit` 与 `rankexceeded` 映射到 `backenderror::resourceexhausted`，把 `cpuerror::backend` 原样透传。

资源回滚：失败的内核在构造输出前返回，无部分分配逃逸；`cpu` 的类型是值类型，拥有的缓冲由作用域释放（OW1、ER4）。

### 6.5 复杂度（DS3）

每个函数低于 DS3 上界（圈复杂度 ≤ 10，有效行数 ≤ 80）。最大者为 `cpuattention::apply` 与 `attentionplan`。（以阅读核对，证据等级 T。）

### 6.6 并发设计（DS4）

`cpu` 不引入共享可变状态与同步原语。`cpubackend` 与各内核类型是无状态值；`cpuexecutor` 持有 `cpubackend` 并只读访问。`compute` 以 `&self` 调用，隐含 executor 可在请求间共享。不存在锁序；不引入线程（PF2 的热路径无分配、无全局锁要求在此阶段以朴素路径满足）。

### 6.7 被否决方案（DS5）

来自本章决策：

- **现在就以二元/归约内核 trait 扩展 `api` 以在同一阶段实现 `matmul`。** 被否决：扩大变更边界并把契约变更与新后端混在一起，违反 AGENTS.md §11 的「每个提交一个可测试阶段」。
- **在 `cpu` 本地用更丰富的 trait 实现 `matmul`。** 被否决：会分叉算子契约，让 `cpu` 与 `api` 漂移，违反 §5.3 要求的单一窄契约。
- **按键名动态内核注册表。** 被否决：静态分发是既定偏好，注册表增加间接层而无运行期选择需求。
- **为逐元素内核引入 SIMD 或线程。** 本次阶段被否决：性能准则要求先有实测瓶颈与正确性计划，尚无此证据。
- **把预算拒绝映射到 `backenderror::executionfailed`。** 被否决：把资源耗尽与执行失败混为一谈，错误处理准则禁止。
- **复用 `tensorerror` 表示预算拒绝。** 被否决：`tensorerror` 由 `tensor` 拥有用于元数据校验，从 `cpu` 返回它会让调用方为 `cpu` 拥有的策略去解释另一个 crate 的错误。

来自存储 dtype / 计算精度决策：

- **保持 `f64` 在外并以「不支持」文档化。** 被否决：保留循环论证，且与 `i64` 不一致。
- **同时增加 `precision::f64`。** 被否决：会暗示不存在的计算能力。

### 6.8 后果

- Layer 1 拥有可用、经测试的 CPU 路径，且不改变 Layer 0；`cpu` 只依赖 `api`、`tensor`、`lib`，图无环，后端实现可替换。
- `kernel` 契约强制每次 `apply` 一次分配；就地或 arena 执行无法经冻结契约表达，被推迟。
- 无法表达为单操作数映射的算子（矩阵乘、attention、embedding gather、带权归一化）在契约被累加扩展前保持未实现，随后经 `compute` 提供。
- 直接使用内核类型的调用方绕过 executor 的预算检查；预算强制属于 `cpuexecutor`，内核只校验自身形状前置条件。

### 6.9 验证

本节的验证于 2026-10-10 执行：

- 仓库根 `tests/cpuunits.rs` 与 `tests/cpucontract.rs`（对应 `cpu`）—— 0 失败。
- 本记录重测（2026-10-10）：`cargo test --workspace --locked -- --list` 枚举 736 个用例；`bash scripts/check.sh` 报告 "All static quality checks passed"；通过/失败总数未在本轮重跑。相关根集成套件位于仓库根 `tests/`。
- `bash scripts/check.sh` —— format、`cargo check --workspace --all-targets`、`cargo clippy --workspace --all-targets -- -D warnings`、`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 全部通过。
- `bash scripts/test.sh` —— 完整工作区单元、集成与文档测试通过。
- 后续补充：仓库根 `tests/cpuunits.rs`/`tests/cpucontract.rs`（对应 `cpu`）与 `tests/testkitunits.rs`（对应 `testkit`）—— 0 失败（见第 5.9 节）。
（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）

基线（executor 级 `benches/cpuops.rs`，可经 `cargo bench -p llm --bench cpuops` 复现；裸内核 `benches/kernels.rs`）：release profile、`lto = "thin"`、`codegen-units = 1`、`debug = 1`；criterion 0.7.0；warm-up 0.2 s、measurement 1.0 s、20 samples；无 SIMD、无线程。环境 `rustc 1.99.0 (b940084d7 2026-09-28)`，macOS 27.0.1，arm64。未记录确切 CPU 型号；为同机同工具链基线，不可移植。

| Benchmark | Workload | Time (median) | Throughput |
| --- | --- | --- | --- |
| elementwise/relu | 1,048,576 f32 | 68.509 µs | 15.31 Gelem/s |
| elementwise/silu | 1,048,576 f32 | 1.0690 ms | 980.9 Melem/s |
| elementwise/gelutanh | 1,048,576 f32 | 1.5330 ms | 684.0 Melem/s |
| softmax/rows512cols4096 | 2,097,152 f32 | 2.5344 ms | 827.5 Melem/s |
| kernel/binary/matmul256 | 256x256x256 f32 | 974.0 µs | 67.3 Melem/s |
| kernel/attention/seq128heads8dim64 | 65,536 f32 | 1.268 ms | 51.70 Melem/s |

这些数字是基线，不证明也不声称任何相对既有实现的改进（VF5）。

### 6.10 复审条件

当具体模型要求第二操作数契约时（则以累加方式扩展 `api`）；当剖析显示超越逐元素内核或 softmax 在某个文档化负载中占主导时（则评估带正确性计划的 SIMD）；当请求级 KV cache 或批路径使逐调用分配成为实测瓶颈时（则在不削弱契约的前提下评估分配复用），重新审议本章。

## 7. 层 1：autograd

### 7.1 上下文

Layer 1 计算扩展把后端无关的算子集放到 `api::compute` 之后。训练与模型层（`nn`、`models`、`train`）需要这些算子的梯度。没有差异化 crate，每个上层要么手写反向传播，要么绑定单一后端，两者都破坏 `architecture.md` 的依赖方向与 AGENTS.md §3.3 的静态组合规则。

因此 `autograd` 只拥有差异化机制。它依赖 `lib`、`tensor`、`api`，永不依赖 `cpu`，因此同一图代码运行在任何实现 `api::compute` 的后端上。

本决策新增的依赖边（无环）：

```text
lib
tensor   -> lib
api      -> lib
api      -> tensor
autograd -> api
autograd -> tensor
autograd -> lib
```

### 7.2 决策

1. **一个所有者，一条磁带。** `graph<T: element>` 是每个 `variable<T>` 的唯一所有者；它按创建顺序把它们存在一个 `Vec` 中。把节点存在向量里使所有权显式：无别名、无内部可变性、无引用环、无 `unsafe`。算子每次调用追加一个节点；反向传播按逆序走一遍向量。

2. **可复制句柄，而非借用。** `variableid(usize)` 是图局部索引，是 `Copy` 且不拥有任何东西。算子以 `getvalue`/`valueof` 取值，以 `setgrad` 安装梯度，因此一次调用期间图不会被同时不可变与可变借用。

3. **静态算子 enum，而非装箱闭包。** `op` 是 enum，其变体命名输入句柄（`matmul { lhs, rhs }`、`rmsnorm { input, weight, eps }`、`softmax { input }`、`relu|silu|gelutanh { input }`）。因此磁带不持 `Box<dyn Fn>`、不持有捕获的堆状态、不做动态分发。`getinputs` 精确返回反向传播消费的句柄。

4. **前向算子是图上的固有方法。** `graph::matmul`、`graph::add`、`graph::mul`、`graph::rmsnorm`、`graph::softmax`、`graph::relu`、`graph::silu`、`graph::gelutanh` 转发到 provider，追加节点并返回句柄。只有当录制开启且至少一个输入需要梯度时才记录节点（`graph::record`），因此纯推理路径不分配梯度簿记。

5. **反向传播是单次逆序扫描。** `graph::backward(id)` 以全 1 播种输出余切，从种子向下走到索引 0，经 `backward.rs` 派发每个节点的向量-雅可比积。梯度累加进每个消费者（`ops::accumulate`），因此被多次使用的值收到其贡献之和。

6. **RAII 录制控制。** `graph::nograd` 返回 `nogradscope<'_, T>`，一个在创建时清除录制标志、在 drop（含 unwind）时恢复的作用域。作用域内产生的输出保留其值但携带 `op::leaf`，因此后续反向传播忽略它们。`graph::detach` 是非作用域形式。

7. **仅 f32，无 `unsafe`，无 panic。** crate 声明 `#![forbid(unsafe_code)]`；每个可失败路径返回 `autoggraderror`，它包装 `backenderror` 与 `tensorerror` 并新增 `unknownvariable`、`unsupportedshape`、`sizeoverflow`。生产路径不含 `panic`、`unwrap`、`expect`。

8. **逐 crate 错误归属。** `autoggraderror` 是 crate 自己的 enum。它无损重导出抽象的区分（`backend`、`tensor`），绝不把后端实现错误泄漏进上层契约（ER2）。

### 7.3 接口边界（DS1）

`autograd` 的 crate root 重导出 6 个公共类型 —— `graph`、`variable`、`variableid`、`op`、`autoggraderror`、`nogradscope` —— 且无自由函数；`arena`、`node`、`variables`、`gradient`、`nograd`、`backward`、`ops` 保持私有模块。反向传播只经固有方法 `graph::backward` 与 `graph::backwardwith` 触达，因此逐算子的 VJP 辅助函数永不面向调用方。公共类型/trait/enum/variant/函数名小写拼接；常量大写拼接。模块文件名（`arena`、`node`、`variables`、`gradient`、`nograd`）不同于其导出的公共类型（`graph`、`op`/`variableid`、`variable`），因为模块与重导出类型在 crate root 共享一个命名空间（E0255 守卫）。

`graph<T>` 的公共接口：`new`、`getlen`、`isempty`、`get(id)`、`getvalue(id)`、`getgrad(id)`、`requiresgrad(id)`、`recording()`、`leaf(value, requiresgrad)`、`constant(value)`、`detach(id)`、`nograd()`、`backward(id)`、`backwardwith(id, seed)`，以及 `gradient.rs` 增加的 `zerograd(id)` 与 `cleargradients()`。

### 7.4 不变量与错误路径（DS2）

算法级不变量：

- 在 `backward(root)` 之后，对每个被记录的节点，累加梯度等于其所有消费者上的向量-雅可比积之和；叶子的梯度留待调用方读取。
- 前向算子经 provider 运行，反向函数在记录值上用普通 `f32` 算术。

实现级不变量：

- 被记录 `op` 中的每个 `variableid` 都 `< vars.len()`（id 由 `push` 按创建顺序发放）。
- 节点的梯度张量存在时与其值同形状。
- 磁带只增长。`accumulate` 在既有梯度形状与增量不一致时返回 `unsupportedshape`。
- `backwardwith` 的显式种子必须与输出同形状，否则 `unsupportedshape`；输出不需要梯度时该调用是 no-op。

错误分支枚举：未知句柄（`unknownvariable`）、被后端或 op 层拒绝的算子形状（`unsupportedshape`）、尺寸算术溢出（`sizeoverflow`），以及任何透明传播的后端/张量失败。

资源回滚：被拒的算子在 `push` 之前返回，因此失败的前向不在磁带上留下部分节点；图拥有每个节点，随图一起 drop。

### 7.5 复杂度（DS3）

每个函数低于圈复杂度 10 与有效行数 80；最大者为 `backward` 中的逆序循环与 `gelutanh` 的 VJP，二者都是短直线体。（以阅读核对，证据等级 T。）

### 7.6 并发设计（DS4）

`autograd` 不引入共享可变状态、全局或 thread-local 磁带（A6/OW2 禁止）。磁带是调用方拥有的值。`nogradscope` 可变借用图，并在 drop（含 unwind）时恢复标志，因此录制状态不会因异常而错乱（CC5 的取消/异常传播语义在此以 RAII 表达）。无锁、无锁序。请求级隔离以所有权表达：每次训练步构建一个新的 `graph`。

### 7.7 被否决方案（DS5）

设计门（DS0/DS5）中列出的被否决方案：

- **逐节点装箱闭包（`Vec<Box<dyn Fn(...)>>`）。** 被否决：静态 enum 携带同样信息时引入动态分发与捕获的堆状态，违反 AGENTS.md §3.3。
- **引用计数 `Rc<RefCell<Node>>` 图。** 被否决：内部可变性与引用环；单一拥有向量更简单且无 `unsafe`。
- **全局可变磁带或 thread-local arena。** 被否决：A6/OW2 禁止全局可变状态，请求级状态必须有显式所有者，因此磁带是调用方拥有的值。
- **在单独一趟或图重写 IR 中录制。** 被否决：没有实测需求；define-by-run 是优化图 IR 必须击败的基线。
- **直接依赖 `cpu` 做反向数学。** 被否决：保持依赖方向 `autograd -> api`，保持后端可替换。

其它被否决方案：

- **前向模式（雅可比-向量）差异化。** 被否决：反向模式适配多参数标量损失训练目标。
- **数值微分作为生产反向路径。** 被否决：中心差分每步 O(参数数)，此处仅用作独立测试预言机。
- **一个算子 trait（`backwardop`）每算子一个 impl。** 被否决：静态 `op` enum 加 `match` 把反向传播放在一处可审计位置，避免每节点一个 trait object。

### 7.8 后果

- `nn` 与 `models` 在 `autograd` 之上构建而不碰后端，并复用推理路径所用的同一 `api::compute` 算子。
- 磁带在 `graph` 生命周期内只追加；长训练循环应每步构建新图（自然的请求/步边界），而非无限保留。这是文档化的使用契约，不是隐藏默认。
- 梯度是真实 `f32` 张量，因此 `backward` 为每个需要梯度的节点分配一个张量。那是 `api::compute` 契约的拥有输出成本，不是隐藏缓冲池。
- `nograd` 同时抑制录制与梯度存储，因此一次评估遍历比开启录制的同一次遍历分配更轻。

### 7.9 验证

2026-10-10 执行（仅 scoped 命令，因同批 crate 处于构建中）：

- 仓库根 `tests/autogradunits.rs` 与 `tests/autograd.rs`（对应 `autograd`）—— 0 失败；1 doctest 忽略（`nograd` 示例为说明性）。
（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
- `cargo test -p llm --test autograd` —— 13 通过，0 失败，exit 0。每个测试把解析 VJP 与中心有限差分 `(f(x+h) - f(x-h)) / (2h)` 在文档化容差内比较；另有测试覆盖多消费者累加、`detach` 与 `nograd` 作用域。
- `cargo clippy -p autograd -p llm --all-targets -- -D warnings` —— 干净，exit 0。
- `cargo check -p llm --all-targets` —— 通过，exit 0。

基线：`cargo bench -p llm --bench autograd`，release profile（`lto = "thin"`、`codegen-units = 1`、`debug = 1`），criterion 0.7.0，warm-up 1.0 s，measurement 2.0 s，20 samples，无 SIMD，无线程。环境 `rustc 1.99.0 (b940084d7 2026-09-28)`，macOS 27.0.1，arm64。

| Benchmark | Workload | Time (median) | Throughput |
| --- | --- | --- | --- |
| autograd/backward/matmul_rmsnorm_softmax | 64x128 @ 128x128, rmsnorm, softmax; forward + reverse | 763.53 µs | 10.73 Melem/s |

测得区间 759.30–769.40 µs；20 个样本中 3 个为高严重度离群。这些数字是基线；不声明改进（VF5），且没有可比较的既有实现。

### 7.10 复审条件

当剖析显示磁带本身（而非算子）主导训练步时（则考虑带正确性计划的图 IR 或算子融合）；当某模型需要 `op` 尚未包含的反向算子时（则以累加方式扩展 enum 与反向 `match`）；当需要批矩阵乘或归约时（则同时扩展 `api::compute` 与磁带）；当请求隔离要求使逐步图所有权规则不再充分时（则带显式所有者重新审议磁带生命期），重新审议本章。

## 8. 层 2：nn

### 8.1 上下文

Layer 1 给工作区提供后端无关算子与差异化机制。模型层（`models`）需要可复用、可组合的层 —— 投影、嵌入、归一化、旋转嵌入、注意力与前馈网络 —— 以及一种命名并携带其参数、以便写入与读取检查点的方式。

`nn` 拥有这些层与容器。它只依赖 `lib`、`tensor`、`api`，因此组合算子契约且永不绑定具体后端。层在运行期经 `api::compute` 到达 CPU，正如 `autograd`。

本决策新增的依赖边（无环）：

```text
lib
tensor -> lib
api    -> lib
api    -> tensor
nn     -> api
nn     -> tensor
nn     -> lib
```

### 8.2 决策

1. **参数是具名、拥有的张量。** `parameter` 把一个叶子名绑定到拥有的 `tensor<f32>`。它经公共接口不可变，因此活跃层无法观察到参数在它下方变化。

2. **`module` 是名字加参数导出，而非前向签名。** `module` trait 暴露 `getname` 与 `collect(prefix, dict)`；默认 `getstatedict` 在 root 前缀收集模块。前向刻意**不**在 trait 上：投影接受一个张量、嵌入接受 `&[usize]` 索引、注意力接受三个操作数，因此单一签名会迫使输入被 enum 包装，增加间接层而不增加含义。每层改为暴露与其操作元数匹配的固有 `forward`。

3. **`statedict` 是名字与值相遇的唯一处。** `statedict` 是有序 `BTreeMap<String, tensor<f32>>`。层通过前缀其叶子名组合出层级名（`mlp` 产生 `gate.weight`、`up.weight`、`down.weight`）。重复插入以 `nnerror::duplicateparameter` 拒绝，而非静默覆盖权重。按名排序使迭代确定，无需单独排序。

4. **层只组合 provider 的原语。** 每次前向调用 `api::compute`；层不引入自己的算子。原语集刻意省略某行为处，层显式执行：`linear` 在 `add` 前把 bias 铺成 `[rows, out]`，因为 `api::compute::add` 要求形状相等，且 `api::compute` 刻意拒绝了广播，以便层显式 reshape。

5. **权重布局为 `[in_features, out_features]`。** 以该顺序存储权重让 `linear` 直接调用 `api::compute::matmul`，无需转置算子、无需逐调用转置拷贝。Bias 为 `[out_features]`。输入的 leading batch 维为秩二 matmul 压平后再恢复，因此 `[..., in]` 映射到 `[..., out]`。

6. **注意力是配置，其数学是后端契约。** `attention` 拥有 `heads`、`kvheads`、`causal`，校验 head 布局后转发到 `api::compute::attention`，后者已组合分组 head、causal mask、缩放、matmul 与 softmax。层不在后端旁重实现该数学；复制它会制造第二份需同步的实现。

7. **前馈网络是 SwiGLU。** `mlp` 是 `down(silu(gate(x)) * up(x))`，Llama 族门控 MLP。它由三个 `linear` 与 `silu`/`mul` 原语构成，并在构造时校验一次投影形状，而非每次前向。

8. **初始化是确定性且局部的。** `initxavier`、`fillnormal`、`filluniform`、`fillvalue`、`fillzero`、`fillone` 各自构建由 `lib::seed` 播种的 SplitMix64 流。无全局生成器，因此同一 seed 与调用顺序的两次运行产生相同参数；无隐藏的进程级 RNG 状态（A6）。

9. **仅 f32，无 `unsafe`，无 panic。** crate 声明 `#![forbid(unsafe_code)]`；每个可失败路径返回 `nnerror`，它包装 `backenderror` 与 `tensorerror` 并新增 `shapemismatch`、`invalidconfig`、`duplicateparameter`、`missingparameter`。生产路径不含 `panic`、`unwrap`、`expect`。

### 8.3 接口边界（DS1）

crate root 导出 10 个公共类型（`nnerror`、`parameter`、`statedict`、`module`、`linear`、`embedding`、`rmsnorm`、`rotary`、`attention`、`mlp`）、7 个函数（`fillzero`、`fillone`、`fillvalue`、`filluniform`、`fillnormal`、`initxavier`、`joinname`）与 2 个常量（`DEFAULTEPS`、`DEFAULTBASE`）—— 低于 A4 上界 20。公共类型/trait/enum/variant/函数名小写拼接；常量大写拼接。模块文件（`parameters`、`dictionary`、`modules`，与层文件 `projection`、`lookup`、`normalize`、`rotate`、`multihead`、`gated`）都不同于其导出类型（`parameter`、`statedict`、`module`、`linear`、`embedding`、`rmsnorm`、`rotary`、`attention`、`mlp`），因为模块与重导出类型共享一个命名空间（E0255 守卫）。

关键接口：

- `module`：`getname`、`collect(prefix, dict) -> Result<(), nnerror>`、默认 `getstatedict() -> Result<statedict, nnerror>`（在 root 前缀收集）。
- `statedict`：`new`、`isempty`、`getlen`、`insert(name, value)`、`add(prefix, parameter)`、`get`、`getmut`、`names`、`iter`；`PartialEq` 按名与 `tensoreq`（形状和数据）比较。
- `parameter`：`new(name, value)`、`getname`、`getvalue`、`getshape`、`getlen`。
- `linear`：`new(weight, Option<bias>)`、`getinfeatures`、`getoutfeatures`、`getweight`、`getbias`、`forward(engine, input)`。
- `embedding`：`new(weight)`、`getvocab`、`getwidth`、`getweight`、`forward(engine, indices)`。
- `rmsnorm`：`new(weight, eps)`、`getfeatures`、`geteps`、`getweight`、`forward(engine, input)`。
- `rotary`：`new(heads, base)`、`getheads`、`getbase`、`forward(engine, input)`。
- `attention`：`new(heads, kvheads, causal)`、`getheads`、`getkvheads`、`getcausal`、`forward(engine, query, key, value)`（无参数，`collect` 为空）。
- `mlp`：`new(gate, up, down)`、`getfeatures`、`gethidden`、`getgate`、`getup`、`getdown`、`forward(engine, input)`。
- 初始化：`fillzero`、`fillone`、`fillvalue`、`filluniform(shape, lo, hi, seed)`、`fillnormal(shape, mean, std, seed)`、`initxavier(shape, seed)`（limit = `sqrt(6/(fan_in+fan_out))`）。

`DEFAULTEPS = 1.0e-5`，`DEFAULTBASE = 10_000.0`。`joinname(prefix, leaf)` 在 prefix 为空时返回 leaf，否则返回 `prefix.leaf`。

### 8.4 不变量与错误路径（DS2）

算法级不变量：

- 层的 forward 输出形状是其文档化布局所隐含的形状，且参数的元素计数等于其形状之积。
- `linear` 输入 `[..., in]` 产生 `[..., out]`；`mlp` 的 `gate`/`up` 共享输入输出特征，`down` 消费 `gate` 的输出特征并产生 `gate` 的输入特征。

实现级不变量：

- 用于切分参数的每个索引受该参数元素计数界定（provider 为其自身算子保证这一点；`linear` 以受检乘法把 bias 铺成 `rows * out`）。
- `statedict` 永不含重复名。
- 层不存请求级可变状态；层在前向期间不可变。

错误分支枚举：rank 不匹配（权重/bias/表）、最后一维不匹配（`linear`、`rmsnorm`）、head 布局为奇或不可整除（`rotary`、`attention`）、序列/head 不匹配（`attention`）、嵌入索引越界（provider）、epsilon/base/head 数无效、形状元素计数溢出。

`nnerror`（6 个 variant）：`backend(#[from] backenderror)`、`tensor(#[from] tensorerror)`、`shapemismatch { layer, expected, actual }`、`invalidconfig { field, detail }`、`duplicateparameter { name }`、`missingparameter { name }`。辅助构造器 `shapes(layer, expected, actual)` 与 `config(field, detail)`。

资源回滚：被拒的 `new` 在层值存在之前返回；被拒的 forward 在层被改变之前返回（层在前向期间不可变），因此无部分状态逃逸。

### 8.5 复杂度（DS3）

每个函数低于圈复杂度 10 与有效行数 80；最大者为 `attention::forward` 与 `linear::forward`，都是短校验后跟 provider 调用。（以阅读核对，证据等级 T。）

### 8.6 并发设计（DS4）

层是运行期无状态的不可变参数值；层不在调用间持前向缓冲，因此请求隔离（OW2）由类型自然成立，无需额外工作。无共享可变状态、无同步原语、无锁序。`statedict` 拥有其张量，不持对层的活引用，因此快照无法观察到层在变更中（OW4）。初始化无全局状态。

### 8.7 被否决方案（DS5）

- **统一的 `module::forward` 签名。** 被否决：异构元数需要 enum 包装输入，增加间接层与一个失败模式（错误变体），而不增加含义。
- **一个 `provider` trait 抽象推理与训练，使一份层实现服务两者。** 本次阶段分析后否决：它是对两个值域（推理用 `tensor`，训练用 autograd 句柄）的通用 provider 抽象，没有共享的图 reshape 算子来桥接它们；会比它解决的问题更大。层是仅前向的，训练路径被推迟（见「偏离」）。
- **PyTorch 权重布局 `[out, in]`。** 被否决：因为 `api::compute` 无转置算子且本阶段不新增，它会在每次前向强制一次转置。
- **按键名的层注册表（`Box<dyn layer>`）。** 被否决：静态组合是既定偏好；模型组合具体层。
- **在 provider 内广播 bias。** 被否决：`api::compute` 已决定层显式扩展，隐藏广播规则会复活该决定所避免的形状缺陷。
- **用 `tensor::PartialEq` 做参数比较。** 被否决：元素相等不是张量元数据语义；`nn` 在本地定义 `tensoreq`（形状与数据）并在自身容器上实现 `PartialEq`。

### 8.8 与计划的偏离

- **本阶段未接上 `autograd` 依赖。** rule.md A2/§3.2 与 AGENTS.md §3.2 禁止仅为便利引入依赖，且计划自身的跨阶段决定否决「每 eval/train 一份实现」（会漂移）。`autograd` 目前不暴露 reshape、embedding 或 rope 算子，因此可微层路径无法在不重复这些前向实现或引入比问题更大的通用 provider 抽象的前提下复用它们（见 DS5）。本阶段因此只交付推理层并声明偏离。预期的统一 —— 层对 provider 泛型化，使推理与训练共享一份实现 —— 推迟到 `autograd` 拥有模型路径所需的 reshape/embedding/rope 算子之后，届时 `nn`（或适配器）获得 autograd 边。条款：rule.md A2、AGENTS.md §3.2。影响：训练路径复用尚不可用。降级方案：先加入缺失的 `autograd` 算子（独立、可测试的阶段），再不重复层数学地接上 provider。按 M4 记录。

### 8.9 后果

- `models` 组合 `linear`、`embedding`、`rmsnorm`、`rotary`、`attention`、`mlp` 并把其参数收集进一个名字稳定且层级的 `statedict`。
- 层是对不可变参数的纯函数：层不在调用间持前向缓冲，因此请求隔离（OW2）由类型自然成立。
- `linear` 每次前向分配一份压平输入拷贝与一份铺平 bias。那是 `api::compute` 契约加显式广播决定的拥有输出成本，不是隐藏缓冲。它是融合内核必须击败的文档化基线。
- 注意力的成本由 `cpu::attention` 主导（O(seq^2 * heads * headdim)）；层只增加校验。
- `rotary` 与 `attention` 不持参数、不收集任何东西，因此模型的 `statedict` 只含学到的权重。

### 8.10 验证

2026-10-10 执行（仅 scoped 命令）：

- 仓库根 `tests/nnunits.rs` 与 `tests/nnlayers.rs`（对应 `nn`）—— 0 失败；0 doctest。
（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
- `cargo test -p llm --test nnlayers` —— 12 通过，0 失败，exit 0。覆盖：`linear` 带/不带 bias 及保留 leading batch 维、`embedding`、`rmsnorm`、`rotary`、`attention`（对照独立标量分组查询 causal 参考）、`mlp`（对照显式 SwiGLU 组合）、形状错误拒绝、加前缀参数收集、初始化可复现。
- `cargo clippy -p nn -p llm --all-targets -- -D warnings` —— 干净，exit 0。
- `RUSTDOCFLAGS='-D warnings' cargo doc -p nn --no-deps` —— 干净，exit 0。

基线：`cargo bench -p llm --bench nn`，release profile（`lto = "thin"`、`codegen-units = 1`、`debug = 1`），criterion 0.7.0，warm-up 1.0 s，measurement 2.0 s，20 samples，无 SIMD，无线程。环境 `rustc 1.99.0 (b940084d7 2026-09-28)`，macOS 27.0.1，arm64。

| Benchmark | Workload | Time (median) | Throughput |
| --- | --- | --- | --- |
| nn/linear/forward | 128x512 in, 512x512 weight | 1.9002 ms | 34.49 Melem/s |
| nn/mlp/forward | 128x512, hidden 1365, SwiGLU | 18.518 ms | 3.539 Melem/s |
| nn/attention/forward | seq 128, heads 8, headdim 64 (causal) | 1.2131 ms | 54.03 Melem/s |

不声明改进（VF5）；这些是这些层的首次测量。

### 8.11 复审条件

当剖析显示 `linear` 的输入压平或 bias 铺平主导一次模型前向时（则加入带前后对比的融合或就地路径）；当 `autograd` 获得 reshape/embedding/rope 算子时（则接上共享 provider 并移除本偏离）；当某模型需要 `[in, out]` 以外的权重布局时（则向 provider 加入显式转置，而非改变层的存储）；当检查点格式固定了不同的参数命名方案时（则调整 `collect` 并记录），重新审议本章。

## 9. 层 2：models

### 9.1 上下文

Layer 2 给工作区提供可复用推理层与参数容器。推理层（`infer`）需要一个具体模型，把词元 id 从不可变权重的角度转成下一词元 logits，以及一个所有者为非模型的请求级键/值缓存契约。`models` 同时提供两者，外加一个把架构名映射到构造器的注册表。

`models` 依赖 `nn`、`api`、`tensor`、`lib`。它只经 `api::compute`（经层）到达后端，永不经过 `cpu`。来源 ADR 记为 Layer 3（依赖 Layer 2 的 `nn`），本章仍单列成章。

本决策新增的依赖边（无环）：

```text
lib
tensor -> lib
api    -> lib
api    -> tensor
nn     -> api
nn     -> tensor
nn     -> lib
models -> nn
models -> api
models -> tensor
models -> lib
```

### 9.2 决策

1. **`causallm` 是模型契约。** `causallm` 暴露 `getconfig` 与 `forward(engine, tokens) -> [tokens.len(), vocab]`。模型不可变地拥有其权重、不持请求级状态，因此一个模型值服务独立请求（OW2）。

2. **`kvcache` 在此定义，在别处实现。** `kvcache` 定义在 `models`，因为模型层拥有它消费的缓存抽象，但缓冲与其生命期属于实现它的调用方（`infer`）。它每层持一个 key 与一个 value 张量，并报告其层数、长度与容量。本阶段只定义契约；确定性前向重算整个序列，不读也不写缓存（见「偏离」）。

3. **架构按职责拆分。** `llamaconfig` 校验维度并计算派生值（`head_dim`、`kv_width`），而非二次存储。`llamaweights` 把每个参数作为 `nn` 层拥有并校验，因此模型读取的正是它校验过的值，无拷贝。`llamablock` 持不依赖权重的配置派生算子（注意力配置与两个 rotary 嵌入）。`llamamodel` 组合它们。

4. **权重在构造时校验，而非在前向期间。** `llamaweights::new` 对照配置检查嵌入、层数、最终归一化与输出投影，任何不一致返回 `modelerror::shapemismatch`。`llamamodel::new` 重新检查同样的不变量，因此模型不能由属于不同配置的权重构建。

5. **前向是 pre-norm 标准 Llama。** 每块是 `x + Wo(attn(rope(Wq(xn)), rope(Wk(xn)), Wv(xn)))`，随后 `x + down(silu(gate(xn2)) * up(xn2))`，其中 `xn`/`xn2` 是两个 RMS 归一化；模型随后施加最终归一化与输出投影。分组查询注意力与 causal mask 是后端的 `api::compute::attention`；层不引入算子。

6. **确定性随机权重用于可复现测试与基准。** `llamaweights::random(config, seed)` 从由单个 `lib::seed` 派生的逐参数子流抽取每个参数，因此同一 seed 与配置产生相同权重。归一化权重从 1 开始，投影为 Xavier-uniform，无 bias。

7. **注册表是受检 `match`，不是按键名表。** `architecture` 是 enum，每个受支持族一个 variant，`build` 经 `match` 派发；`getarchitecture` 把名字映射到 variant 并以 `modelerror::unknownarchitecture` 拒绝未知名。相对动态注册表，静态分发与编译期覆盖（A3）。

8. **仅 f32，无 `unsafe`，无 panic。** crate 声明 `#![forbid(unsafe_code)]`；每个可失败路径返回 `modelerror`，它包装 `backenderror`、`tensorerror`、`nnerror` 并新增 `invalidconfig`、`shapemismatch`、`unknownarchitecture`、`missingweight`。生产路径不含 `panic`、`unwrap`、`expect`。

### 9.3 接口边界（DS1）

crate root 导出 9 个公共类型（`modelerror`、`causallm`、`kvcache`、`architecture`、`llamaconfig`、`llamaweights`、`llamalayerweights`、`llamablock`、`llamamodel`）、1 个函数（`getarchitecture`）与 0 个常量 —— 低于 A4 上界 20。公共类型/trait/enum/variant/函数名小写拼接。模块文件（`causal`、`registry`、`llama/config`、`llama/block`、`llama/weights`、`llama/model`）不同于其导出类型（`causallm`/`kvcache`、`architecture`、`llamaconfig`、`llamablock`、`llamaweights`、`llamamodel`），因为模块与重导出类型共享一个命名空间（E0255 守卫）。

关键接口：

- `causallm`：`getconfig() -> &llamaconfig`、`forward<C: compute>(&self, engine: &C, tokens: &[usize]) -> Result<tensor<f32>, modelerror>`。
- `kvcache`：`getlayers() -> usize`、`getlen() -> usize`、`getmaxseq() -> usize`、`append(&mut self, layer, key, value) -> Result<(), modelerror>`、`keys(layer) -> Option<&tensor<f32>>`、`values(layer) -> Option<&tensor<f32>>`。
- `architecture`：enum `llama`；`getname()`、`newmodel(config, weights) -> Result<llamamodel, modelerror>`（命名为 `newmodel` 而非 `new`，因为返回 `llamamodel` 而非 `Self`，Clippy `new_ret_no_self` 拒绝 `new`）。
- `getarchitecture(name) -> Result<architecture, modelerror>`。
- `llamaconfig`：`new(vocab, dim, nlayers, nheads, nkvheads, hidden, rmseps, ropebase, maxseq)`；getter `getvocab`、`getdim`、`getnlayers`、`getnheads`、`getnkvheads`、`gethidden`、`getrmseps`、`getropebase`、`getmaxseq`、`getheaddim`（`dim/nheads`）、`getkvwidth`（`nkvheads*head_dim`）。
- `llamaweights`：`new(config, embedding, layers, norm, output)`；`getembedding`、`getlayers() -> &[llamalayerweights]`、`getnorm`、`getoutput`、`random(config, seed)`；实现 `module`（收集 `embedding`、`layers.{i}`、`norm`、`output`）。
- `llamalayerweights`：`new(config, attn_norm, wq, wk, wv, wo, ffn_norm, mlp)`；getter `getattnnorm`、`getwq`、`getwk`、`getwv`、`getwo`、`getffnnorm`、`getmlp`；实现 `module`（`getname` 返回 `"llamalayer"`，收集 `attn_norm`、`wq`、`wk`、`wv`、`wo`、`ffn_norm`、`ffn`）。
- `llamablock`：`new(config)`；`forward<C: compute>(&self, engine, weights, x)`。
- `llamamodel`：`new(config, weights)`；`getweights()`；实现 `causallm`。

### 9.4 《模型支持契约》

本节是模型支持契约的完整规范。

**支持的架构。** 目前只支持一个架构族：`llama`（RMSNorm、RoPE、分组查询注意力、SwiGLU）。`architecture` enum 只有一个 variant `llama`，`getarchitecture` 只识别名字 `"llama"`，其它名字返回 `modelerror::unknownarchitecture { name }`。新增架构 MUST 以扩展 `architecture` 及其 `match` 的方式加入，MUST NOT 改为动态注册表。注册表负责把名字映射到构造器，MUST NOT 在此 crate 内解析序列化的 `config.json`（那是 `format` 的职责）。

**配置校验。** `llamaconfig::new` 在构造时校验全部字段，任何不合法返回 `modelerror::invalidconfig { field, detail }`：

| 字段 | 校验 |
| --- | --- |
| `vocab` | 非零 |
| `dim` | 非零，且可被 `nheads` 整除 |
| `nlayers` | 非零 |
| `nheads` / `nkvheads` | 均非零；`nheads` 是 `nkvheads` 的整数倍 |
| `head_dim`（`dim/nheads`） | 偶数（使 rotary 成对存在） |
| `hidden` | 非零 |
| `rmseps` | 有限且非负 |
| `ropebase` | 有限且为正 |
| `maxseq` | 非零 |

派生值 `getheaddim`（`dim/nheads`）与 `getkvwidth`（`nkvheads*head_dim`）由存储字段计算，不二次存储。一个存在的 `llamaconfig` 因此满足：`dim % nheads == 0`、`nheads % nkvheads == 0`、head 维为偶。

**权重映射与命名。** `llamaweights` 把每个参数作为 `nn` 层拥有：一个 `embedding`（`[vocab, dim]`）、恰好 `nlayers` 个 `llamalayerweights`、一个 `[dim]` 最终 RMS 归一化、一个 `[dim, vocab]` 输出投影 `linear`（无 bias）。每个 `llamalayerweights` 持 `attn_norm`/`ffn_norm` 两个 `rmsnorm`（`[dim]`）、四个 `linear` 投影 `wq`（`[dim, dim]`）、`wk`/`wv`（`[dim, kv_width]`）、`wo`（`[dim, dim]`），以及一个 SwiGLU `mlp`（`gate`/`up` 为 `[dim, hidden]`，`down` 为 `hidden, dim`）。`llamaconfig` 提供的 `rmseps` 用于两个 RMS 归一化。

参数名稳定且层级，由 `nn::module::collect` 定义而非在此重新派生：`embedding.weight`、`layers.{i}.attn_norm.weight`、`layers.{i}.wq.weight`、`layers.{i}.wk.weight`、`layers.{i}.wv.weight`、`layers.{i}.wo.weight`、`layers.{i}.ffn_norm.weight`、`layers.{i}.ffn.gate.weight`、`layers.{i}.ffn.up.weight`、`layers.{i}.ffn.down.weight`、`norm.weight`、`output.weight`。

**兼容性。** 权重与配置的一致性是构造时校验：

- `llamaweights::new` 校验嵌入 `[vocab, dim]`、`layers.len() == nlayers`、最终归一化 `[dim]`、输出投影 `[dim, vocab]`，否则 `modelerror::shapemismatch` 或 `modelerror::config`。
- `llamamodel::new` 重新校验层数、嵌入、输出投影，并构建 `llamablock`（`attention`、`rotaryq`、`rotaryk`）。因此模型不能由属于不同配置的权重构建。
- 检查点级别的兼容性（对照显式 `tensorspec` 列表）由第 12 章定义，不在此处。

**KV cache 契约。** `kvcache` 由 `models` 定义（模型层拥有它消费的缓存抽象），由推理层实现（请求级缓冲及其生命期归调用方）。它每层持一个 key 与一个 value 张量，形状 `[cached_len, kv_heads * head_dim]`，并报告 `getlayers`、`getlen`、`getmaxseq`。`append` 把一层的单位置 key/value 追加；当层越界、张量形状与该层不一致或缓存已满时返回 `modelerror`。`keys`/`values` 在层为空时返回 `None`。模型权重不可变，缓存是请求级可变状态，二者分开（OW2/OW4）是 `infer` 实现 `kvcache` 的原因。

**限制（已声明的限制，非静默缺口）。**

- 仅 `f32`；`f16`/`bf16` 是存储 dtype，但无后端执行。
- 仅 `llama` 一个架构；无第二个架构。
- 前向路径提供全序列 `forward` 与增量 `forward_cached`（读/追加 `kvcache`，2026-10-16 接线：`incrementalattention` 2 + `kvcacheforward` 1 + `generation` 10 全绿，Metal 46/46、CUDA 44/44，见 `docs/conformance.md`「本轮重测证据（2026-10-16）」）。全序列 `forward` 每次调用重算每个位置，是 KV-cache decode 阶段必须击败的正确性基线；`forward_cached` 的性能收益尚未实测（同步往返主导），不声称任何加速。
- 无采样、无训练路径；`models` 生产依赖集为 `nn`/`api`/`tensor`/`lib`（无 `autograd`）。
- 注册表解析架构名，不解析序列化配置；`format` 拥有的 `config.json` 解析器在推理里程碑接入前不接线。
- 输出投影是显式、单独校验的权重；不支持绑定输入/输出嵌入。
- 投影无 bias（目标 Llama 架构无 bias）。

### 9.5 不变量与错误路径（DS2）

算法级不变量：

- 对任何合法输入，`forward` 返回 `[tokens.len(), vocab]` 张量，其值有限。
- 一个词元只出现在注意力掩码中位置 `<=` 其自身的位置，因此第一个位置只注意它自己。

实现级不变量：

- 一个存在的 `llamaconfig` 满足 `dim % nheads == 0`、`nheads % nkvheads == 0`、head 维为偶（在 `new` 中检查）。
- 一个存在的 `llamaweights` 有嵌入 `[vocab, dim]`、恰好 `nlayers` 个块、`[dim]` 最终归一化、`[dim, vocab]` 输出投影（在 `new` 中检查）。
- 模型不持可变状态，因此任何前向都无法观察到部分更新的权重；`llamablock` 是 `Copy` 的配置派生值。

错误分支枚举：零/奇/不兼容维度、负 epsilon、非正 rotary base、未知架构名、权重/配置不一致、越界词元（provider 索引错误）、形状元素计数溢出。

`modelerror`（7 个 variant）：`backend(#[from] backenderror)`、`tensor(#[from] tensorerror)`、`nn(#[from] nnerror)`、`invalidconfig { field, detail }`、`shapemismatch { context, expected, actual }`、`unknownarchitecture { name }`、`missingweight { name }`；辅助构造器 `config(field, detail)`。

资源回滚：被拒的构造器在模型值存在之前返回；模型在前向期间不可变，因此无部分状态逃逸。

### 9.6 复杂度（DS3）

每个函数低于圈复杂度 10 与有效行数 80；最大者为 `llamaconfig::new`（九个字段检查）与 `llamaweights::random`（对层循环的直线构造）。（以阅读核对，证据等级 T。）

### 9.7 并发设计（DS4）

模型不可变地拥有其权重、不持请求级状态，因此一个模型值可服务独立请求而无需共享可变状态（OW2/OW4）。无同步原语、无锁序。`kvcache` 是请求级可变状态，由实现者以显式所有权持有，与共享权重分离。`llamablock` 是 `Copy` 的不可变配置，可自由共享。

### 9.8 被否决方案（DS5）

- **在 `llamaweights` 存原始张量并每次前向重建 `nn` 层。** 被否决：重建层会克隆其参数，复制权重；持有层保持一份拷贝并复用 `nn` 的校验。
- **按键名 `Box<dyn causallm>` 注册表。** 被否决：静态分发与编译期穷尽优于动态注册表（A3）。
- **在 `models` 内解析 `config.json`。** 被否决：外部格式解析是 `format` 的职责；`models` 接受已校验配置，注册表映射已解析的架构名。
- **把 KV cache 折进模型。** 被否决：缓存是请求级可变状态而模型拥有不可变权重；保持分离是所有权规则（OW2/OW4），也是 `infer` 实现 `kvcache` 的原因。
- **投影带 bias。** 被否决：目标架构（Llama）无 bias，加入未使用 bias 会无收益地扩大权重契约与检查点格式。
- **绑定输入/输出嵌入。** 本次阶段被否决：输出投影是显式、单独校验的权重；绑定是后续选择，会改变检查点格式，采用时必须记录。

### 9.9 与计划的偏离

- **前向不接收也不填充 `kvcache`。** 计划的签名草图为 `forward(weights, tokens, cache)`。计划也陈述了推理优先次序：P7a 为正确性重算整个序列，P7b 把缓存作为实测优化加入。现在接上缓存参数要么闲置，要么在采样与请求层存在之前伪装增量 decode。本阶段因此定义 `kvcache` 契约并交付全重算前向，由 P7 拥有缓存路径。条款：rule.md VF1（不声明未验证能力）。影响：尚无缓存 decode。降级方案：在 `infer` 实现 `kvcache` 并加入带缓存前向变体，对照本阶段测量做前后基线。按 M4 记录。
- **注册表解析架构名，不解析它们。** 计划称注册表把架构映射到「加载配置」。序列化 `config.json` 解析器由 `format` 拥有，因此 `models` 拥有内存内 `llamaconfig` 与名字到构造器的映射；从 `format` 接入已解析名推迟到推理里程碑。条款：AGENTS.md §3.1（所有权边界）。影响：尚无文件到模型的路径。按 M4 记录。
- **`models` 生产依赖集为 `nn`/`api`/`tensor`/`lib`（无 `autograd`）。** 本阶段模型路径仅推理；加入 `autograd` 会是无用依赖（AGENTS.md §3.2）。训练路径集成随 `nn` 偏离一同推迟。

### 9.10 后果

- `infer` 可从配置与权重构建 `llamamodel`、调用 `forward`，并在日后实现 `kvcache` 而不让 `models` 依赖它。
- 权重是一份拷贝，被每次前向按引用共享；一次前向只分配激活张量。
- 全序列前向每次调用重算每个位置。那是 KV-cache decode 阶段必须击败的正确性基线；`kvcache` 契约为该测量就位。
- 参数名稳定且层级（`layers.0.wq.weight`、`norm.weight` 等），因此检查点的命名方案由 `nn::module::collect` 定义，不在此重新派生。

### 9.11 验证

2026-10-10 执行（仅 scoped 命令）：

- 仓库根 `tests/modelsunits.rs` 与 `tests/modelforward.rs`（对应 `models`）—— 0 失败；0 doctest。
（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
- `cargo test -p llm --test modelforward` —— 5 通过，0 失败，exit 0。核心测试对照由 `testkit::reference` 组装的独立标量参考交叉检查一次完整单层前向；其余钉住固定 seed 的确定性、每个位置的有限性、越界词元拒绝，以及第一个位置的 logits 不依赖后续词元。
- `cargo clippy -p models -p llm --all-targets -- -D warnings` —— 干净，exit 0。
- `RUSTDOCFLAGS='-D warnings' cargo doc -p models --no-deps` —— 干净，exit 0。

基线：`cargo bench -p llm --bench models`，release profile（`lto = "thin"`、`codegen-units = 1`、`debug = 1`），criterion 0.7.0，warm-up 1.0 s，measurement 2.0 s，20 samples，无 SIMD，无线程。环境 `rustc 1.99.0 (b940084d7 2026-09-28)`，macOS 27.0.1，arm64。

| Benchmark | Workload | Time (median) | Throughput |
| --- | --- | --- | --- |
| models/llama/forward | seq 16, dim 64, 2 layers, heads 4 / kvheads 2, hidden 170 | 135.57 µs | 118.02 Kelem/s |

吞吐按词元报告，匹配基准的元素单位。这是模型的首次测量；不声明改进（VF5）。

### 9.12 复审条件

当存在缓存 decode 路径时（则对照本基线比较并记录加速）；当接线检查点格式时（则校验 `collect` 的名字与之匹配并加入带形状检查的加载路径）；当加入第二个架构时（则扩展 `architecture` 及其 `match`，而非注册表表）；当需要权重绑定或 bias 时（则把权重契约与检查点格式作为单个可测试阶段一起改变），重新审议本章。

## 10. 层 2：tokenizer

### 10.1 上下文

路线图阶段 P4 引入 `tokenizer`。`tokenizer` 拥有：词表表示、编码与解码、适用的字节对编码、特殊词元处理、对话模板处理。它 MUST NOT 拥有张量执行、模型权重、KV cache 或请求调度。其唯一工作区依赖是 `lib`（用于 `tokenid`）；它用 `serde_json` 解析 HuggingFace `tokenizer.json` 文档。它刻意**不**依赖 `format`、`data`、`models` 或任何工作流 crate。文本到词元转换是更高层（例如 `infer`）消费的组合，因此此处不导入更高层 crate。

crate 遵循仓库无分隔符命名约定（NM1）、工作区无 `unsafe` 与无 `panic`/`unwrap`/`expect` 规则（AGENTS.md §4 与 §6），并拥有自己的 `thiserror` 错误 enum。

### 10.2 决策

crate 暴露 9 个公共项，根于 crate root 并从无冲突模块文件重导出：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `vocabulary` | `vocab.rs` | 不可变双向 token/id 映射 |
| `bytepair` | `merges.rs` | 有序 BPE 合并表，按 rank 优先级 |
| `specialtokens` | `special.rs` | 不可变结构标记集合 |
| `chatrole` | `chat.rs` | 对话角色枚举 |
| `chatmessage` | `chat.rs` | 一条角色/内容消息 |
| `chattemplateentry` | `chat.rs` | 逐角色前缀/后缀对 |
| `chattemplate` | `chat.rs` | 显式非 Jinja 对话模板 |
| `tokenizer` | `engine.rs` | 加载 JSON 子集并编排转换 |
| `tokenizererror` | `error.rs` | crate 自有错误 enum |

### 10.3 接口边界（DS1）

- `tokenizer` 只依赖 `lib`（`tokenid`）；外部解析是 `serde_json`，在 crate 清单中声明为工作区依赖。不导入 `tensor`、`api`、`cpu`、`format`、`data` 或任何工作流 crate。
- `tokenizer` 是值类型，除 install-once 对话模板（`setchattemplate`）外无内部可变性；它不持请求状态、不持缓存，因此可跨请求不可变共享（OW2）。
- 公共接口宽度：crate root 9 个项，远低于 A4 上界 20。*依据：单模块公共契约应能在一轮评审内完整复核（rule.md A4）。*

关键接口：`tokenizer::fromjson(&str)`、`getvocabulary`、`getbytepair`、`getspecialtokens`、`getchattemplate`、`setchattemplate`、`encode(&str)`、`encodespecial(&str)`、`decode(&[tokenid])`、`applychattemplate(&[chatmessage])`。`vocabulary::fromentries`、`getid`、`getcontent`、`getcount`、`getminid`、`getmaxid`。`bytepair::frommerges`、`getcount`、`getrank`、`apply`。`specialtokens::new`、`getid`、`getcontent`、`contains`、`getcount`、`iter`。`chattemplate::new(bos, eos, entries)`、`getbos`、`geteos`、`render`。`chatrole::getname`。

### 10.4 不变量与错误路径（DS2）

算法级不变量：

- `vocabulary` 是双射：对每个条目，`getcontent(getid(content))` 返回 `content`，`getid(getcontent(id))` 返回 `id`。重复完全相同的 `(content, id)` 对被接受；任何其它冲突被拒绝。
- `bytepair` 的 rank 是首次出现的数组下标；重复的 merge 字符串保留其首个 rank。`apply` 是确定性的：每一步合并最低 rank 的相邻对，平局向左取，直到无规则适用。
- `specialtokens` 在同一 entry 集上按内容与 id 解析。
- `decodetokens` 把 byte-fallback 词元消费进 pending 字节缓冲，并发出单个 UTF-8 字符串；当 id 构成合法 UTF-8 时拼接是字节精确的。

实现级不变量：

- `vocabulary` 把 id 到 token 方向存于 `BTreeMap<u32, String>` 而非稠密 `Vec`，因此来自不可信输入的越界 id 无法触发大分配；id 对照已插入条目校验，绝不对照其数值大小（IN4）。
- `encodespecial` 将候选字面量按最长优先排序，并以内容作平局决胜，因此重叠标记确定性解析。
- `parsebytetoken` 只接受恰好 `<0x` + 两个十六进制数字 + `>`；其它一律当普通词元。

错误分支与返回语义（`tokenizererror`，15 个 variant）：

- `invalidjson { line, column }` —— 文档不是合法 JSON。
- `rootnotobject` —— JSON 根不是对象。
- `missingfield { field }` —— 缺少必需的点分路径。
- `unexpectedtype { field, expected }` —— 存在的字段 JSON 类型错误。
- `unsupportedmodel { modeltype }` —— `model.type` 不是 `"BPE"`。
- `idoutofrange { id }` —— 词元 id 不适合 `u32`。
- `duplicateid { id }` / `duplicatecontent { token }` —— 不一致词表。
- `malformedmerge { index }` —— merge 条目不是含两个非空侧的 `"left right"` 对。
- `malformedspecialtoken { index }` —— `added_tokens` 条目缺 `id` 或 `content` 或 `special` 类型错误。
- `unknowntoken { token }` —— 符号无词表条目、无完整 byte fallback、也无配置的 unknown 词元。
- `unknowntokenid { id }` —— 解码输入 id 不在词表中。
- `invalidutf8` —— 字节词元无法重建合法 UTF-8。
- `nochattemplate` —— 未安装模板却调用 `applychattemplate`。
- `unsupportedrole { role }` —— 消息角色无模板条目。

资源回滚：所有构造都是值式的。`fromjson` 构建局部 map，只在每个阶段校验后才返回 `Ok`，因此被拒文档不分配 `tokenizer`。解码累加进 `String` 与 pending `Vec<u8>`，出错时 drop（作用域释放，OW1、ER4）。

### 10.5 HuggingFace `tokenizer.json` 子集

`fromjson` 只读以下字段；未知字段忽略：

| 路径 | 类型 | 必需 | 处理 |
| --- | --- | --- | --- |
| `model.type` | string | 是 | 必须等于 `"BPE"` |
| `model.vocab` | object | 是 | token 到整数 id；id 必须适合 `u32` |
| `model.merges` | string 数组 | 否 | 默认空 |
| `model.byte_fallback` | boolean | 否 | 默认 `false` |
| `model.unk_token` | string | 否 | 必须在词表中解析 |
| `added_tokens` | object 数组 | 否 | `{id, content, special?}` |

`added_tokens` 条目在双射检查前并入词表条目，因此 `model.vocab` 与 `added_tokens` 之间的重复 id/content 被拒绝。只有 `special == true` 的条目加入 `specialtokens`；缺失 `special` 默认 `false`。

**刻意未实现**（文档化限制，非静默默认缺口）：预分词/切分规则（输入按 Unicode 标量处理）；GPT-2 字节到 unicode 字母表（byte fallback 使用字面 `<0xNN>` 词元形式）；Jinja 对话模板求值（只支持显式 prefix/suffix 形式）；`normalizer`、`pre_tokenizer`、`post_processor`、`decoder` 段。

### 10.6 编码、字节回退与解码

`encode` / `encodeplain` 把文本当作 Unicode 标量序列，应用 BPE 合并表，然后按此顺序解析每个结果符号：① 精确词表匹配；② 若启用 `byte_fallback` 且符号的每个 UTF-8 字节都有 `<0xNN>` 词表条目，则发出那些字节词元；③ 配置的 `unk_token`（若存在）；④ 否则 `unknowntoken`。

`encodespecial` 从左到右扫描；每个位置发出最长的匹配特殊字面量，其间文本用 `encodeplain` 编码。`encode` 永不解释特殊字面量。`decode` 对照词表校验每个 id（失败为 `unknowntokenid`，绝非静默跳过）。`<0xNN>` 词元把一字节追加到 pending 缓冲；其它词元 flush 缓冲为 UTF-8 并原样追加。flush 的字节不是合法 UTF-8 时为 `invalidutf8`。这满足 IN1：词元 id 在使用前被校验。

### 10.7 对话模板

`chattemplate` 持可选 BOS 词元、可选 EOS 词元，以及逐角色 `chattemplateentry { role, prefix, suffix }` 列表。`render` 对每条消息拼接 `prefix + content + suffix`，当消息角色无条目时返回 `unsupportedrole`。`applychattemplate` 发出模板 BOS（若有），然后以特殊感知方式编码渲染字符串；EOS 词元经 `geteos` 暴露给调用方追加，因为是否与何时发出 EOS 是 `infer` 拥有的生成关切，而非 tokenizer 的。

### 10.8 资源限制

- 词表 id 对照已插入条目校验，id 到 token map 是稀疏的（`BTreeMap`），因此恶意 id 不会导致超大分配（IN4）。
- `fromjson` 除调用方提供的 `&str` 外不做额外字节大小上限；限定文档大小是加载方（后续 `format`/`infer` 关切）的责任，此处记录为限制。
- byte-fallback 扩展受输入符号的 UTF-8 长度界定。
- BPE 合并循环为最小堆 + 双向链表（`bytepair::apply`）：每次合并只触及相邻符号，替换原先每轮全表重扫的 O(n²) 实现。基准见 `benches/tokenizer.rs`，基线与加速记录见 `docs/tokenizer-optimization-plan.md` §12。

### 10.9 复杂度（DS3）

每个函数保持低于 rule.md DS3 上界（圈复杂度 ≤ 10、有效行数 ≤ 80）。`fromjson` 分解为 `parsevocab`、`parsemerges`、`parseaddedtokens`；编码分解为 `encodeplain`、`encodespecial`、`flushsegment`、`resolvetoken`；解码分解为 `decodetokens`、`parsebytetoken`、`flushbytes`。*依据：超限函数无法在一轮评审内完成分支验证（rule.md DS3）。*

### 10.10 并发设计（DS4）

`tokenizer` 不引入共享可变状态与同步原语。每个组件都是不可变值；唯一变更是对拥有的 `tokenizer` 在共享前安装对话模板。因此 `tokenizer` 可按引用跨线程共享而无需加锁（OW4），也不存在锁序。crate 不持请求级状态，因此取消与请求隔离不属其关切（OW2）。

### 10.11 命名与模块布局

仓库无分隔符约定使模块与从它重导出的类型在同一名称命名空间冲突（E0255）。模块文件因此是非冲突容器：`vocab.rs` 到 `vocabulary`、`merges.rs` 到 `bytepair`、`special.rs` 到 `specialtokens`、`engine.rs` 到 `tokenizer`、`chat.rs` 到各 chat 类型。这是仓库 rule.md A8 对路线图文件名列表（`lib, tokenizer, vocabulary, encode, decode, bytepair, specialtokens, chattemplate`）在 Rust 上的翻译。

### 10.12 被否决方案（DS5）

- **依赖通用 JSON-schema/反序列化 derive。** 被否决：接受的子集很小，手工解析让每个畸形输入路径都是独立、带类型的错误；`serde_json::Value` 遍历避免为一个文档形状引入 `serde` derive 机制。
- **用稠密 `Vec<String>` 按下标存 id。** 被否决：像 `4294967295` 这样的恶意 id 会强制多吉字节分配（IN4）。`BTreeMap` 稀疏，并对照真实条目校验。
- **实现 GPT-2 字节到 unicode 字母表。** 本次阶段被否决：所需的往返行为可用字面 `<0xNN>` 字节词元表达，字母表会加入大静态表而无当前消费者。
- **求值 HuggingFace Jinja 模板。** 被否决：需要模板引擎依赖与不可信代码求值，远超运行期所需的窄、可测试的 prefix/suffix 组合。
- **让 `applychattemplate` 也追加 EOS。** 被否决：决定序列何时终结属于生成（`infer`）；tokenizer 暴露 EOS 而非静默发出。
- **把分词放进 `data` 或 `infer`。** 被否决：架构契约把文本到词元转换归 `tokenizer`，且 `data` 刻意避免对它的反向依赖。

### 10.13 后果

- `tokenizer` 是 `lib` 之上的依赖叶；加入它不改变任何既有 crate 清单，因为根成员是 glob 式（`members = ["crates/*"]`）。
- 对词表能表示的输入，编码/解码往返确定且字节精确。
- 畸形与不支持的文档以独立、带类型的错误失败，而非 panic 或不透明消息。

### 10.14 验证

命令限于 `tokenizer`，因为同批 crate 正在并发构建。2026-10-10 从 `/Users/fei/Doubao/llm` 执行：

```sh
cargo fmt --all
cargo check -p tokenizer --all-targets
cargo clippy -p tokenizer --all-targets -- -D warnings
cargo test -p tokenizer
RUSTDOCFLAGS='-D warnings' cargo doc -p tokenizer --no-deps
```

结果：`cargo clippy -p tokenizer --all-targets -- -D warnings` 通过；仓库根 `tests/tokenizerunits.rs` 与 `tests/tokens.rs`（对应 `tokenizer`）—— 0 失败；`RUSTDOCFLAGS='-D warnings' cargo doc -p tokenizer --no-deps` 通过。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）

覆盖：`vocabulary`（双向、计数/边界、重复 id、重复内容、重复相同对、空）；`bytepair`（解析/rank、缺分隔符、空侧、最低 rank 优先、重复合并到单符号、不可合并序列、最左平局）；`specialtokens`（双向、contains/count、按内容顺序迭代、空）；`encode`（空输入、先合并后查找、byte fallback、无 fallback 的 unknown、带 `unk` 的 unknown、特殊切分、plain 忽略特殊）；`decode`（普通词元、UTF-8 重建、空、未知 id、不完整 UTF-8、字节词元形状解析器）；`chat`（角色名、渲染包裹、未覆盖角色、空对话）；`engine`（加载/暴露、非法 JSON、非对象根、缺 type、非 BPE 模型、非整数 id、越界 id、未知 `unk_token`、往返、特殊感知编码、缺模板）；`tests/tokens.rs`（ASCII 与 UTF-8 往返、`unk` fallback、结构特殊词元、BOS 前缀对话渲染、缺模板、词表外解码，以及一组必须不 panic 的畸形文档）。

### 10.15 复审条件

当某模型要求预分词（例如 GPT-2 正则切分）或字节到 unicode 字母表时，扩展 `encodeplain` 并对照上游 tokenizer 加入参考测试；当调用方需要在本 crate 内强制 JSON `config.json`/`tokenizer.json` 大小上限时，在 `fromjson` 边界加入显式 `lib::limits` 上界；当运行期加入第二种对话格式时，泛化 `chattemplate` 而非求值 Jinja，重新审议本章。

## 11. 层 2：data

### 11.1 上下文

路线图阶段 P8 引入 `data`。`data` 拥有：数据集接口、记录与文本数据集表示、打包与批处理、采样、数据加载器生命期与错误传播。它 MUST NOT 拥有模型前向执行、优化器更新、推理请求状态或检查点文件格式。其唯一工作区依赖是 `lib`（`tokenid`、`seed`、`limits`）。它刻意**不**依赖 `tokenizer`：把原始文本转成词元标识是更高层拥有的组合，因此本 crate 接受已分词的文档。这保持依赖方向单向，避免低层反向依赖。

crate 遵循仓库无分隔符命名约定（NM1）、工作区无 `unsafe` 与无 `panic`/`unwrap`/`expect` 规则，并拥有自己的 `thiserror` 错误 enum。

### 11.2 决策

crate 暴露 9 个公共项，根于 crate root 并从无冲突模块文件重导出：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `record` | `records.rs` | 拥有有序 `Vec<tokenid>` |
| `dataset`（trait） | `datasets.rs` | `getlen`、`isempty`、`getrecord` |
| `textdataset` | `textdatasets.rs` | 内存内具体 `dataset` |
| `packing` | `packings.rs` | 定长序列打包 |
| `batch` | `batches.rs` | 矩形词元批 |
| `sampler` | `samplers.rs` | 可复现索引采样器 |
| `loader` | `loaders.rs` | 生命期显式的 epoch 迭代器 |
| `loaderconfig` | `loaders.rs` | 纯数据加载器配置 |
| `dataerror` | `error.rs` | crate 自有错误 enum |

### 11.3 接口边界（DS1）

- `data` 只依赖 `lib`。不导入 `tokenizer`、`tensor`、`api`、`cpu` 或任何工作流 crate。
- `dataset` 是返回借用记录的窄能力 trait；它不持迭代状态，因此数据集可被不可变共享，而 `loader` 私下持请求级游标（AGENTS.md §3.3 与 §5）。
- 公共接口宽度：crate root 9 个项，远低于 A4 上界 20。*依据：单模块公共契约应能在一轮评审内完整复核（rule.md A4）。*

关键接口：`record::new`、`gettokens`、`getlen`、`isempty`。`dataset`：`getlen`、`isempty`（默认 `getlen()==0`）、`getrecord(index) -> Result<&record, dataerror>`。`textdataset::new(records, lim)`、`fromtokens(documents, lim)`、`gettotaltokens`。`packing::new(seqlength)`、`getseqlength`、`pack(records, lim) -> Result<Vec<record>, dataerror>`。`batch::new(tokens, batchsize, seqlength, lim)`、`fromrecords(records, seqlength, pad, lim)`、`getbatchsize`、`getseqlength`、`gettokens`、`getrow(row)`。`sampler::new(seed)`、`shuffle(len, lim)`、`sample(len, count, lim)`。`loaderconfig::new(batchsize, seqlength, pad, shuffle, droplast)` 及 getter。`loader::new(data, config, lim, seed)`、`getepoch`、`reset`、`getnext() -> Result<Option<batch>, dataerror>`。

### 11.4 不变量与错误路径（DS2）

算法级不变量：

- 一个 `batch` 恰好持 `batchsize * seqlength` 个词元；每行同长。`batch` 至少有一行。
- `packing` 产生的每个打包序列长度为 `seqlength`，除最后一段部分序列长度在 `1..seqlength`；空输入产生空序列列表。
- `sampler` 的 shuffle 是 `0..len` 的一个排列。
- 构造后，`loader` 的记录顺序恰含下标 `0..data.getlen()`，且 `position`/`offset` 总定位到合法位置或该顺序末尾。

实现级不变量：

- 对每个构造的 `batch`，`gettokens().len() == batchsize * seqlength`；`getrow` 用受检乘法/加法与 slice `get`，而非未检查索引。
- `packing::pack` 最多分配经校验的总词元数。
- `loader::fillsequence` 只在完整消费一条记录后才推进 `position`/`offset`；`remaining` 用 `get(offset..)` 取得，使被违反的不变量成为带类型错误而非 panic。

错误分支与返回语义（`dataerror`，13 个 variant）：

- `indexoutofrange { index, len }` —— `dataset::getrecord`、`batch::getrow` 在 `index >= len`。
- `emptydataset` —— 空范围上以正 `count` 调用 `sampler::sample`。
- `zeroseqlength` / `zerobatchsize` —— `packing`、`batch`、`loader` 中的零维。
- `emptybatch` —— `batch::fromrecords` 零记录。
- `recordeceedseqlength { index, len, seqlength }` —— 记录长于批 `seqlength`。
- `lengthmismatch { expected, actual }` —— 词元缓冲长度与期望元素计数不符。
- `elementcountoverflow` —— 受检乘法/加法溢出 `usize`。
- `elementcountlimitexceeded { requested, limit }` / `sequencelenlimitexceeded { requested, limit }` / `batchlimitexceeded { requested, limit }` —— 请求值严格超过对应 `lib::limits` 上界。

资源回滚：所有构造都是值式的。出错时不返回部分构建值，所有拥有缓冲被 drop（作用域释放，OW1、ER4）。`batch::fromrecords` 在 reserve 前校验维度与元素预算，并以 `Vec::try_reserve_exact` 失败式 reserve，因此被拒输入不分配任何东西。

### 11.5 复杂度（DS3）

每个函数保持低于 rule.md DS3 上界（圈复杂度 ≤ 10、有效行数 ≤ 80）。加载器分解为 `getnext`、`fillsequence`、`finish` 以保持各自小。*依据：超限函数无法在一轮评审内完成分支验证（rule.md DS3）。*

### 11.6 并发设计（DS4）

`data` 不引入共享可变状态与同步原语。`loader` 不可变借用其数据集，私下拥有全部游标、采样器与缓冲状态，因此两个 loader（或一个 loader 与数据集的任何读者）不会竞争。这以类型而非运行期约定满足请求隔离（OW2/CC2）。无需锁序，因为不存在锁。单所有者加载器无需同步；多 worker 加载需求在复审条件中列出。

### 11.7 采样可复现性

`sampler` 拥有由 `lib::seed` 播种的 SplitMix64 生成器。算法固定：

```text
state = seed
nextu64():
    state += 0x9E3779B97F4A7C15   (wrapping)
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9   (wrapping)
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB   (wrapping)
    return z ^ (z >> 31)
```

`shuffle(len)` 从下标 `len-1` 向下到 `1` 就地做 Fisher–Yates，每一步与 `bound(i + 1)` 交换。`sample(len, count)` 经 `bound(len)` 有放回抽 `count` 个下标。`bound(upper)` 用拒绝采样 —— 拒绝低于 `(u64::MAX - upper + 1) % upper` 的抽取，否则返回 `draw % upper` —— 因此下标分布无偏。由于每一步都确定性且平台无关，同一 seed 的两个采样器产生相同序列，且连续的变更调用推进流。

加载器用采样器产生每 epoch 记录顺序。构造从 epoch 1 开始；每次 `reset` 推进流并重建新排列。因此 epoch 序列从一个 seed 可复现，而不同 epoch 彼此不同。

### 11.8 批形状与边界行为

- `batch::new(tokens, batchsize, seqlength, lim)` 要求 `tokens.len() == batchsize * seqlength` 精确成立；其它长度为 `lengthmismatch`。先校验维度，因此零维或超批/序列预算在检查长度前报告。
- `batch::fromrecords(records, seqlength, pad, lim)` 是填充构造器。每条记录被复制并以调用方提供的 `pad` 填充到 `seqlength`。长于 `seqlength` 的记录以 `recordeceedseqlength` 拒绝而非静默截断，因此错配的 `seqlength` 响亮失败。空记录集以 `emptybatch` 拒绝。
- 行按行主序排列；`getrow(r)` 返回 `[r * seqlength, (r + 1) * seqlength)`，`r >= batchsize` 被拒。
- 填充词元由调用方提供；crate 从不发明默认词元（AGENTS.md §5；IN4）。

打包边界行为（`packing::pack`）：空输入 → 空序列列表；词元计数恰为 `seqlength` 整数倍 → 只有完整序列；否则最后一段较短且不填充，需要矩形的调用方经 `batch` 填充；长于 `seqlength` 的记录被切进多个序列，不同记录的词元可共享一个序列。

### 11.9 加载器失败、终止与生命期

`loader::getnext` 返回：

- `Ok(Some(batch))`：epoch 仍在产出批。
- `Ok(None)`：恰在 epoch 耗尽时，及此后每次调用直到 `reset`。
- `Err(dataerror)`：记录访问或批构造失败。

epoch 耗尽且存在尾随部分批时，`droplast` 选择行为：`true` 丢弃它（返回 `Ok(None)`），`false` 用 pad 词元把剩余行填成完整矩形并返回。无论哪种，加载器变为终态并持续返回 `Ok(None)`。

返回错误也让加载器变为终态，因此失败的加载器永不被静默复用；调用方必须 `reset` 或重建。这使失败路径复位请求级状态（OW3）。所有加载器缓冲被拥有并在加载器 drop 时释放；无手工释放动词（NM3）。

### 11.10 资源限制

所有受外部影响的尺寸都对照显式 `lib::limits` 预算校验：

| 操作 | 强制上界 |
| --- | --- |
| `textdataset::new` | 记录数与总词元数 ≤ `maxelements` |
| `packing::pack` | 总词元数 ≤ `maxelements` |
| `batch::new` / `fromrecords` | `batchsize` ≤ `maxbatch`；`seqlength` ≤ `maxsequencelen`；`batchsize * seqlength` ≤ `maxelements` |
| `sampler::shuffle` / `sample` | `len` / `count` ≤ `maxelements` |
| `loader::new` / `reset` | 其配置与记录顺序适用上述 |

任何元素计数乘积的溢出都以受检算术检测并报告为 `elementcountoverflow`。加载器每批缓冲容量为 `batchsize * seqlength`，已被 `maxelements` 界定。限制总由调用方提供；`data` 不发明默认上界。

### 11.11 命名与模块布局

仓库无分隔符约定使模块与从它重导出的类型在同一命名空间冲突（E0255）。模块文件因此是复数容器（`records.rs` → `record`、`datasets.rs` → `dataset`、`textdatasets.rs` → `textdataset`、`packings.rs` → `packing`、`batches.rs` → `batch`、`samplers.rs` → `sampler`、`loaders.rs` → `loader`），与既有 `lib` 和 `tensor` crate 一致。这是 rule.md A8 对路线图文件名列表在 Rust 上的翻译。

### 11.12 被否决方案（DS5）

- **依赖 `tokenizer` 并在 `data` 内分词文本。** 被否决：会让低层数据 crate 依赖高层文本 crate，产生反向边与耦合。边界要求 tokenizer 组合位于 `data` 之上。架构契约与 P8 计划都把 tokenizer 组合归更高层。
- **让 `textdataset` 成为惰性分词数据集。** 被否决：`dataset` 契约返回借用记录，惰性分词实现无法返回对刚构建记录的引用。急切物化加调用方上游提供的 tokenizer 保持 trait object-safe 且简单。
- **打包时在记录间加显式 EOS/BOS 分隔符。** 本次阶段被否决：它发明属于 tokenizer/模型配置的词元语义。打包纯定义为拼接；调用方可在打包前插入分隔符。
- **让加载器持数据集的拥有 `Vec<record>` 拷贝。** 被否决：会复制可能很大的数据并破坏不可变共享模型。加载器改为借用数据集。
- **经 `dyn` 运行期选择采样器。** 被否决：不存在运行期选择需求（A3）；单一具体采样器保持静态派发。
- **引入共享的、独立于 `data` 的 PRNG crate。** 被否决：所需算法只有几行，抽出 crate 会无独立职责地扩大依赖闭包（A5）。

### 11.13 后果

- `data` 是 `lib` 之上的依赖叶；加入它不改变任何既有 crate 清单，因为根成员是 glob 式。
- 采样与加载器 epoch 从 seed 可复现，使训练与评估运行可复现。
- crate 无环且只依赖 `lib`。

### 11.14 验证

2026-10-10 在真实工作区对 `crates/data` 执行（macOS，edition 2024，rust-version 1.99）：

```sh
cargo fmt --all -- --check                                  # exit 0
cargo check -p data --all-targets                           # exit 0
cargo clippy -p data --all-targets -- -D warnings           # exit 0, no warnings
cargo test -p data                                          # 迁移前计数已省略；当前套件位于仓库根 tests/
RUSTDOCFLAGS='-D warnings' cargo doc -p data --no-deps      # exit 0
```

仓库根 `tests/dataunits.rs`（对应 `data`）—— 0 失败。3 个端到端集成测试在后续合并中移到仓库根 `tests/pipeline.rs`。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）

当同批的 `tokenizer`/`format`/`checkpoint`/`context` crate 并发创建时，工作区根清单曾短暂无法解析（glob 成员 `crates/*`）。同样的命令也因此在仅含 `lib` 与 `data` 的隔离工作区中运行，并镜像工作区 lint 与包配置；结果相同（clippy exit 0，测试通过，doc exit 0）。隔离运行是补充而非替代；上述最终结果来自真实工作区。

覆盖：`record`（排序、空记录）；`dataset`/`textdataset`（索引访问、越界、空、记录数与词元数的边界及刚好超过边界）；`packing`（空输入、精确倍数、部分尾、跨记录拼接、长记录切分、空记录、词元预算）；`batch`（矩形构造、长度不匹配、零维、填充、过长记录、空输入、行访问、批/序列/元素预算）；`sampler`（排列性质、同 seed 可复现、流推进、短输入、抽取范围与计数、空数据集、元素预算）；`loader`（跨批填充、`droplast`、跨记录打包、空数据集、恒等与打乱顺序、epoch 可复现、epoch 复位、终态 `None`、配置校验、泛型 `dataset` 约束）；`dataerror`（可区分值与 display）；`tests/pipeline.rs`（经公共 API 端到端组合 dataset、packing、batch、sampler 与 loader）。

### 11.15 复审条件

当更高层需要无法在内存中物化的惰性或流式数据集时，重新审议借用记录的 `dataset` 契约；当需要记录级元数据（文档 id、标签、注意力掩码）时，扩展 `record` 而非重载词元缓冲；当需要多 worker 加载时，重新审议加载器所有权（当前设计为单所有者、无需同步），重新审议本章。

## 12. 层 2：format 与 checkpoint

### 12.1 上下文

运行期需要读取外部模型构件并持久化模型与训练状态。涉及两个职责，且它们不同：

- **解析不可信外部字节**为已校验元数据 —— 一个 `config.json` 与一个 safetensors 容器。这是对不可信输入的纯只读变换，不拥有状态。
- **持久化并重载**我们自己的检查点构件，包括 manifest、模型张量与训练状态，带原子替换与对照期望模型的兼容性检查。

rule.md IN1/IN2/IN3 要求所有外部字节都被当作不可信：每个长度、偏移、维度与元素计数在任何由它派生的分配或访问之前都必须以受检算术校验，截断或畸形输入必须产生带类型错误而非 panic。A2 要求无环依赖图；A5 让这些 crate 留在内核之外。OW1/ER4 要求资源获取与释放作用域绑定，且失败时按逆序释放资源。

架构契约把两个职责分给两个 crate，允许 `checkpoint` 依赖 `format`，反之不可。

### 12.2 决策

创建两个 crate，`format` 与 `checkpoint`，带单向依赖 `checkpoint -> format`。`format` 只依赖 `lib`（资源限制）与 `tensor`（dtype/shape 分类法）；`checkpoint` 新增 `format`。不引入反向边与环。

```text
format     -> lib, tensor
checkpoint -> tensor, format        (checkpoint 不直接导入 lib)
```

### 12.3 接口边界（DS1）

**`format` 拥有**外部构件的解析与结构校验。它 MUST NOT 执行模型、管理请求状态、分配设备缓冲或依赖任何应用 crate。

| 模块（文件） | 导出接口 | 宽度 |
| --- | --- | --- |
| `configs.rs` | `modelconfig`（含 `parse`，14 个 getter）；Llama 族 `config.json` | 1 类型 |
| `metadata.rs` | `tensormetadata`、`dtypetoken`、`tokendtype` | 3 |
| `safetensor.rs` | `safetensorfile`、`HEADERLENGTHSIZE` | 2 |
| `validation.rs` | `checkoverlaps` | 1 |
| `error.rs` | `formaterror`（16 variant） | 1 类型 |

**`checkpoint` 拥有**检查点构件：其 `manifest`、带 load/save 的 `checkpoint` 容器、尽力原子持久化与兼容性校验。它 MUST NOT 拥有训练循环、优化器更新规则或推理调度器。

| 模块（文件） | 导出接口 | 宽度 |
| --- | --- | --- |
| `manifests.rs` | `manifest`、`manifestentry`、`checkpoint` | 3 |
| `training.rs` | `trainingstate` | 1 |
| `atomicwrite.rs` | `writeatomic` | 1 |
| `compatibility.rs` | `tensorspec`、`checkcompat`、`checkversion` | 3 |
| `save.rs` / `load.rs` | `savecheckpoint` / `loadcheckpoint` | 各 1 |
| `error.rs` | `checkpointerror`（15 variant）+ `isrecoverable`/`isincompatibility` | 1 类型 |
| `lib.rs` | `CHECKPOINTVERSION` | 1 常量 |

接口宽度是每个模块导出的不同顶层符号数；固有 `pub fn` 访问器属于其拥有类型。每个模块都远低于 A4 上界 20，因此每个模块的契约可在一轮评审内复核。

**模块文件名刻意不同于路线图。** 路线图按公共类型命名模块（`modelconfig.rs`、`tensormetadata.rs`、`safetensors.rs`、`manifest.rs`、`trainingstate.rs`）。按字面采用在无分隔符约定下不可能：模块与它重导出的同名小写类型共享一个命名空间，`mod manifest` 加 `pub use manifest::manifest;` 会以 `E0255` 失败。这与 rule.md A8 记录的翻译相同。crate 使用无冲突文件名（`configs`、`metadata`、`safetensor`、`manifests`、`training`）并从 `lib.rs` 重导出预期公共类型名。

### 12.4 《检查点格式契约》

本节是检查点格式的完整规范。

**构件布局。**

- safetensors 容器：8 字节小端 header 长度、一个 JSON header（每张量 `{dtype, shape, data_offsets}` 条目）、连续的数据段。`HEADERLENGTHSIZE = 8`。
- 检查点（我们自己的格式，版本 `CHECKPOINTVERSION = 1`）：8 字节 magic `LLMCKPT1`、一个小端 `u64` manifest 长度、UTF-8 JSON manifest、数据段。前缀总长 `PREFIXLEN = 16`（magic + manifest 长度）。两种布局都自描述。

**safetensors manifest schema（每条目）。**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `dtype` | string | 存储 dtype token（经 `dtypetoken`/`tokendtype` 映射） |
| `shape` | 整数数组 | 逻辑维度，非空 |
| `data_offsets` | `[u64; 2]` | `[begin, end)` 半开字节区间 |

`__metadata__` 是保留 key，承载 string 到 string 的元数据映射；其它 key 是张量名。

**检查点 manifest schema。** manifest 是构件的唯一真值来源，命名版本、可选训练状态、可选字符串元数据 map，以及每个张量及其 dtype、shape 与字节区间。其序列化形式为：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `version` | `u32` | 检查点版本 |
| `training` | 可选 `trainingstate` | 训练进度快照 |
| `metadata` | `BTreeMap<String,String>` | 字符串元数据，空则省略 |
| `tensors` | `manifestentry` 数组 | 按插入顺序，亦为磁盘数据布局顺序 |

`manifestentry` 的序列化字段：`name: String`、`dtype: String`（dtype token）、`shape: Vec<usize>`、`offset: u64`、`nbytes: u64`；`getend()` 返回 `offset + nbytes`（由构造不变量保证不溢出）。`trainingstate { step: u64, epoch: u64 }` 记录已完成的优化器步与 epoch；优化器参数状态不在此建模，它属于优化器 crate，并作为普通张量经同一检查点容器持久化。

**dtype token 映射。** `dtypetoken` 对 `dtype` 全域：`F32`、`F64`、`F16`、`BF16`、`I8`、`U8`、`I32`、`I64`。`tokendtype` 只接受这八个 token，显式拒绝布尔、`i16` 与 `u16`/`u32`/`u64`，而非静默加宽或截断。`f16`/`bf16` 作为存储 dtype 被接受以描述其元数据，但本层不声称任何后端能执行它们。

**不变量（算法级）。**

- 每个张量的字节区间 `[begin, end)` 包含在数据段内（`end <= datalength`），且 `end - begin` 等于 `element_count * dtype_width`。
- 没有两个张量字节区间重叠；每个名字唯一。
- `element_count = product(shape)` 以受检乘法计算，必须适合 `usize` 且在配置的元素预算内。
- safetensors header 是合法 UTF-8 JSON，每个声明的 dtype 都在受支持存储分类法内；不支持的 dtype 被显式拒绝。
- 检查点 manifest 是合法 UTF-8 JSON，`version` 与 `CHECKPOINTVERSION` 匹配（由 `checkversion` 判定）。

**不变量（实现级）。** 所有长度/偏移/维度算术使用 `checked_*`，结果在切片前重新校验；不使用 `as` 截断把已校验值加宽为索引。`checkpointerror` 使三类保持区分：可恢复 I/O、结构不兼容、畸形构件。失败的保存绝不发布部分构件：写入同目录 staging 文件，仅在全内容 flush 后 rename 到目标。

**完整性校验。** 解码检查 magic、前缀长度与声明的 manifest 长度（受检算术）；校验 manifest；对每个张量的字节区间检查其落在数据段内且不重叠（`checkoverlaps` 按起始偏移排序并跟踪最大 end，因而嵌套区间也能检出；零长度张量被忽略）。数据段长度由 manifest 之后的字节数确定。保存侧 `checkpoint::addtensor` 校验字节长度等于元素计数乘 dtype 宽度，偏移为数据段当前末尾，因此区间按构造连续且不重叠。

**原子写（`writeatomic`）。** 字节写入同目录下唯一 staging 文件 `.{file}.tmp.{pid}.{attempt}`，flush 并 sync，然后 rename 到目标。读者因此观察到目标的旧内容或完整新内容，绝不观察到部分写。任一步失败则（尽力）移除 staging 并返回错误；既有目标不变。staging 名字尝试上限 `MAXTEMPATTEMPTS = 16`（依据：16 次尝试让偶然碰撞极不可能，同时避免无界循环）。保证与限制：同目录 rename 在工作区目标文件系统上原子（POSIX `rename`）；rename 自身的持久性需要目录 `fsync`，它跨受支持平台不可移植且不执行，因此保证是原子替换，而非目录项的崩溃持久性。

**兼容性。** `checkcompat(ckpt, expected: &[tensorspec])` 校验检查点含每个期望张量且 dtype 与 shape 匹配。检查点中的额外张量被允许；检查是检查点内容的下界。这是纯结构比较，不读任何张量字节、不做 I/O，因此永不返回可恢复 I/O 错误。`checkversion(ckpt, supported)` 校验 manifest 版本等于受支持版本。`tensorspec` 携带 `name`、`dtype`、`shape`，不带字节区间。三类失败 `missingtensor`、`dtypemismatch`、`shapemismatch` 与 `unsupportedversion` 都被 `checkpointerror::isincompatibility` 归类为结构不兼容。

**错误路径。**

`formaterror`（16 个 variant）：`truncated`、`headertoolarge`、`invalidjson`、`invalidentry`、`unsupporteddtype`、`emptyshape`、`shapeoverflow`、`elementlimitexceeded`、`offsetsoutofrange`、`bytelengthmismatch`、`overlappingtensors`、`invalidmetadata`、`missingfield`、`invalidvalue`、`unsupportedmodeltype`、`unsupportedtorchdtype`。

`checkpointerror`（17 个 variant）：`io { path, source }`、`badmagic`、`unsupportedversion { found, supported }`、`truncated { needed, available }`、`invalidmanifestjson { line, column, message }`、`invalidmanifest { reason }`、`shapeoverflow { name }`、`lengthmismatch { name, expected, actual }`、`dtypeunsupported { name, dtype }`、`duplicatetensor { name }`、`offsetoutofrange { offset, nbytes, datalength }`、`overlappingtensors`、`missingtensor { name }`、`dtypemismatch { name, expected, actual }`、`shapemismatch { name, expected, actual }`、`tensorcrcmismatch { name }`，以及 `format(#[from] formaterror)`、`tensor(#[from] tensor::tensorerror)`。`format` 与 `tensor` 错误以保留的上下文传播，而非被压平。

`checkpointerror::isrecoverable()` 只对 `io` 为真；`isincompatibility()` 只对 `unsupportedversion`、`missingtensor`、`dtypemismatch`、`shapemismatch` 为真；其余为畸形构件。

**资源回滚规则。** 所有缓冲是拥有的 `Vec`/`String`，由作用域释放；staging 文件在任何失败路径上于返回前移除。不暴露手工释放调用，符合 OW1/NM3。

**上传/加载工作流。** `savecheckpoint(path, ckpt)` 先 `ckpt.encode()` 再经 `writeatomic` 发布；`loadcheckpoint(path)` 读文件为内存再经 `checkpoint::decode` 校验，可恢复文件系统失败报为 `io`，畸形或结构非法构件报为解码错误。

**配置解析（`modelconfig`）。** `modelconfig::parse(bytes)` 解析并校验 `config.json`：未知字段忽略；`model_type` 必须是 `"llama"`；每个尺寸非零；`hidden_size` 可被 `num_attention_heads` 整除；`num_key_value_heads`（默认为 `num_attention_heads`）可整除前者；`rms_norm_eps` 与 `rope_theta` 有限且为正；可选 `torch_dtype` 映射到存储 `dtype`（`float32`/`f32`、`float16`/`f16`/`half`、`bfloat16`/`bf16`、`float64`/`f64`/`double`）否则拒绝。无未检查的公共构造器：值只由 `parse` 产生。派生 getter `getheadsize`（`hidden_size/num_attention_heads`）与 `getkeyvaluegroups`（`num_attention_heads/num_key_value_heads`）。

**推理 raw 权重格式（`rawsnapshot`/`rawreader`）。** 推理侧独立格式，与 `.u` 训练快照并存：`.u` 优先空间与增量（内容寻址、LZ4、delta），raw 优先加载速度与可寻址性（不压缩、页对齐、区段 CRC32C）。构件布局：8 字节 magic `LLMRAW01`、`u32` 版本（`RAWFORMATVERSION = 1`）、`u32` 张量数、`u64` 数据段偏移（页对齐，`PAGESIZE = 4096`）、二进制目录（每张量：`u32` 名字长 + UTF-8 名、`u8` 存储 dtype token、`u32` 秩 + `u64` 各维度、`u64` 区段偏移、`u64` 字节长、`u32` CRC32C）、页对齐 padding、数据段（区段按目录顺序、每区段起始页对齐）。`rawsnapshot::fromcheckpoint`（内存检查点）与 `rawsnapshot::fromsnapshot`（训练 `.u` 快照，推理导出的主源）导出，重新校验每张量字节长等于元素计数乘 dtype 宽度；`rawexport` 读 `.u` 文件 → `snapshot::decode` → `fromsnapshot` → `saveraw` 原子发布；`encode`/`decode` 全程受检算术；`decode` 校验结构、区段范围、页对齐与重叠，不验 CRC（留给 reader 按需）。`rawreader` 复用 `blocksource`：`open`（内存缓冲）与 `fromfile`（文件按需读区段）在读时验 CRC32C（失败报 `tensorcrcmismatch`，归类损坏）；`readall` 经 `blocksource` 一次性顺序读整个数据段（零 unsafe 下的顺序预读语义）再逐区段验 CRC；`readtensortyped<T>` 先校验存储 dtype 等于 `T::DTYPE`（不符报 `dtypemismatch`，归类结构不兼容），再把目录形状与字节交给 `tensor::tryfrombytes` 解码。raw 格式不携带量化描述符、训练状态与 delta/base 关系；scale/zero-point 张量按普通张量导出，关联由加载器按命名约定推导。

**限制（已声明）。** 检查点为单文件、内存内、未加密；`f16`/`bf16` 作为存储 dtype 接受但不声称后端执行；兼容性对照显式 `tensorspec` 列表而非检查模型对象。

### 12.5 不变量与错误路径（DS2）

本节的不变量与错误路径集中在 12.4 的《检查点格式契约》，此处补充跨切面条目。

错误分支枚举（整体）：截断、header 过大、非法 JSON、非法条目、不支持 dtype、空 shape、shape 溢出、超元素预算、偏移越界、字节长度不符、张量重叠、非法 `__metadata__`、缺字段、非法值、不支持模型类型、不支持 torch dtype；以及检查点侧的坏 magic、不支持版本、非法 manifest JSON、非法 manifest、张量重名、张量区间越界、区间重叠、缺张量、dtype 不符、shape 不符、区段 CRC 损坏（`tensorcrcmismatch`）、`tensor` 层校验错误传播（`tensor(#[from])`）。

实现级不变量：`checkpointerror` 的 `isrecoverable`/`isincompatibility` 语义见 12.4；`safetensorfile::readtensor` 从解析 header 长度与张量偏移以受检算术重算绝对字节区间，再对照传入缓冲校验后复制（copy-validate-use），因此字节不因早先解析过而被信任。

资源回滚：见 12.4「资源回滚规则」。

### 12.6 复杂度（DS3）

`format` 与 `checkpoint` 的每个函数低于圈复杂度 10 与有效行数 80；最大者为 `safetensorfile::parse`、`checkpoint::decode` 与 `manifest::frombytes`，都是校验后跟构造的直线体。（以阅读核对，证据等级 T。）

### 12.7 并发设计（DS4）

`format` 的解析是纯只读变换，不拥有状态、不做 I/O、不引入共享可变状态或同步原语。`checkpoint` 的 `manifest`/`checkpoint`/`tensorspec`/`trainingstate` 都是拥有数据的值类型；`writeatomic` 通过唯一 staging 名（进程 id 加有界尝试计数）允许并发进程不互相覆盖。两 crate 都无锁序、无全局可变状态（A6）。`checkpoint` 本身不在值之间共享可变状态，请求级隔离不适用。

### 12.8 被否决方案（DS5）

- **一个合并的持久化 crate。** 被否决：架构契约把外部格式解析与检查点所有权分开；单一 crate 会让解析代码获得一条指向它并不需要的训练状态与原子写关切的依赖边（A1、A5）。
- **按字面采用路线图的模块文件名。** 被否决：在无分隔符约定下不可行（`E0255`），见 12.3。
- **在计划指向根 `tests/checkpoint_roundtrip.rs` 时保留 crate 本地集成测试。** 该任务的工作区机制要求在并发多 crate 构建期间测试位于拥有 crate 内 `crates/<name>/tests/*.rs`；根 `tests/` 由工作区布局工作拥有，超出本次变更范围。后续已被取代：合并 pass 把套件移到根 `tests/roundtrip.rs`。
- **通过就地截断目标来写检查点。** 被否决：被中断的写会留下截断文件，可能被误认为合法检查点；staging 文件加 rename 保证目标是旧构件或完整新构件。
- **把所有失败归为单一 `Error` enum 而无访问器。** 被否决：调用方必须能区分可重试 I/O 失败与真正的模型不兼容，而无需匹配 variant 名；`isrecoverable`/`isincompatibility` 访问器编码该契约。
- **在热解析路径用 `unsafe` 或未校验切片。** 被否决：工作区禁止 `unsafe`（OW5），不可信输入正确性（IN1/IN2）重于任何微优化。

### 12.9 后果

- `format` 与 `checkpoint` 是 `lib`/`tensor` 之上的依赖安全叶；不引入反向边与环。
- 两 crate 都是 `#![forbid(unsafe_code)]`；所有运行期路径使用 `Result`，生产代码不出现 `panic`/`unwrap`/`expect`/`todo`/`unimplemented`（由工作区 `clippy` lint 强制，并在此复核）。
- 经 `writeatomic`/`savecheckpoint` 写的检查点永不被观察到部分写状态，失败的保存保持先前检查点不变。
- 已报告的限制：检查点单文件、内存内、未加密；`f16`/`bf16` 作为存储 dtype 接受但不声称后端执行；兼容性对照显式 `tensorspec` 列表而非检查模型对象。

### 12.10 验证

2026-10-10 从 `/Users/fei/Doubao/llm` 执行：

- `cargo fmt --all` —— 干净（exit 0）。
- `cargo check -p format --all-targets` —— 通过。
- `cargo clippy -p format --all-targets -- -D warnings` —— 通过。
- 仓库根 `tests/formatunits.rs` 与 `tests/artifact.rs`（对应 `format`）—— 0 失败。集成套件 `tests/artifact.rs` 覆盖合法解析、截断 header、不支持 dtype、重叠区间、畸形 `config.json`。
- `cargo check -p checkpoint --all-targets` —— 通过。
- `cargo clippy -p checkpoint --all-targets -- -D warnings` —— 通过。
- 仓库根 `tests/checkpointunits.rs` 与 `tests/roundtrip.rs`（对应 `checkpoint`）—— 0 失败。`tests/roundtrip.rs` 覆盖 save/load 往返、截断输入、损坏 magic、缺文件报为可恢复 I/O、兼容性匹配/不匹配、无部分构件的原子替换，以及失败的保存保持先前检查点不变。
（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
- `RUSTDOCFLAGS='-D warnings' cargo doc -p format --no-deps` —— 通过。
- `RUSTDOCFLAGS='-D warnings' cargo doc -p checkpoint --no-deps` —— 通过。

### 12.11 复审条件

当检查点构件获得多文件分片、内存映射加载或二进制 manifest 时；当加入第二种外部格式（例如 GGUF）且共享校验应下移到更低 crate 时；当检查点必须对照活的模型对象而非显式 `tensorspec` 列表验证时，重新审议本章。

`.u` 增量检查点契约（§12.12）已实现；其固定字段布局、量化/稀疏/多排布与安全增强落地时，本章整体复审。

### 12.12 《.u 增量检查点契约》

> 状态：**已验证（Verified）**。`.u` 格式的合并/恢复逻辑与哈希/校验体系已在 `checkpoint` crate 实现并有测试覆盖：内容寻址分块（CDC）、增量创建/合并/按需读取、逐块压缩（LZ4/Zstd + bit-shuffle 变换）、链压缩与溯源（`compact`/`verifyprovenance`）、`checkpointerror` 的损坏分类（`iscorruption()`），以及安全增强层（Ed25519 签名、AES-256-GCM 加密）与存储增强（多排布 layout、稀疏 COO、对称 int8 量化、多文件分片、mmap 零拷贝）。本节描述的固定 64 字节条目字段布局为设计草案，实际 manifest 采用变长二进制序列化（以 §12.4 与源码为准）。

**上下文。** 增量检查点通过「仅存相对基础快照变化的块」降低训练存储冗余，通过「块级随机访问」支撑按需加载与显存调度。实现方式是**内容寻址 + 梅克尔 DAG**：块身份 = 内容哈希，快照身份 = 根哈希，溯源链由 `base_id` 显式记录。

**内容寻址数据模型。**

- 块边界由**内容定义分块（CDC）**确定：Gear 滚动哈希（64 位）+ 18 位切分掩码，目标平均块长 256 KiB，`MINBLOCK = 64 KiB`、`MAXBLOCK = 1 MiB`（末块可短）。插入/删除只扰动邻近几个块边界，未变块内容保持稳定从而去重。数据大于 16 MiB（`SEGMENT`）时按段并行分块，段边界强制切分，去重仅在段边界附近的少量块损失跨段身份（有界、可忽略）。
- 逻辑数据段 = manifest 张量按插入序拼接的字节序列（延续 §12.4「插入序即布局序」不变量），按 `BLOCKSIZE` 切块，末块可短。
- 块身份 = BLAKE3-128 内容哈希（对未压缩内容计算，BLAKE3-256 的前 128 位截断），与所在文件、偏移无关。

**快照 manifest（逻辑真值）。**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `version` | `u32` | 格式版本 |
| `flags` | `u32` | bit0 增量 / bit1 全量 / bit2 签名 / bit3 加密 / bit4 变换，其余保留 |
| `base_id` | `[u8; 32]` | 全 0 = 无基础；否则 = 基础根哈希（BLAKE3-256） |
| `training` | 可选 `trainingstate` | 复用 §12.4 |
| `metadata` | `BTreeMap<String, String>` | 复用 §12.4，键序确定 |
| `tensors` | `manifestentry[]` | 复用 §12.4，`offset` 相对逻辑数据段；每条目可选 layout 与 sparse |
| `blocks` | `blockref[]` | 有序块引用 `{ hash: [u8; 16], len: u32 }` |
| `root_hash` | `[u8; 32]` | BLAKE3-256，见下 |
| `signature` | 可选 `[u8; 64]` | Ed25519，覆盖 `root_hash`，由 `flags` bit2 标记 |

**根哈希与溯源。** `root_hash = BLAKE3_256(canonical(manifest 除 root_hash、base_id、物理块表外的全部字段))`。根哈希是逻辑状态的内容身份，不覆盖 `base_id` 与物理块表（那是「如何存放/产生」，非「是什么状态」），故同一逻辑状态的全量/增量根哈希相同、可互换；溯源链由 `base_id` 显式记录，与内容身份正交。块列表本身是 128 位内容哈希，`root_hash` 经块哈希传递承诺到全部张量字节（梅克尔 DAG）。根哈希是全局锚点、无兜底，用密码学级 BLAKE3-256（128 位碰撞安全界）。Ed25519 签名覆盖 `root_hash` 而非 canonical 本身：root_hash 已承诺全部块内容，对根签名即对整个快照签名；签名与 `base_id`、物理块表同列存储元数据、不参与 root_hash 计算，故签名只认证「这是什么状态」，不认证「从哪产生」，与全量/增量可互换不变量一致。

**物理块表（每文件，哈希键）。** 全量文件覆盖全部逻辑块，增量文件仅覆盖本文件新块：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `block_offset` | `u64` | 数据区绝对偏移（16 EiB 寻址） |
| `compressed_size` | `u32` | 压缩后长度 |
| `crc32c` | `u32` | 块存储态 CRC32C |
| `content_hash` | `[u8; 16]` | BLAKE3-128 内容身份 |

键是内容哈希（非位置），故解析某逻辑位置时先查增量自身再查基础，偏移漂移不受影响。每条 32 字节；70B FP16（140 GB、平均 256 KiB 块）约 53 万块，表约 17 MB（≈0.01%）。

**哈希/校验四层体系。** 原则：不跨级滥用；校验/去重/溯源三类需求不混用。BLAKE3-64/128/256 是同一次 BLAKE3-256 的不同截断，成本一致。

| 层 | 算法 | 长度 | 持久化 | 职责 |
| --- | --- | --- | --- | --- |
| 块级检错 | CRC32C（Castagnoli） | 4B | 是 | 随机位翻转/坏块定位，不参与身份判定 |
| 保存期去重过滤 | BLAKE3-64 | 8B | 否（仅内存） | 快速筛选，不做最终判定 |
| 块/张量内容身份 | BLAKE3-128 | 16B | 是 | 跨文件解析、去重、版本比对 |
| 快照根锚点 | BLAKE3-256 | 32B | 是 | 快照身份、溯源、完整性公证 |

**签名（Ed25519 认证）。** 签名是「认证」层，叠加在哈希「完整性」层之上：`signature = Ed25519(seed, root_hash)`，64 字节，由 `flags` bit2（`FLAGSIGNED`）标记存在性，manifest 中紧随 `root_hash`。`encode_signed(seed)` 写入签名，`decode` 解析并保留（不验签）；`verifysignature(public)` 由调用方外部提供公钥、显式验签（离线分发，文件不自含公钥——内嵌公钥不提供真实性，反徒增自证语义）。验签失败报 `signaturemismatch`（归 `iscorruption()`），无签名验签报 `signaturemissing`（归 `isincompatibility()`），全零 seed 报 `invalidsignaturekey`（不归三类，与 `invalidmanifest` 同列）。merge 产物无签名（纯内容变换），由调用方决定是否重签；reader 只解析不验签（职责仍是按需读块）。

**加密（AES-256-GCM 认证加密）。** 加密是「机密性」层，叠加在哈希与签名之上：每块独立 `AES-256-GCM(compress(content))`，密钥 32 字节外部提供、不进文件。`nonce = blockref.hash[..12]`（BLAKE3-128 明文内容身份前 12 字节）确定性派生——GCM nonce 只需唯一、不需保密，确定性使「相同明文块→相同密文块」保持内容寻址去重；96 位 nonce 对 140 GB 模型（~2^19 块）远低于生日界 2^48。加密在压缩之后、不改明文 content 的块哈希，故**加密不改变 root_hash**，同一逻辑内容的加密/未加密/签名快照 root_hash 相同、可互换，签名与加密正交。全快照统一加密（`flags` bit3 = `FLAGENCRYPTED`），codec 保持 raw/lz4/zstd、变换独立；每 `storedentry` 追加 16 字节 tag（未加密 tag 全 0、不占字节，旧文件字节级兼容）。`encode_encrypted(key)`/`encode_signed_encrypted(seed, key)` 写入密文，`decode_encrypted(bytes, key)` 解密（无密钥遇加密报 `missingkey`，归 `isincompatibility()`）；tag 不匹配报 `decryptionfailed`（归 `iscorruption()`）。reader 带密钥打开变体按需解密；`mergefiles_encrypted` 解密合并后重加密输出。

**多排布（layout）。** `tensorentry` 可选携带显式 `layout`（`strides: usize[]` + `offset: usize`，复用 `tensor::layout`），`None` = 规范行主序（向后兼容）。非 contiguous 布局（如转置存储）让权重以计算友好的排布落盘、加载时零拷贝重建；`setlayout` 校验 `isvalidfor(shape)`，decode 时再验，非法报 `invalidmanifest`。数据段字节仍按插入序连续拼接，layout 仅描述逻辑张量到连续缓冲的映射；`readtensortyped` 用 `tryfrombyteswithlayout` 恢复。

**稀疏存储（COO / bitmap）。** `tensorentry` 可选携带 `sparsedesc`（`nnz` + `format`）。COO：数据段字节 = `nnz` 个 `u32` 线性索引（小端）+ `nnz` 个非零值；bitmap：`ceil(elementcount/8)` 字节位图（LSB 优先，bit 置位 = 非零）+ `nnz` 个非零值（元素序）。`addsparse`/`addsparsebitmap` 分别构造，校验索引/位图合法、位图高位清零且（COO）索引唯一，重复报 `invalidmanifest`；`readtensortyped` 经 `desparsify`/`desparsifybitmap` 零填充重建稠密后再解 dtype/layout。索引/位图是逻辑元素坐标（shape 行主序线性化），layout 仅作用于重建后的稠密缓冲，故稀疏与多排布正交。bitmap 对结构化稀疏（如 2:4，50% 密度）比 COO 省约 16× 索引开销。

**量化内核（对称 + 非对称 int8）。** `quantize`/`dequantize` 实现对称 int8 量化：`scale = max(|x|)/127`（`groupsize=0` 为 per-tensor，否则 per-group/per-channel），`q = round(x/scale)` clamp `[-127,127]`，反量化 `x ≈ q×scale`（重建误差 ≤ scale/2）。`quantizeasymmetric`/`dequantizeasymmetric` 实现非对称：`scale = (max-min)/255`，`zeropoint = round(-128 - min/scale)`，`q = round(x/scale)+zp` clamp `[-128,127]`，反量化 `x ≈ (q-zp)×scale`，对偏斜分布（如全正数据）重建精度更高；常量组 scale=0、zeropoint=0。与 `quantdesc`（scale/zeropoint 一等张量）配套，量化输出经 `addtensor`/`setquant` 落盘。

**多文件分片（sharded）。** `shards::split`/`join` 把编码后快照的数据段切成固定大小分片（manifest 前缀保持完整，`block_offset` 语义不变），`join` 按序拼接精确重建原始工件。分片与压缩/加密/签名正交；适合超大模型拆文件存储/传输。

**Zstd 压缩。** 逐块压缩支持 LZ4 与 Zstd（`CODECZSTD`），`compressblock` 同时尝试两者选最小、`decompressblock` 按 codec 解压；解压点预留新 codec。

**bit-shuffle + delta 变换（`FLAGTRANSFORMED`）。** 压缩前的无损预处理：把块的字节按元素宽（2 = f16、4 = f32）转置成字节平面，再在每个平面内做字节级差分，把相关浮点数据的符号/指数/尾数字节分组并差分，显著提升后续 codec 的可压缩性。变换采用 floor+tail 语义——覆盖最大的整元素前缀（`elem*n` 字节），尾部 0..`elem` 字节原样携带在输出末尾——因此任意块长的 CDC 块都可无损变换（CDC 边界是任意字节偏移，实测 16 MiB 剖面 50 块中 32 块、70.4% 字节不整除元素宽；若像早期版本那样整块跳过，真实管线压缩率会从可达的 38.7% 掉到 25.4%）。`compressblock` 候选集刻意最小化——逐块试 {无变换, f16, f32} × Zstd-1 + raw 取最小存储：相关 f32 权重 `sd4+Zstd-1` 省 38.7%（裸 `Zstd-1` 仅 17%），f16 用 `sd2` 微赢，量化/随机剖面由选择器自然回落 raw/plain；LZ4 与 Zstd-3/9 已按扫参移除（每个实测剖面 Zstd-1 比率最优或持平，Zstd-3/9 最多多省 0.4pp 但压缩慢 5–10×，`CODECLZ4` 解码路径为旧文件兼容保留）。格式上 `FLAGTRANSFORMED`(0x10) 门控每条 `storedentry` 追加 1 字节 transform（0/2/4），无该 flag 的旧文件字节级兼容；块哈希计算基于未变换原始字节，故 root_hash、内容寻址去重、加密与签名语义全部不变，变换与其正交。可复现基准：`benches/checkpoint_compress.rs`（快照级：correlated-f32 61.29% 剩余/省 38.7%，random-f32 100.02% raw 回退；decode 3.35ms、encode 73ms @ 16 MiB）与 `/tmp/zbench`（块级扫参）。

**decode 加载路径优化（2026-10-16）。** 快照级 decode 实测分解（16 MiB 相关 f32 / 9.7 MiB 存储）：手写逐字节 CRC32C 17.1ms（567 MiB/s，占 43%）、zstd 解压 10.3ms、BLAKE3 内容哈希 6.6ms、变换逆 5.1ms，合计 ≈39ms 即旧 decode 40ms。两项优化：① CRC32C 改 slicing-by-8 查表（8×256 编译期表，纯 safe Rust 零新依赖，570 → 2583 MiB/s）；② `verifyfull` 并行化——块间完全独立，元数据一致性/数据边界串行先验（manifest 错误保持全序），数据段按块数均分 worker、各写预分配 buffer 的互不重叠区间（`split_at_mut`，零拼接拷贝），`PARALLELMINBLOCKS=8`/`PARALLELMINBYTES=1MiB` 阈值以下保持串行原路径；数据错误报最早失败段内首个坏块（错误语义差异已 doc 注明）。结果：**decode/correlated 40ms → 3.35ms（快 ≈12×）**，压缩率与格式字节级不变，936 测试全过、clippy 零警告。可复现：`cargo bench --offline --bench checkpoint_compress`。

**encode 并行化。** 对称 decode 的并行：`encode_with` Phase 1 逐块压缩（候选集 `{无变换,f16,f32}×Zstd-1`）与 Phase 2 逐块加密（AES-GCM）+ CRC 均块间独立，按核数均分块 span，`thread::scope` + `split_at_mut` 各 worker 写互不重叠槽，压缩/加密结果字节精确不变。`encode/correlated-f32` 73ms → 28ms（快 ≈2.6×），decode 3.35ms 不变。

**mmap 零拷贝。** `checkpointreader` 的文件变体（`openfullfile`/`opendeltafile` 等）经 `mmap-guard`（safe API，unsafe 封装在依赖内）内存映射文件，`mmapsource` 按块返回借用切片（`Cow::Borrowed`），`readtensorcow` 对单块 raw 且无变换的张量零拷贝返回。workspace 保持 `forbid(unsafe_code)`。

**字段布局（设计草案）。** 下列固定字节步长的条目为早期设计草案；实际 manifest 采用变长二进制序列化（`version`/`flags`/张量/块列表按 `pushu*`/`read*` 编解码，见 §12.4 与 `crates/checkpoint/src/uformat.rs`）。草案张量条目 64 B 定长：

| 长度 | 字段 | 类型 |
| --- | --- | --- |
| 8 | `name_hash` | `u64`（FNV-1a） |
| 1 | `dims` | `u8` |
| 16 | `shape` | `[u16; 8]` |
| 1 | `dtype` | `u8` |
| 1 | `quant_granularity` | `u8` |
| 1 | `layout_id` | `u8` |
| 1 | `sparse_type` | `u8` |
| 2 | `group_id` | `u16` |
| 8 | `data_offset` | `u64` |
| 4 | `total_size` | `u32` |
| 2 | `block_count` | `u16` |
| 16 | `content_hash` | `[u8; 16]` |
| 3 | `_reserved` | `[u8; 3]` |

字节和 = 64。已知限制：`total_size: u32` 单张量上限 4 GiB（当前前沿模型最大单张量 < 2 GiB，满足；超限需扩展字段，保留位预留升级空间）。

**增量 checkpoint 语义。** 增量 = 完整逻辑 manifest（含全部块 128 位哈希列表）+ 仅新块物理数据 + `base_id`。**无删除列表**：逻辑内容由块列表完全决定，未引用块自然消失。保存去重：新块哈希已存在于基础块集（64 位过滤键匹配 + 128 位身份一致）即不重复写，仅在 manifest 保留引用；相同张量（嵌入层/输出层）全局去重。

**防静默错误。** 默认生产模式以 CRC32C（检损坏）+ BLAKE3-128（身份判定）为最终依据，不逐块字节比对（与 ZFS、Git LFS 默认一致）。字节精确比对为可选严格模式（张量级、一次性），用于发布前终验、基础不可信等场景。

**合并（物化）。** `merge(base, delta, out)`：校验 delta 根哈希 → 校验 base 存在且 `root_hash(base) == delta.base_id` → staging 写 out（`base_id := 0`，物理块表全量，`root_hash(out) == root_hash(delta)`）→ 逐逻辑块按「delta 自身 → base」解析来源并校验 CRC32C + BLAKE3-128 → 存储态字节原样拷贝（不重新编码）→ flush + fsync + 原子 rename（复用 `writeatomic`）。任一校验失败即中止并删 staging，源文件不变；合并是复制 + 校验的纯变换，绝不把损坏 base 传播进 out。

**恢复（按需读取）。** `checkpointreader` 提供与 `checkpoint::readtensor` 同构接口：载入 manifest 并验根哈希 → 建块解析表（逻辑位置 → 来源/偏移/CRC32C）→ `readtensor` 仅读被请求张量覆盖的块（受检算术算块区间），验 CRC32C（首次验 BLAKE3-128 并缓存）。类型化读取 `readtensortyped<T>` 在同一路径上先校验存储 `dtype` 等于 `T::DTYPE`（不符报 `dtypemismatch`，归类结构不兼容），再把 manifest shape 与字节交给 `tensor::tryfrombytes` 解码为 `tensor<T>`，调用方不可能观察到静默错误的 dtype 或 shape；解码失败经 `checkpointerror::tensor` 传播。文件变体经 `mmap-guard` 内存映射（见「mmap 零拷贝」）。

**压缩 / 保留 / 溯源。** 文件一旦 rename 即不可变（copy-on-write），压缩只产新文件、不原地改。链长上限（连续增量 > 32 或增量体积 > 全量 50%）触发压缩。GC 仅删不可达且过宽限期的文件。`verifyprovenance` 沿 `base_id` 链验证各根哈希。

**并发。** 训练写 = 新文件 + 原子 rename；读者按名打开持 fd 读旧完整内容（POSIX，`writeatomic` 既有保证）。基础删除由 GC 宽限期 + 打开时验根哈希兜底；不做锁/引用计数（§12.7）。

**错误分类。** `checkpointerror` 新增 13 变体（15 → 28）：`rootmismatch`、`blockcrcmismatch`、`blockhashmismatch`、`truncatedblock`、`missingblock`、`blockoutofrange`、`signaturemismatch`、`decryptionfailed`（归入新访问器 `iscorruption()`）；`missingbase`、`basehashmismatch`、`signaturemissing`、`missingkey`（并入 `isincompatibility()`）；`invalidsignaturekey`（不归三类，与 `invalidmanifest` 同列 malformed input）。`isrecoverable()` 不变（仅 `io`）。`iscorruption()` 区分「损坏」于「不兼容」与「可重试 I/O」（engineering.md §5.11）。

**不变量（验收判据）。** 块 `len` 与 BLAKE3-128/CRC32C 一致；块列表总长 = 逻辑数据段长（受检算术，`blockoutofrange` 兜底）；逻辑数据段 = 张量字节连续拼接且区间不重叠（复用 `checkoverlaps`）；合并输出根哈希 == 增量根哈希、数据段与原始全量字节相等；校验失败不发布部分产物；恢复只读目标块；所有文件不可变。

**明确排除（本契约范围外）。** LZ4 与 Zstd 逐块压缩已实现（codec 与解压点预留扩展）；bit-shuffle + delta 变换已实现（`FLAGTRANSFORMED`）；Ed25519 签名已实现（`encode_signed`/`verifysignature`）；AES-256-GCM 分块加密已实现（`encode_encrypted`/`decode_encrypted`）；mmap 零拷贝已实现（`mmapsource`，safe API）。块来源以 `blocksource` trait 预留扩展点。

---

## 13. 层 3：optim

### 13.1 上下文

路线图阶段 P9 引入 `optim`。按本仓为 `optim` 定义的职责边界，它拥有：优化器契约、AdamW 与 SGD 实现、梯度裁剪、学习率调度，以及优化器专属的可变状态。它 MUST NOT 拥有模型架构、数据集加载、checkpoint 文件编码或训练循环编排。

其工作区依赖仅为 `tensor` 与 `lib`，另加 `thiserror` 用于错误枚举。它刻意**不**依赖 `autograd`、`nn` 或 `checkpoint`：

* 不依赖 `autograd`：计划要求它与任何具体 autograd 图解耦。更新规则作用于被拥有的参数张量及其梯度张量，以借用切片传入；梯度是任意 `tensor<f32>`，无论由何种方式产生。
* 不依赖 `nn`：`nn` 目前只暴露 forward-only 参数，更小的基于 `tensor` 的抽象对优化器状态与更新面已足够。
* 不依赖 `checkpoint`：`optim` 既非 checkpoint 编码者，也非训练状态所有者。它以命名状态张量暴露快照，由更高层映射到 checkpoint 张量契约。

角色与主责（IDN3）：本 crate 是优化规则的所有者。它在实现前定义边界与不变量，逐条把更新对照手算参考与边界用例验证，其唯一的性能声明是测量基线而非优化。

触发域（M2）：域 3（架构与契约——新 crate、trait 与公共符号）、域 4（数值与性能——浮点更新规则加基准）、域 1（失败、输入与资源——操作数校验与不可信恢复状态）。本 crate 遵循仓库的无分隔符命名约定（NM1）、工作区无 `unsafe` 与无 `panic`/`unwrap`/`expect` 规则，并自有 `thiserror` 错误枚举。

### 13.2 决策

crate 根暴露十一个公共项，从互不冲突的模块文件重导出：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `optimizer`（trait） | `optimizers.rs` | 更新规则契约与共享校验 |
| `adamw` | `adamws.rs` | 带偏置修正与解耦权重衰减的 AdamW |
| `adamwconfig` | `adamws.rs` | 经校验构造器构造的 AdamW 超参数 |
| `sgd` | `sgds.rs` | 带耦合权重衰减的动量 SGD |
| `sgdconfig` | `sgds.rs` | 经校验构造器构造的 SGD 超参数 |
| `clipconfig` | `gradientclip.rs` | 经校验构造器构造的裁剪方法 |
| `clipgradients` | `gradientclip.rs` | 原地全局范数/数值裁剪 |
| `lrschedule` | `scheduler.rs` | 纯 step-to-rate 调度 |
| `optimizerstate` | `state.rs` | 可序列化的优化器快照 |
| `statetensor` | `state.rs` | 一个命名 `f32` 状态张量 |
| `opterror` | `error.rs` | crate 自有错误枚举 |

**更新方程。** 优化器状态以参数位置为键。完成的步数 `t` 在首次更新时从 1 开始。

AdamW（`adamw`）：

```text
m = beta1 * m_prev + (1 - beta1) * g
v = beta2 * v_prev + (1 - beta2) * g * g
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
p = p - lr * ( m_hat / (sqrt(v_hat) + eps) + weightdecay * p )
```

`m` 与 `v` 从零开始，故首次更新为 `m = (1 - beta1) * g`、`v = (1 - beta2) * g * g`。偏置修正除以严格正因子，因为 `beta1, beta2 < 1` 且 `t >= 1`。权重衰减是解耦的：作用于参数而非加入梯度，与 Loshchilov & Hutter 的 AdamW 一致。

SGD（`sgd`）：

```text
g_tilde = g + weightdecay * p
b = momentum * b_prev + g_tilde
p = p - lr * b
```

速度缓冲 `b` 从零开始。权重衰减是耦合的（在动量累积前加入梯度），与经典 L2 惩罚及 PyTorch 的 SGD 一致。当 `momentum = 0` 时 `b = g_tilde`，规则退化为普通 SGD。本阶段刻意不支持 Nesterov 动量与 dampening。

**梯度裁剪。** `clipconfig` 经校验构造器构建，故 `clipgradients` 不会失败：

* `clipconfig::none` 不改变梯度。
* `clipconfig::byvalue(min, max)` 将每个元素独立钳制进闭区间；构造器拒绝非有限边界与 `min > max`。
* `clipconfig::bynorm(maxnorm)` 把每个梯度的每个元素视为一个向量，当其 L2 范数超过 `maxnorm` 时按 `maxnorm / norm` 缩放每个元素。范数用 `f64` 累积以保证确定性与余量。范数为零或在 `maxnorm` 内时梯度不变，故 `maxnorm = 0` 精确地把非零梯度置零且无除零；构造器拒绝负数或非有限 `maxnorm`。

优化器可通过其配置的 `withclip` 挂接一个 `clipconfig`；设置后，`update` 在应用更新前原地裁剪梯度。

**学习率调度。** `lrschedule` 是零基步索引到学习率的纯函数；它不持状态，由调用方经 `optimizer::setlr` 应用。设 `s` 为步索引：

```text
constant:          lr
steplr:            max(min, initial * gamma^(s / steplen))
exponential:       max(min, initial * gamma^s)
cosineannealing:   min + (initial - min) * (1 + cos(pi * min(s, period) / period)) / 2
warmupcosine:      initial + (peak - initial) * s / warmup                  if s < warmup
                   min + (peak - min) * (1 + cos(pi * min(s - warmup, period) / period)) / 2   otherwise
```

构造器校验每个域：非负学习率、`gamma ∈ [0, 1]`、非零 `steplen`/`period`，以及 `warmup <= period`。具有 `min` 界的调度在 warmup 后单调不增且从不低于 `min`。

**状态与 checkpoint 兼容。** `optimizerstate` 携带已完成步数、当前学习率、参数数量与矩张量。它刻意独立于模型架构与 checkpoint 编码。每个矩以命名 `statetensor` 暴露：

* AdamW：参数索引 `i` 对应 `m.{i}` 与 `v.{i}`。
* SGD：`momentum.{i}`。

`statetensor::tobytes`/`frombytes` 恰好编码 checkpoint 张量所需的小端 `f32` 字节，故上层通过把每个张量加入 checkpoint 来持久化快照，并通过读回字节重建 `optimizerstate` 来恢复。`restore` 在采纳前重新校验快照（数量、名字、元素数、学习率），把恢复数据视为不可信（IN1）。拥有 `step`/`epoch` 的训练状态仍是 checkpoint crate 的 `trainingstate`；`optim` 与调用方保持二者同步。`checkpoint` 因而仅是用于在集成测试中证明该契约的开发期依赖，不属于生产闭包。

**命名与模块布局。** 仓库的无分隔符约定使得模块与从其重导出的类型在同一名字空间冲突（`E0255`）。因此计划的文件列表被翻译（A8），使模块不与导出的类型同名：`optimizer.rs` → `optimizers.rs`（trait `optimizer`）、`adamw.rs` → `adamws.rs`（struct `adamw`）、`sgd.rs` → `sgds.rs`（struct `sgd`）。`gradientclip.rs`、`scheduler.rs`、`state.rs`、`error.rs` 与 `lib.rs` 保留原名，因为它们不导出同名项。

### 13.3 接口边界（DS1）

* `optim` 仅依赖 `tensor`（拥有的张量与形状）与 `lib`（基础类型），另加 `thiserror` 用于错误枚举。不导入任何工作流 crate。
* `optimizer` trait 有七个方法（`getname`、`getlr`、`setlr`、`getstep`、`update`、`getstate`、`restore`）。`update` 取两个平行 `&mut [tensor<f32>]` 切片，分别为参数与梯度，顺序由调用方定义且稳定；优化器以该位置为每参数状态键。
* 公共接口宽度：11 个 crate 根项加 7 个 trait 方法，共 18 个计数符号，低于 `A4` 上界 20。*依据：单模块公共契约应能在一轮评审内完整复核（A4）。*

### 13.4 不变量与错误路径（DS2）

算法级不变量：

* 一次更新逐元素读每个参数/梯度对；梯度形状必须等于其参数形状，两切片长度必须相等。
* 矩缓冲在首次非空更新时惰性初始化为零，随后数量与形状固定；随后参数集不同的更新被拒绝。
* 步计数器在每次成功非空更新中恰好递增一次，使偏置修正与状态持久化观察到同一计数。
* `lrschedule::getlr` 是全函数且永不返回负率；有 `min` 界的调度在 warmup 后单调不增且从不低于 `min`。

实现级不变量：

* `update` 在变更任何张量前校验长度、形状与既有状态，故被拒的调用保持参数、梯度与状态不变。
* 元素循环 zip 等长切片；长度相等由形状检查推出，因为张量的缓冲长度等于其形状的元素数。
* `statetensor` 构造与 `frombytes` 对照形状校验元素数，故 `tobytes`/`frombytes` 精确往返，`optimizerstate` 不可能携带畸形张量。

错误分支与返回语义（`opterror`）：

* `tensor`——从形状、元素数或张量构造透传的 `tensorerror`。
* `mismatchedcount`——参数与梯度切片长度不同。
* `shapemismatch`——梯度形状不同于其参数形状。
* `invalidstate`——已初始化状态或恢复快照不一致（张量数量错、未知或重复槽名、越界索引、缺槽）。
* `invalidhyperparameter`——学习率、衰减、epsilon、动量、界、周期或 warmup 超出支持域。

资源回滚：所有构造基于值；出错时不返回半成品，拥有的缓冲被 drop（作用域释放，`OW1`/`ER4`）。`update` 在首次变更前完成校验，故被拒调用是 no-op。梯度仅在校验成功后才原地裁剪。

### 13.5 复杂度（DS3）

每个函数都保持在 `DS3` 上界之内：圈复杂度 ≤ 10、有效行数 ≤ 80。配置校验拆分为小助手（`checklr`、`checknonnegative`、`checkunitinterval`、`fraction`），更新循环用迭代器 zip 而非带分支的嵌套索引。*依据：超限函数无法在一轮评审内完成分支验证（DS3）。*

### 13.6 并发设计（DS4）

`optim` 不引入共享可变状态，也不引入同步原语。优化器完全拥有其缓冲并经 `&mut self` 被变更，故两线程不能在无外部锁的情况下共享一个优化器，类型系统表达了该约束（`CC2`）。张量在一次更新的持续期内被可变借用，调用后归还调用方；无引用存活于调用之外。无锁，故无锁序要求。

### 13.7 被否决方案（DS5）

* **依赖 `autograd` 并取 `variable`/图句柄。** 否决：计划要求 `optim` 与任何具体 autograd 图解耦，且 `autograd` 目前仅通过自己的 tape 暴露梯度。借用普通 `tensor<f32>` 切片使更新规则可用于任意梯度生产者。
* **依赖 `nn` 并更新 `parameter` 值。** 否决：`nn::parameter` 拥有不可变张量且无可变更新路径，从 `optim` 依赖 `nn` 会拉入更大闭包而无收益；基于 `tensor` 的切片接口更小且足够。
* **依赖 `checkpoint` 并直接序列化整个优化器状态。** 否决：`optim` 将因此拥有 checkpoint 编码并反转 `train` 协调持久化的边界。命名 `statetensor` 把持久化留在调用方。
* **运行期选择的 `dyn` optimizer trait object。** 否决：不存在运行期选择需求；对 `optimizer` trait 的具体泛型分发已足够（`A3`）。
* **把学习率调度嵌入优化器。** 否决：会把调度演化耦合到优化器状态持久化，使两者更难测试；经 `setlr` 应用的纯 step-to-rate 函数更小且可独立验证。
* **把梯度裁剪作为 `update` 的强制步骤并取不可变梯度切片。** 否决：会强制每步复制梯度。经校验构造器构造的可选 `clipconfig` 原地裁剪调用方缓冲，独立的 `clipgradients` 仍可用于显式控制。
* **SGD 的 Nesterov 动量与 dampening。** 本阶段否决：它们增加分支与方程却暂无消费者；文档化的规则集是最小代表对（AdamW 解耦、SGD 耦合）。

### 13.8 后果

* `optim` 是 `tensor` 与 `lib` 之上的依赖叶子。加入它不改变任何既有 manifest，因为根成员是 glob 形式（`members = ["crates/*"]`）。
* 更新规则无需图即可测试，状态可经现有 checkpoint 张量契约持久化而无生产依赖边。
* 基准最初是 crate 本地的（`crates/optim/benches/optim.rs`），因为 P9 指令把新算子基准限定在所属 crate，`criterion` 声明为开发依赖并在 `crates/optim/Cargo.toml` 添加 `[[bench]]` 项。后续整合把基准移到仓库根 `benches/optim.rs`，故 `crates/optim` 现在不保留 `[[bench]]` 目标与 `criterion` 依赖。
* crate 无环；其唯一开发期边是集成测试到 `checkpoint`。

### 13.9 验证

在真实工作区中对 `crates/optim` 于 2026-10-10 执行（macOS 27.0.1、`aarch64-apple-darwin`、rustc 1.99.0、edition 2024）：

```sh
cargo fmt --all                                     # exit 0
cargo check -p optim --all-targets                  # exit 0
cargo clippy -p optim --all-targets -- -D warnings  # exit 0, no warnings
cargo test -p optim                                 # 迁移前计数已省略；当前套件位于仓库根 tests/
cargo bench -p optim --no-run                       # exit 0
RUSTDOCFLAGS='-D warnings' cargo doc -p optim --no-deps   # exit 0
```

仓库根 `tests/optimunits.rs` 与 `tests/optim.rs`（对应 `optim`）—— 0 失败。3 个集成测试在后续整合中移到仓库根 `tests/optim.rs`。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）

覆盖：

* `adamw`——首次更新手算；恒定梯度下步长等于学习率；零梯度下的解耦权重衰减；原地裁剪；`setlr`；操作数长度/形状拒绝且不变更；改变参数集拒绝；状态往返；快照计数与槽名拒绝；无效超参数。
* `sgd`——普通下降；动量累积（`-0.1, -0.29, -0.561`）；耦合权重衰减；数值裁剪；状态往返；计数拒绝；无效超参数。
* `gradientclip`——none；按值钳制；跨多个梯度的全局按范数缩放；界内 no-op；零范数与零上限且无除零；无效配置。
* `lrschedule`——constant；steplr 衰减与其 `min`；exponential；cosine 边界与周期后钳制；warmup 上升与 cosine 衰减；无效域。
* `state`——构造与长度拒绝；含 `f32::MAX` 的字节往返；短缓冲拒绝；标量与张量访问器。
* `opterror`——`tensorerror` 转换；助手格式化；变体可区分性。
* `tests/optim.rs`——AdamW 与 SGD 状态经 `checkpoint::encode`/`decode` 的 checkpoint 往返、继续更新等价性，及驱动 `setlr` 的调度。

基准（不声称改进）。命令：

```sh
cargo bench -p optim --bench optim -- --warm-up-time 1 --measurement-time 2 --sample-size 30
```

负载：一个 `[1024, 1024]` `f32` 参数块（1,048,576 元素）与同形梯度；每个计时迭代一次 `update`，缓冲在 criterion 的 setup 阶段新建。Release profile：`opt-level = 3`（默认）、`lto = "thin"`、`codegen-units = 1`、`debug = 1`。环境：macOS 27.0.1、`aarch64-apple-darwin`、rustc 1.99.0（确切 CPU 品牌字符串在命令沙箱下不可读）。

| 基准 | 中位数 | 95% 区间 | 吞吐 |
| --- | --- | --- | --- |
| `optim/adamw/step` | 762 µs | 753–775 µs | ~1.38 Gelem/s |
| `optim/sgd/step` | 160 µs | 159–162 µs | ~6.54 Gelem/s |
| `optim/clipgradients/bynorm` | 576 µs | 574–578 µs | ~1.82 Gelem/s |

AdamW 的每元素工作量约为 SGD 的五倍（两次矩更新、偏置修正与一个平方根），与观察到的比例相符。按范数裁剪基准读取每个元素两次（累积、再缩放），仅作为承载正确性的基线被测量。

### 13.10 复审条件

* 当 `autograd` 暴露稳定的公共梯度访问器时，复审把从图变量到 `tensor` 切片的薄适配器放入 `train`（优先），而非新增 `optim` → `autograd` 边。
* 当消费者需要 Nesterov 动量、dampening 或第二种自适应方法时，在现有 trait 后扩展规则集，而非拓宽 `update`。
* 当训练层持久化优化器状态时，对照 checkpoint 张量契约确认 `m.{i}` / `v.{i}` / `momentum.{i}` 命名与 `optimizerstate` 布局；改名是兼容性变更。
* 当需要分布式或混合精度优化器时，复审 `f32`-only 与单进程假设。

## 14. 层 3：context

### 14.1 上下文

路线图阶段 P6 引入 `context`。它拥有：执行配置、取消契约、内存预算抽象、所需的工作者协调原语与指标接口。它 MUST NOT 拥有模型架构、领域专属采样算法、无关应用状态的通用容器，或隐藏的全局可变状态。其唯一工作区依赖是 `lib`（用于 `requestid`）；同步仅来自 Rust 标准库。它刻意**不**依赖 `infer`、`train`、`models` 或任何其他工作流 crate，以便被推理与训练同时消费而无反向边。

本 crate 遵循仓库的无分隔符命名约定（NM1）、工作区无 `unsafe` 与无 `panic`/`unwrap`/`expect` 规则，并自有 `thiserror` 错误枚举。

### 14.2 决策

crate 根暴露十一个公共项，从互不冲突的模块文件重导出：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `executionconfig` | `execution.rs` | 经校验构造器构造的不可变上限 |
| `executioncontext` | `execution.rs` | 每次执行的控制所有者 |
| `cancellation` | `cancel.rs` | 可克隆的 `AtomicBool` 句柄 + 检查点 |
| `memorybudget` | `budget.rs` | 容量 + 原子字节记账 |
| `memoryreservation` | `budget.rs` | drop 时释放的 RAII 字节守卫 |
| `workergroup` | `worker.rs` | 有界作用域线程作业运行器 |
| `workererror` | `worker.rs` | 通用协调错误 |
| `metrics` | `metric.rs` | 窄计数/仪表接收 trait |
| `collectingmetrics` | `metric.rs` | 用于测试的内存接收器 |
| `noopmetrics` | `metric.rs` | 丢弃型默认接收器 |
| `contexterror` | `error.rs` | crate 自有错误枚举 |

### 14.3 接口边界（DS1）

* `context` 仅依赖 `lib`（`requestid`）。它不导入更高层 crate，原子、线程与互斥量只用 `core`/`std`。
* `executionconfig` 是 `Copy` 且不可变；`executioncontext` 拥有请求级控制并按执行创建。二者均非全局，也均不存于模型权重，满足 `OW2` 与 `context` 的职责边界。
* `metrics` trait 有两个方法，名字为 `&'static str`，保持指标基数有界，且不对基础 crate 施加后端、注册表或 crate 耦合。
* 公共接口宽度：crate 根 11 项，低于 `A4` 上界 20。*依据：单模块公共契约应能在一轮评审内完整复核（A4）。*

### 14.4 不变量与错误路径（DS2）

算法级不变量：

* `cancellation` 起始为未取消；`cancel` 幂等，且一旦置位，每个 clone 都观察到已取消。
* `memorybudget`：`reserved ≤ capacity` 恒成立；一次成功的 `tryreserve` 恰好按请求字节增加 `reserved`，返回守卫的 drop 恰好减少这些字节。
* `executioncontext`：`steps ≤ maxsteps`；一次成功的 `checkpoint` 使 `steps` 递增一；取消在步骤预算被查询之前报告。
* `workergroup::run` 在无失败时恰好返回 `count` 个输出，按作业索引排序；失败时返回单个首个观察到的失败且无部分结果。

实现级不变量：

* `memorybudget` 记账使用 `AtomicUsize::try_update`（Rust 1.99 中已弃用 `fetch_update` 的替代）并在更新闭包内用 `checked_add`，故溢出请求被拒绝（`budgetoverflow`）而非回绕；将超过容量的请求被拒绝（`budgetexceeded`）且不预留任何字节。
* `executioncontext::checkpoint` 对其步计数器使用相同的 compare-and-update 模式，故并发检查点不能超过步骤预算。
* `workergroup` 用 `AtomicUsize::fetch_add` 分发作业索引，并使用 `stop` 标志加首次写入者胜出的 `Mutex<Option<workererror<E>>>`；工作者各自持有本地结果，仅在所有 join 后合并，故无锁在作业执行期间被持有。

错误分支与返回语义（`contexterror`，5 个变体）：

* `cancelled`——一个检查点观察到已取消的执行。
* `stepbudgetexceeded { used, max }`——步骤预算耗尽。
* `budgetexceeded { requested, reserved, capacity }`——预留超过容量。
* `budgetoverflow`——字节记账会使 `usize` 溢出。
* `invalidconfig { detail }`——`executionconfig::new` 或 `workergroup::new` 中零 `maxworkers`；一个无法运行任何工作的执行是接线错误而非有效资源上限。

协调失败由 `workererror<E>` 建模，含 `context { source }`（一个 `contexterror`）、`job { index, source }`（作业自身的错误 `E`）与 `workerpanicked`。区分这三者可让调用方对取消、调用方提供的作业失败与工作者 panic 作出不同反应。

资源回滚：预留在每个终止路径（正常返回、`?` 早退、取消循环、线程 unwind）于 `Drop` 释放。这是唯一的释放机制；无显式释放调用（`OW1`、`ER4`、`NM3`）。失败的 `workergroup::run` 不返回部分输出，故中止的运行无法把结果泄漏进调用方状态（`OW3`）。

### 14.5 复杂度（DS3）

每个函数都保持在 `DS3` 上界内：圈复杂度 ≤ 10、有效行数 ≤ 80。`workergroup::run` 是最大的函数体，被保持为一个作用域线程循环加一个 join 循环，除作业循环外无嵌套控制流。*依据：超限函数无法在一轮评审内完成分支验证（DS3）。*

### 14.6 并发设计（DS4）

共享状态及其同步，每个资源一个原语：

| 状态 | 原语 | 顺序 |
| --- | --- | --- |
| `cancellation` 标志 | `Arc<AtomicBool>` | store `Release`，load `Acquire` |
| `memorybudget::reserved` | `AtomicUsize` | `try_update(AcqRel, Acquire)`，drop `fetch_sub(AcqRel)` |
| `executioncontext::steps` | `AtomicUsize` | `try_update(AcqRel, Acquire)` |
| `workergroup` 作业游标 | `AtomicUsize` | `fetch_add(AcqRel)` |
| `workergroup` stop 标志 | `AtomicBool` | store `Release`，load `Acquire` |
| `workergroup` 首个失败 | `Mutex<Option<..>>` | 作用域局部，从不跨作业持有 |
| `collectingmetrics` 映射 | 两个 `Mutex<BTreeMap>` | 每指标，容忍中毒 |

取消可见性由原子本身建立：`Release` store 与 `Acquire` load 构成所需的 happens-before 对（`CC3`）。互斥量是作用域局部的且从不嵌套，故不存在锁序，也不可能有死锁环（`CC1`、`DS4`）。`collectingmetrics` 用 `PoisonError::into_inner` 从中毒锁恢复而非 panic，故 panic 的写者不能卡住后续读者。无全局锁、无静态可变状态、无跨执行共享计数器（`A6`）。

`workergroup::run` 是阻塞且可取消的调用：它阻塞至每个工作者 join，在每个作业前检查 `exec::checkpoint` 以使取消与步骤预算传播（`CC5`），且在观察到取消后不产生任何可观察副作用。作业返回自己的错误值；组不翻译 `E`。

### 14.7 被否决方案（DS5）

* **把 `context` 做成通用 `Vec<Box<dyn Any>>` 请求容器。** 否决：`context` 的职责边界明确禁止通用容器；显式类型字段保持所有权与生命周期清晰。
* **用 `AtomicBool` 加全局取消句柄注册表。** 否决：全局注册表是全局可变状态（`A6`）；句柄改为按值克隆，故生命周期跟随所有权。
* **用条件变量 / `park`/`unpark` 实现取消。** 否决：取消是单向闩锁，release/acquire 的 `AtomicBool` 已足够且更简单；在显式检查点轮询使取消确定且可测（`CC5`）。
* **用 `Mutex<usize>` 计数器跟踪内存。** 否决：compare-and-update 原子给出相同记账而无锁、无锁序义务，且 RAII 守卫使释放不会失败。
* **让 `memoryreservation` 拥有字节并经显式 `release()` 归还。** 否决：`Drop` 释放由 `OW1`/`NM3` 要求，且是唯一在早退与 unwind 上同样正确的路径。
* **让 `metrics` 接受 `&str` 标签。** 否决：无界标签基数是可观测性隐患；`&'static str` 从构造上限定它。
* **为 `workergroup` 依赖 `tokio`/`rayon` 执行器。** 否决：该原语是有界阻塞扇出，`std::thread::scope` 已足够，并避免在基础 crate 上引入异步运行时依赖。
* **让 `workergroup` 在失败时返回部分结果。** 否决：调用方绝不应能把中止运行的输出误当成完整输出（`OW3`）；失败只返回 `Err`。

### 14.8 后果

* `context` 是 `lib` 之上的依赖叶子；加入它不改变任何既有 crate 的 manifest，因为根成员是 glob 形式（`members = ["crates/*"]`）。
* 取消、步骤预算与内存预算显式、可测且可区分；调用方能分辨「已取消」「内存耗尽」与「步骤预算耗尽」。
* 未引入全局可变状态；每个共享字段都是原子或带文档化顺序的作用域局部互斥量。
* `workererror` 对作业错误泛型，故更高层保留自己的错误类型而无需翻译。

### 14.9 验证

命令限定到 `context`，因为兄弟 crate 正在并发构建。于 2026-10-10 自 `/Users/fei/Doubao/llm` 执行：

```sh
cargo fmt --all
cargo check -p context --all-targets
cargo clippy -p context --all-targets -- -D warnings
cargo test -p context
RUSTDOCFLAGS='-D warnings' cargo doc -p context --no-deps
```

结果：

* `cargo clippy -p context --all-targets -- -D warnings`——通过。
* 仓库根 `tests/contextunits.rs` 与 `tests/context.rs`（对应 `context`）—— 0 失败。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
* `RUSTDOCFLAGS='-D warnings' cargo doc -p context --no-deps`——通过。

覆盖：

* `cancel`——初始状态、向 clone 传播、幂等性、跨线程传播。
* `budget`——预留/报告、超过容量的请求、等于容量的请求、计数器溢出、错误路径上的释放、并发预留从不超过容量（四线程争两个槽）。
* `metric`——计数累积、仪表覆盖、缺失指标、noop 接收器。
* `execution`——零工作者拒绝、配置暴露、检查点推进与步骤预算强制、取消优先级、预算 sizing、指标转发、可选指标。
* `worker`——零工作者拒绝、零作业、按索引排序的结果、作业错误传播（单工作者）、步骤预算耗尽、任何作业前取消。
* `error`——`Display` 中的结构化上下文、不同起因。
* `tests/context.rs`——经公共 API 对取消、步骤预算、内存预算、工作者协调与指标的端到端组合。

### 14.10 复审条件

* 当必须在共享结构上持有多于一把锁时，定义全局锁序并记录（`CC1`/`DS4`）。
* 当取消必须中断阻塞叶子调用（而非仅作业之间）时，新增可取消等待原语而非全局注册表。
* 当需要生产指标后端时，作为新的 `metrics` 实现加入，保持 trait 无后端耦合。
* 当工作者协调需要异步执行或带优先级的 work stealing 时，复审 `std::thread::scope` 设计并论证新增依赖。

## 15. 层 3：train

### 15.1 上下文

路线图阶段 P10 引入 `train`。它拥有：训练循环编排、训练配置、损失计算与工作流集成、梯度累积调度、评估工作流与恢复编排。它 MUST NOT 拥有 CPU kernel、张量存储、优化器内部或外部 checkpoint 编码。`train` 协调 save 与 restore，`checkpoint` 拥有持久化语义；checkpoint crate 从不依赖训练循环来解释其格式。

角色与主责（IDN3）：本 crate 是训练工作流所有者。它在实现前定义边界与不变量，逐步对照测量的 loss 下降与继续训练等价性验证，其唯一性能声明是它断言的测量 loss 值。

触发域（M2）：域 1（失败、输入与资源——不可信恢复状态与梯度校验）、域 2（并发与共享状态——取消与请求级训练状态）、域 3（架构与契约——新 crate 及其公共符号）、域 5（交付与治理——恢复契约与披露的局限）。

**塑造设计的现实约束。** `nn` 与 `models` 目前是 **forward-only**：它们构建并运行前向，但不暴露 autograd 反向路径，且 `autograd` 仅对 `matmul`、`add`、`mul`、`rmsnorm`、`softmax`、`relu`、`silu`、`gelutanh` 提供向量-雅可比积。`optim` 刻意与图无关，从借用梯度切片更新被拥有的 `tensor<f32>` 参数。因此在此阶段无法在没有发明穿过 `nn` 的 autograd 路径的前提下实现完整的「训练 `models` Llama 反传」循环，而本阶段不得如此。crate 围绕真实可用的契约设计。

### 15.2 决策

crate 根暴露十三个公共项，从互不冲突的模块文件重导出：

| 公共项 | 模块文件 | 职责 |
| --- | --- | --- |
| `trainconfig` | `trainconfigs.rs` | 经校验构造器构造的运行配置 |
| `trainer` | `trainers.rs` | 对参数与梯度的 loss/step 循环 |
| `stepresult` | `step.rs` | 一个已完成步的 loss、step 与学习率 |
| `crossentropy` | `loss.rs` | 稳定的交叉熵前向与梯度 |
| `lossresult` | `loss.rs` | 标量 loss 加 logits 梯度 |
| `gradientaccumulator` | `gradientaccumulation.rs` | 窗口平均的梯度缓冲 |
| `evalsample` | `evaluation.rs` | logits 与目标类别配对 |
| `evalresult` | `evaluation.rs` | 行加权 loss 与困惑度 |
| `evaluate` | `evaluation.rs` | 评估聚合 |
| `saveresume` | `resume.rs` | 写参数、优化器状态与进度 |
| `loadresume` | `resume.rs` | 原子地校验并恢复上述内容 |
| `resumestate` | `resume.rs` | 恢复的 step 与 epoch |
| `trainerror` | `error.rs` | crate 自有错误枚举 |

**命名与模块布局。** 仓库的无分隔符约定使模块与从其重导出的类型在同一名字空间冲突（`E0255`）。计划的文件列表被翻译（A8），使模块不与导出的类型同名：`trainer.rs` → `trainers.rs`（类型 `trainer`）、`trainconfig.rs` → `trainconfigs.rs`（类型 `trainconfig`）。`step.rs`（导出 `stepresult`）、`loss.rs`、`evaluation.rs`、`resume.rs`、`gradientaccumulation.rs`、`error.rs` 与 `lib.rs` 保留原名，因为它们不导出同名项。

### 15.3 接口边界（DS1）

* `train` 在生产中依赖 `lib`、`tensor`、`optim`、`checkpoint` 与 `context`，另加 `thiserror` 用于错误枚举。`autograd` 与 `cpu` 是**开发期依赖**，仅用于在集成测试中证明 loss 下降契约；它们不属于生产闭包。
* 它**不**依赖 `nn` 或 `models`。二者都是 forward-only，故本 crate 中没有任何东西能为训练调用它们；加入该边会为无消费者扩大依赖闭包。其所代表的局限在下方声明，而非掩盖。
* 它**不**依赖 `data`。评估刻意与数据集表示解耦：因为 `train` 不拥有前向，调用方从它拥有的任意 batch 或 dataset 表示构造 `evalsample` 值（logits 加 target）。这使 `train` 不获取它不使用的依赖。
* 损失函数按步以 `FnMut(&[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror>` 传入，故手写梯度、`crossentropy` 结果与 `autograd` 计算的梯度都以静态分发驱动同一循环（`A3`）。
* 公共接口宽度：13 个 crate 根项，低于 `A4` 上界 20。*依据：单模块公共契约应能在一轮评审内完整复核（A4）。*

### 15.4 不变量与错误路径（DS2）

算法级不变量：

* 每 `accumulationsteps` 个 micro-batch 应用一次优化器更新。窗口按 `1 / count` 平均，故参差 micro-batch 的窗口仍被平衡。
* 应用到一步的学习率是 `schedule.getlr(stepindex)`，其中 `stepindex` 等于已完成步数；未配置调度时为优化器的当前率。
* `crossentropy` 返回其自身损失的精确解析梯度（softmax 减 one-hot，按行取均值）；`loss.rs` 中的有限差分测试把它钉在 `1e-3` 以内。
* `loadresume` 仅当优化器快照自身的步计数等于持久化的训练步且优化器名匹配该 trainer 时才恢复。

实现级不变量：

* 状态仅在完成的 `optimizer::update` 上推进。任何 loss、梯度、取消、调度或优化器错误下，缓冲窗口被置零，且局部步计数器与优化器计数器都不移动。
* `optimizer::update` 在变更前校验长度、形状与既有状态，故被拒的更新保持参数与状态不变；`train` 依赖该契约，不自加部分变更。
* `gradientaccumulator::accumulate` 在加入任何元素前校验数量与形状，故被拒的 micro-batch 保持缓冲和不变。
* `loadresume` 对 trainer 是原子的：优化器状态先被恢复（可能失败），参数与计数器之后才被替换，故被拒的加载对 trainer 是 no-op（`OW3`）。

错误分支与返回语义（`trainerror`）：

* `tensor`——从形状、元素数或张量构造透传的 `tensorerror`。
* `optim`——从更新或状态操作透传的 `opterror`。
* `checkpoint`——从 load、save 或兼容性检查透传的 `checkpointerror`。
* `cancelled`——观察到取消令牌。
* `invalidaccumulation`——零长度累积窗口。
* `emptyparameters`——以空参数集构造 trainer。
* `steplimitreached`——配置的步骤预算耗尽。
* `gradientcountmismatch` / `gradientshapemismatch`——损失函数返回的梯度数量错，或梯度形状不同于其参数。
* `nonfiniteloss`——非有限的 loss 值。
* `accumulatorfull`——梯度被加入已满窗口。
* `invalidstate`——局部不一致，包括空 `take`、元数据不一致的恢复、恢复时的形状不匹配。
* `lossnotmatrix` / `lossemptymatrix` / `targetcountmismatch` / `targetoutofrange` / `lossnonfinite`——交叉熵契约违背。
* `unsupportedtensordtype`——必须为 `f32` 的 checkpoint 张量声明了其他 dtype。

资源回滚：缓冲基于值；丢弃部分构造的值会释放其存储（`OW1`/`ER4`）。`take` 在同一调用中把平均后的梯度移出并重新置零缓冲，故没有窗口跨越失败。

### 15.5 训练契约

**训练步契约。** `trainer::step(loss)` 恰好运行 `accumulationsteps` 个 micro-batch 并应用一次更新：

1. 对每个 micro-batch：检查步骤预算与取消；以当前参数调用损失函数；拒绝非有限 loss；校验并把梯度加入窗口。
2. 在填满窗口的那个 micro-batch：再次检查取消，取平均梯度，解析并应用调度率，调用一次 `optimizer::update`，并递增步计数器。
3. 返回一个 `stepresult`，携带窗口 micro-batch loss 的均值、新步计数、应用的学习率与窗口大小。

`trainer::accumulatemicrobatch` 逐个 micro-batch 暴露同一逻辑，填充时返回 `None`，关闭窗口的 micro-batch 上返回 `Some(stepresult)`。`trainer::step` 要求进入时窗口为空，故部分窗口绝不被静默丢弃；`trainer::step` 实现为对 `accumulatemicrobatch` 的有界循环。

**损失与梯度校验。** `crossentropy(logits, targets)` 要求秩二矩阵 `[rows, classes]`，至少一行一列，且每行恰好一个在界内 target。它用最大减法计算 `logsumexp(row) - logit[target]`，故大 logits 向零平移而非使 `f32` 溢出，并返回 mean loss 加解析梯度 `(softmax - onehot) / rows`。它拒绝非有限 loss 或梯度，且从不修改输入。梯度在模块测试中对照中心有限差分校验。

**失败或取消不推进状态。** 状态仅通过完成的 `optimizer::update` 推进。在取消或任何错误下，缓冲窗口经 `resetwindow` 丢弃，步计数器不动；优化器的「变更前校验」契约保持其自身状态不变。集成测试断言：在 between-steps 取消与 mid-window 取消之后，参数、局部步计数器与优化器步计数器均不变且窗口为空。

**针对持久化状态的恢复。** `saveresume` 写一个 `checkpoint` 容器，持有 `p.{i}` 参数张量、`trainingstate { step, epoch }` 与优化器快照张量（按自身名字 `m.{i}`、`v.{i}`、`momentum.{i}`），并把优化器名、步、学习率、参数数量与张量名列表记为 manifest metadata。`loadresume` 检查容器版本，用 `checkcompat` 验证参数 spec，解码参数，从 metadata 与命名张量重建 `optimizerstate`，并在替换参数与计数器前恢复它。为不同优化器写的 checkpoint，或优化器步与训练步不一致者，被拒绝而非静默应用。形状不匹配与缺失文件都在不变更 trainer 的前提下被拒绝。

### 15.6 声明局限：nn/models 无反向路径

通过完整 `nn`/`models` Llama 栈训练**尚未接线**，因为 `nn` 与 `models` 没有反向路径，且 `autograd` 缺少表达可微 Llama 前向（特别是交叉熵反向）的算子。本 crate 不捏造一个。剩下的是穿过 `nn`/`models` 层的反向路径——为 forward-only 层（归一化、注意力、旋转、嵌入与模型组装）提供 autograd 算子与 VJP，使一次前向产生可逆图——之后此处的同一 `trainer` 循环将无需改动地消费逐参数梯度。这是一条声明的局限（M4/VF5），在 crate 文档中同样声明。

因此 loss 下降证据取自真实之处：在仅由当前已有 VJP 的算子构建的 `autograd` 图上（`matmul`+`add`+`mul` 最小二乘，已知最优 `w = 2`），以及在 crate 自己的交叉熵损失与解析梯度上。

### 15.7 复杂度（DS3）

每个函数都保持在 `DS3` 上界内：圈复杂度 ≤ 10、有效行数 ≤ 80。步骤循环的可失败工作用 `match`/早退与单个 `resetwindow` 助手表达；配置与损失校验拆为小助手。*依据：超限函数无法在一轮评审内完成分支验证（DS3）。*

### 15.8 并发设计（DS4）

`train` 不引入自身的共享可变状态，也不引入同步原语。`trainer` 拥有其参数、优化器、累加器与计数器，并经 `&mut self` 被变更，故类型系统阻止两线程在无外部锁的情况下共享一个 trainer（`CC2`）。取消是唯一的共享信号；它使用 `context::cancellation`，一个基于 `AtomicBool` 的廉价可克隆句柄，采用 `Release`/`Acquire` 顺序，是显式同步原语而非全局状态（`A6`/`CC3`）。取消在每个 micro-batch 前检查，并在更新前再次检查。

### 15.9 被否决方案（DS5）

* **发明穿过某一 `nn` 层的最小 autograd 路径以演示端到端训练。** 否决：指令禁止捏造未请求的反向路径，且基于私有临时反向的演示会误述栈的就绪度。
* **依赖 `nn`/`models` 并在文档中引用它们但代码中不用。** 否决：未使用的依赖边为无消费者扩大闭包；改为记录该局限。
* **把损失做成 trainer 上的 boxed `dyn FnMut` 字段。** 否决：不存在运行期可扩展性需求；按调用泛型方法保持静态分发并避免堆分配（`A3`）。
* **暴露一个应用部分窗口的独立 `flush`。** 否决：部分窗口正是绝不能静默推进训练的状态；`step` 要求空窗口，`accumulatemicrobatch` 暴露该原语。
* **把优化器快照的 `f32` 标量存入张量。** 否决：checkpoint 容器已携带任意字符串 metadata 与类型化 `trainingstate`；把 step/rate/count 编码为 metadata 避免发明假标量张量。
* **依赖 `data` 以在评估中消费 loader。** 否决：`train` 中无前向，调用方无论如何必须产生 logits，故 `data` 会是未使用边；解耦的 `evalsample` 契约已足够。

### 15.10 后果

* `train` 在 `lib`、`tensor`、`optim`、`checkpoint` 与 `context` 之上无环。加入它不改变任何既有 manifest，因为根成员是 glob 形式（`members = ["crates/*"]`）。
* 训练步无需图即可测试，同一循环消费来自测试中开发依赖图的 `autograd` 梯度。
* crate 不依赖 forward-only 层，故不编码端到端训练的虚假声明。
* 与计划的根 `tests/training_resume.rs` 的偏差在此阶段是有意的（P10 指令把测试限定在 `crates/train/tests/*.rs`）；后续整合把测试移到仓库根 `tests/`，名为 `trainingresume.rs`、`lossdecrease.rs`、`cancellation.rs`。

### 15.11 验证

于 2026-10-10 在真实工作区对 `crates/train` 执行（macOS 27.0.1、`aarch64-apple-darwin`、rustc 1.99.0、edition 2024）：

```sh
cargo fmt --all                                          # exit 0
cargo check -p train --all-targets                       # exit 0
cargo clippy -p train --all-targets -- -D warnings       # exit 0, no warnings
cargo test -p train                                      # 迁移前计数已省略；当前套件位于仓库根 tests/
RUSTDOCFLAGS='-D warnings' cargo doc -p train --no-deps  # exit 0
```

仓库根 `tests/trainunits.rs`、`tests/lossdecrease.rs` 与 `tests/trainingresume.rs`（对应 `train`）—— 0 失败；1 个 doc-test。7 个集成测试移到仓库根（`tests/cancellation.rs` 2、`tests/lossdecrease.rs` 2、`tests/trainingresume.rs` 3）。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）

**测得的 loss 下降（E1）。** 两个下降都在固定 RNG 种子下对照测量值断言；下列数字由测试打印。

* `autograd_least_squares_decreases_the_loss`：SGD（`lr = 0.01`、无 momentum、seed 2024）在 `matmul`+`add`+`mul` 上对 `y = 2x` 运行 300 步。测得 loss `56.0 → 2.98e-13`，最终权重 `1.9999999`，对照解析最优 `2`。
* `cross_entropy_descent_decreases_the_loss`：SGD（`lr = 0.3`、seed 7）在 crate 自己的交叉熵与解析梯度上，两个可分行、三类，500 步。测得 loss `1.0986123`（`ln 3`，均匀 logits 值）`→ 0.009226322`。

**覆盖的行为。**

* `loss`——均匀 logits 给出 `ln(classes)`；自信正确的 loss 更低；梯度匹配中心有限差分；巨 logits 不溢出；非矩阵、空、target 数量错与 target 越界被拒。
* `gradientaccumulation`——窗口上的算术均值；零容量、空参数、溢出、数量错、形状错与空 `take` 被拒；reset 重新置零缓冲。
* `trainer`——二次损失在 20 步内下降；累积为 3 时一次更新后得到手算的 `w = 0.8`；空参数被拒；`step` 要求空窗口；取消不推进状态；非有限 loss 不推进状态；步骤预算被强制；梯度数量错会重置窗口。
* `crossentropy`/`evaluate`——行加权聚合与困惑度；空样本被拒。
* `step`/`resume` 单元测试——调度应用、step-result 字段、`f32` 字节往返与坏字节长度拒绝。
* `tests/lossdecrease.rs`——上述两个测得下降。
* `tests/trainingresume.rs`——参数与优化器状态往返、继续训练等价性、形状不匹配拒绝且不变更、缺失文件报告。
* `tests/cancellation.rs`——between-steps 与 mid-window 取消都保持参数、步计数器与窗口不变。
* `error`——透传 `tensorerror`/`opterror` 转换、助手格式化与取消可区分性。

### 15.12 复审条件

* 当 `nn`/`models` 获得反向路径（或 `autograd` 获得缺失算子）时，把真实前向接进每步 loss 闭包，并用 `models` 的端到端 loss 下降测试替换最小二乘演示。
* 当训练循环直接从 `autograd` 图获得梯度时，在 `train`（而非 `optim`）中加入薄的图到 `tensor` 适配器。
* 当需要数据并行或多设备训练时，复审单所有者 trainer 并为梯度归约加入显式同步。
* 当需要混合精度时，复审 `f32`-only 的 loss、累加器与 checkpoint 编码。

## 16. 层 3：infer

### 16.1 上下文

层 3 为工作区提供生成工作流所需的部件：`models` 拥有因果语言模型契约 `causallm` 与请求作用域的 `kvcache` 契约，`context` 拥有执行检查点与取消（`executioncontext`、`cancellation`）。`infer` 是在它们之上把 prompt 变为生成 token 的工作流 crate，既不把请求状态存入模型，也不依赖具体后端。

`infer` 依赖 `models`、`api`、`tensor`、`context` 与 `lib`。它仅通过 `models::causallm::forward` 重导出的泛型 `api::compute` 约束到达后端；`cpu` 是仅由集成测试使用的开发依赖。它不依赖 `nn`、`autograd`、`optim`、`train`、`cli` 或 `serve`。

本决策新增的依赖边（无环）：`infer → models`、`infer → api`、`infer → tensor`、`infer → context`、`infer → lib`；既有边 `models → api`、`models → tensor`、`models → lib`、`api → lib`、`api → tensor`、`context → lib`。

### 16.2 决策

1. **引擎无状态；请求拥有全部可变状态。** `generationengine<'a, M: causallm, C: compute>` 借用一个不可变模型与一个计算提供者，不持其他东西。生成期间变化的一切都活在 `generationrequest` 中：不可变 prompt、累积上下文、采样器位置、停止准则与请求拥有的 `kvstate`。故两个请求对一个模型值并发运行而不共享可变状态（`OW2`），模型权重从不是存放请求数据之处。
2. **`generate` 是先 prefill，再 sample/append/decode 循环。** 它先把请求重置到一个新尝试，检查取消，对 prompt 运行一次 prefill 前向，然后重复：采样一个 token，遇 EOS 停止，追加 token，交给 sink，达到新 token 上限则停止，否则 decode 完整累积上下文。每次迭代都以 `executioncontext` 检查点开始，故取消与步骤预算耗尽在下一次前向之前停止循环并返回错误而非结果（`CC5`）。
3. **正确性优先：主路径重算完整上下文。** `prefillprompt` 与 `decodestep` 都归约为对给定 token 的一次 `causallm::forward` 加最后位置 logits。缓存未接线时，decode 前向整个上下文。这是缓存式 decode 必须超越的已测基线（`PF1`/`PF5`）。
4. **采样是由 `lib::seed` 播种的确定性宿主逻辑。** 温度恰为 `0.0` 时是 greedy（argmax，平局取最低索引）。否则 logits 按 `1 / temperature` 缩放，可选地限制到 `topk` 个最大者与降序概率累积质量达到 `topp` 的最小前缀，并从由 SplitMix64 流驱动的分类分布抽一个 token。由相同 `samplingparams` 与 `seed` 构造的两个采样器对相同 logits 序列返回相同 token；`reset` 回绕流。
5. **停止条件显式且已验证。** `stopcondition` 命名 EOS token 与非零新 token 上限；`stopreason` 报告 `endoftoken`、`maxlength` 或 `stopped`（sink 要求停止）中的哪一个结束了运行。无法发出任何 token 的请求作为接线错误被拒，而非被当成退化上限接受。
6. **输出契约是调用作用域的 sink。** `streamsink::ongenerated` 返回 `true` 继续或 `false` 停止，`onstop` 在正常终止时恰好触发一次。失败或取消的运行返回 `infererror` 且从不调用 `onstop`，故流不能静默地暗示成功。引擎对 `S: streamsink` 泛型（无 `Box<dyn ...>`，`A3`）；`collectingsink` 是 crate 内记录实现。
7. **`kvstate` 实现 `models` 缓存契约；`infer` 拥有缓冲。** `kvstate` 每层持有一个 key 与一个 value 张量，各为 `[cached_len, width]`，通过 `append`/`appendall` 锁步推进每层，并用 `clear` 重置。它在每次 append 校验 `[1, width]` 行形状、层索引与容量，且由 `generationrequest` 创建、拥有、重置。请求作用域缓存正是 `models` 只拥有契约而 `infer` 拥有实现的原因。
8. **`reset` 使每个终止路径可复用。** 在完成、取消或失败的运行后，`reset` 把上下文恢复到 prompt，回绕采样器并清空缓存，故无残留状态可被错误复用（`OW3`）。所有资源由 `Drop` 释放；无手工释放路径。
9. **`#![forbid(unsafe_code)]`、一个自有错误枚举、无 panic。** 每个可失败路径返回 `infererror`，它透传包装 `modelerror` 与 `contexterror`（`ER2`），并新增 `invalidrequest`、`invalidparams`、`tokenoutofrange`、`emptylogits`、`degeneratelogits`。生产代码不含 `panic`、`unwrap`、`expect`、`todo` 或 `unimplemented`。

### 16.3 接口边界（DS1）

crate 根导出十二个公共类型（`generationengine`、`infererror`、`generationresult`、`kvstate`、`generationparams`、`generationrequest`、`sampler`、`samplingparams`、`stopcondition`、`stopreason`、`collectingsink`、`streamsink`），无常量或自由函数——低于 `A4` 上界 20。公共类型/trait/enum/变体/函数名是小写连接；常量若有大写。模块文件名不同于其导出的类型，因为模块与重导出类型共享一个名字空间（`E0255` 守卫）：`engine`→`generationengine`、`error`→`infererror`、`generation`→`generationresult`、`kvcache`→`kvstate`、`request`→`generationparams`/`generationrequest`、`sampling`→`sampler`/`samplingparams`、`stopping`→`stopcondition`/`stopreason`、`stream`→`collectingsink`/`streamsink`。`prefill` 与 `decode` 只暴露 crate 私有函数。

### 16.4 不变量与错误路径（DS2）

算法级不变量：

* 一个 `generationresult` 恰好按序拥有累积的生成 token，故 `gettokens().len() == getcount()` 且其中不出现 prompt token；对于空 EOS、上限 `N` 的 greedy 请求，引擎的 token 等于对增长上下文做 `N` 次独立 `argmax` 步；采样器是 `(params, seed, logits sequence)` 的纯函数，故相同输入复现相同 token。
* 实现级：由 `new` 创建（或由 `reset` 恢复）的 `generationrequest` 有 `context == prompt`、`getgeneratedlen() == 0`、长度为 0 的缓存，以及处于播种起点的采样器；上下文绝不超过 `maxseq`，因为构造拒绝 `prompt.len() > maxseq` 或 `prompt.len() + maxnewtokens > maxseq`，且 `pushgenerated` 在 `context.len() >= maxseq` 后拒绝；存在的 `samplingparams` 有有限非负温度、`topk != Some(0)`、`topp ∈ (0, 1]`；存在的 `stopcondition` 有 `maxnewtokens != 0`；存在的 `kvstate` 有非零 layers/width/maxseq，并拒绝任何非 `[1, width]` 行或会超容量的 append；模型在整个调用期间被不可变借用，故无请求状态可经它被观察。

错误分支：空 prompt；prompt 长于 `maxseq`；prompt 加 cap 长于 `maxseq`；零 `maxseq`/`kvlayers`/`kvwidth`；零 `maxnewtokens`；非有限或负温度；`Some(0)` top-k；top-p 在 `(0, 1]` 之外；空 logits；非有限 logits；无有限质量的分布；越界的 prompt 或生成 token（作为 `modelerror` 经 `infererror::model` 浮现）；缓存层越界、坏行形状或容量溢出；以及取消或步骤预算耗尽（`infererror::context`）。资源回滚：被拒的构造器在请求值存在前返回；`reset` 在任何终止路径后恢复新状态；部分运行将其全部状态留在请求与 sink 中，绝不留在模型中。

### 16.5 推理契约

**请求生命周期。** `generationrequest::new(prompt, params)` 校验 prompt 非空、不超过 `maxseq`、且 prompt 加 `maxnewtokens` 不超过 `maxseq`，并构造请求拥有的 `kvstate`。`engine.generate(exec, request, sink)` 先 `request.prepare(model.getconfig())`（重置并对照词汇表校验 prompt 与 EOS token），再做一次 `exec.getcancellation().checkpoint()`，然后 prefill。循环中每步 `exec.checkpoint()`、采样、判 EOS、追加、交给 sink、判上限、decode。正常结束时 `sink.onstop(reason)` 并返回 `generationresult`；失败或取消返回 `infererror` 且不调用 `onstop`。

**prefill/decode。** `prefillprompt` 对 prompt 做一次前向并返回最后位置的下一个 token logits；`decodestep` 对累积上下文做一次前向。二者都归约为 `nextlogits`（`causallm::forward` 加 `lastrow`）。正确性路径在每步重算完整上下文；命名分开的两个阶段使缓存式 decode 能替换其中之一而不改变另一个。

**KV cache 所有权与隔离。** `kvstate` 实现 `models::kvcache`：每层一个 key 与一个 value 张量，形状 `[cached_len, width]`，`append`/`appendall` 锁步推进每层，`clear` 重置。每次 append 校验 `[1, width]` 行形状、层索引与容量，越界 token 或索引不能损坏缓冲。它由 `generationrequest` 创建、拥有、重置，绝不存入共享模型权重（`OW2`）。正确性路径重算完整上下文，尚未读写此缓存；它存在的目的是让缓存式 decode 能在当前全重算基线之上加入而不改变所有权模型。

**采样。** 温度为 `0.0` 时 `greedy` 返回最大有限 logit 的索引，平局取最低索引。否则 `samplecategorical` 用 `f64` 把 logits 除以温度、做 max 减法求稳定 softmax，按降序概率稳定排序，`keptcount` 应用 top-k 与 top-p（top-p 取降序概率累积质量首次达到阈值的最小前缀，至少保留 1），把其余概率置零后按 SplitMix64 生成的单位变量做累积抽样。`samplingparams::new` 在构造时一次性校验温度有限非负、`topk != Some(0)`、`topp ∈ (0, 1]`。

**停止。** `stopcondition::new(eostokens, maxnewtokens)` 拒绝零 `maxnewtokens`。`stopreason` 有三个变体：`endoftoken`（采样到配置的 EOS token）、`maxlength`（达到新 token 上限）、`stopped`（sink 的 `ongenerated` 返回 `false`）。检查顺序为：EOS → 追加并交给 sink（可能 `stopped`）→ 达到上限（`maxlength`）→ 否则 decode。

**流式错误语义。** `streamsink::ongenerated(index, token)` 返回 `true` 继续 / `false` 停止；`onstop(reason)` 仅在正常终止时调用一次。失败或取消的生成返回 `infererror` 且从不调用 `onstop`，故流绝不静默地终止以暗示成功。

**取消。** 取消经调用方拥有的 `executioncontext` 句柄观察，在每个 decode 步（循环起点的 `exec.checkpoint()`）检查。取消或步骤预算耗尽在下一次前向之前停止循环并返回 `infererror::context`，而非成功结果。`reset` 在成功、取消或失败后恢复上下文到 prompt、回绕采样器并清空缓存。

### 16.6 复杂度（DS3）

每个函数都低于圈复杂度 10 与有效行数 80；最大者为 `sampler::samplecategorical`（若干守卫加对 logits 向量的单次排序/过滤）与 `generationengine::generate`（一个带三个出口的循环）。（经阅读复核，证据等级 T。）

### 16.7 并发设计（DS4）

`infer` 自身不引入共享可变状态或同步原语。`generationengine` 只借用不可变模型与提供者；`generationrequest` 拥有全部可变状态并经 `&mut` 传入，故两请求不能共享可变生成状态（`OW2`）。取消使用调用方拥有的 `context::cancellation`（`AtomicBool`，`Release`/`Acquire`），是唯一的共享信号且为显式同步原语而非全局状态（`A6`/`CC3`）。因此一个不可变模型与提供者可服务并发请求，且无锁、无锁序。

### 16.8 被否决方案（DS5）

* **把请求状态存入模型。** 否决：模型权重不可变且跨请求共享；可变请求数据不得进入其中（`OW2`）。
* **在 `models` 内实现缓存缓冲。** 否决：`models` 拥有缓存抽象，但缓冲是请求作用域可变状态；把它留在 `infer` 是所有权规则，并避免模型 crate 依赖请求状态。
* **把 `Box<dyn streamsink>` 存入请求。** 否决：sink 是调用作用域的泛型借用（`S: streamsink`），故无动态分发，也无请求类型中的生命周期耦合（`A3`）。
* **在后端经 `api::compute` 采样。** 否决：采样是对小 logits 向量的宿主逻辑；留在 `infer` 避免拓宽 compute 契约并把可复现种子集中在一处（`lib::seed`）。
* **预分配缓存并信任调用方形状。** 否决：`kvstate` 在每次 append 校验形状、层与容量，故坏 token 或索引不能损坏缓冲（`IN1`/`IN2`）。
* **现在就把 KV cache 接进 `forward`。** 否决：尚无缓存感知的 forward 入口，也无测量基线，故无法诚实地声称加速（`PF1`/`PF5`）。全重算路径是正确性基线；缓存路径是独立的、被测量的阶段。
* **每步用 insert/splice 重建上下文。** 否决：`Vec::push` 摊销 O(1)，模型按引用接收上下文，故循环不执行冗余拷贝。
* **`maxnewtokens == 0` 的 no-op 请求。** 否决：无法发出 token 的请求是接线错误，在构造时被拒。

### 16.9 后果与声明局限

* 一个不可变模型与提供者可服务并发请求：所有可变状态都在 `generationrequest` 与 sink 中，二者均为请求作用域。
* 每个 decode 步都重算完整上下文。这是缓存式 decode 必须超越的基线；缓存契约及其 `infer` 实现已就位以待测量。
* `generationresult` 是自包含值，在请求被重置或丢弃后仍然有效。
* 调用方看到一个 `infererror` 枚举，保留可操作的区分（模型失败、执行控制失败、无效请求、无效采样参数、越界 token、不可采样 logits），同时无损携带底层 `modelerror`/`contexterror`（`ER2`）。

与计划的偏差与声明局限：

* **停止条件模块是 `stopping.rs` 而非 `stopcondition.rs`。** 计划文件列表名为 `stopcondition.rs`，但 crate 重导出名为 `stopcondition` 的类型，模块与重导出类型共享一个名字空间，故该文件会冲突（`E0255`）。同样的理由命名 `error.rs`（非 `infererror.rs`）与 `stream.rs`（非 `streamsink.rs`）。子句：`NM2` / `E0255` 守卫。影响：公共接口无影响。
* **集成测试最初是 crate 本地的。** 计划名为根 `tests/generation_greedy.rs` 与 `tests/request_isolation.rs`；此阶段把 `crates/infer/tests/generation.rs` 与 `crates/infer/tests/isolation.rs`（含 `tests/common/mod.rs`）放在 crate 本地，因为任务把该阶段限定为 crate 本地测试，而后续整合拥有根 `tests/` 与 `benches/`。后续整合已把它们移到根 `tests/generation.rs` 与 `tests/isolation.rs`，共享 `tests/common/mod.rs`。
* **本阶段无基线基准。** 计划 P7b 要求缓存基线基准；根 `benches/` 由后续阶段拥有且本阶段不作性能声明，故未加基准；全重算路径是缓存式 decode 将被对照的基线。子句：`PF1`/`PF5`，显式指令。影响：缓存路径暂无前后对比数字。
* **缓存已实现但尚未被前向读取。** `kvstate` 满足 `models::kvcache` 契约并由请求拥有、重置，但 `causallm::forward` 无缓存参数，故 decode 重算完整上下文。这与 `models` 的偏差一致；缓存感知的 decode 是独立的、被测量的阶段。子句：`VF1`/`PF1`。影响：无缓存式 decode，未声称加速。

### 16.10 验证

于 2026-10-10 执行（仅限范围命令，因为兄弟 `optim` crate 在此阶段构建中）：

* `cargo check -p infer --all-targets`——干净，exit 0。
* `cargo clippy -p infer --all-targets -- -D warnings`——干净，exit 0。
* `cargo fmt -p infer -- --check`——干净，exit 0。
* 仓库根 `tests/inferunits.rs`、`tests/generation.rs`、`tests/isolation.rs`、`tests/pipeline.rs` 与 `tests/cancellation.rs`（对应 `infer`）—— 0 失败；0 doc-tests。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
* `RUSTDOCFLAGS='-D warnings' cargo doc -p infer --no-deps`——干净，exit 0。

中心正确性测试 `generation::greedy_generation_matches_an_independent_recomputation` 用仅依赖公共模型 API（`causallm::forward` 加 argmax）的独立重算交叉核对引擎的 greedy token。其余测试钉住跨请求确定性、EOS 与最大长度停止、要求停止的 sink、首次前向之前的取消与生成中取消及 `reset` 后复用、词表外拒绝、独立请求状态与缓存隔离，以及共享模型上并发生成匹配串行结果。不声称加速（`VF5`）。

环境：`rustc 1.99.0 (b940084d7 2026-09-28)`、macOS 27.0.1、arm64（Apple Silicon）。所有命令均以 `-p infer` 限定运行。

### 16.11 复审条件

当以下情况发生时重新考虑：存在缓存感知的 forward 入口（则把 `kvstate` 接进 decode 并对照全重算路径记录前后基线）；`format`/`checkpoint` 提供序列化模型（则加入 load 路径与黄金 token 生成测试）；需要新的采样规则（则扩展 `samplingparams` 并配参考测试，而非改动 greedy 路径）；流式 sink 需要拥有模型或跨调用存活（则复审调用作用域的 `S: streamsink` 约束）；或根 `tests/`/`benches/` 整合发生（则移动集成测试到位并加入生成基准）。

## 17. 层 4：cli

### 17.1 上下文

层 0–3 已拥有 `cli` 所需的可复用工作流。`format` 解析并结构校验 `config.json`（`modelconfig`）与 safetensors 头（`safetensorfile`）；`checkpoint` 拥有 checkpoint 容器、其 manifest 与 `decode`；`models` 拥有 Llama 配置/权重与 `architecture` 注册表；`infer` 拥有 `generationengine`、`generationrequest`、`generationparams`、`samplingparams`、`stopcondition`、`collectingsink` 与 `infererror`；`train` 拥有 `trainconfig` 与 `trainer` 编排循环；`optim` 拥有优化器配置与学习率调度；`tokenizer` 拥有 HuggingFace BPE 加载与编码；`context` 拥有 `executionconfig`/`executioncontext`；`cpu` 是具体的 `api::compute` 提供者；`lib` 拥有 `limits`、`seed`、`requestid` 与 `tokenid`。

`cli` 是把 `argv` 变为对这些工作流的调用、并把结果映射到 stdout/stderr 与进程退出码的薄层 4 应用。它不添加张量数学、模型内部、训练循环或推理引擎（`A1` 归属唯一：每个能力只归属一个 crate）。

**声明局限：nn/models 无反向路径。** 在写本 crate 前已重新核实工作区。`nn`/`models` 是 forward-only，`autograd` 没有 Llama 或交叉熵反向路径。因此端到端 `cli train` 微调 Llama **不可达**。`cli` 不捏造它：`train` 加载并校验 `trainconfig`（委托 `train::trainconfig` 与 `optim`），任何真正更新模型权重的请求返回独特的非零 `clierror::notsupported`（M4/VF5）。`train --check-only` 暴露今天真正运行的部分（配置加载 + 校验）并成功。

同样，推理需要真实模型与权重。`generate` 不要求一个并不存在的工件，而是用固定配置与显式 `seed` 构建确定性的微小 Llama，然后在真实 `cpu` 提供者上驱动**真实** `infer` 引擎。`--text` prompt 由真实 `tokenizer` 编码；缺失或无效的 tokenizer 文件被精确报告。

本决策新增的依赖边（无环）：`cli → cpu`、`cli → checkpoint`、`cli → context`、`cli → format`、`cli → infer`、`cli → lib`、`cli → models`、`cli → optim`、`cli → tokenizer`、`cli → train`。`cli` 是层 4 应用：它可以依赖工作流 crate 与具体 `cpu` 后端。它不是任何层 0–3 crate 的依赖。

### 17.2 决策

1. **`cli` 是薄适配器，不是引擎。** 每个命令解析参数、选择配置、调用一个真实库工作流并渲染报告。crate 不含张量运算、前向/反向、训练循环或采样算法。工作流无法运行处（端到端 Llama 训练），命令返回类型化错误而非合成成功。
2. **命令树可在进程内测试。** 包是库加一行二进制。`src/lib.rs` 暴露 `parse`（`clap` `try_parse_from` 包装）、`run`（分发，返回 crate 自有的 `commandoutput` 或 `clierror`）、`exitcode`（单一错误到码映射）与 `runcli`（完整 `argv` 到退出码适配器，写入调用方提供的 `Write` 接收器）。`src/main.rs` 只锁定 `stdout`/`stderr` 并调用 `runcli`，故参数解析、分发、输出与退出码都在不 spawn 进程的情况下被覆盖。
3. **退出码显式且集中定义。** 一个函数 `clierror::exitcode` 把每个错误变体映射到码，usage 码是命名常量 `USAGEEXITCODE`。表：

| 码 | 常量 | 含义 |
| ---- | -------- | ------- |
| 0 | （成功） | 命令完成；报告写入 stdout。 |
| 1 | `OPERATIONALEXITCODE` | 运营失败：I/O、无效路径、超大工件、未识别工件、模型/推理/训练/tokenizer/优化器/context 失败。 |
| 2 | `USAGEEXITCODE` | 用法：`clap` 解析诊断、语义无效参数值（`invalidarguments`）、或畸形训练配置文档（`invalidconfigjson`）。 |
| 3 | `VALIDATIONEXITCODE` | 目标工件结构无效（`formaterror`、`checkpointerror`），由 `inspect`/`validate` 报告。 |
| 4 | `UNSUPPORTEDEXITCODE` | 已声明的结构性局限（`notsupported`）：端到端 Llama 训练尚无反向路径。 |

`--help`/`--version` 不是错误：`runcli` 把它们写到 stdout 并退出 `0`，与 `clap` 自身约定一致。

4. **每个外部输入在使用前校验（IN1/IN2/IN4）。** 路径用 `std::fs::metadata` 检查为存在常规文件，长度在读取前对照 `DEFAULTMAXARTIFACTBYTES` 检查，并在读取后复核读取长度（复制-校验-使用，IN3）。token 列表受 `MAXPROMPTTOKENS` 限制并使用受检转换解析；`--maxnewtokens` 受 `MAXNEWTOKENSLIMIT` 限制；`--vocab` 必须非零；模型词表外的 prompt 或 EOS token 在模型运行前作为用法错误被拒。safetensors/CPU 的元素、字节、序列与批预算为显式常量 `DEFAULTLIMITS`，执行步骤预算由 `--maxnewtokens` 派生（从不无界）。
5. **`inspect` 与 `validate` 委托工件解析。** 首个非空白字节选择解析器：`{` → `modelconfig::parse`；恰好的 checkpoint magic `LLMCKPT1` → `checkpoint::decode`；否则 `safetensorfile::parse`。`inspect` 打印人类可读摘要；`validate` 打印成功摘要或原样返回结构化 `formaterror`/`checkpointerror`，故底层区分不被折叠（`ER2`）。
6. **`generate` 在确定性微小模型上运行真实引擎。** 配置经 `llamaconfig::new` 构建，权重经 `llamaweights::random(&config, seed)`，模型经 `getarchitecture("llama")?.newmodel(...)`；请求使用 `stopcondition`/`samplingparams`/`generationparams::formodel`、请求作用域 `generationrequest`、调用方拥有的 `executioncontext`、`collectingsink`，并以 `cpuexecutor` 作为 `api::compute` 提供者。所有可变状态都是请求作用域且 drop 时释放（`OW2`/`OW3`）。
7. **`train` 委托配置并拒绝伪造训练运行。** CLI 自有的 `trainconfigfile` JSON 用 `serde` 解析，再经 `trainconfig::new` 与真实优化器值（`sgd::new`/`adamw::new` 经校验构造器构造的 `sgdconfig`/`adamwconfig`）校验。带 `--check-only` 时报告经校验的计划并 exit `0`；不带时命令返回 `clierror::notsupported` 并 exit `4`。
8. **一个自有错误枚举；生产代码无 panic。** `#![forbid(unsafe_code)]`；`clierror` 透传携带每个工作流错误（`#[from]`）并拥有仅适配器变体（`io`、`invalidpath`、`sizeexceeded`、`unrecognizedartifact`、`invalidarguments`、`invalidconfigjson`、`notsupported`）。生产代码不含 `panic`、`unwrap`、`expect`、`todo` 或 `unimplemented`；测试模块可允许 `unwrap`/`expect`。一个刻意丢弃的写（对 stderr 的尽力诊断）以 `let _ = ...` 与理由注释标注（`ER5`）。

### 17.3 接口边界（DS1）

crate 根导出重导出的 clap 类型（`cli`、`command`、`inspectargs`、`validateargs`、`generateargs`、`trainargs`）、`clierror`、`commandoutput`，以及四个函数 `parse`、`run`、`exitcode`、`runcli`，加五个退出码常量——远低于 `A4` 上界 20。公共类型/函数名为小写连接；常量为大写连接。模块文件名不同于其导出的类型，因为模块与重导出类型共享一个名字空间（`E0255` 守卫）：`args` 导出 `cli`/`command`，`error` 导出 `clierror`，复数模块 `commands` 不导出类型，故 `command`（`args` 中的类型）绝不与模块冲突。

### 17.4 不变量与错误路径（DS2）

算法级：

* `inspect`/`validate` 仅按字节分类工件，故相同字节总到达相同解析器；`validate` 恰在底层解析器接受字节时返回成功；固定 `--seed` 与固定参数的 `generate` 跨运行返回相同 token；`train` 在无 `--check-only` 时永不报告成功；`exitcode` 对 `clierror` 是全函数。

实现级：

* `commandoutput` 只持渲染后文本；`clierror` 值总携带足以打印的上下文；每个 `clierror` 变体可从至少一个命令路径或参数解析器到达。

错误分支：用法（clap、坏 token 列表、越界 token、坏配置 JSON），工件校验（`formaterror`、`checkpointerror`），路径/读取（`io`、`invalidpath`、`sizeexceeded`、`unrecognizedartifact`），模型与生成（`modelerror`、`infererror`、`contexterror`），tokenization，以及已声明的 `notsupported` 局限。资源回滚：文件读入有界缓冲并 drop；大于界的文件在读取前被拒；生成状态活在请求与 context 中并在 drop 时释放。

### 17.5 复杂度（DS3）

每个函数都低于圈复杂度 10 与有效行数 80；最大者为各命令的 `run` 函数（一串短的经校验调用）与 `runcli`（单 `match` 两臂）。（经阅读复核，证据等级 T。）

### 17.6 并发设计（DS4）

`cli` 不引入共享可变状态或同步原语。它是一个进程内单线程适配器：一个进程一次处理一个命令。命令选择的执行控制（`executioncontext`、步骤/内存预算）是 `context` 提供的显式同步与资源原语，而不是 CLI 自有的状态。故无锁、无锁序、无跨请求共享。

### 17.7 被否决方案（DS5）

* **`cli` 只依赖后端无关提供者且无法运行。** 否决：二进制必须选择具体 `api::compute` 提供者以执行 `generate`；工作区自己的 `cpu` crate 是最小具体选择且不增加外部依赖。`serve` 是泛型的，因为它是被交给提供者的库；`cli` 是选择提供者的组合根。
* **用动态分发或插件/提供者注册表选择后端。** 否决：恰有一个受支持后端且无运行期可扩展性需求，故使用 `match`/直接构造（`A3` 静态分发优先）。
* **重实现训练循环以使 `train`「成功」。** 否决：那会在应用层复制训练引擎，并伪造结果；缺失的反向路径被报告。
* **要求用户为 `generate` 提供真实模型 + safetensors。** 作为默认否决：那会使命令在没有工件时不可用。确定性微小模型驱动同一真实引擎；`--tokenizer`/`--text` 路径仍行使真实 tokenization。
* **`validate` 返回 `0` 并打印布尔结果。** 否决：失败校验是独特的、机器可观察的结果；进程 exit `3` 并打印结构化错误。
* **把所有失败折叠进退出码 1。** 否决：调用方必须区分用法错误（`2`）、无效工件（`3`）与已声明局限（`4`），且映射集中故不会逐命令漂移。

### 17.8 后果与与计划的偏差

* 命令树、报告与退出码在进程内被行使；二进制本身只是对 `runcli` 的一次调用。
* `inspect`/`validate` 不会与库解析器漂移，因为它们直接调用它们并原样浮现其错误。
* `generate` 限于微小确定性模型；把它指向真实 checkpoint/权重是复用同一引擎接缝的未来工作。
* `train` 对缺失的反向路径诚实：配置工作流运行、权重更新工作流不运行，进程退出码如实说明。

与计划的偏差：

* **计划的 `src/command/` 目录是 `src/commands/`。** 子命令枚举是名为 `command` 的类型；把模块也命名 `command` 会把模块与重导出类型放进一个名字空间（`E0255`）。子句：`NM2` / `E0255` 守卫。影响：公共接口无影响。
* **`src/main.rs` 是对 `cli::runcli` 的一个表达式。** 计划列出 `main.rs`；此阶段把所有逻辑留在库中，使退出码映射与输出可单元测试，`main.rs` 只获取标准流。子句：显式任务指令（薄 `main`）。影响：无。
* **集成测试最初在 `crates/cli/tests/`。** 后续整合拥有根 `tests/` 与 `benches/`，故此阶段不触碰它们。后续整合已把它们移到根 `tests/cli.rs`。
* **无基准、无性能声明。** `cli` 自身不做热路径工作；所有计算被委托，且不声称加速（`VF5`）。子句：`PF1`/`PF5`。影响：无。
* **`train` 尚不支持学习率调度文件字段。** `trainconfig` 接受可选 `lrschedule`，但 CLI schema 为保持配置面最小而省略它；计划不要求它。子句：显式任务指令 / `A4`。影响：尚不能从 CLI 选择调度。

### 17.9 验证

于 2026-10-10 对工作区根以 `-p cli` 执行：

* `cargo fmt --all`——干净，exit 0。
* `cargo check -p cli --all-targets`——干净，exit 0。
* `cargo clippy -p cli --all-targets -- -D warnings`——干净，exit 0。
* 仓库根 `tests/cliunits.rs` 与 `tests/cli.rs`（对应 `cli`）—— 0 失败（doc-tests：0）。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
* `RUSTDOCFLAGS='-D warnings' cargo doc -p cli --no-deps`——干净，exit 0。
* 二进制冒烟经 `cargo run -q -p cli -- ...`：`--help` exit 0；`generate --tokens 1,2 --maxnewtokens 2` exit 0 且两次运行字节相同（`shasum d789e15910d58b271348fb070c0e77c3d5d79354`），打印 `generated: [3, 5]`、`stopreason: maxlength`、`steps: 2`；`train --config /nonexistent.json` exit 1 并给出 `i/o error` 诊断（E1）。

中心集成测试 `generate_runs_the_real_engine_deterministically` 经 `runcli` 两次运行命令并断言输出相同，证明微小模型与引擎确定。其余测试钉住工件分类与报告（对真实 `config.json` 与对用 `tensor` 开发依赖构建的真实编码 checkpoint 运行 `inspect`/`validate`）、退出码常量、token 解析界、`--text` 需要 `--tokenizer` 的规则、`train --check-only` 成功路径，以及已声明的 `notsupported` 局限（exit 4）。不声称加速（`VF5`）。

环境：`rustc 1.99.0 (b940084d7 2026-09-28)`、macOS 27.0.1、arm64（Apple Silicon）。

### 17.10 复审条件

当以下情况发生时重新考虑：`nn`/`models` 的反向路径落地（则 `train` 在同一命令后长出真实权重更新路径，本节的局限被移除）；`generate` 的真实工件加载器落地（则加入 `--checkpoint`/`--weights` 并重新测量）；根 `tests/`/`benches/` 整合发生（则移动集成测试到位）；或加入第二个具体后端（则复审后端选择决策）。

## 18. 层 4：serve

### 18.1 上下文

层 3 已拥有可复用的生成工作流：`infer` 提供 `generationengine`、`generationrequest`、`generationparams`、`samplingparams`、`stopcondition`、`streamsink`/`collectingsink` 与 `infererror`，`context` 提供 `executioncontext`、`cancellation`、`executionconfig` 与 `metrics`。`serve` 是把 HTTP 请求变为该工作流、并把 HTTP 响应变回去的薄层 4 应用。它不添加模型逻辑，也不拥有自己的请求作用域生成状态。

`serve` 依赖 `models`（`causallm` 契约）、`api`（`compute` 提供者契约）、`context`（执行控制）、`infer`（工作流）与 `lib`（标识符与种子）。`cpu` 是仅由集成测试使用的开发依赖。它不依赖具体后端，也不是任何层 0–3 crate 的依赖。

本决策新增的依赖边（无环）：`serve → infer`、`serve → models`、`serve → api`、`serve → context`、`serve → lib`；既有边 `infer → models`、`infer → api`、`infer → context`、`infer → lib`、`models → api`、`models → lib`、`api → lib`、`context → lib`。

### 18.2 决策

1. **HTTP 栈是最小、仅标准库的 HTTP/1.1 适配器。** 服务器构建于 `std::net::TcpListener`/`TcpStream` 与 `std::io::BufRead`/`Read`/`Write`，带有界请求解析器与阻塞的每连接一线程 accept 循环（`std::thread::scope`）。不使用异步运行时、HTTP 框架或 TLS。这保持层 4 应用的依赖闭包最小，使整个适配器无需 socket 即可单元测试，并尊重「静态组合优于不必要间接」的偏好（`A3`）。被否决的 `tokio` 方案记录在 DS5。
2. **传输与应用之间由一个值类型分隔。** `httprequest` 是传输校验过的请求（方法、目标、小写 header、原始 body），`httpresponse` 是传输中立的响应（status、reason、content type、拥有的 body）。`inferenceservice` 把一个 `httprequest` 映射为一个 `httpresponse` 且从不触碰 socket，故路由、校验、映射、准入与取消都可在不开端口的情况下测试。
3. **传输错误与模型错误保持区分。** `serveerror` 是 crate 唯一的自有错误枚举。传输/协议失败（畸形请求行或 header、不支持版本/方法、超大行/header/body、无效 content length、不支持媒体类型、字段太多、缺失/无效字段、过载、未找到、分发失败、I/O）是不同变体；模型执行失败作为 `serveerror::inference(#[from] infererror)` 透传携带（`ER2`）。传输失败绝不报告为模型失败，反之亦然。从 `serveerror` 到 HTTP status 的单一映射在 `httpresponse::fromerror`，把协议关注留在应用边界。
4. **每个外部输入在到达模型前被校验。** 请求解析器对请求行、header 行长度、header 数量、content-length 与 body 大小施加显式、调用方提供的 `transportlimits`（`IN1`/`IN2`/`IN4`）；表单解析限定字段数量与每个字段值长度；prompt 与 EOS token 列表受 `maxprompttokens` 限制；`maxnewtokens` 非零且受 `maxnewtokens` 限制；数值字段在溢出或非数值输入时返回字段错误而非 panic（`IN1`/`IN2`）。服务器额外用 `Read::take(maxrequestbytes)` 限制每连接读取的总字节。
5. **准入控制在昂贵工作前限定并发。** `admissions` 是对固定 `maxconcurrent` 上限的原子 compare-and-update 计数器。`inferenceservice::generate` 在构造生成请求或运行模型前取一个 `admission` RAII 守卫，故超过上限的请求以 HTTP 429 快速失败且不预留任何东西。守卫在每个终止路径于 drop 释放准入（`OW1`），零上限在配置时作为接线错误被拒。
6. **取消在传输边界映射并由请求拥有。** `inferenceservice::handlewith` 在准入成功后、模型运行前立即用请求的 `context::cancellation` 句柄调用传输的 `onstart` 回调。当配置了 deadline 时，服务器在作用域工作者上运行请求并用 `mpsc::Receiver::recv_timeout` 等待；超时则经该句柄取消并返回 HTTP 504，而工作者在其下一个引擎检查点观察到取消并终止（`infererror::context(cancelled)`）。在首个检查点之前请求的取消映射到 HTTP 408。没有终止路径留下可复用的残留状态：所有可变状态都活在每请求的 `executioncontext`/`generationrequest`/sink 中并在 drop 时释放（`OW2`、`OW3`、`CC5`）。
7. **请求隔离且服务可共享。** `inferenceservice` 借用不可变模型与提供者，仅拥有不可变 `serviceconfig` 加一个原子计数器，故一个服务值服务并发请求而不共享可变生成状态（`OW2`）。每个响应以 `connection: close` 写出，故服务器不在请求之间保持 keep-alive 或会话状态。
8. **`#![forbid(unsafe_code)]`、一个自有错误枚举、无 panic。** 生产代码不含 `panic`、`unwrap`、`expect`、`todo` 或 `unimplemented`；集成测试（可以合法地大声失败）携带 `#![allow(clippy::unwrap_used, clippy::expect_used)]`。I/O 失败只保留其 `ErrorKind`，故错误响应绝不暴露内部文件系统或 socket 细节。

### 18.3 接口边界（DS1）

crate 根导出十个公共类型（`admission`、`admissions`、`serveerror`、`httpmethod`、`httprequest`、`transportlimits`、`httpresponse`、`inferenceservice`、`serviceconfig`、`httpserver`），无常量或自由函数——低于 `A4` 上界 20。公共类型/trait/enum/变体/函数名为小写连接；常量若有大写。模块文件名不同于其导出的类型，因为模块与重导出类型共享一个名字空间（`E0255` 守卫）：`admit`→`admission`/`admissions`、`error`→`serveerror`、`request`→`httpmethod`/`httprequest`/`transportlimits`、`response`→`httpresponse`、`routes`→`inferenceservice`/`serviceconfig`、`server`→`httpserver`。

### 18.4 不变量与错误路径（DS2）

算法级：

* 一个 `httprequest` 恰好映射为一个 `httpresponse`；`/generate` 路由绝不在未持有 `admission` 时运行模型；成功的 `/generate` 响应体恰好编码等价 `infer` 调用下引擎的生成 token、停止原因与步数；每个请求终止（成功、错误或取消）后 in-flight 准入计数为 `0`。

实现级：

* 存在的 `httprequest` 是在 `transportlimits` 内被解析的且其 body 长度等于声明的 content-length；存在的 `httpresponse` 有落在 HTTP 范围内的 status 且 body 长度匹配发出的 `content-length`；`admission` 守卫仅在控制器计数低于上限时存在并在 drop 时恰好减一；`serviceconfig` 拒绝零 `maxconcurrent`；模型与提供者在整个服务生命周期被不可变借用，故无请求状态可经它们被观察。

错误分支：畸形/超大请求行或 header；header 太多；不支持 HTTP 版本或方法；不支持媒体类型；无效 content-length；超大或不完整 body；表单字段太多或过大；缺失或无效的 `prompt`/`maxnewtokens`/`eos`/采样字段；未知路由（404）；达到准入上限（429）；配置错误；分发工作者失败；I/O 失败；deadline 到期（504）；以及每个原样携带的 `infererror`。资源回滚：body 与 header 缓冲有界且每连接；准入守卫在 drop 释放；校验或准入失败时服务在模型运行前返回；作用域分发工作者总会终止，因为取消经请求句柄投递。

### 18.5 复杂度（DS3）

每个函数都低于圈复杂度 10 与有效行数 80；最大者为 `httprequest::parse`（一串短的有界读取器）与 `inferenceservice::handlewith`（至多两臂的单 `match`）。（经阅读复核，证据等级 T。）

### 18.6 并发设计（DS4）

`serve` 的共享状态是 `admissions` 的 `AtomicUsize`（用 `try_update(AcqRel, Acquire)`，drop 用 `fetch_sub(AcqRel)`）与 `inferenceservice` 的 `AtomicU64` 请求 id（`fetch_add(Relaxed)`）。这是每个资源单一同步原语，无嵌套、无锁序，故无死锁环（`CC1`）。取消可见性由 `context::cancellation` 的 `Release`/`Acquire` 原子对建立（`CC3`）。请求并发由每连接一线程（`std::thread::scope`）承载，可变生成状态严格请求作用域，故一个不可变模型与提供者可被并发共享（`OW2`）。deadline 路径用 `mpsc::Receiver::recv_timeout` 等待作用域工作者，并在超时经请求句柄投递取消；工作者在下一检查点退出。

### 18.7 被否决方案（DS5）

* **基于 `tokio` 的异步 HTTP 服务器。** 否决：会给一个并发上限已由原子准入计数器限定的层 4 适配器加入异步运行时及其传递依赖闭包；生成工作流是同步且 CPU 密集的，故异步 I/O 增加间接而无测量收益（`A3` 静态分发优先、依赖最小化、`PF1` 先测量后优化）。工作区声明了 `tokio`，但 `serve` 只用 `std`。
* **现有 HTTP 框架（hyper/axum/warp）。** 否决：框架加其运行时的依赖面远大于有界 HTTP/1.1 解析器，而本应用所需的协议面只有两条路由。
* **把传输错误与模型错误折叠进单一 status 映射。** 否决：客户端必须能区分协议错误（400/413/415）与模型失败（400/500），`serveerror::inference` 保持 `infererror` 区分完好（`ER2`）。
* **接受无界输入并依赖模型失败。** 否决：资源上限必须在昂贵工作前强制（`IN4`）；解析在与其成正比的分配之前拒绝超大输入。
* **把请求的取消令牌存入服务或全局。** 否决：取消是请求作用域状态，属于 `executioncontext` 而非共享配置（`OW2`、`A6`）。
* **keep-alive 会话与连接状态机。** 否决：工作流是请求作用域且无状态，故每响应 `connection: close` 避免无收益的会话生命周期。
* **从 accept 线程直接服务模型且无准入限制。** 否决：无界并发会破坏执行上下文的资源控制；准入使上限显式且可观察（`IN4`）。

### 18.8 后果与与计划的偏差

* 应用边界是一小的阻塞 HTTP/1.1 适配器；超出单进程的扩展依赖部署层，而非 `serve` 内部的运行期并发。
* 所有路由、校验、映射、准入与取消行为都可在无需 socket 的情况下被单元与集成测试，真实服务器在临时回环端口上被行使。
* 一个不可变模型与提供者服务并发请求：所有可变状态都是请求作用域且 drop 时释放。
* 客户端对每类失败收到一个由 `serveerror` 派生的 status，而模型失败保留其 `infererror` 区分；无错误响应泄漏 I/O 内部。

与计划的偏差：

* **传输是仅标准库而非基于 `tokio`。** 计划提供异步栈或更小的 std-only 适配器；此阶段选择更小的 std-only 适配器并记录被否决的 `tokio` 方案（DS5）。子句：显式任务指令。影响：无异步 I/O，无 TLS；不支持 keep-alive。
* **准入模块是 `admit.rs` 而非 `admission.rs`。** crate 重导出名为 `admission` 的类型，模块与重导出类型共享一个名字空间，故该文件会冲突（`E0255`）。同样的理由命名 `error.rs`（非 `serveerror.rs`）、`request.rs`（非 `httprequest.rs`）、`response.rs`（非 `httpresponse.rs`）、`routes.rs`（非 `inferenceservice.rs`）、`server.rs`（非 `httpserver.rs`）。子句：`NM2` / `E0255` 守卫。影响：公共接口无影响。
* **集成测试最初是 crate 本地的。** 因为后续整合拥有根 `tests/` 与 `benches/`，此阶段把 `crates/serve/tests/service.rs`、`crates/serve/tests/server.rs` 与 `crates/serve/tests/common/mod.rs` 放在 crate 下而非根 `tests/`。后续整合已把它们移到根 `tests/service.rs` 与 `tests/server.rs`，共享 `tests/common/mod.rs`。
* **本阶段无基准或性能声明。** 根 `benches/` 由后续阶段拥有且不作性能声明，故未加基准。子句：`PF1`/`PF5`，显式指令。影响：不声称吞吐或延迟数字（`VF5`）。

### 18.9 验证

于 2026-10-10 对工作区根以 `-p serve` 执行，在兄弟 `train` crate 的在建构建完成后进行。`train` 在建期间其 manifest 无 `src/lib.rs`，故 `cargo metadata` 对整个工作区的每个 `-p` 命令都失败（兄弟状态，非 `serve` 缺陷）；因此同样四条命令先对一个镜像 `serve` 及其依赖 crate 的临时虚拟工作区验证，然后在真实工作区加载后原样重复。

* `cargo fmt --all`——干净，exit 0。
* `cargo check -p serve --all-targets`——干净，exit 0。
* `cargo clippy -p serve --all-targets -- -D warnings`——干净，exit 0。
* 仓库根 `tests/serveunits.rs`、`tests/server.rs` 与 `tests/service.rs`（对应 `serve`，覆盖 `admit`、`error`、`request`、`response`、`routes`、`server`）—— 0 失败；0 doc-tests。（迁移前阶段记录：黑盒测试迁移已于 2026-10-10 完成，`crates/*/src` 现无内联单元测试，当前套件位于仓库根 `tests/`。）
* `RUSTDOCFLAGS='-D warnings' cargo doc -p serve --no-deps`——干净，exit 0。

中心集成测试 `service::generation_matches_the_engine` 用同一模型与参数上独立的 `infer` `generationengine` 运行交叉核对 HTTP 服务的生成 token 与停止原因。其余测试钉住路由（health、未知路由、错方法）、媒体类型与字段校验、越界 prompt 拒绝、超过上限的准入拒绝、传输取消映射（408）、deadline 映射（504）、并发请求隔离，以及临时回环端口上的真实服务器。不声称加速（`VF5`）。

环境：`rustc 1.99.0 (b940084d7 2026-09-28)`、macOS 27.0.1、arm64（Apple Silicon）。

### 18.10 复审条件

当以下情况发生时重新考虑：部署目标需要异步 I/O 或 TLS（则在相同 `inferenceservice` 边界后加入异步传输并对照本基线测量吞吐变化）；根 `tests/`/`benches/` 整合发生（则移动集成测试到位并加入请求吞吐基准）；需要流式响应（则扩展 `httpresponse`/sink 契约而非传输解析器）；或缓存感知的 `infer` decode 落地（则在不改变本边界的前提下重新测量请求延迟）。

## 19. 基础设施：testkit

### 19.1 上下文

`testkit` 是开发与测试基础设施。它 MUST NOT 成为生产依赖，也 MUST NOT 进入生产依赖闭包。它只依赖 `tensor`；仓库根 `tests/` 中的跨 crate 集成测试另外引入 `lib`、`api` 与 `cpu`，经公共接口行使整个栈。它提供三块窄能力，并保留一处待建位置：

* `fixtures`——构建确定性张量输入，使测试可复现。
* `assertions`——在文档化容差内比较数值结果。
* `reference`——独立标量实现，用于交叉核对优化过的 kernel。
* `tinymodel`——为微小确定性模型夹具保留。

### 19.2 决策

* `fixtures` 的每个构造函数固定形状与填充值，使测试逐字节可复现；形状被 `tensor` 拒绝时返回 `tensorerror` 而非 panic。提供：`sequential(rows, cols)`（行主序 `0,1,2,...`）、`ramp(n)`（秩一斜坡）、`constant(n, value)`、`matrix(rows, cols, value)`。
* `assertions` 提供 `assertallclose(actual, expected, abstol, reltol)`，逐索引满足 `|actual[i] - expected[i]| <= abstol + reltol * |expected[i]|`；相对项按 `|expected[i]|` 缩放，故近零值由 `abstol` 支配、大值由 `reltol` 支配。两个容差必须有限且非负。
* `reference` 是刻意用普通 `f32` 算术与标准库数学、独立于 `cpu` crate 编写的标量参考：`relu`、`silu`、`gelutanh`、`map`、`softmax`（行最大减法）、`matmul`（确定性从左到右 `f32` 累积）、`rmsnorm`、`rope`、`embedding`。它们不是生产算子实现，也不在任何运行期路径上。
* `tinymodel` 是为微小确定性模型夹具保留的空间：一个微小模型夹具需要 `nn` 与 `models` 层来定义层、参数与权重初始化。该模块当前刻意不定义任何项，把该保留记录在类型系统中而非仅在散文中。

### 19.3 接口边界（DS1）

`testkit` 只依赖 `tensor`（生产 crate 用于构造夹具是被允许的）。它导出四个公共模块：`assertions`、`fixtures`、`reference`、`tinymodel`。它不导出 trait，也不引入任何算子或后端。它 MUST NOT 成为生产依赖，也 MUST NOT 进入生产依赖闭包。仓库根 `tests/` 的集成测试经由 `lib`/`api`/`cpu` 行使完整栈，但那些是测试期依赖，不改变 `testkit` 自身闭包。

### 19.4 不变量与错误路径（DS2）

算法级不变量：

* 每个 `fixtures` 构造函数对给定参数产生逐字节相同的张量；`sequential`/`matrix` 的元素数是 `rows * cols`（经张量元素数校验）。
* `assertallclose` 在长度不等、容差为负或非有限、或首个超出容差的值处失败；失败时 panic 消息命名索引、两个值与所超容差。
* `reference` 的每个函数是纯函数且不分配超出其返回值所需的内存；`matmul` 用与生产 kernel 相同的确定性从左到右 `f32` 累积顺序。

错误路径：`fixtures` 构造函数在形状被拒（如元素数溢出）时返回 `tensorerror`。`assertions` 刻意 panic——这是测试基础设施，在此「带描述信息快速失败」是正确行为；它从不在生产运行期路径上使用。`reference` 是全域纯函数，无错误路径（`softmax` 对空输入返回空向量，需非空末维的调用方自行校验前置条件）。

### 19.5 复杂度（DS3）

每个函数都是短小的直线标量或构造逻辑，远低于圈复杂度 10 与有效行数 80。`matmul` 是最大的函数，为一个三重循环而无额外分支。

### 19.6 并发设计（DS4）

`testkit` 不引入共享可变状态或同步原语。全部项都是纯函数或返回拥有值的构造函数，故可被并发测试无锁调用，也无锁序要求。

### 19.7 被否决方案（DS5）

* **把 `reference` 实现建立在 `cpu` 之上。** 否决：参考实现必须独立于它交叉核对的优化 kernel；依赖 `cpu` 会使参考与被测共享同一错误源，无法检测错误。
* **把 `assertions` 做成返回 `Result` 的容差比较。** 否决：测试基础设施中带描述信息的快速失败正是期望行为；返回 `Result` 会把断言失败的责任推给每个调用点。
* **把夹具散布到各测试文件或放进生产 crate。** 否决：共享夹具集中在一处避免重复并使每个 crate 的构建保持快速；放进生产 crate 会让测试专用代码进入生产闭包。

### 19.8 后果

* 测试可复现：夹具固定形状与填充值，参考实现独立于被核对 kernel。
* `testkit` 是依赖叶子且仅依赖 `tensor`；它不扩大任何生产闭包，因为它是开发依赖。
* `tinymodel` 的保留使「微小模型夹具待 `nn`/`models` 落地后补齐」这一缺口在代码中可见，而非仅在散文中。

### 19.9 验证

本文件的 `reference` 实现被 `cpu`、`autograd`、`nn`、`models` 的交叉核对测试消费；`fixtures` 与 `assertions` 被多个 crate 的数值测试消费。仓库级验证在整合记录中给出。本记录重测（2026-10-10）：`cargo test --workspace --locked -- --list` 枚举 736 个用例；`bash scripts/check.sh` 报告 "All static quality checks passed"；通过/失败总数未在本轮重跑。`cargo bench --workspace --no-run` exit 0（7 个基准目标）；`scripts/test.sh` 成功。`assertallclose` 在容差被违反时以命名索引、两个值与所超容差的描述性消息 panic，这是测试基础设施的预期行为（E1）。

### 19.10 复审条件

* 当 `nn`/`models` 落地且需要一个共享的微小确定性模型夹具时，在 `tinymodel` 中定义它，并使其与 `tests/common/mod.rs` 的既有夹具不重复。
* 当出现新的、被多个 crate 共享的确定性输入或独立参考实现时，优先加入 `fixtures`/`reference` 而非在各测试中重复。
* 当测试需要新的数值比较语义（如相对误差上界、逐元素掩码）时，在此扩展 `assertions` 并保持其仅用于测试基础设施。

## 20. 基础设施：bench

### 20.1 上下文

`bench` 拥有可复用基准负载。它拥有基准负载设置、执行 harness 与可复现元数据。它 MUST NOT 拥有生产算子实现或与生产行为相偏离的替代实现，也 MUST NOT 成为生产依赖。它的工作负载以确定性构造函数暴露，使每个基准目标构建相同输入。除被测量契约返回的输出外，计时区内不发生分配。

本库只依赖 `lib`、`tensor` 与 `cpu`，故 crate 保持为依赖叶子。基准可执行文件位于 `benches/`；本库持有它们组合的共享负载生成器。

### 20.2 决策

* 负载常量：`ELEMENTWISE = 1 << 20`（逐元素激活）、`SOFTMAXROWS = 512`、`SOFTMAXCOLS = 4096`（softmax）、`MATMULDIM = 256`（方阵乘）、`NORMROWS = 512`、`NORMCOLS = 4096`（RMS 归一化）、`SEQLEN = 128`、`HEADS = 8`、`HEADDIM = 64`（RoPE 与注意力）、`EMBEDDIM = 1024`、`GATHER = 128`（嵌入）。
* `OPLIMITS` 是准入每个基准负载而不拒绝它的资源预算，刻意放大（`limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18)`）：基准测量 kernel 成本而非上限执行，上限执行由 `cpu` 契约测试覆盖。它是 `cpuexec` 的内部细节，故保持 crate 私有（`A4` 接口最小化）。`VOCAB = 4096` 只被 `embeddinginput` 读取，故也保持 crate 私有（`A4`）。
* 构造函数：`cpuexec()`（带基准资源预算的 CPU 执行器）、`elementwiseinput()`（`ELEMENTWISE` 个 `f32`）、`softmaxinput()`（`SOFTMAXROWS × SOFTMAXCOLS`）、`matmulinput()`（两个 `[MATMULDIM, MATMULDIM]`）、`norminput()`（`[NORMROWS, NORMCOLS]` 输入与权重）、`ropeinput()`（`[SEQLEN, HEADS * HEADDIM]`）、`embeddinginput()`（`[VOCAB, EMBEDDIM]` 表与 `GATHER` 个索引 `(i * 7) % VOCAB`）、`attentioninput()`（q/k/v 三个 `[SEQLEN, HEADS * HEADDIM]`）。

### 20.3 接口边界（DS1）

`bench` 只依赖 `lib`、`tensor` 与 `cpu`，故保持依赖叶子。它导出负载常量与确定性构造函数，以及 `attentionoperands` 类型别名。它 MUST NOT 导出生产算子实现或与生产行为偏离的替代实现。`OPLIMITS` 与 `VOCAB` 保持 crate 私有以最小化接口宽度（`A4`）。每个构造函数在固定形状被 `tensor` 拒绝时返回 `tensorerror`。

### 20.4 不变量与错误路径（DS2）

算法级不变量：

* 每个负载构造函数对相同编译构建产生逐字节相同的输入（固定形状与固定填充值），故基准可复现。
* `OPLIMITS` 准入每个基准负载而不触发上限拒绝；改动负载常量或预算时二者必须一致。
* 计时区内除被测量契约返回的输出外不发生分配；输入在 criterion 的 setup 阶段或在计时外构建。

错误路径：构造函数在形状被 `tensor` 拒绝时返回 `tensorerror`。`bench` 不执行计算本身的错误处理；它调用生产契约，其错误由被测量的 API 决定。

### 20.5 复杂度（DS3）

每个构造函数都是短小直线逻辑，远低于圈复杂度 10 与有效行数 80；`embeddinginput` 中的索引生成是唯一循环且无嵌套分支。

### 20.6 并发设计（DS4）

`bench` 不引入共享可变状态或同步原语。负载构造函数是确定性的纯构造；`cpuexec()` 返回一个 CPU 执行器值。并发由各基准目标与 `cpu` 后端处理，不属本库。

### 20.7 被否决方案（DS5）

* **在 `bench` 中保留生产算子的替代实现。** 否决：会与生产行为静默偏离并破坏基准的意义；基准必须行使生产代码路径。
* **在计时区内分配输入。** 否决：会污染测量；输入在 setup 阶段或计时外构建。
* **把负载常量留在各基准目标中重复。** 否决：共享负载生成器避免重复并使每个基准目标构建相同输入，从而可横向比较。

### 20.8 后果

* 所有基准目标从同一组确定性构造函数构建输入，故结果可横向比较且可复现。
* `bench` 是依赖叶子；它不进入任何生产闭包。
* 基准仅建立可测量性，不声称性能改进。

### 20.9 验证

整合记录于 2026-10-10 对全工作区执行（macOS 27.0.1、`aarch64-apple-darwin`、rustc 1.99.0、edition 2024）：

```sh
cargo bench --workspace --no-run                             # exit 0 (7 benchmark targets)
```

`cargo bench --workspace --no-run` 构建 `cpuops`、`kernels`、`metadata`、`autograd`、`nn`、`models` 与 `optim` 共 7 个基准目标。`scripts/check.sh` 与 `scripts/test.sh` 成功。不作性能声明；基准目标只建立可测量性（`PF1`/`VF5`）。`optim` 基准的测量数字见 13.9。

### 20.10 复审条件

* 当加入新的生产算子时，为其在根 `benches/` 添加基准目标，并复用 `bench` 的确定性负载生成器。
* 当出现新的、被多个基准共享的负载时，把生成器加入本库而非在各目标中重复。
* 当资源预算或负载维度改变时，重新确认 `OPLIMITS` 仍准入每个负载且计时区内无分配。

## 21. 层 1：GPU 后端（metal 与 cuda）

本章是一次**累加式**扩展：新增两个 Layer 1 后端 crate `metal`（macOS）与 `cuda`（Linux/Windows），二者都实现 Layer 0 已冻结的 `backend`、`kernel` 与 `compute` 契约，因此不修改任何既有 crate 的公共符号。本章同时记录工作区 `unsafe` 禁令的**唯一**一次修订。为不重编既有章号（第 7–20 章为冻结的 crate 章），本章与两个元章节顺次后移一位。

### 21.1 上下文

`cpu` 是当前唯一后端，无法利用 GPU。要在不牺牲既有纯洁性的前提下接入 GPU，有两个硬约束：

1. **工具链是稳定版。** 工作区固定 `1.99.0`（`rust-toolchain.toml`）。以 nightly 为前提的 `rust-cuda` / `rustc_codegen_nvvm` 直接排除；CUDA 只能走运行时编译（NVRTC）加运行时加载。
2. **原生 GPU 派发处必然需要 `unsafe`。** 这不是偏好问题，而是已取证的平台事实（证据等级 E1）：`objc2-metal` 0.3.2 的 crate 文档自述着色器「调用不安全」、含偏移/长度的 API 常为 `unsafe`、`MTLBuffer` 的 CPU/GPU 同步 "often unsafe"、且 `MTLBuffer` 无类型；`cudarc` 0.19.10 的 `LaunchArgs::launch` 与 cuBLAS `gemm` 均为 `pub unsafe fn`。二者都无法在全仓 `forbid(unsafe_code)` 下实现，因为 `forbid` 不可被内层 `allow` 覆盖，故本章附带一次受限的宪法修订（见 21.8）。

已执行的一次性尖刺验证（代码置于被 git 忽略的 `work/`，不提交；证据等级 E1）：

* **Metal 在本机真实运行成功。** 设备 `Apple M5 Pro`；MSL 经 `newLibraryWithSource_options_error` **运行时编译成功且无需 Xcode**；2×3 乘 3×2 的 f32 matmul 输出 `[58, 64, 139, 154]` 与参考一致。
* **Metal 的真实 `unsafe` 面仅 4 处**：host→device 上传、device→host 下载、`setBuffer_offset_atIndex`、`setBytes_length_atIndex`。`MTLCreateSystemDefaultDevice`、`newLibraryWithSource_options_error`、`newComputePipelineStateWithFunction_error`、`newBufferWithLength_options`、`newCommandQueue`、`commandBuffer`、`computeCommandEncoder`、`setComputePipelineState`、`dispatchThreads_threadsPerThreadgroup`、`endEncoding`、`commit`、`waitUntilCompleted` 全部是**安全**的（实测判据：编译器对多余的 `unsafe` 块报 `unused_unsafe`）。（尖刺时刻观测；现值已收敛至 3 个驱动交接块，见 21.8 表。）
* **cudarc 可在无 CUDA 工具链的 macOS 上类型检查通过。** feature 为 `driver`、`cublas`、`cudnn`、`nvrtc`、`dynamic-loading`，且**必须**显式指定一个 `cuda-<version>`（本仓库取 `cuda-12060`；缺省时 `build.rs` 直接失败）。因此 `cargo check -p cuda` 可作为无 GPU 的 CI 门禁。
* **cuDNN 同样可在无工具链的宿主上类型检查。** `dynamic-loading` 下 `cudarc` 的 `build.rs` 不发出任何 `rustc-link-lib`，`cudnn` 的符号改由 `libloading` 在运行时解析（`src/cudnn/sys` 的 `culib()`），故 cuDNN 头文件与库都不参与本机构建。代价是符号缺失从链接期错误变为 `cudarc` 内部的 `panic!`，我们无法消除，只如实记录（证据等级 E1，读源码所得）。
* CUDA 的运行时数值验证**已于 2026-10-12 在真机执行**（RTX 2080 Ti；`cudacontract` 36 passed / 0 failed、`cargo test --workspace` 758 passed / 0 failed、输出 0 条 `E4 未执行`），见 21.10.3。

两个 crate 都只在目标平台编译出实体代码：`metal` 仅在 `cfg(target_os = "macos")` 下启用依赖并导出实现，其他平台编译为空 lib；`cuda` 仅在 `cfg(target_os = "linux")` 下列表依赖，其他平台编译为空 lib。这样 `cargo check --workspace --all-targets` 在任何宿主与 CI 上都保持通过。

### 21.2 决策

1. **采纳原生双后端，不引入 `wgpu`。** 用户要求 CUDA 专用能力（cuBLAS，后续可扩 cuDNN），`wgpu` 的 Vulkan 后端不满足该诉求；且 `wgpu` 只是把 `unsafe` 移入第三方，并未消除，同时引入一层与 `api` 契约不直接对应的抽象。
2. **单文件 `unsafe` 豁免模型。** 两个 crate 使用 `#![deny(unsafe_code)]`，crate 内恰好一个文件 `src/ffi.rs` 声明 `#![allow(unsafe_code)]`；所有平台调用被封在此文件内，对外只暴露安全 Rust 函数。详见 21.8。
3. **Metal 走 `objc2-metal`，不走 `metal-rs`。** 实测 `objc2-metal` 的绝大多数 API 是安全的，`unsafe` 面仅 4 处；`metal-rs` 0.33 把大量 API 标为 `unsafe`，暴露面显著更大。
4. **CUDA 走 `cudarc`，feature 固定为 `driver` + `cublas` + `cudnn` + `nvrtc` + `dynamic-loading` + `cuda-12060`。** `dynamic-loading` 使 `cargo check` 无需 CUDA 工具链；`cuda-12060` 是 `cudarc` 的强制要求；`cudnn`（即 `cudnn-09021`）提供 `SoftmaxForward` 等安全封装。
5. **条件编译为空 lib。** 非目标平台不引入任何平台依赖，也不导出实现类型，保证跨平台 workspace 绿。
6. **同步、运行至完成的执行语义。** 与 `tensor` 与 `cpu` 的同步模型一致：每次算子调用完成 host→device→dispatch→device→host 全流程后返回拥有张量；不引入流式或异步 API。
7. **首个算子为 `f32` matmul。** MSL 与 CUDA C 源码以独立文件（`kernels/*.metal` / `kernels/*.cu`）经 `include_str!` 内嵌并在运行时编译，避免构建期依赖 `xcrun`/`nvcc`。
8. **设备身份在 `api` 侧累加。** `devicekind` 新增 `metal` 与 `cuda` 变体，`device` 新增对应构造与判定；不改动既有 `cpu` 语义，也不加 `#[non_exhaustive]`（工作区内需要穷举 match 以获得编译期完整性）。
9. **依赖边仅为后端到 `api`/`tensor`/`lib`。** `api` MUST NOT 依赖 `metal` 或 `cuda`；上层（`models`/`infer`）通过既有的 `C: compute` 泛型槽接入，零改动。
10. **CUDA 的 `unsafe` 站点收于三类平台交互、共三处。** NVRTC 内核启动一处（以闭包注入实参，服务于全部在仓内核）、cuBLAS `gemm` 一处、cuDNN 末轴 softmax 一处；不做 `unsafe` 的通用化封装，仅按 21.8 登记的三处具体调用。
11. **算子按机制分工。** `matmul` 走 cuBLAS（并保留 NVRTC 路径作独立参考）；`softmax` 走 cuDNN；其余算子（`add`/`mul`/`silu`/`rmsnorm`/`rope`/`embedding`/`attention`）走内嵌 CUDA C++ 源码经 NVRTC 运行时编译。`silu` 刻意**不**走 cuDNN 的 `CUDNN_ACTIVATION_SWISH`：它坐落于 `ActivationForward::launch` 这一自带 `unsafe` 的入口，为一个逐元素算子新增一类平台交互不值得，而 NVRTC 内核族复用同一站点、零成本。

### 21.3 接口边界（DS1）

`metal` 在 crate root 导出 4 个公共项，`cuda` 导出 5 个，均远低于 A4 上界 20：

| crate | 公共项 | 模块文件 | 职责 |
| --- | --- | --- | --- |
| `metal` | `METALMAXRANK` | `backend.rs` | 后端接受的最大张量秩 |
| `metal` | `metalbackend` | `backend.rs` | macOS 后端身份与能力声明，返回 `device::metal()` |
| `metal` | `metalexecutor` | `executor.rs` | 实现 `api::compute` 的 f32 算子集 |
| `metal` | `metalexecutor::getcachedpipelinecount` | `executor.rs` | 已缓存的编译产物数，使「每内核只编译一次」可从 crate 外机械核对 |
| `cuda` | `CUDAMAXRANK` | `backend.rs` | 后端接受的最大张量秩 |
| `cuda` | `CUDAMAXHEADDIM` | `backend.rs` | 单个注意力头的最大宽度，由内核的共享内存预算决定 |
| `cuda` | `cudabackend` | `backend.rs` | Linux 后端身份与能力声明，返回 `device::cuda()` |
| `cuda` | `cudaexecutor` | `executor.rs` | 实现 `api::compute` 的 f32 算子集 |
| `cuda` | `matmulpath` | `executor.rs` | `matmul` 走 cuBLAS 还是 NVRTC，构造期固定 |

`src/ffi.rs` 是私有模块，不导出任何符号；它只向 `executor.rs` 提供安全的封装函数，`backend.rs` 不接触它。模块文件名刻意不同于导出的类型名以满足 E0255 守卫。

依赖图（累加，无环）：

```text
metal -> lib
metal -> tensor
metal -> api
cuda  -> lib
cuda  -> tensor
cuda  -> api
```

`api`、`tensor`、`lib` 均不依赖 `metal` 或 `cuda`。目标平台的表驱动依赖（`[target.'cfg(...)'.dependencies]`）不产生新的 workspace 内边。

两个后端的预算与秩判定不各自实现：`api` 的 `checkbudget`/`checkgatherbudget` 是唯一实现，二者在 `executor.rs` 中各持一个转发包装。`api` 的 crate root 导出项由 10 增至 12，仍低于 A4 上界 20。

### 21.4 不变量与错误路径（DS2）

算法级不变量（每个算子都复用 `cpu` 同名契约的形状规则，逐条对应）：

* `matmul`：相等 rank `r >= 2`，前导批维逐一相等，尾两维 `[m,k]` 与 `[k,n]` 相乘得 `[batch..., m, n]`；无广播。
* `add`/`mul`：两侧维度逐一相等，无广播；输出形状等于操作数形状。
* `rmsnorm`：末轴非空；权重元素计数等于末轴长度（比较的是元素计数而非形状，故 `[width]` 与 `[1, width]` 均合法）。
* `softmax`：末轴非空；输出与输入同形。
* `rope`：rank ≥ 2、`heads ≥ 1`、末轴为 `heads` 的倍数，且每头宽度为正偶数；位置是该行在倒数第二轴上的下标（`seqaxis == 0` 时恒为 0）。
* `embedding`：表的第一轴是行空间，每个下标须落在 `[0, vocab)`；输出形状为 `[indices.len(), table.shape[1..]]`。
* `attention`：query/key/value 均为 rank 2；key 与 value 同形且序列长度等于 query 的；`heads ≥ 1` 且是 `kvheads` 的倍数；query 末轴是 `heads` 的倍数；key/value 末轴等于 `kvheads * headdim`。
* 输出元素计数等于输出维度之积，且在任何分配之前完成溢出检查。

实现级不变量：

* 设备缓冲的元素类型与长度由 Rust 侧 `CudaSlice<f32>` / Metal 缓冲长度确定；GPU 侧索引由内核内的显式边界判断收束（`row < m`、`col < n`、平铺的 `index < total`）。
* `attention` 的每头宽度不超过各后端上限（`metal` 为 `METALMAXHEADDIM`、`cuda` 为 `CUDAMAXHEADDIM`），因为内核为每个头维保留一个累加槽；超出是设备容量不足，不是契约违规。
* 每次调用结束后设备缓冲与命令对象随作用域析构释放；无跨调用的设备状态残留（与 `tensor`/`cpu` 的 OW1/OW3 一致）。
* `metal` 的 `ffi::cache` 是唯一跨调用存活的对象，它只持有设备句柄与已编译的 pipeline，**不持有**张量、缓冲、索引或任何请求作用域的值，因此跨请求共享它不可能传递数据（OW2/OW3）。
* 一个算子调用的输出元素计数由输出形状给出，且其结果预算在任何分配之前校验；`matmul` 的**全部** batch block 在一次 dispatch 内完成，故其成本与 block 数无关（见 21.10.2）。
* `unsafe` 站点只在 `ffi.rs`，且每个站点的安全前置在 `// SAFETY:` 中显式陈述。

错误分支与返回语义（复用 `backenderror`，不新增错误类型，遵循 ER2）：

* 形状不匹配（上述任一形状规则不满足）→ `shapemismatch { op }`；
* `embedding` 下标越界 → `indexoutofrange { index, len }`；
* 末轴为空（`softmax`、`rmsnorm`）→ `unsupportedoperation { op }`；
* 元素计数溢出（秩、批、元素数、字节数任一算不出）→ `sizeoverflow`；
* 秩超过 `METALMAXRANK`/`CUDAMAXRANK` → `resourceexhausted`（与 `cpu` 对 `CPUMAXRANK` 的分类一致：被拒绝的是声明的能力上界，不是执行失败）；
* 超出能力声明的元素或字节预算（`embedding` 的按其 gather 输出 `rows * width` 计；`matmul` 的除两个操作数外还含其**结果**）→ `resourceexhausted`；
* 每头宽度超过 `METALMAXHEADDIM`/`CUDAMAXHEADDIM` → `resourceexhausted`；
* 设备初始化失败 → `devicenotavailable`；MSL 或 NVRTC 编译失败 / 模块或内核解析失败 → `kernelcompilefailed`（带 backend/kernel/驱动诊断）；缓冲分配失败 / 内核启动失败 / cuBLAS 调用失败 / cuDNN 调用失败 / host-device 传输失败 / 输出张量构造失败 → `executionfailed`（携带结构化诊断 `{ backend, kernel, message }`，`kernel` 为失败阶段名，`message` 为驱动或宿主诊断——执行失败可定位到阶段与根因）；
* `metal` 与 `cuda` 都已实现 `compute` 的全部算子（`matmul`、`add`、`mul`、`silu`、`rmsnorm`、`softmax`、`rope`、`embedding`、`attention`、`scan`），故两者唯一的 `unsupportedoperation` 来源是上一条的空末轴（`softmax`、`rmsnorm`）。`scan` 为每线程一通道的串行递推内核（β 通道，`docs/beta-scan-plan.md`），复用既有上传/下载/绑定/参数绑定站点，`unsafe` 站点数不变（metal 3/3、cuda 3/3）。

`compute` 契约的每个方法都只接受 `f32` 张量视图，故 GPU 后端不可能因 dtype 失败；`capabilities` 声明 `f32` 是为了让上层在派发前就能判定，`unsupporteddtype` 不由这两个后端产生。

预算与秩的判定由 `api::checkbudget` / `api::checkgatherbudget` 单点实现：两个 GPU 后端在 `executor.rs` 只保留转发到该谓词的私有包装，因此同一操作数不可能在两个后端得到不同的分类；`cpu` 保留自己更丰富的策略类型（`cpuerror` 携带 `limiterror` 的 `requested`/`limit` 与 `rankexceeded`），但把这两类拒绝映射到同一个 `backenderror`，故三个后端的**可观测**分类一致。

`matmul` 的结果由后端自行分配，故其结果形状在**任何分配之前**也过同一预算谓词；这是 `embedding` 早已有的输出预算检查在 `matmul` 上的对齐（此前 `matmul` 只受操作数约束，`m*k <= max` 且 `k*n <= max` 仍可要求 `m*n` 级别的输出）。

资源回滚：任何失败在构造输出张量之前返回；已获取的设备缓冲、命令缓冲、库与管线对象按作用域逆序析构（ER4）。

### 21.5 复杂度（DS3）

各 crate 的单函数圈复杂度 ≤ 10、有效行数 ≤ 80：

* `geometry.rs` 把每个算子的形状推理与拒绝条件收进单个函数，使 `ffi.rs` 的每个启动封装只做「取已验证的几何 → 一次平台调用 → 一次错误映射」，判定点不超过 3 个。
* `executor.rs` 的每个算子在预算校验后委派给 `geometry` 与 `ffi`，自身不做算术；`matmul` 不再在宿主侧切分 batch block，因此不含逐块循环。
* 预算与秩判定不是 `executor.rs` 自己的逻辑：两个后端都转发到 `api::checkbudget`，连判定顺序都无法分歧；`geometry.rs` 的 `matmulshape` 在两侧同形，并各自提供 `outshape()` 以在分配前校验结果预算。
* `source.rs` 只存放 `include_str!` 引用与 entry 名常量，不参与 Rust 复杂度（MSL/CUDA 原文在 `kernels/` 目录）；两个后端的 `ffi.rs` 都只有一条通用 `execute` 启动路径，编译、管线、队列、命令缓冲与编码器的失败各由一个 `?` 上报，判定点不超过 5 个；`attention` 的融合内核把规约树拆成 `reducemax`/`reducesum` 两个辅助函数，避免单个内核函数越过 80 行。

宿主 macOS 的 `scripts/checkthresholds.sh` 看不到被 `cfg(target_os = "linux")` 门控的 CUDA 代码，因此 CUDA 侧以同一组阈值单独在 Linux 目标上核对（`cargo clippy -p cuda --target x86_64-unknown-linux-gnu --lib --locked -- -D clippy::cognitive_complexity -D clippy::too_many_lines`，exit 0）。两者均为机械检查（证据等级 E1）。

### 21.6 并发设计（DS4）

两个 executor 都持有**恰好一个**锁，且都只用来保护「设备句柄 + 已编译内核」这份不含请求数据的缓存（`cuda` 于 2026-10-12 实施，见 21.10.3；`metal` 于同日稍早实施，见 21.10.2）。此前 `cudaexecutor` 不引入任何共享可变状态：它只持有 backend 与一个构造期固定、之后只读的 `matmulpath`。

`metalexecutor` 引入**恰好一个**锁。MSL 的运行时编译成本（实测约 0.85–1.0 ms/内核实，见 21.10.2）此前落在每次调用上，要消除它必须让编译产物跨调用存活，因此 `metal` 持有一个 `ffi::cache`：

* **内容与所有权。** `ffi::cache` 只含设备句柄与按 `(source, entry)` 索引的 pipeline；它由 executor 拥有，executor 的 clone 通过 `Arc` 共享同一份，独立构造的 executor 各编译一份。它不是全局单例，故 §21.11 的「显式所有权而非全局单例」得到满足。
* **锁序。** 全 crate 只有一个锁，且从不在持锁期间申请第二个锁，故锁序为平凡单元素偏序，不存在环。
* **持锁时间。** 锁内只做查表、可能的编译、以及句柄的 retain 克隆；命令缓冲创建、dispatch、commit 与 `waitUntilCompleted` 都在**锁外**执行，因此并发调用不会在 GPU 等待上互相阻塞。命中缓存的稳态调用持锁约百纳秒量级。
* **首次并发。** 同一内核的并发首次调用在锁内串行编译一次，而不是各自重复编译；`tests/it/metalcontract.rs` 的 `the_executor_is_shareable_across_threads` 与 `every_kernel_is_compiled_once_and_then_reused` 分别钉住「并发结果一致且只编译一次」与「九次调用恰好九个 pipeline、重复调用不增长」。
* **失败与中毒。** 设备打开失败时不缓存 `None`，下次调用重试（瞬态失败可恢复）；锁中毒后取回内部值继续使用，因为缓存不持有任何可被 panic 破坏的不变量——最坏情况是一个部分填充但仍合法的映射。
* **声明局限。** 设备在进程生命周期内被缓存，**不重新探测**：eGPU 移除或 GPU 重置后，本 crate 只如实上报底层 Metal 错误，不自动重建上下文。该行为记录为复审条件（见 21.11）。

`cuda` 的缓存与 `metal` 同构，差异只在被缓存的对象与「执行队列故意不缓存」的实现方式：

* **内容与所有权。** `ffi::cache` 只含 device 0 的**已 retain 主上下文**与按 `(source, entry)` 索引的已解析内核；键里带着内核**原文**，因此改动 `kernels/` 源码不可能复用陈旧模块；每个 `CudaFunction` 持有其 `CudaModule`、进而持有上下文，故缓存的内核不可能比它编译所在的设备活得更久。它同样由 executor 拥有、clone 经 `Arc` 共享，不是全局单例。
* **执行队列不缓存。** 每次调用从缓存上下文取一条**新流**并在返回前释放（`metal` 的对应选择是不缓存命令队列）：共享一条流会把并发调用的 GPU 工作排成队，而这正是 §21.6 在 Metal 侧否决过的做法。新流的创建/销毁成本记账在下面的前后对比里。
* **锁与持锁时间。** 全 crate 一个锁，从不在持锁期间申请第二个；锁内只做查表、可能的 NVRTC 编译与模块装载；传输、内核启动与同步**都在锁外**。同一内核的并发首次调用在锁内编译一次，命中缓存的稳态调用持锁约百纳秒量级。
* **失败与中毒。** 与 `metal` 相同：设备打开失败不缓存 `None`（下次调用重试），锁中毒后取回内部值继续使用。
* **验证。** `tests/it/cudacontract.rs` 的 `every_kernel_is_compiled_once_and_then_reused`（七个 `(source, entry)` 对服务九个算子；NVRTC 路径多一个 matmul 内核）、`a_clone_shares_the_compiled_kernels`、`the_executor_is_shareable_across_threads`（`Send + Sync`，四线程并发 `silu` 结果逐位一致且只编译一个内核）三例在真机上钉住这些声明。

每次调用仍在调用作用域内创建执行流（`cuda`）或命令缓冲与命令队列（`metal`）以及设备缓冲，调用返回即析构；请求作用域状态不进入缓存。

### 21.7 被否决方案（DS5）

* **改用 `wgpu` 纯安全抽象。** 被否决：用户显式要求 CUDA 专用能力（cuBLAS/cuDNN），Vulkan 不满足；且 `wgpu` 不消除 `unsafe`，只是把它移入第三方。
* **工作区级放宽 `unsafe_code = "deny"`。** 被否决：会静默影响全部 crate，无法机械区分「预期的 FFI `unsafe`」与「意外引入的 `unsafe`」。
* **整 crate 级豁免（`#![allow(unsafe_code)]` 置于 lib.rs）。** 被否决：豁免面覆盖整个 crate 的所有未来代码，无法收敛。
* **推迟 CUDA 至出现纯安全绑定。** 被否决：用户已具备 NVIDIA GPU 环境且明确要求 CUDA；`cudarc` 的 `safe` 模块已把大部分生命周期与同步责任收拢，剩余 `unsafe` 面仅 2 处，可被精确登记。
* **以 nightly `rust-cuda` 在编译期生成 PTX。** 被否决：工作区固定稳定版工具链；且会引入 nightly 与 `unsafe` 之外的构建复杂度。
* **`#![forbid(unsafe_code)]` 加内层函数级 `#[allow]`。** 被否决（技术上不可行）：`forbid` 不可被内层 `allow` 覆盖，编译期即拒绝；这正是豁免必须精确到 crate 边界的原因。

### 21.8 `unsafe` 禁令的受限修订与站点登记

这是工作区 `unsafe` 禁令的**唯一**一次修订，范围严格受限：

* 工作区 lint 保持 `unsafe_code = "deny"`（`Cargo.toml`），一行不改。
* 全部既有 crate 保持 `#![forbid(unsafe_code)]`，一行不改。
* `metal` 与 `cuda` 使用 `#![deny(unsafe_code)]`，且每个 crate 内**恰好一个文件** `src/ffi.rs` 声明 `#![allow(unsafe_code)]`；其余模块仍受 `deny` 约束。
* `src/ffi.rs` 中每一处 `unsafe` 构造前必须有 `// SAFETY:` 注释，陈述该 API 的安全前置为何被满足。
* 站点总数必须与下表条目数**恰好相等**，多一处或少一处都视为豁免扩散并触发评审。实测登记（证据等级 E1，来自 21.1 的尖刺）：

| crate | 站点 | 位置 |
| --- | --- | --- |
| `metal` | 同步驱动交接块（`dispatch`：上传、绑定缓冲、绑定标量、启动、`endEncoding`/`commit`/`waitUntilCompleted`、下载，全部收于一个 `unsafe` 块） | `ffi.rs` |
| `metal` | 会话编码块（`session::dispatch`：同一 command buffer 上绑定并启动，不等待） | `ffi.rs` |
| `metal` | 会话提交块（`session::apply`：`commit`/`waitUntilCompleted` + 下载全部结果） | `ffi.rs` |
| `cuda` | `LaunchArgs::launch`（NVRTC 内核启动） | `ffi.rs` |
| `cuda` | `CudaBlas::gemm`（cuBLAS 矩阵乘） | `ffi.rs` |
| `cuda` | `SoftmaxForward::launch`（cuDNN 末轴 softmax） | `ffi.rs` |
| `musa` | `unsafe extern "C"` 声明块（edition 2024 强制；MUSA driver 全部 FFI 声明） | `ffi.rs` |
| `musa` | 驱动交接 `unsafe` 块（`drive`：模块加载/函数解析/内存传输/内核启动/同步，全部 `mu*` 调用收于一个块） | `ffi.rs` |

站点登记：`metal` = 3，`cuda` = 3，`musa` = 2；门禁断言「实测站点数**恰好等于**本表条目数」，多一处或少一处都失败。因此新增任何站点都必须先更新本表并在此说明理由，删除站点同样必须同步本表。

依据：这些数字由**调用面结构**推导，而非经验取值。Metal 的口径是**每次驱动交接一个 `unsafe` 块**：同步路径的 `dispatch` 把上传、缓冲绑定、标量绑定、启动、`endEncoding`/`commit`/`waitUntilCompleted`、下载收进**单一交接块**；累积路径的 `session` 拆成**编码块**（绑定并启动、不等待）与**提交块**（`commit` + 等待 + 下载）两个块——command buffer 单次使用、跨算子共享，故提交只发生一次。每个块内的全部平台调用共享同一个 `// SAFETY:` 前置论证（缓冲均为本块持有或调用者传入的存活 `MTLBuffer`、长度与内核读写一致、`setBytes` 拷贝而非保留、`waitUntilCompleted` 排序读写）。CUDA 的口径是**每类平台交互恰一处**，而不是每个算子一处：

* `LaunchArgs::launch` 只登记一次，因为整个在仓内核族共用它。启动辅助以闭包注入实参、以 `(source, entry)` 选择内核，故 `matmul`/`add`/`mul`/`silu`/`rmsnorm`/`rope`/`embedding`/`attention` 八个内核**不增加站点**。这是设计选择而非巧合：把「一类平台交互」而非「一个算子」作为登记的原子单位，使豁免面不随算子数增长。
* `CudaBlas::gemm` 与 `SoftmaxForward::launch` 各一处，因为二者是各自库里唯一自带 `unsafe` 的入口。

MUSA 的口径是「自写 FFI 的 2024 edition 理论下限」：`musa` 没有像 `cudarc`/`objc2-metal` 那样的第三方安全封装，必须自己写 `extern "C"` 声明与调用（`docs/musa-backend-plan.md` §5.0/§5.4）。在 edition 2024（本工作区 `rust-version = 1.99`）下，`extern` 块声明本身强制为 `unsafe extern "C"`——这是**一个不可消除的 `unsafe` 构造**；所有 `mu*` 调用（`muModuleLoadData`/`muModuleGetFunction`/`muMemAlloc`/`muMemcpy*`/`muLaunchKernel`/`muCtxSynchronize`）被收进**同一个 `unsafe` 块**（`drive` 驱动交接函数），块内调用不再需要逐处标记——这是**第二个、也是最后一个**构造。故 `musa` = 2 是自写 FFI 条件下的理论下限。`musa` 刻意不引入 `muBLAS`/`muDNN`：matmul 由 WMMA 内核（`include/crt/mma.h`，Tensor Core）服务、softmax 由减最大式内核服务（与 `cpu`/`metal` 同式），因此没有「逐库」站点可加——这也是 `musa` 比 `cuda` 少一个站点的原因。

任何使数字上升的改动都意味着引入了一类新的平台交互或一个自带 `unsafe` 入口的第三方库（例如 cuDNN 的 attention frontend、cuDNN 的 `ActivationForward`），属于豁免扩散，必须先改本表并说明理由；数字下降则意味着某类交互被移除，表也必须同步。

机械门禁 `scripts/checkunsafe.sh` 断言四项：白名单文件是唯一出现 `allow(unsafe_code)` 之处；全部既有 crate 根仍含 `forbid(unsafe_code)`；代码中每处 `unsafe` 前一行（含注释块）含 `// SAFETY:`；`unsafe` 站点计数与本表条目数**恰好相等**（非上界语义：多一处或少一处均失败，本表与代码不允许单向漂移）。该脚本接入 `scripts/check.sh`。

回滚：删除 `crates/metal`、`crates/cuda` 与 `crates/musa`、撤销 `api` 的设备变体与门禁条目，即完全恢复修订前状态。

### 21.9 后果

* GPU 能力以两个独立、可单独删除的 crate 提供；删除任一 crate 不影响 `cpu` 路径与依赖图。
* `unsafe` 面被压缩到 8 个已登记站点（`metal` 3 + `cuda` 3 + `musa` 2），且由机械门禁强制；既有 crate 的 `forbid` 未被触碰。`metal` 3 个站点均是一次驱动交接块（同步 1 + 累积 2），`musa` = 2 是 2024 edition 下自写 FFI 的理论下限（`unsafe extern` 声明 + 单一驱动交接块）。
* `api` 与上层（`models`/`infer`）零改动即可通过 `C: compute` 使用 GPU。
* CUDA 的运行时正确性依赖 NVIDIA 驱动与设备：本文件与 `docs/conformance.md` 只声明**真机执行过**的部分（2026-10-12 的 RTX 2080 Ti 会话，见 21.10.3），未执行的部分仍如实标注，绝不声称通过。
* 维护成本：两个 crate 各自绑定一个易变的第三方 GPU 绑定（`objc2-metal`、`cudarc`），版本升级需重新核对 21.8 的站点表。

### 21.10 验证

* **Metal（E1）。** `tests/it/metalcontract.rs` 的 43 个契约测试在本机 Apple GPU 上全部真实运行并通过（MSL 经 `newLibraryWithSource` 运行期编译）：身份/设备/能力声明；`matmul` 数值（含前导批维）与形状/预算拒绝；`add`/`mul`/`silu` 数值、形状不匹配与预算拒绝、空操作数；`softmax` 逐行数值、极端 logits 稳定性、空末轴拒绝、零行空输出；`rmsnorm` 逐行数值、空末轴与权重长度拒绝、单行权重接受；`rope` 逐位置数值与零头数/奇数头宽/秩一拒绝；`embedding` 数值、越界下标拒绝、gather 预算拒绝、空下标输出；`attention` 因果与非因果数值、单位置恒等、形状不一致拒绝、每头宽度超设备上限拒绝。数值用例均对照 `testkit::reference` 的同名标量参考，容差 1e-6–1e-5。首轮运行时 `rope` 内核因把局部变量命名为 `half`（与 MSL 内建标量类型同名）而编译失败并报 `executionfailed`；改名为 `pairs` 后通过，该失败与修复均在本机复现（E1）。
* **CUDA 编译与静态门禁（E1）。** `cargo check -p cuda --target x86_64-unknown-linux-gnu --all-targets`、`cargo clippy -p cuda --target x86_64-unknown-linux-gnu --all-targets -- -D warnings`，以及 `-p llm --target x86_64-unknown-linux-gnu --test it`、`--bench cuda` 的同名检查均 exit 0。全部在无 CUDA toolkit、无 NVIDIA 驱动的 macOS 宿主上完成，`dynamic-loading` 使其不必要。
* **CUDA 契约（E1，2026-10-12 真机）。** `tests/it/cudacontract.rs` 共 36 例，在 RTX 2080 Ti 主机上以 `LLM_CUDA_REQUIRE_DEVICE=1` 全部执行通过（36 passed / 0 failed，输出中 0 条 `E4 未执行`）：18 例定案于宿主侧（身份与能力声明、matmul 路径选择、八个算子的形状/下标/空末轴拒绝、元素与 gather 预算、`CUDAMAXHEADDIM` 上限、退化空输入），15 例走设备并与 `testkit::reference` 的标量参考对照（cuBLAS 与 NVRTC 两条 `matmul` 路径、`softmax`、`rmsnorm`、`rope`、`embedding`、`attention`、逐元素，容差 1e-4–1e-5），其余 3 例断言与共享向量列表一致。首次执行暴露并修正了一个用例缺陷，命令与结果见 21.10.3。
* **CUDA 基准（E1，2026-10-12 真机）。** `benches/cuda.rs` 的十个工作负载在真机上运行，CPU 与 CUDA 两侧在同一进程、同一轮内测量；数字、每调用固定成本的分解与结论见 21.10.3。**不声明任何加速**（VF5）：CUDA 只在三个由 cuBLAS/cuDNN 服务的工作负载上快于 CPU，其余（由 NVRTC 服务）仍慢于 CPU，原因已由测量定位到每次调用的固定成本。
* **门禁（E1）。** `scripts/checkunsafe.sh` 四项断言全绿（`metal` 4/4 站点、`cuda` 3/3 站点，各有 SAFETY 注释）；`scripts/checknaming.sh`、`scripts/checkthresholds.sh` 与汇总的 `scripts/check.sh`（fmt、全工作区 check/clippy、命名、内联测试、unsafe、阈值、文档）全部通过。
* **回归（E1）。** `cargo test --workspace` 在 macOS 宿主上 44 个测试目标全部 `test result: ok`、0 failed（exit 0），其中 `tests/it/main.rs` 单个二进制 748 passed / 0 failed；`metalcontract` 的 30 例真实运行，`cudacontract` 因平台 cfg 不参与本机构建。

基线（E1；`cargo bench -p llm --bench metal`，release profile，criterion 0.7.0，warm-up 0.5 s，measurement 1.0 s，10 samples；环境 macOS、arm64、Apple M5 Pro、`rustc 1.99.0`）：

| Benchmark | Workload | Time (median) | Throughput |
| --- | --- | --- | --- |
| gpu/matmul/cpu256 | 256×256×256 f32（`cpu` 基线） | 957.31 µs | 68.458 Melem/s |
| gpu/matmul/metal256 | 256×256×256 f32 | 2.5830 ms | 25.372 Melem/s |

该尺寸下 Metal **慢于** CPU（约 2.7×）。21.10.1 把同一基线扩展到九个算子与运行间方差。原因是本阶段每次调用都重新创建设备上下文并运行时编译 MSL，固定开销远超 256³ 的计算量；这不是 GPU 的计算劣势，而是未做编译器缓存与管线复用的直接后果。本阶段**不声明任何加速**（VF5），该表是后续优化必须击败的基线；缓存与管线复用是复审条件之一（见 21.11）。

#### 21.10.1 一致性与边界整改（2026-10-12）

环境与 2026-10-11 相同（macOS 27.0.1 arm64 / Apple M5 Pro / `rustc 1.99.0`）；基线 commit `55d8efb`。

* **共享的预算与秩谓词（E1）。** `api` 新增 `checkbudget`/`checkgatherbudget`，`metal` 与 `cuda` 的 `executor.rs` 改为转发；此前两处逐字节重复的 32 行策略（含「秩超限 → `executionfailed`」）被删除，取而代之的是与 `cpu` 一致的 `resourceexhausted`。
* **`matmul` 结果预算（E1）。** 三个后端（`cpu`/`metal`/`cuda`）此前只校验操作数，`m*k <= max` 且 `k*n <= max` 仍可要求 `m*n` 级别的输出分配；现在结果形状在**任何分配之前**过同一谓词。`cpu` 侧的改动是把 `matmulplan` 与 `cpumatmul` 的计划拆开，使 executor 能在内核分配输出前校验，`cpuerror` 的公开语义未变。
* **共享的无设备契约向量（E1）。** `tests/it/gpucontract.rs` 定义 30 个在任何机器上都定案于宿主侧的向量（秩/元素/字节预算、`matmul` 结果预算与其 inclusive 边界、七个算子的形状与下标拒绝、gather 输出预算），`metal` 与 `cuda` 两个套件各自断言「与本列表一致」，并各自断言「与 `cpu` 的观测逐项相同」。
* **证伪性验证（E1）。** 把 `api::checkbudget` 的秩分类改回 `executionfailed`：`metal` 的两个向量用例都变红，报文为 `case \`silu/rank\` returned Some(executionfailed), the contract requires Some(resourceexhausted)`；把 `metal` executor 里 `self.checkbudget(&outshape)?` 一行删掉：`case \`matmul/outelements\` returned None, the contract requires Some(resourceexhausted)`。两处探针随后还原。
* **Metal 真机数值扩展（E1）。** `metalcontract` 由 30 增至 40 例并在本机 Apple GPU 上全部通过：非方阵且各维互不相同（17×19×13，网格在两方向都跨越多个 threadgroup）、单元素 matmul、宽度 1000（跨 threadgroup 且非 2 的幂）的 `rmsnorm`/`softmax`、`heads == kvheads` 的 attention、位置 0 的 `rope` 恒等、embedding 首尾行。新增的 `softmax_of_a_negative_infinity_row_is_not_a_number` 钉住全 `-inf` 行传播 NaN（既有文档声称「保持全零」与此不符，已更正，见 6.3）。
* **CUDA 侧（E1）。** `cargo check -p llm --target x86_64-unknown-linux-gnu --test it --locked`、`cargo clippy -p cuda --target x86_64-unknown-linux-gnu --all-targets --locked -- -D warnings` 与同目标的 `-D clippy::cognitive_complexity -D clippy::too_many_lines` 均 exit 0；`cudacontract` 新增的 3 例（与向量列表一致、与 `cpu` 一致、cuBLAS 与 NVRTC 两条路径一致）随后在 2026-10-12 的真机上与其余 33 例一同运行通过（36 passed / 0 failed），见 21.10.3。
* **门禁（E1）。** `bash scripts/check.sh` 报 `All static quality checks passed.`（exit 0）；`cargo test --workspace` 758 passed / 0 failed。

* **Metal 全算子基线（E1）。** `benches/metal.rs` 由只测 `matmul` 扩为九个算子，其工作负载与组名与 `benches/cuda.rs` 一致（共享语义、各自内核）。命令：`cargo bench -p llm --bench metal -- --warm-up-time 0.5 --measurement-time 1.0 --sample-size 10`，release profile，criterion 0.7；环境同上表。数值为 2026-10-12 一次运行的 median：

| Benchmark | 工作负载 | CPU median | Metal median |
| --- | --- | ---: | ---: |
| `gpu/matmul/…256` | 256×256×256 f32 | 954.92 µs | 2.3370 ms |
| `gpu/elementwise/…silu` | 2^20 f32 | 1.0953 ms | 3.2692 ms |
| `gpu/elementwise/…add` | 2^20 f32 | 101.50 µs | 3.6305 ms |
| `gpu/elementwise/…mul` | 2^20 f32 | 101.59 µs | 3.7550 ms |
| `gpu/softmax/…rows512cols4096` | 512×4096 f32 | 2.5660 ms | 4.2474 ms |
| `gpu/rmsnorm/…rows512cols4096` | 512×4096 f32 | 1.1909 ms | 4.0132 ms |
| `gpu/rope/…seq128heads8dim64` | 128×8×64 f32 | 81.416 µs | 2.3351 ms |
| `gpu/embedding/…gather128` | 128 行 × 1024 | 6.7247 µs | 3.9703 ms |
| `gpu/attention/…seq128heads8dim64` | 128×8×64 f32 | 1.0883 ms | 2.9206 ms |

Metal 在**每个**工作负载上都慢于 CPU，原因与 21.10 已记录的一致：每次调用都重新编译 MSL 并重新创建命令队列与设备上下文，该固定开销与工作负载无关，故在 `embedding`（CPU 6.7 µs）上表现为约 590×。**不声明任何加速**（VF5）；这张表是 G4/G5 优化必须击败的基线。

* **运行间方差（E1）。** 同一二进制重复测量 `gpu/matmul/metal256` 四次得到 2.337 / 2.617 / 2.672 / 2.850 ms（散布约 ±10%），而 `cpu256` 四次为 954.9 / 956.4 / 958.6 / 974.3 µs（约 ±1%）。Metal 侧的高方差与「每次调用重新编译并重新初始化设备」一致，故任何后续的 Metal 性能声明 MUST 报告多次运行的散布，而不是单次 median。

#### 21.10.2 运行时编译缓存与 dispatch 几何（2026-10-12）

环境同上（macOS 27.0.1 arm64 / Apple M5 Pro / `rustc 1.99.0`），基线为 21.10.1 的测量。

**1. 逐阶段分解（E1，临时插桩后已移除）。** 对 `silu` 的 2^20 元素调用（读 4 MiB、写 4 MiB、一次 dispatch）在稳态下各阶段耗时：

| 阶段 | 稳态耗时 | 说明 |
| --- | ---: | --- |
| 取设备 | 0 µs | 已缓存 |
| 分配 2 个缓冲 | 15 µs | |
| host→device 4 MiB | 265 µs | |
| 取/编译 pipeline | 0 µs | 已缓存；冷启动首次为 846 µs |
| 创建命令队列 | 60 µs | 每次调用新建 |
| 编码器建立 | 56 µs | |
| `commit` | 20 µs | |
| `waitUntilCompleted` | **2088 µs** | 唯一的支配项 |
| device→host 4 MiB | 230 µs | |

同一插桩在 **6 元素**输入上给出 `htod` 5 µs、`dtoh` 9 µs、`wait` 2088 µs：等待时间与负载大小无关。把同一内核在一次命令缓冲内 dispatch 1/2/4/8/16 次，`wait` 分别为 2163 / 2456 / 2280 / 1744 / 2432 µs：**与 dispatch 次数无关**。故该环境下一次命令缓冲往返的固定成本为 **约 1.7–2.4 ms**，且 `commit` 与第二次 `wait`（缓冲已完成）分别为 20 µs 与 4–16 µs，排除「等待实现低效」的解释。

结论：这是 §21.2 决策 6（同步、运行至完成）施加在**每次算子调用**上的地板，不是内核质量问题。跨过它需要把多次 dispatch 合入同一命令缓冲，即一个跨调用的执行上下文——那是契约改动（见 21.11）。

**2. 编译缓存前后对比（E1，`cargo bench -p llm --bench metal -- --warm-up-time 0.5 --measurement-time 1.0 --sample-size 10`）。** median，同一二进制、同一机器：

| Benchmark | 基线（每次调用重编译） | 缓存后 | 变化 | CPU 对照 |
| --- | ---: | ---: | ---: | ---: |
| `gpu/matmul/…256` | 2.3370 ms | 1.779 ms | −24% | 0.951 ms |
| `gpu/elementwise/…silu` | 3.2692 ms | 2.606 ms | −20% | 1.106 ms |
| `gpu/elementwise/…add` | 3.6305 ms | 2.920 ms | −20% | 0.100 ms |
| `gpu/elementwise/…mul` | 3.7550 ms | 2.934 ms | −22% | 0.101 ms |
| `gpu/softmax/…rows512cols4096` | 4.2474 ms | 3.444 ms | −19% | 2.500 ms |
| `gpu/rmsnorm/…rows512cols4096` | 4.0132 ms | 3.410 ms | −15% | 1.147 ms |
| `gpu/rope/…seq128heads8dim64` | 2.3351 ms | 1.534 ms | −34% | 0.087 ms |
| `gpu/embedding/…gather128` | 3.9703 ms | 3.000 ms | −24% | 0.0067 ms |
| `gpu/attention/…seq128heads8dim64` | 2.9206 ms | 2.199 ms | −25% | 1.101 ms |

缓存把编译成本移出关键路径（每个内核一次，而非每次调用），但每个工作负载的残余成本仍由第 1 条的往返地板决定，故 Metal 仍慢于 CPU——**这不是内核劣势**。此前 21.10 把 Metal 更慢归因于「未做编译器缓存」，该归因已被本次测量修正：编译约 0.85–1.0 ms/内核，往返约 1.7–2.4 ms/调用，后者更大且与负载无关。

**3. dispatch 几何（E1）。** `matmul` 的 batch 循环此前每 block 提交一个命令缓冲，即在往返地板上乘以 batch。改为**一次 dispatch、batch 作为网格第三维**（内核以 `gid.z` 计算三个指针偏移）。同内核、同数据、同机器的对照测量（临时恢复旧调用模式，随后还原）：

| `gpu/matmulbatch/…8x256`（8 × 256³） | median |
| --- | ---: |
| 每 block 一个命令缓冲 | 13.969 ms |
| 一次 dispatch（batch 在网格第三维） | **2.878 ms** |
| CPU 对照 | 7.519 ms |

即 **−79.4%（×4.85）**，且该工作负载下 Metal 由「慢于 CPU」转为**快于 CPU 约 2.6×**。这是本轮唯一一处 Metal 超过 CPU 的测量点，其原因正是它把 8 次往返压缩为 1 次。

**4. 端到端代价（E1）。** `benches/metal.rs` 新增 `gpu/modelforward`：同一微型 Llama（2 层、`dim=64`、`seq=16`）的前向在**同一进程内**分别跑 CPU 与 Metal 两个 provider。median：**CPU 129–130 µs，Metal 50.5–53.5 ms（约 390–414×）**。该前向是一条约 35 次 `compute` 调用的链，53.5 ms ÷ 35 ≈ 1.5 ms/调用，与第 1 条的往返地板一致。这是 21.13 的直接依据，也是任何批处理实现必须击败的基线。

**5. 验证（E1）。** `metalcontract` 43 例在本机 Apple GPU 上全部通过，其中 `every_kernel_is_compiled_once_and_then_reused` 断言「九个算子恰好九个 pipeline，重复一遍仍为九」；`a_clone_shares_the_compiled_pipelines` 断言 clone 共享；`the_executor_is_shareable_across_threads` 断言 `Send + Sync` 且四线程并发结果逐位一致、只编译一次。`unsafe` 站点仍为 4/4（本次改动未新增任何 `unsafe`）。CUDA 侧当时未随动，理由是其实测基线不存在（E4）。21.10.3 现已建立该基线：CUDA 的每调用固定成本约 12.3 ms，其中约 10.8 ms 是 NVRTC 编译，故同一类优化（复用编译产物与设备上下文）对 CUDA 的收益比 Metal 更大；实施次序见 21.11。

#### 21.10.3 CUDA 真机验证与基线（2026-10-12）

环境（E1）：Linux 5.4.0-182-generic x86_64，96 vCPU，375 GiB RAM；GPU 为 NVIDIA GeForce RTX 2080 Ti（`sm_75`，11264 MiB），驱动 580.105.08；CUDA runtime 13.0（libcudart / libnvrtc / libcublas / libcublasLt 13.0、libcudnn.so.9），**无 `nvcc`**（内核由 libnvrtc 在运行期编译）；`rustc 1.99.0 (b940084d7)` 与宿主一致。被测树以 `git archive` 传输并按 sha256 校验（`3cc4fae` → `95d6c494…`；含本次用例修正的树 → `88da3d95…`），故下面每个数字都对应一个可复核的提交。

**1. 契约（E1）。** `LLM_CUDA_REQUIRE_DEVICE=1 cargo test -p llm --test it cudacontract`：

| 轮次 | 结果 |
| --- | --- |
| 首次真机执行（用例修正前） | 35 passed / **1 failed** |
| 用例修正后 | **36 passed / 0 failed** |

失败用例为 `the_two_matmul_paths_agree_on_a_non_square_case`，报文 `lengthmismatch { expected: 28, actual: 20 }`：用例以 shape `[7, 4]` 描述右操作数，却只提供 20 个元素。该套件整体 `#[cfg(target_os = "linux")]`，且设备用例在探测不到 GPU 时提前返回，因此该缺陷自 67ff611 起潜伏，在无 NVIDIA GPU 的开发机上从未被执行。已修正为 28 个元素（提交 `test(cuda): 修正非方阵 matmul 对照用例的右操作数长度`）。这正是「未执行不得记为通过」的价值：Linux 目标上的编译绿灯掩盖了一个必然失败的用例。

全量执行：`LLM_CUDA_REQUIRE_DEVICE=1 cargo test --workspace --locked` → **758 passed / 0 failed**，输出中 **0 条 `E4 未执行`**（即 15 个需要设备的用例确实运行）；`bash scripts/check.sh` → `All static quality checks passed.`（exit 0）。同一树在宿主 macOS 上为 765 passed / 0 failed，两侧均 0 failed。

**2. 每调用固定成本的分解（E1，临时探针，随后从被检树删除）。** 探针只经 `compute` 公共接口计时（每项 20 次取均值，先做一次预热），把与负载无关的部分与负载相关的部分分开：

| 调用 | 工作量 | 每调用耗时 |
| --- | --- | ---: |
| `cudaexecutor::new` | 不触碰设备 | 0.000 ms |
| `matmul` 2×2（cuBLAS） | 4 次乘加 | 1.468 ms |
| `softmax` 1×64（cuDNN） | 64 元素 | 2.373 ms |
| `matmul` 2×2（NVRTC） | 4 次乘加 | 12.255 ms |
| `add` 2×2（NVRTC） | 2 个元素 | 12.309 ms |
| `add` 2^20（NVRTC） | 2^20 元素、8 MiB 往返 | 15.711 ms |

同一进程、同一设备、同一次运行给出的结论：

* `cudaexecutor::new` 不触碰设备，构造本身没有成本；设备上下文是在**每次算子调用内部**创建的。
* 由 NVRTC 服务的算子每调用约 **12.3 ms**，且与内核和负载都无关（2×2 的 `matmul` 与 2 个元素的 `add` 同为 12.3 ms）。其中约 1.5 ms 是设备上下文与库句柄——该量由 cuBLAS 2×2 的 1.468 ms 与 cuDNN 1×64 的 2.373 ms 界定——其余约 **10.8 ms 是 `compile_ptx` 加模块装载**。
* 负载相关部分是二阶项：2^20 元素的 `add` 只比 2 元素的 `add` 多 3.4 ms，其中已包含 8 MiB 的 htod/dtoh 往返与一次 dispatch。
* 结构与 Metal 同构（21.10.2 第 1 条：往返约 1.9–2.0 ms 且与负载无关），但 CUDA 的固定成本高约 6–7 倍，且最大的一项是**运行期编译**而不是提交往返。

**3. 基准（E1）。** `cargo bench -p llm --bench cuda --locked -- --measurement-time 2 --warm-up-time 0.5`，release profile，criterion 0.7.0；每项取一次运行的 median，CPU 与 CUDA 在同一进程、同一轮内测量（故环境漂移同时移动两侧，比值仍可比）：

| Benchmark | 工作负载 | CPU median | CUDA median | 比值 |
| --- | --- | ---: | ---: | ---: |
| `gpu/matmul/…256` | 256×256×256 f32 | 2.7010 ms | 1.4363 ms | CUDA 快 1.88× |
| `gpu/matmulbatch/…8x256` | 8 × 256³ | 20.912 ms | 11.535 ms | CUDA 快 1.81× |
| `gpu/softmax/…rows512cols4096` | 512×4096 f32 | 10.871 ms | 6.2688 ms | CUDA 快 1.73× |
| `gpu/elementwise/…silu` | 2^20 f32 | 4.5173 ms | 13.927 ms | 慢 3.08× |
| `gpu/rmsnorm/…rows512cols4096` | 512×4096 f32 | 3.6619 ms | 16.713 ms | 慢 4.57× |
| `gpu/attention/…seq128heads8dim64` | 128×8×64 f32 | 5.0748 ms | 14.309 ms | 慢 2.82× |
| `gpu/elementwise/…add` | 2^20 f32 | 499.31 µs | 14.859 ms | 慢 29.8× |
| `gpu/elementwise/…mul` | 2^20 f32 | 501.92 µs | 14.829 ms | 慢 29.5× |
| `gpu/rope/…seq128heads8dim64` | 128×8×64 f32 | 327.98 µs | 12.402 ms | 慢 37.8× |
| `gpu/embedding/…gather128` | 128 行 × 1024 | 26.264 µs | 15.941 ms | 慢 607× |

该表与第 2 条一致：CUDA 胜出的三项（`matmul`、`matmulbatch`、`softmax`）全部由 cuBLAS/cuDNN 服务，固定成本约 1.5–2.4 ms；每一个劣于 CPU 的工作负载都由 NVRTC 服务，其 12.3 ms 固定成本中约 10.8 ms 是运行期编译。`embedding` 的 CPU 侧仅 26 µs，故表现为 607×；而 2^20 元素的 `add`（14.9 ms）与 128×8×64 的 `rope`（12.4 ms）耗时几乎相同，正是「固定成本与负载无关」的直接体现。**不声明任何加速**（VF5）：本表是 CUDA 侧下一步优化必须击败的基线。

**4. 与 Metal 的差异（E1）。** 两者都是「一次调用一个提交边界」的后果，但支配项不同：Metal 的地板是命令缓冲往返（约 1.9–2.0 ms，21.10.2 第 1 条），CUDA 的地板是「设备上下文 + NVRTC 编译 + 模块装载」（约 12.3 ms，其中约 10.8 ms 是编译）。故 CUDA 侧收益最大的第一步是**缓存编译产物与设备上下文**（即 21.10.2 第 2 条对 Metal 所做的同类改动），跨调用 dispatch 批处理（21.13）是第二步。

**5. 声明局限。** 上述数字来自单台 RTX 2080 Ti 主机的一次会话，未做多次运行的散布统计（Metal 侧的散布见 21.10.1）；`gpu/modelforward` 的 CUDA 侧未测量（该基准目前只在 `benches/metal.rs`）；`cuda-test` CI job 仍未激活（自托管 runner 未注册，见 21.12），本次为手工执行。

**6. 三后端不一致的一处（E1，已修）。** `softmax` 的契约（见 6.3 与 `crates/cpu/src/softmax.rs`）规定：整行为 `-inf` 时 `max - max` 为 NaN，故该行传播 NaN，而不是被掩成零。首次真机执行发现 cuDNN 的 `CUDNN_SOFTMAX_ACCURATE` **返回全零**，即同一输入在 `cpu`/`metal` 上得 NaN、在 `cuda` 上得 0——违反「同一操作数在三个后端给出同一可观测结果」。修法是最小改动：输入**含任何非有限值**时不再委托 cuDNN，改用与 `cpu`/`metal` 相同的标量式（减行最大值、指数化、和大于零才除）。`tests/it/cudacontract.rs` 新增两例钉住该行为：全 `-inf` 行传播 NaN；五个非有限行（全 `-inf`、含 `+inf`、含 NaN、`-inf` 与有限混合）与 `cpu` 逐元素一致（NaN 与 NaN 记为一致）。

代价（E1，同机同会话 A/B）：该守卫每次调用多做一遍输入的有限性扫描。2^21 元素的 `gpu/softmax/cudarows512cols4096` 在**临时禁用守卫**时为 4.9417 ms，启用后为 6.8184 ms，即约 +1.9 ms；对 8 MiB 输入独立测得一次扫描约 1.18 ms。这是为契约一致付出的显式代价，不是内核变慢。

该守卫是权宜之计，不是终局：`softmax` 的 cuDNN 路径另有约 **2.9 ms/调用**的固定成本（探针实测 `softmax` 1×64，主要是句柄创建，见本节第 2 条的量级）。下一步应把 `softmax` 改由在仓 NVRTC 内核服务——既按构造等于 `cpu` 的算式（守卫随之删除），也免掉每次调用的句柄创建；该改动会变更 §21.2 决策 11 的机制选择与 §21.8 的站点表（`cuda` 3 → 2），**尚未实施**。

**7. 编译产物与设备上下文缓存（E1，2026-10-12 实施）。** 第 2 条把固定成本拆成「设备上下文 1.5 ms + NVRTC 编译与模块装载 10.8 ms」，两者都是每调用一次；据此把 device 0 的已 retain 主上下文与按 `(source, entry)` 索引的已解析内核收进由 executor 拥有、clone 经 `Arc` 共享的 `ffi::cache`（设计、锁序与「执行流故意不缓存」的理由见 §21.6）。同一二进制、同一机器、同一次会话的 median 前后对比（`cargo bench -p llm --bench cuda --locked -- --measurement-time 2 --warm-up-time 0.5`）：

| Benchmark | 缓存前 | 缓存后 | 变化 | 同轮 CPU |
| --- | ---: | ---: | ---: | ---: |
| `gpu/matmul/…256` | 1.4363 ms | 1.5081 ms | +5%（噪声；cuBLAS 路径本就不编译） | 2.6424 ms |
| `gpu/matmulbatch/…8x256` | 11.535 ms | 11.276 ms | −2%（噪声，同上） | 20.712 ms |
| `gpu/elementwise/…silu` | 13.927 ms | 2.5729 ms | **−81.5%** | 4.4583 ms |
| `gpu/elementwise/…add` | 14.859 ms | 2.9617 ms | **−80.1%** | 513.33 µs |
| `gpu/elementwise/…mul` | 14.829 ms | 2.7269 ms | **−81.6%** | 519.16 µs |
| `gpu/softmax/…rows512cols4096` | 6.2688 ms | 6.8184 ms | +8.8%（其中约 +1.9 ms 是第 6 条的守卫） | 10.893 ms |
| `gpu/rmsnorm/…rows512cols4096` | 16.713 ms | 4.4286 ms | **−73.5%** | 3.6833 ms |
| `gpu/rope/…seq128heads8dim64` | 12.402 ms | 0.95834 ms | **−92.3%** | 321.70 µs |
| `gpu/embedding/…gather128` | 15.941 ms | 4.1326 ms | **−74.1%** | 26.410 µs |
| `gpu/attention/…seq128heads8dim64` | 14.309 ms | 1.2974 ms | **−90.9%** | 5.1487 ms |

由 NVRTC 服务的七个工作负载全部大幅下降（−74% 至 −92%），两个 cuBLAS 工作负载不变（路径本就不编译），符合第 2 条的归因。**CUDA 快于 CPU 的工作负载由 3 个增至 5 个**（`matmul` 1.75×、`matmulbatch` 1.84×、`silu` 1.73×、`attention` 3.97×、`softmax` 1.60×）；仍慢于 CPU 的是五个小或中等负载（`add` 5.8×、`mul` 5.3×、`rmsnorm` 1.20×、`rope` 2.98×、`embedding` 156×）。

**缓存后的固定成本（E1，探针，同一会话两次运行）。** `cudaexecutor::new` 仍为 0.000 ms（不触碰设备）；NVRTC 服务的算子 2 元素调用 **0.98–1.08 ms/调用**（缓存前 12.3 ms），其中已含设备上下文（缓存命中）、一条新执行流、两次传输、一次 dispatch 与同步；同一个 `add` 加到 2^20 元素为 2.66–2.75 ms，即 8 MiB 往返约 1.7 ms。cuBLAS 2×2 为 1.47–1.59 ms，cuDNN 1×64 为 2.85–2.98 ms：这两个库路径的固定成本如今由**句柄创建**主导，是剩余的下一项（第 6 条已记录 `softmax` 的替代方案；`matmul` 的同名问题记为下一个复审点）。

**验证（E1）。** `tests/it/cudacontract.rs` 在真机上 41 passed / 0 failed，其中三例钉住缓存本身：`every_kernel_is_compiled_once_and_then_reused`（七个 `(source, entry)` 对服务九个算子，NVRTC 路径多一个 matmul 内核；第二遍不增长）、`a_clone_shares_the_compiled_kernels`、`the_executor_is_shareable_across_threads`（`Send + Sync`，四线程并发 `silu` 结果逐位一致且只编译一个内核）。`unsafe` 站点仍为 `cuda` 3/3（缓存没有新增任何平台交互）。

### 21.11 复审条件

* 若某种纯安全的 GPU 绑定成熟且能满足 CUDA 专用能力诉求 → 评估撤销 21.8 的豁免。
* 若 `metal` 或 `cuda` 的 `unsafe` 站点数需要增长 → 先更新 21.8 的登记表并说明理由，再改动代码。
* 若再引入第三个自带 `unsafe` 入口的 CUDA 专用库（例如 cuDNN 的 attention frontend 或 `ActivationForward`）→ 先更新 21.8 的登记表并说明理由，再改动代码。
* 若出现流式/异步执行需求 → 复审 21.2 决策 6 的同步语义与 21.6 的并发设计。
* 若需消除每调用 MSL 编译与设备上下文创建开销 → 引入由显式所有权持有的编译产物/上下文缓存（而非全局单例），并对照 21.10 的基线测量前后差异。**已于 2026-10-12 实施**（21.10.2）：设备与 pipeline 缓存属于 executor，命令队列**故意不缓存**（共享队列会串行化并发调用的 GPU 工作；被否决的备选）。残余开销已是命令缓冲往返本身。
* 若需要跨过「每次调用一次命令缓冲往返」的地板（21.10.2 第 1 条）→ 需要把多次 dispatch 合入同一命令缓冲，即一个跨调用的执行上下文。**设计已于 2026-10-12 提出（见 21.13），实现分三阶段、尚未开始**；在实施前不得改动冻结的 `compute` 语义。
* 若需消除 CUDA 每次调用的设备上下文创建与 NVRTC 编译开销 → 引入由显式所有权持有的编译产物/上下文缓存（而非全局单例），并对照 21.10.3 的基线测量前后差异。**已于 2026-10-12 实施**（21.10.3 第 7 条）：七个 NVRTC 工作负载 median 下降 74–92%，`unsafe` 站点仍为 3/3。残余固定成本已是执行流创建、传输往返与 cuBLAS/cuDNN 的句柄创建。
* 若需消除 `matmul`/`softmax` 每次调用的库句柄创建开销（21.10.3 第 7 条实测 cuBLAS 1.47–1.59 ms、cuDNN 2.85–2.98 ms 的固定成本）→ 复审「句柄是否可以跨调用复用」：句柄绑定执行流，复用要么共享执行流（会串行化并发调用，已在 21.6 否决），要么在句柄上做同步。替代方案是把 `softmax`（以及必要时的 `matmul`）改由在仓 NVRTC 内核服务——按构造与 `cpu` 算式一致，并同步 §21.2 决策 11 与 §21.8 站点表（3 → 2）。**优先实施 softmax 一项，尚未开始。**
* 若设备在运行期可能消失（eGPU 移除、GPU 重置）→ 复审 21.6 的「设备缓存不重新探测」局限，设计失效重建与已编译 pipeline 的作废规则。
* 若 `objc2-metal` 或 `cudarc` 的主版本升级 → 重新执行尖刺并核对 21.1 与 21.8 的实测结论。

### 21.12 CI 门禁

在既有 `quality` job 之外新增三个 job：

| job | runner | 作用 | 现状 |
| --- | --- | --- | --- |
| `quality` | `ubuntu-latest` | 不变：fmt、全工作区 check/clippy/test/doc 与基准编译。因 `cuda` 在 Linux 上非空，它在 Linux 上已覆盖 CUDA 的编译与 lint | 既有 |
| `macos` | `macos-14` | `cargo test --workspace`，使 `metal` 真实编译并运行 MSL 内核 | 新增 |
| `cuda-check` | `ubuntu-latest` | `cargo check -p cuda --all-targets`；运行 `cudacontract` 的设备无关一半；外加 `scripts/checkunsafe.sh` 与 `scripts/checkthresholds.sh`。**故意不安装 CUDA toolkit**，`dynamic-loading` 使其不必要 | 新增 |
| `cuda-test` | `[self-hosted, linux, x64, cuda]` | `cargo test -p llm --test it cudacontract`，并设 `LLM_CUDA_REQUIRE_DEVICE=1`，使「探测不到设备」成为硬失败而非静默通过 | **未激活**（runner 未注册；2026-10-12 已在 RTX 2080 Ti 上手工执行同一命令，见 21.10.3） |

`cuda-check` 为何仍需单独存在：宿主 macOS 的 `checkthresholds.sh` 看不到被 `cfg(target_os = "linux")` 门控的代码，而 `checkunsafe.sh` 与 `checkthresholds.sh` 本来就不在 `quality` 的步骤里；该 job 把这两项放到 CUDA 代码确实非空的平台上以求值——在该 runner 上 `checkthresholds.sh` 的 `cargo clippy --workspace --lib --bins` 原生覆盖 CUDA 的 Linux 代码，故 DS3 在那里是机械强制的，而非仅在宿主上另跑一次交叉目标 clippy。此外 `quality` 的 `cargo test --workspace` 与 `cuda-check` 的 `cudacontract` 步骤都会执行该套件的设备无关一半。

`cuda-test` 当前只在 `workflow_dispatch` 下运行，因为自托管 runner 尚未注册，而在 `push`/PR 上加入一个无法调度的 job 会让整个工作流永久处于 pending。runner 注册后删除该 job 的 `if:` 一行即可启用。

默认 feature 不引入任何 GPU 依赖，故 `quality` job 的语义未变。

未决（需仓库所有者提供）：自托管 CUDA runner 的标签与配置。

## 21.13 执行上下文与 dispatch 批处理（**已验证**）

> 状态：**已验证（Verified）**。本节记录一个由测量确定的架构缺口、被采纳的设计方向、被否决的备选方案与分阶段的实施顺序。该契约已按 21.13.5 的顺序在三个后端（`cpu`、`metal`、`cuda`）实施；三后端的「同一链经上下文与经 `compute` 逐位相同」测试与全部静态门禁均已执行并通过（见 21.13.7）。性能验收也已达成：`gpu/modelforward` 的 Metal 数字由 52.60 ms（35 次同步往返的 `compute` 基线）降至 2.81 ms（整链一次 commit，`forwardsubmission`），**改善 ≈18.7×**，远超 10× 标准；且 `forwardsubmission` 与 `forward` 逐位一致，不是以正确性换速度。

### 21.13.1 上下文（测量事实）

一次同步算子调用的固定成本是**一次命令缓冲往返**，实测约 1.9–2.0 ms，且与下列因素**无关**：

| 变量 | 取值 | 完成延迟 |
| --- | --- | --- |
| 负载大小 | 6 元素 / 2^20 元素 | 1689 / 2088 µs |
| 一次命令缓冲内的 dispatch 数 | 1 / 2 / 4 / 8 / 16 | 2163 / 2456 / 2280 / 1744 / 2432 µs |
| 等待方式 | `waitUntilCompleted` / 忙等 `status()` | 2088 / 1754–2001 µs |
| 缓冲存储模式 | `StorageModeShared` / `StorageModePrivate` | 2037 / 1920 µs |
| 宿主虚拟化 | `kern.hv_vmm_present = 0`（Apple M5 Pro 裸机） | — |

`commit` 仅 20 µs，缓冲完成后的第二次等待仅 1–16 µs，故该延迟是提交到完成的真实往返，不是等待实现的产物。

因此**调优 kernel 与「小/中工作负载慢于 CPU」无关**：256³ matmul（17M 次乘加）与 6 元素 `silu` 的总耗时几乎相同（1.79 ms 与 1.75 ms）。

端到端代价（`benches/metal.rs` 的 `gpu/modelforward`，2 层、`dim=64`、`seq=16` 的 Llama 前向）：**CPU 129–130 µs，Metal 50.5–53.5 ms（约 390–414×）**。该前向是一条算子**链**，每层 16 次 `compute` 调用（`rmsnorm` ×2、`matmul` ×6、`rope` ×2、`attention` ×1、`add` ×2、`silu` ×1、`mul` ×1），两层加模型头共约 35 次；53.5 ms ÷ 35 ≈ 1.5 ms/调用，与往返地板一致。

结论：**在「一次调用一次提交、返回拥有张量」的语义下，GPU 后端无法在小/中工作负载上胜过 CPU，与内核质量无关**。要改变这一点，只能把多次 dispatch 合入同一命令缓冲。

### 21.13.2 决策（已实施）

新增一个**累加式**的执行上下文契约，`compute` 本身保持冻结、一行不改：

* **位置。** 契约定义在 Layer 0（`api`），因为它是「后端无关的提交边界」，与 `compute` 同级；具体实现分散到各后端。
* **形状。** 调用方先**声明**一串步骤（每个步骤引用先前步骤的结果或自己提供的输入），再提交一次；提交返回被声明为输出的那些张量。步骤的操作集合是 `compute` 已有算子的复述，不引入新数值语义。
* **可选性。** `compute` 是默认路径；上下文只在调用方明确选择时使用。CPU 以「按序调用既有 `compute`」实现同一契约，因此三个后端语义一致，性能上按各自硬件取最优。

### 21.13.3 不变量与所有权（已实施）

* **所有权。** 上下文由调用方拥有、请求作用域；它拥有该请求的全部中间设备缓冲与一次提交的命令缓冲。它**不**拥有模型权重，也不得把请求数据写入任何跨请求存活的结构。
* **请求隔离（OW2/OW3）。** 两个并发上下文不共享任何可变状态；executor 的编译缓存（只含设备句柄与已编译 pipeline，见 21.6）可继续共享，因为它不含请求数据。
* **预算。** 每个中间结果的形状在**声明时**校验（复用 `api::checkbudget`），因此没有任何分配发生在预算校验之前；声明阶段的失败不产生设备操作。
* **顺序。** 步骤按声明顺序执行；同一 encoder 内相邻 dispatch 之间的依赖由 Metal 对 `MTLResource` 的默认 hazard tracking 保证，不依赖调用顺序假设。
* **错误归属（ER2）。** 任一步失败即中止本次提交并返回该步的算子名与既有 `backenderror` 变体，不新增错误类型；已获取的设备资源由上下文析构释放，不泄漏、不跨请求复用。
* **取消。** 提交前丢弃上下文即释放全部资源；提交**期间**的取消不由同步契约表达，明确列为范围之外。
* **上限。** 上下文可声明的步骤数与中间元素总量都受调用方提供的 `limits` 约束；超限在声明时拒绝。

### 21.13.4 被否决方案（DS5）

* **继续调优 kernel / 提高 kernel 并行度。** 被否决：实测固定成本与负载无关，256³ matmul 与 6 元素 `silu` 同价，内核改进无法触及该成本。
* **缓冲池或改变存储模式。** 被否决：实测存储模式对完成延迟无影响（2037 对 1920 µs）；传输仅占大负载总耗时的约 19%（4 MiB 上传 265 µs + 下载 230 µs 于 2.6 ms 中），且对 `embedding` 这类小负载为 0%。它是二阶项，不是瓶颈。
* **把 `compute` 改成异步/流式。** 被否决：那会改动一个所有后端与所有上层都已依赖的冻结契约，并要求每个调用方改为管理未完成结果的生命周期；累加式上下文以零破坏达成同一目标。
* **全局单例上下文 / 进程级设备管理器。** 被否决：破坏测试隔离与所有权（21.6 已排除全局单例），且无法为不同请求提供不同的预算。
* **为特定算子对新增融合算子（如 `rmsnorm`+`matmul`）。** 被否决：需要每对一次 API 扩张，且对 16 步的链缺乏一般性；批处理把任意长度的链都压缩为一次提交。

### 21.13.5 分阶段实施顺序（每阶段独立可测、可回滚，均已实施）

1. **契约与 CPU 实现。** 在 `api` 定义提交边界；`cpu` 以「按序调用既有 `compute`」实现之。验收：同一链经上下文与经 `compute` 的结果**逐位相同**；声明阶段的拒绝（形状、下标、预算、秩）与 `compute` 一致。
2. **Metal 实现。** 把 `ffi.rs` 的每个算子拆成「分配槽位 → 编码进 encoder → 下载 sink」三段，全部步骤共用一个 compute encoder 与一次 commit/wait。验收：`unsafe` 站点仍为 4/4（复用同一组上传/下载/绑定站点）；结果与阶段 1 逐位相同；`gpu/modelforward` 的 Metal 数字相对约 51 ms 的基线至少改善 10×。
3. **CUDA 实现。** 在自托管 GPU runner 可用后以其自身提交语义实现同一契约（不预设与 Metal 相同的机制）。

### 21.13.6 验收标准与回滚

* **正确性。** 任何链经上下文与经 `compute` 逐位相同（同一内核、同一累加顺序）；全部既有测试保持通过。
* **性能。** `gpu/modelforward` 的 Metal 数字相对约 51 ms 的基线至少改善 10×；CUDA 侧以 21.10.3 的真机基线为准，且应先实施 21.11 的编译产物/上下文缓存（≈10.8 ms/调用），否则跨调用批处理的收益会被该固定成本掩盖。
* **门禁。** `unsafe` 站点数不变；`api`/`metal` 的 A4 导出宽度仍在上界内；`check.sh` 全绿。
* **回滚。** 该契约是累加式的：删除上下文类型与其实现即完全回到当前状态，调用方无迁移成本。

### 21.13.7 实施与验证（2026-10-15）

契约在 `api`（`submission`/`stepref`/`operand`）定义，`compute` 冻结一行未改；`cpu` 以「按序重放 `compute`」实现，`metal` 以「一个 command encoder + 一次 commit/wait、中间结果驻留设备」实现，`cuda` 以「一个 stream 串行 + 一次 synchronize、中间结果驻留设备」实现。

* **cpu。** `tests/it/submissioncontract.rs` 8 例通过：多算子链经 `submission` 与经 `compute` 逐位相同（`assert_eq!` 精确）、声明阶段拒绝与 `compute` 同变体。
* **metal。** `tests/it/metalcontract.rs` 45 例在 Apple M5 Pro 上通过，含 2 例 `submission`（逐位相同、越界 `stepref`）；`unsafe` 站点仍 4/4（复用 `write`/`read`/`bind`/`setbytes`）。首轮暴露 `session` 缺 `impl Drop`：中途错误路径丢弃 session 时 encoder 未 `endEncoding` 即被释放，Metal 在 dealloc 时断言（SIGTRAP）；补 `ended` 标志 + `Drop` 兜底后 45/45 全过。
* **cuda。** `tests/it/cudacontract.rs` 43 例在 NVIDIA T600（Windows WSL2 Ubuntu 24.04）真机上通过，含 `submission_reproduces_a_compute_chain_bit_for_bit` 的 bit-for-bit 复现；`unsafe` 站点仍 3/3（复用 `submit`/`gemm`/`softmax_forward`，后两者为单算子路径抽出的 helper）。
* **平台修复。** `metal` 新增 `build.rs` 显式链接 `CoreGraphics` 框架：`MTLCreateSystemDefaultDevice` 要求 CoreGraphics 链接，而 `objc2-core-graphics` 依赖无本 crate 使用的符号，其 `#[link]` 被死代码消除，设备探测会静默返回空。
* **公共化。** 平台无关的 `geometry`（shape 校验）从 `metal`/`cuda` 各一份复制抽到 `api::geometry`，`attentionshape::new` 增 `maxheaddim` 参数承载唯一的设备差异（Metal 4096 vs CUDA 8192）；消除 607 行重复。

* **接入与性能。** `llamamodel` 与 `llamablock` 增 `forwardsubmission`，把前向链的 35 步算子内联进一个 `submission`；`tests/it/submissionforward.rs` 2 例在 CPU 上钉住 `forwardsubmission` 与 `forward` 逐位相同。`gpu/modelforward` 实测（Apple M5 Pro）：`metal` 52.60 ms（35 次同步往返的 `compute` 基线）、`metalsubmission` 2.81 ms（整链一次 commit），**改善 ≈18.7×**，远超 10× 验收，且逐位一致。无需权重驻留设备即达成。

## 22. 跨领域主题与声明局限

### 22.1 请求隔离（OW2 / OW3 在层间的体现）

* **infer。** 模型权重不可变且不持请求状态。所有生成状态（prompt、累积上下文、采样器位置、停止准则、`kvstate`）活在 `generationrequest` 中，由引擎借用的模型与提供者不持每请求状态。故两个请求可对一个模型值并发运行而不共享可变状态。`reset` 在成功、取消、失败三条路径后都恢复请求到新尝试，使残留状态不可被错误复用。
* **serve。** `inferenceservice` 只借用不可变模型与提供者，拥有不可变 `serviceconfig` 加原子计数器；每请求的 `executioncontext`/`generationrequest`/sink 是请求作用域并在 drop 时释放。每个响应 `connection: close`，故无会话状态在请求间保留。取消经请求的 `cancellation` 句柄映射，不存入服务或全局。
* **cli。** `generate` 的请求使用请求作用域 `generationrequest`、调用方拥有的 `executioncontext` 与 `collectingsink`；所有可变状态是请求作用域且 drop 时释放。进程级单线程一次处理一个命令。
* **train。** `trainer` 拥有其参数、优化器、累加器与计数器并只经 `&mut self` 被变更；取消是唯一共享信号且经显式 `context::cancellation` 句柄。失败的加载对 trainer 是 no-op，失败或取消的步骤丢弃窗口且不推进任何计数器。

### 22.2 资源上限在层间的体现

* **context** 是上限的原语所有者：`executionconfig { memorybytes, maxworkers, maxsteps }`（零 `maxworkers` 被拒）、`memorybudget` 的字节记账（`reserved ≤ capacity`，溢出与超容量均显式失败）、`workergroup` 的有界扇出、`cancellation`。
* **infer** 显式限定：`generationparams { maxseq, kvlayers, kvwidth }` 全非零；`stopcondition` 的非零 `maxnewtokens`；`generationrequest` 拒绝 prompt 超过 `maxseq` 或 prompt 加 cap 超过 `maxseq`，且 `pushgenerated` 在上下文达到 `maxseq` 后拒绝；`kvstate` 每次 append 校验容量。每次循环起点 `executioncontext.checkpoint()` 强制步骤预算。
* **serve** 在昂贵工作前强制：调用方提供的 `transportlimits` 限定请求行、header 行、header 数量、content-length 与 body；`Read::take(maxrequestbytes)` 限定每连接总字节；表单字段数量与长度有界；`admissions` 以固定 `maxconcurrent` 上限在模型运行前拒绝（HTTP 429）；可选 deadline 映射到取消（HTTP 504）。
* **cli** 使用显式常量：`DEFAULTLIMITS`（元素/字节/序列/批）、`MAXNEWTOKENSLIMIT`、`MAXPROMPTTOKENS`、`DEFAULTMAXARTIFACTBYTES`、`DEFAULTMEMORYBYTES`、`DEFAULTMAXWORKERS`；路径在读前按大小检查、读后复核（复制-校验-使用）。
* **optim / train** 以经校验构造器限定超参数与配置域；`trainconfig` 拒绝零累积窗口，`trainer::step` 强制 `maxsteps` 预算，`gradientaccumulator` 拒绝零容量与满窗口。

### 22.3 错误归属在层间的体现（ER2）

每个 crate 拥有其职责的错误类型，跨层传播时以 transparent 方式保留上下文，不把底层错误类型泄入上层契约：

| crate | 自有错误 | 透传携带的下层错误 |
| --- | --- | --- |
| `optim` | `opterror` | `tensorerror` |
| `context` | `contexterror` | （无；含 `workererror<E>` 泛型协调错误） |
| `train` | `trainerror` | `tensorerror`、`opterror`、`checkpointerror` |
| `infer` | `infererror` | `modelerror`、`contexterror` |
| `cli` | `clierror` | `formaterror`、`checkpointerror`、`modelerror`、`infererror`、`trainerror`、`contexterror`、`tokenizererror`、`opterror` |
| `serve` | `serveerror` | `infererror`（`serveerror::inference`） |

应用边界在各自层做协议映射：`clierror::exitcode` 是到退出码的单一映射；`httpresponse::fromerror`（经 `statusfor` 与 `inferstatus`）是到 HTTP status 的单一映射。传输失败与模型失败永不互相折叠：`serveerror` 的协议变体保持独立，而 `serveerror::inference(#[from] infererror)` 保持 `infererror` 的可操作区分。

### 22.4 声明局限汇总

* **`nn`/`models` 是 forward-only，无反向路径。** `autograd` 只为 `matmul`、`add`、`mul`、`rmsnorm`、`softmax`、`relu`、`silu`、`gelutanh` 提供 VJP。因此不存在从 Llama 前向回到其权重的可微图；端到端 Llama 训练不可达，`train` 与 `cli train` 均不捏造它。`cli train` 在无 `--check-only` 时返回 `clierror::notsupported` 并 exit 4。
* **KV cache 已实现但尚未被前向读取。** `kvstate` 满足 `models::kvcache` 并由请求拥有、重置，但 `causallm::forward` 无缓存参数，故 decode 重算完整上下文。缓存路径尚无前后对比数字，不声称加速。
* **`cli generate` 限于确定性微小模型。** 把 `generate` 指向真实 checkpoint/权重是复用同一引擎接缝的未来工作。
* **`serve` 是仅标准库的阻塞 HTTP/1.1。** 无异步 I/O、无 TLS、无 keep-alive；超出单进程的扩展依赖部署层。
* **`testkit::tinymodel` 当前为空。** 微小确定性模型夹具待 `nn`/`models` 层提供定义层、参数与初始化的能力后补齐。
* **数值与并行范围。** `optim` 是 `f32`-only 且单进程；不支持 Nesterov 动量、dampening 或 AdamW 之外的自适应方法。`train` 的 loss、累加器与 checkpoint 编码是 `f32`-only。分布式、混合精度与多设备训练均未实现。
* **`train` 尚不能从 CLI 选择学习率调度文件字段。** `trainconfig` 接受可选 `lrschedule`，但 CLI schema 省略它。

## 23. 复审条件汇总

以下汇总第 13–21 章的复审触发点，按 crate 归类。当任一条件出现时，回到对应章节并对照本节的验收标准复核。

### 23.1 optim（见 13.10）

* `autograd` 暴露稳定公共梯度访问器 → 把图变量到 `tensor` 切片的薄适配器放入 `train`（优先），而非新增 `optim → autograd` 边。
* 需要 Nesterov、dampening 或第二种自适应方法 → 在现有 `optimizer` trait 后扩展规则集，而非拓宽 `update`。
* 训练层持久化优化器状态 → 对照 checkpoint 张量契约确认 `m.{i}`/`v.{i}`/`momentum.{i}` 命名与 `optimizerstate` 布局；改名是兼容性变更。
* 需要分布式或混合精度优化器 → 复审 `f32`-only 与单进程假设。

### 23.2 context（见 14.10）

* 需在共享结构上持有多于一把锁 → 定义并记录全局锁序（`CC1`/`DS4`）。
* 取消需中断阻塞叶子调用 → 新增可取消等待原语而非全局注册表。
* 需要生产指标后端 → 作为新 `metrics` 实现加入，保持 trait 无后端耦合。
* 工作者协调需要异步执行或带优先级的 work stealing → 复审 `std::thread::scope` 设计并论证新增依赖。

### 23.3 train（见 15.12）

* `nn`/`models` 获得反向路径，或 `autograd` 获得缺失算子 → 把真实前向接进每步 loss 闭包，并用 `models` 端到端 loss 下降测试替换最小二乘演示。
* 训练循环直接从 `autograd` 图获得梯度 → 在 `train`（而非 `optim`）加入图到 `tensor` 适配器。
* 需要数据并行或多设备训练 → 复审单所有者 trainer 并加入梯度归约的显式同步。
* 需要混合精度 → 复审 `f32`-only 的 loss、累加器与 checkpoint 编码。

### 23.4 infer（见 16.11）

* 存在缓存感知的 forward 入口 → 把 `kvstate` 接进 decode 并对照全重算路径记录前后基线。
* `format`/`checkpoint` 提供序列化模型 → 加入 load 路径与黄金 token 生成测试。
* 需要新的采样规则 → 扩展 `samplingparams` 并配参考测试，而非改动 greedy 路径。
* 流式 sink 需拥有模型或跨调用存活 → 复审调用作用域的 `S: streamsink` 约束。
* 根 `tests/`/`benches/` 整合发生 → 移动集成测试到位并加入生成基准。

### 23.5 cli（见 17.10）

* `nn`/`models` 的反向路径落地 → `train` 在同一命令后长出真实权重更新路径，移除 `notsupported` 局限。
* `generate` 的真实工件加载器落地 → 加入 `--checkpoint`/`--weights` 并重新测量。
* 根 `tests/`/`benches/` 整合发生 → 移动集成测试到位。
* 加入第二个具体后端 → 复审后端选择决策。

### 23.6 serve（见 18.10）

* 部署目标需要异步 I/O 或 TLS → 在相同 `inferenceservice` 边界后加入异步传输，并对照本基线测量吞吐变化。
* 根 `tests/`/`benches/` 整合发生 → 移动集成测试到位并加入请求吞吐基准。
* 需要流式响应 → 扩展 `httpresponse`/sink 契约而非传输解析器。
* 缓存感知的 `infer` decode 落地 → 在不改变本边界的前提下重新测量请求延迟。

### 23.7 testkit（见 19.10）

* `nn`/`models` 落地且需要共享微小确定性模型夹具 → 在 `tinymodel` 中定义，并与 `tests/common/mod.rs` 既有夹具不重复。
* 出现被多个 crate 共享的新确定性输入或独立参考实现 → 优先加入 `fixtures`/`reference`。
* 需要新的数值比较语义 → 扩展 `assertions` 并保持其仅用于测试基础设施。

### 23.8 bench（见 20.10）

* 加入新的生产算子 → 在根 `benches/` 添加基准目标并复用 `bench` 的确定性负载生成器。
* 出现被多个基准共享的新负载 → 把生成器加入 `bench` 库而非在各目标中重复。
* 资源预算或负载维度改变 → 重新确认 `OPLIMITS` 仍准入每个负载且计时区内无分配。

### 23.9 metal 与 cuda（见 21.11）

* 出现可满足 CUDA 专用能力的纯安全 GPU 绑定 → 评估撤销 21.8 的 `unsafe` 豁免。
* `metal` 或 `cuda` 需新增 `unsafe` 站点 → 先更新 21.8 登记表再改代码。
* 再引入第三个自带 `unsafe` 入口的 CUDA 专用库 → 先更新 21.8 登记表并说明理由，再改代码。
* 出现流式/异步执行需求 → 复审 21.2 决策 6 与 21.6。
* `objc2-metal` 或 `cudarc` 主版本升级 → 重跑尖刺并核对 21.1/21.8。
