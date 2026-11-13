# 0018：精度（dtype）单一事实源

- 状态：已采纳
- 日期：2026-11-12

## 背景

重构前精度闭集在仓内有三套内在枚举，各自手写全量表、位宽与跨层映射：

- `tensor::DType`：5 变体（F32/F16/BF16/I8/U4）；
- `prof::atom::DType`：9 变体（含 F8e4m3、F4 等）；
- `weights::dtype::Wire`：21 变体，带稳定数字 code。

命名也不统一（`BF16` vs `Bf16`、`U4` vs `I4`），新增一种精度要在三处同步并维护
映射函数，重复且必然漂移。外部码镜像（GGUF、torch bin、C ABI 的 CType）是合理防腐层，
不属于本次重复。

## 决策

1. 在 L1 新建零依赖 crate `dtype`（与 `diagnostic` 同级），定义单一全量枚举 `DType`，
   共 22 变体：旧 Wire 的 21 个 + prof 独有的 `F4`。
2. 前 21 个变体的数字 code 与旧 Wire 逐位相同（1..21），`F4` 追加为 22，code 0 保留；
   于是 LLPK 容器字节布局不变，容器字节兼容。
3. 判别值 fit 一个 `u8`，code/位宽/名字/分类/字节数全部为 `const fn`，零运行时开销；
   各层不再写跨层映射函数。
4. 局部能力以子集表达：`tensor::SUPPORTED` 五个 + `supports()`，`Tensor::new` 构造期校验；
   `weights` 以 `pub use dtype::DType as Wire` 薄模块保留容器词汇；`prof` 以
   `impl Closed for DType` 接上词汇表契约。
5. 外部码镜像（GGUF/bin/CType）保留为薄防腐层，不并入。

## 已否决的备选

- 把精度放进 `proto`/`comm`：它们在 L5 且职责不符，低层反而看不到；
- 各层保留小 enum：Rust 无法对子集化 enum，跨层映射函数无法消除；
- 加 I3/I5/I6/I7：无硬件对应，闭集不携带无消费者的变体；
- 把 TF32 作为 dtype：TF32 没有独立存储布局，应走算子属性（`AttrKey::Accuracy`）。

## 后果

- 三套枚举与手写映射删除，新增精度只改 `dtype` 一处；
- LLPK 容器字节兼容由 `dtype` 的 legacy code 对齐测试持续保证；
- 架构门禁（成员/边/所有权）与词汇表门禁（白名单过期）随事实源同步更新；
- 新增 `dtype → （无工作区依赖）`，被 `tensor`/`prof`/`weights` 依赖，依赖方向全部向下。

## 补充（2026-11-13）：精度类型按三个正交轴分文件

本 ADR 采纳时 `dtype` 只有一个 `DType` 枚举（22 变体），把「标量元素」与「块量化」混在
同一张表里。后续按业界三类正交轴拆分，仍保持「唯一事实源」不变：

1. `dtype.rs` 的 `DType`：标量元素类型（一个元素怎么存、无 scale）。补 `F8E4M3FNUZ` /
   `F8E5M2FNUZ` 两个 AMD/OCP 变体，追加为 code 23/24，既有 1..=22 一个不动。
2. `qtype.rs` 的 `QType`：GGML 块量化（一块元素怎么打包、块内共享 `f16`/`f32` 线性 scale），
   25 个格式覆盖传统 Q、K-quant、IQ 查表、TQ 三元四族。此前散在 `weights::gguf` 的私有
   `Kind` 由它取代。
3. `mxtype.rs` 的 `MxType`：微缩放（低精度元素 + 块共享纯指数 scale），OCP MX v1.1 六个
   标准格式（MXFP8 E4M3/E5M2、MXFP6 E3M2/E2M3、MXFP4 E2M1、MXINT8）+ NVFP4 厂商扩展。

三者互不引用，`lib.rs` 只做薄入口与再导出。分文件判据是**语义轴不同**，不是大小：MX 的
scale 是 E8M0 纯指数、块由 OCP 规范固定，与 GGML 的 `f16` 线性 scale、可变块（32/128/256）
是两套独立标准，混进同一枚举会让反量化分支互相污染。

**分工线**：三个文件只放「格式身份 + 布局元数据」（块大小/每块元素数/每块字节数/位宽/
scale 种类/家族），反量化与解码算法仍留在 `weights`，`dtype` 维持零依赖。

**不入类型层的两个边界**：`NF4`（Meta QLoRA 的 NormalFloat4）是「数据编码算法」不是硬件
格式，留在 `weights` 的反量化逻辑；`NVFP4` 是厂商扩展，作为 `MxType` 的一个变体列出而非
另立标准码表。块布局与字节数逐项以 `ggml-common.h` 的 `static_assert` 为据，并由
`qtype`/`mxtype` 的单测锁死，防止与上游规范漂移。

## 补充（2026-11-13）：闭集的泛型视角

三个格式闭集的身份读法（闭集全量、判别值即下标、规范名、位宽、按名解析）此前靠
`blocked_accessors!` 在 `qtype`/`mxtype` 各展开一次、`DType` 再手写一份；上层也只能重造：

- `prof::atom` 只能在 L2 手写 `impl Closed for DType`（`ALL`/`as_str`/`index`）；
- `weights::gguf::Layout` 自带一份 `elems`/`span` 的「标量手臂 + 块手臂」匹配。

契约因此下沉到 `dtype`：

1. `Format`：闭集身份（`ALL` / `index` / `name` / `bits` / 派生的 `parse_name`），三个格式闭集各实现一次（`Element` 见下一节）；
   `prof` 用一条 `impl<T: dtype::Format> Closed for T` blanket 接上，闭集表不再按类型手写。
2. `BlockFormat: Format`：块布局（`block_len` / `block_bytes` / 派生的 `block_count` 与饱和的
   `bytes_for`），`QType`/`MxType` 实现。`DType` 刻意不实现——亚字节标量按位跨元素打包，
   硬套「每块字节」会把 `U4` 的字节数算成两倍。
3. `Layout<B>`：`Scalar(DType)` 或 `Block(B)`，几何只有一份；`weights` 用 `Layout<QType>` 别名，
   MXFP4/NVFP4 解码接入时换 `Layout<MxType>`，不必再写第二份 `elems`/`span`。

**不变量没变**：trait 方法是普通 `fn`（stable 还没有 const trait），常量求值仍走各枚举的
inherent `const fn`，两条路径读同一张 `SPECS`，由 `crates/dtype/tests/formats.rs` 逐项对拍。
`dtype` 依旧零依赖、`no_std`，新增抽象不引入分支、分配或堆。

## 补充（2026-11-13）：`Element` 进入闭集纪律，互操作下沉为可选 feature

`mxtype::Element` 此前是唯一还在用 `match` 表达身份的子闭集（`bits` 两条手臂、`dtype` 五条手臂），
于是 `DType` 上已经修掉的「第四份身份清单」在它身上又出现一次。现在它与另外三个轴同构：
`ELEMENT_SPECS` 行表（name / variant / bits / dtype）按判别值索引，`Element::ALL` 锚定表长，
`Format` / `Display` / `FromStr` 由同一份行表派生。

四个闭集同时补上标准互操作，且都不引入运行时成本：

1. `FromStr` 转调 `Format::parse_name`，错误是零分配的 `UnknownName`；`DType` 另实现
   `From<DType> for u16` 与 `TryFrom<u16>`（错误 `UnknownCode`），并由穷举 65536 个 `u16`
   的测试锁定「合法 code 恰好是 `1..=ALL.len()`」。`QType`/`MxType` 的判别值只是行表下标、
   不是线上码，所以不提供数字转换——给一个不存在的字节码一个名字才是撒谎。
2. 刻意不实现 `Default`：一个默认精度会静默宣布一个没有出现在文件头里的类型；读不到就失败
   是本 crate 的取舍。
3. `serde` / `bytemuck` 作为**默认关闭**的可选 feature，默认特性集的零依赖承诺不变。
   `serde` 的线上名字直接取 `Format::name`——逐变体 `rename` 会是第二份名字表，而 `MxType`
   的 `mxfp8_e4m3` 也不是变体名的小写；`bytemuck` 给 `NoUninit` 与 `CheckedBitPattern`，
   不给 `Pod`：`repr(u8)` 的枚举只有 24/25/7/6 个合法字节值，其余是非法位模式。
4. `MxType` 仍未接上解码器。`tests/mxfp4.rs` 用 OCP v1.1 的数值表与手写字节块对拍了块长、
   每块字节与 E2M1/E4M3 解码，但真实模型字节流仍待 `weights` 接入 MXFP4 反量化后补。
