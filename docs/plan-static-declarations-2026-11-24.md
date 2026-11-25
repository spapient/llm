# 重构计划：对齐 llm.u 的强类型静态声明，拆除动态树生成

日期：2026-11-24　状态：待审阅

## 1. 参照基线：llm.u 怎么做（已核实 `/Users/fei/code/llm.u`）



* **没有&#x20;**`build.rs`（全仓 `find build.rs` 为空），没有代码生成、没有 `include!` 生成件。

* 声明是**手写强类型 struct + 闭集枚举**：`config::spec::Op { name, device: Option<Device>, dtype: Option<DType>, algo: Option<Algo>, ... }`；`Algo`/`Device`/`Prec` 是 Copy 枚举，`#[serde(rename)]` 映射字符串。

* 配置语料留在 `config/*.toml`（`[[ops]]` 行），装载期 serde 直反序列化进强类型 struct；**不存在通用&#x20;**`Table/Value`**&#x20;动态树、没有 OnceLock 注册表**。

* 内核分派 = `enum Impl` + `dispatch!` match 臂，编译期闭合。

## 2. 当前仓现状（已核实）

三条 `build.rs` 链共用 `diagnostic::data::{Table, Value}` 通用动态树：



| 链          | 语料                              | 产物                                                         | 运行期 parse                  |
| ---------- | ------------------------------- | ---------------------------------------------------------- | -------------------------- |
| prof 算子 70 | `src/ops/*.toml`                | `declared.rs`：`OnceLock<Vec<(&str, Table)>>` + 死代码 `NAMES` | `OpDecl::parse(&Table, …)` |
| prof 档案 6  | `src/<vendor>/*.toml`           | `embedded.rs`：同 OnceLock 树                                 | `Profile::parse(&Table)`   |
| tools 门禁   | `.config/lints/vocabulary.toml` | `vocabulary.rs`：`OnceLock<Table>`                          | `Gate::from_table(&Table)` |

链条：TOML → build.rs 生成 `Table::from_pairs(vec![…])` 表达式文本 → 运行期首访在 OnceLock 闭包里整表堆出 String/Vec 动态树 → 再遍历树 parse 回强类型。树是一次性中转。算子侧 `NAMES` 与 `TEMPLATES` 名字键重复且无外部消费（死代码）。

## 3. 硬约束（ADR 0022，不可违反）



* **运行期不再有任何 TOML 解析**；`toml` 只能作 `[build-dependencies]`。因此**不能**照搬 llm.u 的 "运行期 serde/toml 反序列化"。

* 对齐点取 llm.u 的**强类型 struct + 闭集枚举 + 无通用动态树**；解析只能发生在编译期（build.rs）。

## 4. 推荐路线（路线 2：编译期直产强类型 const 视图）

build.rs 保留（0022 要求编译期解析），但职责改为：解析 TOML 后**直接产出可 const 构造的强类型只读视图字面量**，不再产通用 `Table/Value` 表达式树。



* 视图字段：`&'static str`、`&'static [..]`、闭集枚举（`Skeleton`/`Prim`/`AttrKey`/`LayoutSpec`/`Schedule`/`Emit`/`Space`）、标量 `u32/f64`、位集 `Set`、const `Span`。

* 产物形态：`pub const OPS: [OpView; N] = [ OpView { … }, … ]`；模板正文沿用 `include_str!`。

* 运行期：**零 OnceLock、零动态树堆构造**；视图 → owned `OpDecl`/`Profile`/`Gate` 按需一次转换。

* 删除：算子侧冗余 `NAMES`/`names()`；`buildgen::emit` 表达式树生成；生产链对 `diagnostic::data` 树的依赖。

### 备选路线 1（更激进，需明确拍板）

彻底删 `ops/*.toml`、`vendor/*.toml`、三个 `build.rs`，把声明直接写成手写 `const` 强类型静态表。最贴近 "无生成式"，但 70 件算子 × 多字段手工搬运，量大、中文 summary / 表达式逐字落字面量，编译期校验要改为测试断言。

## 5. 子类型 const 可行性（已核实）

`Span` 全 const fn；`Set` 位表示、`const empty()`；闭集枚举 `Copy + Eq + 'static`。`Attr.values: Vec<String>`、`Need.attrs: Vec<Attr>` 在视图层用 `&'static [(&'static AttrKey, &'static [&'static str])]` 表示，owned 化时转 `Vec`。

## 6. 分层批次（每批收尾跑一次全门禁，不边改边跑）



* **批次 A — 死代码清理**：删算子侧 `NAMES`/`names()`/`catalogue.rs` 重导出；`declared`/`embedded` 查找风格统一。不碰树模型。

* **批次 B — prof 算子链静态化**：定义 `OpView`；build.rs 直产 `pub const OPS: [OpView; N]`；`OpDecl` 从视图构造；拆 `declared.rs` 的 OnceLock/Table 树。

* **批次 C — prof 档案链静态化**：定义 `ProfileView`；拆 `embedded.rs`；`Profile` 从视图构造。

* **批次 D — tools 门禁链静态化**：定义 `GateView`；拆 `vocabulary.rs` 的 OnceLock/Table 树。

* **批次 E — 收尾**：生产链 `data` 依赖清零；测试构造迁移到视图；删 `buildgen::emit` 与 OnceLock 骨架；`diagnostic::data` 树去留按拍板处置。

## 7. 待拍板



1. **路线**：路线 2（build.rs 产强类型 const，推荐）还是路线 1（删 TOML/build.rs，手写静态表）？

2. **范围**：只 prof 两链，还是 tools 门禁链一起？

3. `diagnostic::data`**&#x20;树去留**：彻底拆除（测试迁移）还是生产剥离后留测试构造？

## 8. 联网调研：方案谱系与契合度对照（2026-11-24）

约束前提：0022 运行期无 TOML 解析（`toml` 仅 build-dep）；许可证白名单仅 MIT/Apache-2.0/NCSA/Unicode-3.0；来源仅 crates.io；`unsafe_code = deny`；依赖最小化；workspace 已有 `proc-macro2/quote/syn`，`annotation` 仓本身是 proc-macro。



| 方案                                                    | 形态                                                                                     | 运行期                             | 契合度                                                                               | 结论             |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------- | -------------- |
| **1 手写 const 静态表**                                    | 删 TOML 语料与 build.rs，声明直接写成 Rust `const` 强类型数组；模板 `include_str!`                        | 零解析 / 零 codegen / 零第三方 / 零树堆构造  | 最彻底的非生成式，对齐 llm.u 静态强类型；但 70 件手工搬运、新增算子写 Rust、校验改测试断言，与 "语料唯一事实源" 设计冲突            | 备选（路线 1）       |
| **2 build.rs 产强类型 const 视图**                          | 保留 build.rs 编译期解析，删通用树表达式生成；产物为 `pub const OPS: [OpView; N]` 强类型字面量                    | 零 OnceLock / 零动态树；视图→owned 一次转换 | 守 0022；去动态树 / 重复 NAMES / 堆构造；不引第三方；语料仍在 TOML                                      | **推荐主干（路线 2）** |
| **3 自研 proc-macro 属性宏**                               | 复用 workspace `proc-macro2/quote/syn`，`#[prof_config(src=…)]` 编译期读 TOML 展开 const struct | 无 build.rs；编译期类型检查              | 体验贴近 concrete-config 且不引第三方；但自研解析 TOML / 闭集 / 位集 / 区间投入大，宏增量编译与错误可读性成本高           | 远期可选           |
| **4 include\_str! + 运行期 serde/toml**                  | llm.u 现状：内嵌文本装载期反序列化                                                                   | 运行期带 toml/serde 解析器             | 违反 0022                                                                           | **排除**         |
| **5 静态查找库** phf / cphf / frozen-collections / pickmap | 完美哈希静态 map                                                                             | O (1) 零分配查找                     | 只解决 "名字→条目" 查找，不解决嵌套结构化数据静态内嵌；70 条 const 数组二分已 O (log n) 零分配，收益小；新依赖需过 deny 复核许可证 | 查找层可选增强，不进主方案  |
| **6 const fn 自解析 TOML**                               | 编译期 const fn 解析极简子集                                                                    | —                               | const fn 解析能力受限，等于自造解析器，违背 0022 删解析器初衷                                            | **排除**         |

### 被许可证 / 约束排除的第三方配置宏



* `concrete-config`：EUPL-1.2 copyleft，不在白名单；虽支持 struct/enum/`&'static str`/`&'static [T]`/ 定长数组，仍**不可用**。

* `cruct`：GPL-3.0，**排除**。

* `forgeconf`/`inline-config`/`cfgy`：运行期 loader 或成熟度 / 许可证待核，均非首选。

### 核心判断

现状真正该消的是 "**动态树中转这一层**"（先堆建树、再 parse 二次拷贝、通用弱类型 `Value`）；查找方式（Vec 二分 vs 完美哈希）是次要优化。llm.u 可对齐的是它的**强类型 struct + 闭集枚举**，不是它的运行期 serde 反序列化（那条被 0022 排除）。社区共识：键集合编译期已知 → 静态表；运行期输入 → 动态 map；能用类型系统表达就不上宏。

## 9. 撇开合规：业界顶尖架构对照与技术最优解（2026-11-24 联网调研）

### 9.1 分派层：闭集用静态分派，dyn 只给开放集

Rust 官方与微软训练材料的一致口径：



| 手段                         | 分派         | 集合     | 开销                   |
| -------------------------- | ---------- | ------ | -------------------- |
| 泛型 `impl Trait` 单态化        | 编译期        | 开放集    | 零、可内联                |
| **enum dispatch（match 臂）** | 编译期        | **闭集** | 零 vtable、零堆、连续内存、可内联 |
| `dyn Trait`                | 运行期 vtable | 开放集    | vtable 指针 + 间接调用，挡内联 |

判据：实现集合编译期已知且可控 → enum / 泛型静态分派；只有运行期插插件、异构集合存一处、热点外的抽象边界才用 `dyn`。后端 / 算子族这种闭集小集合，match 穷尽性由编译器兜底，加变体漏改 match 直接编译失败。

### 9.2 注册表：运行期注册表是开放集场景才需要



* PyTorch 用 `TORCH_LIBRARY` + DispatchKey (CPU/CUDA/MPS) 运行期注册表，是因为 3500+ 算子、第三方运行时插后端、跨进程插件 —— 开放集 + 运行期择后端。

* oneDNN/TVM：算子注册 + topi compute/schedule，codegen 成 `runtime.Module`；oneDNN 运行期探测硬件再 JIT 专化内核。

* 本仓是封闭小集合（6 后端、70 算子、编译期固定），无运行期插后端需求，不需要这套运行期注册表。

### 9.3 内核层：原生语言实现，构建期静态挂载（用户直觉成立）

各后端内核用原生语言写（CUDA C++ / Metal Shading / C / OpenCL / MUSA），上层在构建期 / 编译期拼成该后端源码交后端编译器，或静态链接预编译符号；运行期不解析配置、不动态查找内核实现。

### 9.4 对照本仓现状（已核实）



* **内核分派层已经是静态绑定**：`nn::registry` 的 `for_each_operator!` 宏清单一行一个 `名字: 内核类型`，编译期展开，两张表同序、错位编译期断言失败；无动态注册表、无 `dyn` 内核指针。这一层方向正确，保持现状。

* **问题只在声明数据层**：prof 算子 / 档案、tools 配置走了 build.rs → 通用 `Table/Value` 动态树 → OnceLock 运行期堆建树 → 再 parse 强类型这条重中转；这是要改的部分。

### 9.5 技术最优解（撇开许可证）



1. **声明 = 强类型 const 静态表**：`const` 结构体数组 + 闭集枚举（`Skeleton`/`Prim`/`AttrKey`/`Schedule`/`Emit`/`Space`），字段 `&'static str`/ 标量 / 位集 /const `Span`；模板正文 `include_str!`。编译期闭合，零 OnceLock、零动态树堆构造、零二次拷贝。

2. **分派 = enum dispatch / 泛型单态化**：延续 `closed_enum!` 与 `for_each_operator!` 静态清单风格；不引入 `dyn` 内核指针。

3. **内核 = 原生源文件按后端目录静态挂载**：现状 `.cu/.c/.cpp/.cl/.metal` 沿用；拼装在构建期完成。

4. **查找**：70 条 const 数组二分已 O (log n) 零分配；闭集小集合上完美哈希收益有限，不为它引依赖。

5. 第三方配置宏与查找库都不是必需；自研 proc-macro 仅在 "要彻底去掉 build.rs 外壳" 时才考虑。

### 9.6 与方案 2 的关系

9.5 的声明表形态即方案 2 的 `OpView/ProfileView/GateView` const 视图；build.rs 仅承担 "编译期把 TOML 抽取 / 校验成强类型字面量" 这一外壳，不再产通用树。若要求连这层外壳也去掉，落到方案 1 手写静态表。

## 9.5 Ground 后的边界修正（2026-11-24 读 build.rs/buildops 现状）

动手前重读 `../crates/prof/build.rs`（包根构建脚本；设备扫描在 `src/devices.rs`）与 `buildops.rs` 后，必须把 build.rs 的职责拆成两类：

**A 类——构建期文件系统 / 数据加工（不适合、也不该搬进 proc-macro）**

- 目录扫描：列 `src/<vendor>/` 文件、判 `witness.<ext>`、配对 fragment/entry/epilogue/witness/archive；
- 跨文件一致性校验：签名 marker、`provides (prim, impl)` 配对、尾块恰六段且六设备一致、扩展名一致、目录名 == vendor、标识符形状；
- 语料数据变换：`base` 递归继承 + 深度合并、`tiers` 档展开、`dt` 精度替换、模板 C 类型替换、模板存在性校验；
- 设备镜像外部编译（`buildimage` 调 recipe）、`cfg(nightly)` 探测、`rerun-if-changed` 跟踪。

**B 类——要换掉的生成坏味道（本次目标）**

- 通用 `Table/Value` 树表达式（`Table::from_pairs(vec![…])`）；
- `OnceLock` 运行期堆建树 + 遍历树二次 parse 回强类型；
- 算子侧重复且无消费的 `NAMES`/`names()` 死代码。

**proc-macro 在 stable Rust 的硬限制**（调研已核）：无 `tracked_path`（宏读的文件改动不自动触发重编）、cwd 未指定、目录遍历 + 递归继承 + 档展开放进宏既别扭又难做诊断。宏擅长的是"在被标注类型上展开契约/样板"（沿用 `annotation` 现有 `spec/model/parallel` 模式），不擅长目录流水线。

**修正后的最优落地形态**：

- build.rs **保留 A 类外壳**（扫描 / 校验 / 继承合并 / 档展开 / rerun 跟踪），这部分不是"生成式坏味道"，删了反而更差；
- **产物 B 类**：build.rs 解析后直接产出强类型 const 视图字面量（`pub const OPS: [OpView; N]`），不再调 `buildgen::emit` 生成通用树；运行期拆 OnceLock 树与二次 parse；
- **闭集序号**：const 数组 + 编译期生成的名字 → 序号索引，O(1) 定位（借鉴方案 5，不引 phf）；
- 自研宏只落在适合的契约展开点，不承担目录扫描流水线。

**视图体量提示**：`OpDecl`/`Profile` 嵌套 owned 字段深（Variant/Fusion/Axis/Grid/Chunk/Need/Schedule/LayoutSpec/CType；Meta/MatchRule/Mem/Simt/Constant/Peak/Unit/Provides…），const 视图需逐字段映射为 `&'static str`/`&'static [..]`/枚举/标量/位集/const `Span`，是本重构最大工作量与风险点。

## 10. 方案 3 细化：自研 proc-macro，配置保留、编译期直接展开 const

### 11.1 目标形态（用户选定）

- 配置语料（TOML）保留为唯一事实源；**不用 build.rs 拼源码文本落盘**，不产 `$OUT_DIR/*.rs` 中间件，不生成 OnceLock 动态树。
- 自研属性/函数宏在**编译期**读配置、做闭集/范围/链单调/字段齐全校验，直接展开成 `pub const` 强类型视图数组的 token 流，由编译器直接编译进只读段。
- 运行期零解析、零堆构造；视图 → owned `OpDecl`/`Profile`/`Gate` 一次转换（或下游直读视图）。

### 11.2 与 build.rs codegen 的界限

| | build.rs（方案 2 现状路径） | 自研 proc-macro（方案 3） |
|---|---|---|
| 产物 | 源码文本写 `$OUT_DIR/*.rs`，再 `include!` | token 流直接展开 const 项，不落盘中间 .rs |
| 中间层 | OUT_DIR 文本 + OnceLock 树 | 无文本中转、无动态树 |
| 校验 | 运行期 parse 动态树 | 编译期宏内校验，失败 `compile_error!` 指到字段 |
| 依赖跟踪 | `cargo:rerun-if-changed` 显式 | stable 无 `tracked_path`，需兜底（见 11.4） |

注：proc-macro 本质仍是编译期展开 token（与 serde derive 同类）；要消的是"builder 拼字符串落盘 + 运行期建树"那套，不是取消编译期代码展开。

### 11.3 可复用的本仓基础（已核实）

- `annotation` crate 已是 `proc-macro = true`，workspace 已有 `proc-macro2/quote/syn`。
- 现有分层可照搬：`syntax`（token 模型，不碰 `proc_macro`、可单测）/ `decl`（解析产物）/ `render`（产物 token）/ `Streams` trait 依赖倒置（测试用字符串实现）。
- 闭集枚举已有 `closed_enum!`；宏把配置字符串映射到枚举。
- `toml` 作为**宏 crate 的编译期依赖**，不进被宏 crate 的运行期依赖图，不违反 0022。

### 11.4 主要坑（落地前必须定）

1. **文件重编译跟踪**：stable Rust 的 proc-macro 没有 `tracked_path`（跟踪 issue 仍未稳定），宏自己 `fs::read_to_string` 读的 TOML 改动不会自动触发重编。兜底：保留一个**极薄 build.rs 只发 `cargo:rerun-if-changed=ops/` 等路径**（不生成任何代码），或在源里用 `include_str!` 把文件纳入 cargo 依赖跟踪。
2. **路径锚定**：宏内 cwd 未指定，路径以 `CARGO_MANIFEST_DIR` 锚定，不依赖 cwd。
3. **方向依赖**：宏 crate 不能反向依赖被应用的 prof；视图类型定义在 prof，宏只传全限定路径 `::prof::OpView`。
4. **诊断可读性**：非法配置要把 span 指到具体文件/字段，`compile_error!` 文案清晰；文法/解析层脱离 `proc_macro` 单测。
5. **增量编译**：70 件 TOML 在宏展开期一次解析，需验证增量重编表现。

## 11. 六方案排名（不考虑成本：纯运行期性能 + 类型安全 + 编译期保障 + 配置分离 + 可维护性）

### 11.1 总排名

| 名次 | 方案 | 运行期 | 类型/编译期保障 | 配置分离 | 点评 |
|---|---|---|---|---|---|
| **1** | **方案 3 自研 proc-macro** | const 零构造、只读段 | 编译期强类型 + 宏内校验，非法即编译错误 | 配置留 TOML | 同时拿到配置独立、编译期闭合、零运行期构造；无落盘文本中转 |
| 2 | 方案 1 手写 const 静态表 | const 零构造、只读段 | 编译器直接全类型检查，链路最短 | 配置写进 Rust，混层 | 零生成中间层最硬，但 70 件手写、新增算子要改 Rust，事实源不独立 |
| 3 | 方案 2 build.rs 产 const 视图 | const 零构造、只读段 | 产物经编译器检查，生成器正确性靠测试 | 配置留 TOML | 多一道"字符串拼源码 → OUT_DIR → include!"文本中转 |
| 4 | 方案 5 静态查找库（phf/frozen） | 查找 O(1) 零分配 | 不承载结构化数据校验 | — | **叠加层，不是数据底座**；可叠在 1/2/3 之上，70 条闭集上二分已很快，收益有限 |
| 5 | 方案 6 const fn 自解析 TOML | const 零构造 | 受 const 表达式能力限制，自造解析器 | 配置留文件 | const fn 无 HashMap/动态 Vec，难承载嵌套位集/区间/闭集，表达力最弱、最脆弱 |
| 6 | 方案 4 include_str + 运行期 serde/toml | 启动期解析 + 堆分配，最差 | 非法配置推迟到运行期才暴露 | 配置留文件 | 运行期带解析器与堆构造，错误不在编译期拦 |

### 11.2 分维度结论

- **运行期数据构造**：1 ≈ 2 ≈ 3 ≈ 6（均 const 只读段、零堆构造）> 4（运行期解析堆构造）。
- **运行期查找**：5 完美哈希 O(1) > const 数组二分 O(log n) > 4 运行期 Map。
- **编译期保障**：1 最直接（无生成层）≈ 3（宏内校验 + 产物类型检查）> 2（生成器正确性靠测试）> 4（运行期才报错）。
- **配置化/可扩展**：3 ≈ 2 ≈ 4（配置独立文件）> 1（配置进 Rust 代码）。
- **综合最优**：方案 3——唯一在"配置独立"与"编译期强类型闭合、零运行期构造"两端同时拿满的方案。方案 5 可作为其查找层的可选叠加。

### 11.3 为什么方案 3 压过方案 1

方案 1 链路最短、编译器直接检查，单看"无生成中间层"最硬；但它把 70 条声明写进 Rust 代码，配置与实现混层，新增算子要改 Rust、评审兜底漂移。方案 3 保留独立 TOML 事实源，宏在编译期把配置映射成 const 强类型，产物同样过编译器类型检查；代价是多一个自研宏层——不考虑成本时，换来的是配置化、可扩展性与编译期校验，综合质量更高。

### 11.4 方案 5 查找层怎么借鉴（可选后置）

- **本质**：方案 5 只优化"名字→条目"查找，不承载结构化数据内嵌，是叠在方案 3 const 底座之上的加速层。
- **现状**：`TEMPLATES` 已是 const 数组 + `binary_search_by_key`，只读遍历、**零堆分配**；方案 5 不改善分配，只把 O(log 70)≈7 次字符串比较降到 O(1) 一次哈希 + 一次下标。70 条闭集、热点在内核计算而非名字查表，收益是纳秒级，优先级低。
- **三条落地路径**：
  1. 引 `phf`（`phf_map!`，MIT）或 `frozen-collections`（`fz_hash_map!`，MIT）编译期宏出静态表；许可证可过白名单，但多一个第三方依赖。
  2. **借鉴思想、不引依赖（推荐）**：闭集算子编译期定一个 `名字 → 枚举序号/const 索引`，match 臂或数组下标 O(1)；延续 `for_each_operator!` 名字同序清单的现有风格。
  3. 保持二分：算子规模未涨、压测未证明查表是热点前，不做查找优化。
- **结论**：先落方案 3 数据层；查找优化后置，路径 2 优先；仅当算子规模明显变大或压测证明名字查表是热点时，再上路径 1。

## 12. 实现蓝图（纯方案 3 + 闭集序号，续作基线）

### 12.1 OpDecl → OpView const 视图映射（已核字段）

| owned `OpDecl` 字段 | const `OpView` 字段 | 说明 |
|---|---|---|
| `name: String` | `name: &'static str` | |
| `skeleton: Skeleton` | `skeleton: Skeleton` | 闭集枚举 Copy |
| `summary: String` | `summary: &'static str` | |
| `axes: Vec<Axis>` | `axes: &'static [AxisView]` | `AxisView { name: &'static str, values: &'static [u32] }` |
| `needs: Vec<Need>` | `needs: &'static [NeedView]` | 见 12.2 |
| `layouts: Vec<(String, LayoutSpec)>` | `layouts: &'static [(&'static str, LayoutSpec)]` | `LayoutSpec` 枚举 |
| `grid: Option<Grid>` | `grid: Option<GridView>` | `GridView { dims: &'static [(&'static str, &'static str)] }` |
| `chunk: Option<Chunk>` | `chunk: Option<ChunkView>` | `ChunkView { size: usize }` |
| `consts: Vec<(String,String)>` | `consts: &'static [(&'static str, &'static str)]` | |
| `variants: Vec<Variant>` | `variants: &'static [VariantView]` | `VariantView { name: &'static str, order: u32, schedule: Schedule, mma_share: f64, extra_needs: &'static [NeedView] }` |
| `template: String` | `template: &'static str` | `include_str!` |
| `signature: Signature` | 视图不存 | 转换期从模板重解 |
| `phases: Vec<Phase>` | `phases: &'static [PhaseView]` | `PhaseView { name: &'static str, reads: &'static [&'static str], writes: &'static [&'static str] }` |

### 12.2 Need → NeedView

| owned `Need` | const `NeedView` | 说明 |
|---|---|---|
| `prim: Prim` | `prim: Prim` | 闭集枚举 |
| `extent/n/k: Span` | 同 `Span` | `Span` 全 const fn |
| `dt: DTypeSet` | `dt: &'static [DType]` 元素切片 | owned 化 `Set::of(..)` 重建 |
| `space: SpaceSet` | `space: &'static [Space]` 元素切片 | 同上 |
| `attrs: Vec<Attr>` | `attrs: &'static [AttrView]` | `AttrView { key: AttrKey, values: &'static [&'static str] }` |

`Set::empty()` 是 const、`insert` 非 const，故视图层持元素切片（非整数掩码），owned 化时 `Set::of` 重建位集。批次 B 已按此落地于 `decl.rs`。

### 12.3 宏 crate 分层（沿用 `annotation` 的 Streams 倒置范式）

- `fs`：`CARGO_MANIFEST_DIR` 锚定路径，扫 `ops/` 目录、读 toml 与 `.k`；
- `resolve`：`base` 递归继承 + 深度合并（纯函数，脱离 proc_macro 单测）；
- `tiers`：档展开 + `dt` 精度替换 + 模板 C 类型替换（纯函数）；
- `model`：`toml::Table` → 视图字段，闭集/范围/链单调/字段齐全校验，失败 `compile_error!` 指到字段；
- `render`：字符串拼 `pub const OPS: [OpView; N]` 字面量（沿用 `annotation/render.rs` 的 `String` + `write!` 范式，不引 `quote!`），产物再解析成 token 流；
- 闭集序号：编译期生成 `名字 → u16` 索引，查找 O(1) 替代二分（借鉴方案 5，不引 phf）。

#### 12.3.1 闭集词 → prof 变体路径映射（render 依据，已核枚举标识）

| 闭集 | TOML 短名 | prof 变体 | 全限定路径 |
|---|---|---|---|
| 骨架 | `elementwise` / `reduce` / `gemm` / `scan` | `Elementwise` / `Reduce` / `Gemm` / `Scan` | `::llm_prof::decl::Skeleton::*` |
| 原语 | `mma` `dot` `load` `load_async` `commit_wait` `store` `bar` `shuffle` `broadcast` `reduce` `scan` `exp2` `rcp` `cvt` `quant` `atomic` | `Mma` `Dot` `Load` `LoadAsync` `CommitWait` `Store` `Bar` `Shuffle` `Broadcast` `Reduce` `Scan` `Exp2` `Rcp` `Cvt` `Quant` `Atomic` | `::llm_prof::atom::Prim::*` |
| 存储空间 | `GM` / `FAST` / `REG` | `Gm` / `Fast` / `Reg` | `::llm_prof::atom::Space::*` |
| 数据类型 | `f32` / `f16` / `bf16` | `F32` / `F16` / `BF16` | `::llm_prof::atom::DType::*`（atom 重导出 `dtype::DType`） |
| 调度 | `direct` / `pipe{n}` / `warp_specialized{p,c}` / `ping_pong{g}` | `Direct` / `Pipe{stages}` / `WarpSpecialized{producer,consumer}` / `PingPong{groups}` | `::llm_prof::sch::Schedule::*` |
| 布局 | `row` / `col` / `swz<b,m,s>` / `pad(p)` | `Row` / `Col` / `Swz{b,m,s}` / `Pad(p)` | `::llm_prof::sch::layout::LayoutSpec::*` |

`AttrKey` 词表随 `model` 校验层一起核（`Src`/`Dst`/`Align`/`Axis`/`Op`/`Sat`/`Stages`/`Group`/`Kind`/`Accuracy`/`Lanes`/`Exec` 等）。闭集词非法时 `model` 层在编译期 `compile_error!`，render 兜底分支不可达。

#### 12.3.2 AttrKey 词 → `::llm_prof::atom::AttrKey::*` 映射（已核 `atom.rs:290`）

| TOML 键 | 变体 | TOML 键 | 变体 |
|---|---|---|---|
| `op` | `Op` | `axis` | `Axis` |
| `kind` | `Kind` | `sat` | `Sat` |
| `accuracy` | `Accuracy` | `group` | `Group` |
| `lanes` | `Lanes` | `src` | `Src` |
| `dst` | `Dst` | `stages` | `Stages` |
| `exec` | `Exec` | `align` | `Align` |

属性取值是自由字符串列表（如 `axis = ["GMAX"]`、`op = ["sum"]`、`accuracy = ["fp32"]`），不进闭集，原样渲染成 `&'static [&'static str]`。某属性键是否被该原语接受（`Prim::accepts`）属协商校验，保留在 prof owned 化/装载路径，宏进程只做键 → 变体映射。

#### 12.3.3 Span 写法 → const 构造子（`atom.rs:340`）

`Span` 的 `range(lo, hi)` 非 const fn，视图层只能走四个 const 构造子：

| TOML 写法 | 构造子 | const |
|---|---|---|
| 缺省 | `Span::any()` | 是 |
| `16`（纯数字） | `Span::exact(16)` | 是 |
| `<=4096` | `Span::at_most(4096)` | 是 |
| `16..` | `Span::at_least(16)` | 是 |
| `16..64`（闭区间） | 无 const 构造子 | 否——当前语料未使用；若后续出现，需先给 `Span` 加 const `range` 或在视图层改存 `(lo, hi)` |

实测 70 份 `ops/*.toml`：主维度 span 只有 `<=N`（`bytes`/`m`/`n`/`k`），无闭区间、无纯数字 exact、无 `N..`。

#### 12.3.4 Schedule / Layout 参数解析口径

- **Schedule**（`sch.rs:33`）：`direct` → `Schedule::Direct`；`pipe<N>` → `Pipe { stages: N }`；`ws<p,c>` → `WarpSpecialized { producer: p, consumer: c }`；`pingpong<N>` → `PingPong { groups: N }`。`<...>` 内逗号分隔 u32。当前语料只有 `direct`（25）与 `pipe<2>`（3），无 `ws` / `pingpong`。
- **LayoutSpec**（`sch.rs:313`）：`row` → `Row`；`col` → `Col`；`swz<b,m,s>` → `Swz { b, m, s }`（u8×3）；`pad(p)` → `Pad(p)`（u32）。当前语料只有 `row`（63 处），无 `col` / `swz` / `pad`。

#### 12.3.5 语料实况分布（2026-11-24 扫描 70 份 TOML）

- `tiers` 档展开：仅 `abs_f32.toml`（`f32`/`f16`）；其余 69 份无档。
- `extra_needs`：变体段无实值（grep 命中均为注释文字）；`VariantView::extra_needs` 先渲染空切片。
- `[[phases]]`：无显式阶段；`OpView::phases` 渲染空切片，owned 化按签名派生单阶段。
- 模板正文（`template`）：本批不进视图；`OpView.template` 由 #46 prof 接线时经既有 `TEMPLATES` 表按名配对（`include_str!` 属编译期 const 数组，不在要拆的 OnceLock 树内）。


### 12.7 #45 开工骨架决策（2026-11-24 定：并入 annotation，不开新 crate）

- **宏落点**：不新开 proc-macro crate；prof 语料展开宏并入既有 `crates/annotation`（`llm-annotation`，已是 `proc-macro = true`，且不依赖任何本地 crate，方向无环）。prof 在 `[dependencies]` 引入 annotation；annotation 生成代码用全限定路径 `::llm_prof::decl::{OpView, NeedView, …}`，**不反向依赖 prof**（方向约束 §11.4-3）。
- **展开路径（用户拍板：annotation 宏直读 TOML，重议 0022）**：宏为 `#[proc_macro]` 函数宏，编译期 `fs::read_to_string` 读 `src/ops/*.toml`（路径锚 `CARGO_MANIFEST_DIR`），用 `toml` 解析后直接展开 `pub const OPS: [OpView; N]` token 流；不产 OUT_DIR `.rs` 中间件、不建树。`toml` 加为 annotation 的 normal 依赖——它只在编译期宏进程内运行，不链接进 prof 运行期二进制，但 `cargo tree -e normal` 会出现 toml/serde 子树，故需在 0022 补一条重议注记区分「编译期宏依赖」与「运行期解析器」。
- **分层**（照搬 annotation 既有 syntax/decl/render 倒置范式，但本宏是文件目录而非 Rust 属性）：`fs`（锚定路径、扫目录、读 toml/`.k`）/ `resolve`（`base` 递归继承 + 深度合并，纯函数可单测）/ `tiers`（档展开 + `dt` 精度替换 + 模板 C 类型替换，纯函数）/ `model`（toml 表 → 视图字段中间模型，闭集/范围/链单调/字段齐全校验，失败 `compile_error!` 指到文件字段）/ `render`（字符串拼 const 视图字面量，不碰 `proc_macro`，纯文本可断言测试）。文法/解析/render 层脱离 `proc_macro` 可独立单测。
- **薄 build.rs 壳**：现有 build.rs 的 A 类外壳（签名 marker/`(prim,impl)` 配对、尾块核验、设备镜像外部编译、`cfg(nightly)` 探测、`rerun-if-changed`）保留；B 类文本产物（`declared.rs`/`embedded.rs`/`tables.rs`/`template_list.rs`）随批次 C/D 拆除。继承合并与档展开迁到宏内 `resolve/tiers` 纯函数层。
- **重编跟踪兜底**：stable 宏无 `tracked_path`，宏内 `fs` 读的 TOML 改动不自动重编；由薄 build.rs 发 `cargo:rerun-if-changed=src/ops` 等路径兜底，模板 `.k` 经 `include_str!` 纳入 cargo 跟踪。
- **render 路径**：宏不感知 prof 内部模块布局，只传全限定 `::llm_prof::decl::…`；`Set`/枚举等由视图字段类型在 prof 侧解析。
- **序号**：`OPS: [OpView; N]` 数组下标即编译期序号；另生成按序的 `(&'static str, u16)` 索引表供名字 O(1) 定位，不引 `phf`。
- **门禁注意**：annotation 新增生产文件受 `file_min_lines = 300` 硬卡，baseline 只豁免 `dsl/expr.rs`；不得往 `min_lines_baseline` 扩列。落点与行数预算（`file_max_lines = 899`，现状 `decl.rs` 581 / `render.rs` 325）：
  - 中间模型族 + 闭集短名常量 → 既有 `decl.rs`（达标，追加后估 ~700 行，余量充足）。
  - 视图字面量渲染（`render_ops` 及辅助）→ 既有 `render.rs`（达标，追加后估 ~520 行）。
  - TOML 解析纯函数（`resolve` 继承合并 / `tiers` 展开 / 表 → model / span·schedule·layout·attrs 解码）→ 独立新文件 `src/ops.rs`，预计 ≥300 行合规；不塞回 `decl.rs`（模型 + 解析合计会顶到 899 硬顶），也不挂 `syntax.rs`（那是 Rust token 文法，与 TOML 解析正交）。
  - `toml` 依赖与解析层同批加入并被实际使用，避免 `cargo machete` 判 unused dependency。

- **闭环约束（2026-11-24 回退教训）**：annotation 是 lib crate，`pub(crate)` 项无生产调用点即触发 `dead_code`，`clippy -D warnings` 下变红。因此 **宏入口、fs/resolve 解析、model 构造、render 必须同批落地闭环**，不能先落半截 render/model 等后续接入；空壳与未构造结构同样告警。`OpModel`/`VariantModel` 含 `f64`（`mma_share`），只 derive `PartialEq` 不 derive `Eq`。
- **落地顺序**：一批内先通 `fs`（锚 `CARGO_MANIFEST_DIR` 扫 `ops/*.toml`、读 `.k`）→ `resolve`（`base` 递归继承 + 深度合并）→ `model`（toml 值 → 中间模型，闭集/范围/链单调校验）→ `render`（拼 const 视图字面量）→ `#[proc_macro]` 入口把模型接进 token 流；单测覆盖单件 model→render 文本往返与配平。`tiers` 档展开作为紧随的第二刀。

### 12.4 批次（每批收尾跑一次全门禁）

- **A**：删算子侧死代码 `NAMES`/`names()` 与 `catalogue.rs` 重导出（已核零消费）。
- **B**：prof 内定 `OpView/NeedView/…` 与 owned 化转换；先以手写视图测试，不接宏。
- **C**：宏 crate 打通算子链 const `OPS`；拆 `declared.rs` OnceLock/树。
- **D**：档案链 `ProfileView`，拆 `embedded.rs`。
- **E**：tools 门禁链 `GateView`，拆 `vocabulary.rs`。
- **F**：拆 `buildgen::emit` 表达式生成；生产链 `diagnostic::data` 依赖清零；测试构造迁移；全门禁复核。

### 12.6 批次 B 落地记录（2026-11-24）

- **落点**：视图族并入既有 `crates/prof/src/decl.rs`（未单开 `view.rs`）。原因：`file_min_lines = 300` 对新增生产文件硬卡，且 `min_lines_baseline` 只准缩短不准扩列；视图本就是声明层的编译期镜像，归 `decl.rs` 内聚。并入后 `decl.rs` 754 行（< 899 硬顶）。
- **视图结构**（全 `Copy`，字段仅 `&'static str` / `&'static [..]` / 闭集枚举 / 标量 / `Span`）：`AxisView`、`AttrView`、`NeedView`、`GridView`、`ChunkView`、`VariantView`、`PhaseView`、`OpView`。
- **位集口径**：`DType` / `Space` 在视图层用 `&'static [..]` 元素切片；owned 化时 `Set::of(..)` 重建，不在视图层持 `Set`（`insert` 非 const）。
- **owned 化**：子视图 `into_owned()` → 既有 `Axis/Attr/Need/Grid/Chunk/Variant/Phase`；`OpView::into_decl()` 组装 `OpDecl`。
  - 签名：`Signature::parse(name, template)` 仍从模板解析（ABI 唯一事实源，视图不存）。
  - 阶段：视图 `phases` 为空时按签名派生单阶段（`buf_names(false/true)`）；非空时逐条校验读写槽位落在签名缓冲形参内（`check_phase_slots`，与 `ingest::slot_names` 同口径，错误走 `decl::field`）。
- **命名坑**：转换方法定名 `into_owned`，不用 `to_owned`——后者命中 `ToOwned` trait 保留名，迭代借用项会解析成克隆而非 owned 转换。
- **实测**：`cargo check -p llm-prof`、`clippy -p llm-prof`（零告警）、`cargo test -p llm-prof`（56 单测 + 3 doctest 全绿）、`check --workspace --all-targets` 全绿；旧 `Table → OpDecl::parse` 路径未动。
- **门禁处置**：`into_decl` 初版 Ccn=6 超 `ccn_max=5`，按口径（`?`/`if`/`match`/`&&/||` 各 +1，闭包内条件归属外层）把阶段构造拆成 `declare_phases`（空视图派生单阶段）+ `view_phase`（单阶段槽位校验）两层辅助，父函数条件数降到阈值内；8 个新公开视图类型按门规进 `ARCHITECTURE.md` 第三节所有权表（不走 `unlisted` baseline）。收尾全门：`fmt --check` 干净、`clippy --workspace --all-targets -D warnings` 零告警、`nextest --workspace` 2844 passed / 17 skipped / 0 failed、doc-test 绿、tools gate 15 过、testkit 架构/所有权 100 过、name-lint 0 命中。
- **未做（归后续批次）**：`OpView` 尚未接生产链（`declared.rs` OnceLock 树仍在）；闭集序号「名字→序号」O(1) 定位随批次 C 宏展开时由数组下标天然给出，prof 侧不另建索引结构；新公开视图类型的 ownership baseline 登记归批次 F 收尾。

### 12.6.1 批次 C 落地记录（2026-11-24，算子链接线）

- **路线（ADR 0022 重议条款）**：模板正文不进视图，`OpView` 不持 template；模板由调用方按算子名从既有 `TEMPLATES` const 表（`include_str!` + 档级内联，编译期 const 数组，非 OnceLock 树）配对传入。签名仍从模板解析（ABI 唯一事实源）；模板侧三条校验（槽位 / needs / 调用参数）保留在 prof owned 化与装载路径。
- **接线落点**：`prof/src/lib.rs` 加 `extern crate self as llm_prof;` 自别名（宏产物硬编码全限定 `::llm_prof::…`，crate 内部展开无此外部 crate 名）；`ops_generated` 模块 `annotation::prof_ops!()` 展开 `pub const OPS: [OpView; N]`（按名升序，`#[allow(missing_docs)]`）。`catalogue.rs` 重写：`Op{ name, view: &'static OpView, template: &'static str }`，`named(name)` 走 `TEMPLATES` 二分得模板 + `OPS` 二分得视图，`all()` 按名配对；`Op::to_decl()` → `OpDecl::from_view(view, template)`（`into_decl` 后接 `verified`）。
- **拆动态树**：删 `declared_generated` 模块 include 与 `pub use declared`；`buildops.rs` 精简为只产 `template_list.rs` + `templates_table.rs`（删 `emit_declared`/`tier_row`/`Entry.row`，`Generated.rows` 四元组改三元组，`declarations_of` 返回 `(String, String)`）；`build.rs` 主循环删 `declared.rs` 写盘。保留 buildops 内 resolve 继承合并、`stem_of` 校验、`expand_tiers` 档展开、`emit_templates` 档级 C 类型替换。
- **宏修正**：`annotation/src/decl.rs` 删 `template: String`；`render.rs` 删 template 行，layouts/consts/grid dims 项包 `(…)` 元组括号，`mma_share` 渲染加 `f64` 后缀。
- **生产消费点**：`cutile/src/builtin.rs` `assemble` 由 `OpDecl::parse(op.declaration(), op.template())` 改为 `op.to_decl()`。
- **测试迁移**：prof `tests/decl.rs` 内置自洽/compose 用例改 `to_decl()`；cutile plan/emit/reference/builtin/cap 用例改 `to_decl`，轴顺序用例改直改 owned `op.axes[0].values`，`Op::new` 三参改 `(name, source.view(), source.template())`；shard `build_tests.rs` 篡改用例改 owned：`declared_bytes` 遍历 needs 改 `extent = Span::at_most(limit)`，`declared_swizzle` 遍历 layouts 改 `LayoutSpec::Swz{40,3,3}`，`builtin()` 改 `to_decl()`。
- **停用用例（归 #47 按 owned 恢复）**：cutile `one_signature_written_in_either_layer_stays_the_same`（Table 解析器标量/数组、bytes/m 归一化同构）；prof `a_tile_overflow_names_the_axis_to_prune`、`an_attr_key_must_belong_to_its_primitive`（最小表过不了 `verified` 的 needs/槽位交叉校验）。旧"clone 内置 Table 篡改字段"手法在视图链下无桥接：静态镜像不含动态树，无法反解回 Table。
- **实测**：`cargo check --workspace --all-targets` 全绿；`clippy -p llm-prof/-cutile/-llm-shard/-llm-nn/-llm-train --lib --all-targets` 零告警；`cargo fmt --check` 干净。测试：annotation 55 lib + 6 集成、prof 162 lib + 56 集成（1 doc ignored）、cutile 141 lib、nn 536 lib（4 ignored）、shard 161 lib、train 236 lib，全绿。
- **未做（归后续批次）**：档案链 `ProfileView` 拆 `embedded.rs`（批次 D）；tools 链 `GateView` 拆 `vocabulary.rs` OnceLock<Table>（批次 E）；ingest Table 段解析器与 `buildgen::emit` 拆除、`diagnostic::data` 树依赖清零（批次 F）。
- **#47 收尾（2026-11-25）**：三条停用用例已恢复并进入当前测试；`declarations_of` 已改为 `template_tables_of`，内部实现改为 `render_template_tables`；重复代码、命名、复杂度与兼容路径均已复核，`bash scripts/ci.sh` 完整序列通过，未新增豁免。

### 12.5 已知坑

- stable proc-macro 无 `tracked_path`：宏读的 TOML 改动不自动重编；兜底用源内 `include_str!` 纳入 cargo 跟踪，或保留极薄 build.rs 只发 `rerun-if-changed`（不产代码）。
- 宏 crate 不能反向依赖 prof；视图类型定义在 prof，宏只传全限定路径。
- 继承链多级（深度 ≤3）、档展开、模板类型替换必须在纯函数层可单测。

## 13. 风险



* 本机约 31 个并行会话进程在飞，工作树可能被周期 reset；每批落地以磁盘现状复核，改动集中收尾再跑门禁。

* 每批保持 `fmt`/`clippy`/`nextest`/`gate`/`testkit`/`jscpd` 绿，不豁免存量棘轮。

## 14. 批次 D/E/F 完成账本（2026-11-25）

### 14.1 批次 D — prof 档案链

- `buildtables.rs` 直接生成 `PROFILE_VIEWS: &[ProfileView]`；`ProfileView` / `ProvideView` 平铺收进 `prof.rs`，仅保留内联 `view` 重导出路径，不新增碎文件。
- 字段只含静态文本、切片、闭集与数值。位集、字符串与供给只在首次使用时 owned 化；`builtin_profiles` 缓存 `Result`，不静默丢设备，也不要求生产 `expect`。
- `MatchRule::parse`、`Span::parse`、峰值推导式对拍、三级存储与见证校验仍各有唯一入口；生成器额外拒绝非有限浮点。

### 14.2 批次 E — tools 门禁链

- `tools/build.rs` 直接生成 `VOCABULARY: VocabularyView`，覆盖门禁、闭集 / 位集登记与 ownership 基线；`testkit` 两个消费方均读静态字段。
- `GateView<'a>` / `DebtView<'a>` 用生命周期泛化同时服务静态生成物与局部测试，不泄漏内存；`Gate::from_table` 与旧动态阈值读取已删除。
- 正阈值、上下限关系、未知键、无度量债务和具体登记理由仍在构建期或 owned 化时拒绝。

### 14.3 批次 F — 解析器与数据树拆除

- 删除 `prof/src/{schema,ingest,decode}.rs`：有效模板 / 阶段 / 变体校验并入 `decl.rs`，数值推导式并入既有 `formula.rs`。
- 删除 `diagnostic/src/data.rs`、`buildgen::emit` 及旧数据树测试；`buildgen` 只保留 `collect` / `read` / `relative`，其测试迁移为真实 TOML 读取契约。
- 新公开视图全部登记 `ARCHITECTURE.md` 所有权表，没有新增 `unlisted` 豁免。旧停用用例已在 #47 恢复，不重开路线。

### 14.4 实测与闭环

- 工作区 `--all-targets --all-features` 编译零警告；`prof` 169 项单测、65 项设备片段测试，`testkit` 116 项，诊断底座 20 项单测与 2 项文档测试通过。
- 静态质量门 13 项通过，覆盖复杂度、行数、禁止调用、命名与文档链接；最终全量结果以 `bash scripts/ci.sh` 的当次输出为准，不预写分数或沿用历史记录。
- 保持平铺、配置优先与单源校验；不得靠调宽阈值、删除有效覆盖或新增基线完成闭环。

## 15. 收尾复核与 #26 / #53 台账（2026-11-26）

### 15.1 当次实测基线（命令输出为准，不复用历史记录）

- `cargo check --workspace --all-targets --all-features --locked`：0 error。
- `cargo nextest run --workspace --all-features --locked`：3031 passed / 18 skipped / 0 failed。
- `cargo test --doc --workspace`：1 + 6 + 3 passed；`cargo fmt --all --check`：0；`cargo clippy --workspace --all-targets --all-features -- -D warnings`：0。
- ruff `check` 与 `format --check validation`：0；`cargo deny check`：advisories/bans/licenses/sources ok；`cargo machete`：无未用依赖。
- jscpd：535 文件 / 163,094 行 / 0 clone / 0.00%；tools 门 13 passed；`complexity.sh` 0；testkit 架构门 117 passed；命名门 0 命中；`cargo expand` 双产物非空；typos 0；actionlint 0；`cargo bench --workspace --no-run` 0；regress 12 passed；cloc 2.12 统计步 0（`.tools/cloc`，须 `.tools` 前置 PATH，非系统命令）。
- 密度门 `loc-budget.sh`：**未通过**，src 104,164 行 / 313 文件，三条硬阻断同时命中。常量脱节已取证（`git log -S SRC_MAX` + 父提交比对）：重建提交 `b65a0966` 将 `SRC_MAX` 从 **96,100** 回退为 **60,000**，注释由棘轮口径改为"项目统一目标，未达标必须阻断"；但脚本顶部 7–145 行棘轮注释原样保留，停在 95,800（第 127 行，实测 95,687/410 文件）。`FILE_VETO=600`、`CRATE_MAX=8200` 该提交未动。当前 104,164 行：超目标值 44,164、超重建前代码基线 96,100 为 8,064、超注释棘轮 95,800 为 8,364。
  - **总量（N6，`SRC_MAX=60000`）**：104,164 行，超 44,164。
  - **>600 行一票否决（N3，`FILE_VETO=600`）**：45 个文件。Top：`cutile/src/parser.rs` 900、`prof/src/opencl.rs` 898、`weights/src/gguf.rs` 876、`weights/src/quant.rs` 874、`tensor/src/layout.rs` 871、`weights/src/seg.rs` 864、`annotation/src/render.rs` 860、`prof/src/decl.rs` 857、`annotation/src/ops.rs` 857、`cutile/src/cap.rs` 854；其余 35 个 602–850 行（清单可由脚本 157 行当次复算）。
  - **>8200 行 crate（N2，`CRATE_MAX=8200`）**：4 个——nn 15,303 行/37 文件、prof 12,237/32、weights 10,019/15、cutile 8,658/21。
  - 均为存量大层；本批增量仅为注解、文档与台账注释，未新增能力代码，也未调宽阈值掩盖超标。超棘轮基线的 8,364 行属后续并发会话能力增量，需另行取证登记或无损缩减，不在本批处置范围。注释棘轮只登记了 N6 总量；N3 文件线、N2 crate 线无棘轮注释，是独立硬线。

### 15.2 #26 pedantic 分层机制与样板层

- 机制约束（实测确认）：Cargo lints 不支持工具表内 `workspace = true` 子表继承；`[lints] workspace = true` 与 crate 级 `[lints.clippy]` 覆盖不能并存；子表写 `workspace = true` 被拒。可行通道是源码级 lint 属性，沿用仓内既有 `#[allow]` 先例。
- 作用域口径（更正上一轮判断）：源码 `#![deny(clippy::pedantic)]` 只作用本 crate 编译单元，不继承给依赖；ci.sh 的 clippy 步是 `-D warnings` 但**不开** pedantic flag。因此依赖 crate 的 pedantic 告警不阻断真实门禁；只有手动用 `-W clippy::pedantic -D warnings` 整图编译时，未整改依赖才会被命令行 flag 升级。
- 样板一 `llm-dtype`：`src/lib.rs` 顶部 `#![deny(clippy::pedantic)]`；独立 test target `tests/dtype_contracts.rs` 同源属性。本体 lib 与 tests pedantic 0 告警；补 `#[must_use]`，测试数值转换改 `u32::try_from(..).expect(..)`（沿用仓内测试惯例，生产仍禁裸 expect）。
- 样板二 `llm-diagnostic`：`src/lib.rs` 顶部 `#![deny(clippy::pedantic)]`；`autotests = false`，四个测试经 `#[path]` mod 编入 lib test target，受 lib 属性覆盖，无需独立 target 属性。整改：9 处 `#[must_use]`、2 处 `push_str(&format!)` 改 `writeln!`（`use std::fmt::Write as _;`）、3 段 `# Errors` 文档、1 处私有 `io_err` 参数改引用、1 处 `u64→f64` 展示换算局部 `#[allow(clippy::cast_precision_loss)]` 并注释理由。pedantic 全 target 0，20 单测 + 2 doc-test 绿，全量 nextest 无回归。
- 样板三 `llm-annotation`：`src/lib.rs` 顶部 `#![deny(clippy::pedantic)]`；独立 test target `tests/audit.rs` 同源属性；其余 fixture/decl_tests/render_tests/adapter_tests 经 `#[path]` 编入 lib test target 受 lib 属性覆盖。整改：2 处 private `&Option<T>`→`Option<&T>`（调用点改 `as_ref()`）、5 处 `format_push_string` 改 `write!`/`writeln!`（`use std::fmt::Write;`）、2 处分号、1 处 `format_implicit`、1 处 match 同臂合并、1 处 let-else；测试片段 22 处无内嵌 `"` 的 `r#"…"#` 去哈希、7 处 `assert!(x.is_empty())` 改 `assert_eq!(x, [] as [T; 0])`（空数组标注元素类型消歧义）、3 处 `contains("x")` 改 `contains('x')`。3 条 `pass_by_value` 经核为 clippy 误报不可改签名，1 条占比 `i64→f64`、1 条测试夹具 u64→usize 取模截断，均局部 `#[allow]` + 注释。pedantic 全 target 0，59 单测 + 6 集成绿，下游 `llm-prof`/`llm-nn` check 绿。
- 样板四 `llm-tensor`：`src/lib.rs` 顶部 `#![deny(clippy::pedantic)]`；四个测试片段经 `#[path]` 编入 lib test target 受 lib 属性覆盖。lib 46 条处置：14 处返回 `Result` 函数补 `# Errors` 段（含 `elementwise_binary!` 宏统一追加）、14 处查询/构造方法补 `#[must_use]`、8 条 `iN→i128` lossless cast 改 `i128::from(..)`、1 处 `layout` 参数按值改 `&Layout<R>`（两处调用点传引用）、2 处 doc 反引号。语义债 8 条局部 `#[allow]` + 守卫注释：`layout.rs` 基址/stride/dim `usize→isize`（`cast_possible_wrap`，布局校验约束在缓冲宽度内）、`offset isize→usize`（`cast_sign_loss`，反向落点非负不变量）、`ops.rs` 容差 `iN→u128`（`cast_sign_loss`，前置 `tolerance >= 0` 守卫）。测试片段 6 条夹具 cast（计数小整数 `i32/ usize→f32`、计数缓冲 `f32→usize`）局部 `#[allow]` + 注释。本机 macOS：lib+test pedantic 本体 0，标准 `-D warnings` 绿，124 单测 + 1 proptest 绿，下游 nn/engine/serve/train/autograd/weights/storage check 绿。**win 1.100.0 对拍已实测**（主机 192.168.0.65，`1.100.0-x86_64-pc-windows-msvc`；工作树打包同步至 `C:\Users\fei\llm-verify-tensor`，不含 target/.git/docs/validation）：`check -p llm-tensor` Finished 绿；`clippy -p llm-tensor --all-targets --all-features --locked -- -D warnings` EXIT=0；`test -p llm-tensor` 124 单测 + 1 doc-test、0 failed；`fmt --check` EXIT=0；下游 nn/engine/serve/train/autograd/weights/storage check EXIT=0。win 侧 `prof/src/runtime.rs` `sdot_probe_ready` 2 条 dead_code 系 win 平台 cfg 下该 prof 在途函数未被引用（macOS 有调用路径），与 tensor 整改无关，不阻断 tensor 回归。手动 `-W pedantic` 整图连带暴露 testkit 30 条，属 testkit 独立 crate 单元，不在 tensor 源码属性作用域内，归 testkit 层。
- 全仓存量（lib 级粗估，以当次 clippy 为准）：nn 600、weights 318、train 327、prof 280、cutile 151、algorithm 125、serve 106、engine 94、tools 88、shard 83、storage 80、autograd 49、tokenizer 45、cli 34、testkit 26、fuzz 17；dtype、diagnostic、annotation、tensor 已清零。cast 族（截断/精度/符号/指针对齐）属数值内核与 FFI 固有语义，不得为消告警改转换口径；签名扩散类（`&Option<T>`、pass_by_value）需全调用点核对，均留后续批次逐条裁决。
- 候选层逐点裁决（2026-11-26 当次 `-W clippy::pedantic` 实测，结论：本批均不开门）：
  - `llm-fuzz`：lib 17 条（10 `must_use`、4 `errors_doc`、1 `doc_markdown`、2 有意截断 cast——`Rng::below` u128→usize 为 Lemire 归一化取模到 `[0,bound)`，`Rng::byte` u64→u8 取低 8 位，可局部 `#[allow]` 注释）。但经 `#[path]` 包含的 `seeds.rs`/`drivers.rs` 在多个 harness/test target 重复暴露 150 条 `lifetime tied str`（登记表 `&'static str` 被误判）、10 条 strict float 比较、10 条 u32→f32 精度损失、20 条 `doc_markdown`。整改面大且含 float 比较语义债，留后续批次。
  - `llm-annotation`（已开门，见样板三）：3 条 `pass_by_value` 经核为 clippy 误报，不可改签名——`dsl.rs:63 expand(attribute/item: TokenStream)`、`expr.rs:149 receiver: TokenStream2` 是 proc-macro 按值消费输入流，`lib.rs:244 resolve<S,D>` 泛型 `S` 要移动进返回元组，三者本就该按值，局部 `#[allow]`；其余机械项与测试片段告警已全部清零。
  - `llm-cli`：34 条。含 8 条返回值包装类（6 条 `Result` 不必要包装、2 条返回值多余），属 API 返回类型重构判断，不能机械改；4 条 cast 精度债（usize→f32/f64）；其余 5 `errors_doc`、13 `must_use`、2 `pass_by_value`、1 `doc_markdown`、1 布尔冗余。返回类型重构需逐条评估契约，留后续批次。
  - `llm-testkit`：26 条（dev-dep 测试契约层）。7 条 cast 债（精度/符号/截断）、1 条 strict float 比较；其余 3 扩展名比较、3 match bool、3 `must_use`、3 `Panics` 文档、4 `doc_markdown`、1 long literal、1 绑定名相似。cast 来自测试夹具数值，非生产内核；dev-dep 不进入运行期二进制，优先级低于生产底座，留后续批次。
  - `llm-tensor`（已开门，见样板四）：lib 46 条与测试片段 6 条已清零；8 条布局/容差语义债按样板局部 `#[allow]` + 守卫注释，未改转换口径。三平台回归证据待 win 1.100.0 对拍机补充。
  - `llm-tokenizer`：45 条。15 `errors_doc`、9 `must_use`、3 `doc_markdown`、2 redundant closure、1 lossless `From`、1 unused self、1 返回值包装、1 `pass_by_value`；语义债 12 条截断 cast（7 `usize→u32` 截断 + 5 64 位指针下截断）。留后续批次。
  - `llm-autograd`：46 条。11 `errors_doc`、9 `must_use`、3 match bool、1 needless for_each、1 2 字节按引用、2 unused self；语义债 22 条 cast（精度 1、丢符号 3、截断 2、32 位指针截断 5、64 位指针截断 10、环绕 1），近半为 cast 债，留后续批次。

### 15.3 #53 train/serve 适配协议裁决

- 三适配文件（`train/kernel.rs`、`serve/kernels.rs`、`serve/dense.rs`）定向 jscpd 1309 行 0 clone，逐字比对无真实可合并重复。事实源已统一：`storage::LaneView/LaneViewMut` 取槽、`tensor::View::{one,matrix,matrix_triple}` 装配、`nn::registry` 注册表。
- 不可抹平的执行缝必须留入口层：train 单 `&mut ViewMut` + `Kernel`（Geometry/Running/反向）vs serve 多 `&mut [ViewMut]` + `engine::Program`（Launch block/标量槽）；几何来源与反向核心各异。
- 裁决：当前基线无可合规下沉块；七个候选留入口层或已是事实源。明细见 `validation/train-serve-adapter-protocol-design.md`。第二批前置：先对齐 slot 取槽与错误口径、几何契约，满足后再评估下沉。

### 15.5 #58 构建期公共原语下沉与 nightly 机制配置化（2026-11-26）

- **落点裁决**：build 期公共原语归既有 `diagnostic::buildgen` 通道，不在 `crates/tools` 新建平行 build 内核。tools 定位是运行期静态门禁库（被 testkit normal-dep），其 `build.rs` 只是 vocabulary 生成外壳；prof/tools 已 build-dep `diagnostic(build)`，下沉零新依赖边。AGENTS §2.2/§4：优先扩展已有抽象。
- **feature 拆分**：`diagnostic` 新增 `build-std = []`（纯 std，路径/扫描/相对路径）与 `build = ["build-std", "dep:toml"]`（再带 TOML 读取/合并）；`read`/`merge_tables` 挂 `#[cfg(feature = "build")]`，其余原语挂 `build-std`。default 仍纯 std 零依赖；`--features build-std` 与 default 分别 check 绿。
- **下沉并复用**：`file_stem`/`extension`（prof `build.rs` 两处本地副本删除，11 处调用点切 `buildgen::`）；`merge_tables` 就地递归合并（prof `build.rs::merge_tables` 与 `buildops::merge` 两处同构合一，`buildops` 两处 owned 调用改就地形式）。新增 `merge_tables` 递归覆盖/数组整体替换与路径原语两条单测，挂既有 `buildgen_tests.rs`。
- **nightly 机制从探测脚本改为编译 flags 注入**：`cfg(nightly)` 原由 `nn/build.rs` 与 `prof/build.rs` 跑 `rustc --version` 探测发出。现撤回 `buildgen` 的 `is_nightly`/`emit_nightly_cfg`（无消费），删除唯一薄壳 `crates/nn/build.rs` 及其 build-dep；`prof/build.rs` 保留生成职责（档案/模板表/设备镜像），仅移除 nightly 探测段。源码 `#[cfg(nightly)]` 一行未改。
  - cfg 注入落点：CI `nightly` job 加 `env: RUSTFLAGS: "--cfg=nightly --check-cfg=cfg(nightly)"`；workspace `[workspace.lints.rust]` 登记 `unexpected_cfgs = { check-cfg = ['cfg(nightly)'] }`，消除默认构建的未知 cfg 告警。
  - **为什么没做成 Cargo feature**：`--all-features` 会把空 feature 一并打开，stable 门禁就会编 `std::simd`/SME 通道而 E0554；Cargo 无法把单个 feature 排除出 `--all-features`。改 feature 必须把 stable 门禁换成脆弱的显式 feature 清单，故选用与工具链解耦、不被 all-features 误开的 RUSTFLAGS 注入。
- **实测（本机 macOS，stable 1.100.0，默认不注入 nightly cfg）**：`cargo check --workspace --all-features --locked` 0 error；`clippy --workspace --all-targets --all-features -- -D warnings` 0；`fmt --check` 0。单测：diagnostic 22、nn 563（4 ignored）、prof 179、tools card 39 / gate 13 / lex 27 / words 10，全绿。
- **追加复用**：`buildops::file_stem`（fail 硬终止版）内部改为转调 `buildgen::file_stem`，只保留非 UTF-8 时的 `fail!` 错误适配，消除第二份 `file_stem().and_then(to_str())` 解析链；`build.rs` 的 `stem`/`extension` 副本此前已删。
- **第二轮下沉（纯 std、零口径耦合）**：`buildgen` 再补四个 `build-std` 原语——`string_slice`（带引号字符串切片字面量 `&["a", "b"]`）、`expr_slice`（不加引号的表达式切片 `&[a,b]`）、`read_text`（正文读取，I/O 错误归 `Error::from_io`）、`env_required`（必填构建环境变量，缺失归 `Code::Missing`）。tools 本地 `strings`/`expressions` 两份渲染壳与两处 `env::var(...).expect` 删除、切 `buildgen::`；prof `build.rs::read` 与 `buildtables::archive_with_preamble` 内联 `read_to_string` 收口到 `read_text` + `fail!` 适配；prof `variable` 转调 `env_required`。新增切片渲染、正文读取、缺失环境变量三条单测。
- **build feature 取值原语**：`buildgen::subtable`（必填子表）、`buildgen::string`（必填字符串）返回 `diagnostic::Result`；prof `buildtables`/`devices` 以 `unwrap_or_else(|_| fail!(…))` 保留 `fail!` 硬终止出口，tools 仍走自身 `Result` 口径独立壳，未跨口径强收。两者均补 `# Errors` 文档段以过 `missing_errors_doc` 门禁。
- **nightly 本机实测（2026-11-26）**：本机 `nightly-aarch64-apple-darwin` + `RUSTFLAGS="--cfg=nightly --check-cfg=cfg(nightly)"`，`cargo +nightly check -p llm-nn --lib` 与 `-p llm-prof --lib` 均 0 error：nn `portable_simd` 车道、prof `stdarch_aarch64_feature_detection`（nightly + aarch64）两条通道在 nightly 下可编，stable 退路不注入 cfg 时走标量分支。非 macOS aarch64 的 SME 运行时探测路径本机未覆盖，仍由 CI nightly job 守卫。
- **未收簇（留后续）**：prof/tools 的 TOML 取值辅助（`table/required_text/required_value/integer/optional/strings/expressions` 中带领域判定的部分）错误口径分两套——tools 走 `diagnostic::Result`，prof 走 `fail!` + `exit(1)`；取值 helper 与 prof 渲染链深度耦合，统一口径是独立重构批次，本轮不强收。各强类型视图渲染、prof 设备族/片段/见证扫描、native C 槽位与驼峰命名转换、标识符关键字校验属领域装配，留外壳。
- **复核取证（2026-11-26）**：`cargo machete` 无未用依赖；jscpd 0 clone。全仓 `cargo check/clippy --workspace --all-targets --all-features` 与 `fmt --check` 0 告警；本机单测 diagnostic 31（其中 buildgen 12）+ ffi 2、nn 563（4 ignored）、prof 185、tools card 45 / gate 13 / lex 27 / words 10 / 1，全绿。剩余取值 helper 体量小、未达 clone 阈值，且类型约束分叉（tools 正整数 `usize` vs prof 带缺省 `i64`；tools 可选整数产 `Some/None` 字面量 vs prof 可选串产 `-` 占位），强收会把领域渲染语义塞进构建内核，违反"抽象消除真实重复、不做平行包装"。

### 15.4 未完成项（如实披露）

- 密度门存量红账：超目标值 44,164 行、超注释棘轮 95,800 为 8,364、超重建前代码基线 96,100 为 8,064；删除等于砍能力，违反正确性优先，非本批可消除。注意：即使把 `SRC_MAX` 改回重建前的 96,100，当前仍超 8,064 行，门不会直接转绿——不存在"改回棘轮口径即绿"的捷径，只能无损缩减或对超基线增量逐项取证登记。`SRC_MAX=60000` 系重建提交 b65a0966 回退的项目目标值，与棘轮注释脱节；阈值口径与增量归属属仓库门禁治理决策，本批不擅自调宽。
- #26 除 dtype、diagnostic、annotation、tensor 四个样板层外，全仓 pedantic 存量未整改；大层 cast 语义债与签名扩散留后续批次。tensor 为数值底座，macOS 与 win 1.100.0 对拍机编译/ clippy / test / fmt 均实测绿；linux 云主机未单独跑 tensor 专项。
- #53 第二批前置条件未满足，保持待评估。
- `--perf` 固定硬件基准、`--fuzz`、`--miri`、`--coverage` 未跑；18 项 skipped 平台专项未执行；`tensor::View::one` 维度/步长 Vec 分配缺 release 基准证据。
- 并发在飞项现状（2026-11-26 复核）：`docs/gpu-tuning-candidates-cuda-intel.md`、`crates/prof/src/opencl.rs` 已并入并发提交；`crates/shard/tests/pipeline_tests.rs` 的 fmt 红账由本批 `rustfmt` 单文件修正（仅折行排版，逻辑零变化；该用例 `#[cfg(windows)]`，本机 macOS 不编译运行）。nextest 已在含上述改动的树上跑绿（3031 passed / 18 skipped / 0 failed）。
- prof 构建脚本拆分（并发在途，非本批 annotation 工作）：native 编译段迁出至 `crates/prof/src/nativebuild.rs`，薄入口 `build.rs` 经 `#[path]` 接线，设备扫描归入新增 `devices.rs`；`buildops.rs` 保留 `template_tables_of`。该拆分未提交，归属并发会话，本批不回滚也不纳入交付结论。
- #58 遗留：prof/tools 带领域判定的 TOML 取值簇待错误口径统一后再评估下沉（见 15.5）；nightly 通道（nn `gemv::simd`、prof `native::sme`）已在本机 nightly aarch64 工具链 + `--cfg=nightly` 下 check 绿，非 macOS aarch64 探测路径仍由 CI nightly job 守卫；linux 云主机回归阻塞——vm 项目工作区未挂载本仓库，既有两个检出属其他会话且基线落后，本批改动未提交未同步，待代码落库并同步后由 linux/CI 矩阵执行，不在旧基线伪称通过。
- #58 密度门画像（2026-11-26 实测 `scripts/loc-budget.sh`，真实退出码 1）：src 105,124 行 / 316 文件，超 `SRC_MAX=60000` 目标 45,124 行；>400 行文件 105 个、>600 否决文件 43 个、>8200 行 crate 4 个。本批为 build 期去重减法，未把任何运行期大文件推过线；红账归属存量与其他在途批次，阈值口径与增量归属属门禁治理决策，本批不调宽。
- **全门禁复核（2026-11-26，本机 macOS stable 1.100.0）**：`nextest run --workspace --all-features` 3064 passed / 18 skipped / 0 failed；`cargo test --doc`、`cargo deny check`（advisories/bans/licenses/sources ok）、`cargo machete`（无未用依赖）、jscpd 0 clone、tools 静态门 13、`complexity.sh`（Lizard 1.24.1 / rust-code-analysis-cli 0.0.25）、testkit 架构门 117、`name-lint.sh` 0 命中、`cargo expand`（annotation/train 非空）、typos 0、actionlint 0、`cargo bench --no-run` 编译通过、benchmarks regress 16、ruff check/format、cloc 报告均通过；唯一硬门失败为上述密度门存量超线。`--perf`/`--fuzz`/`--miri`/`--coverage` 专项未跑。