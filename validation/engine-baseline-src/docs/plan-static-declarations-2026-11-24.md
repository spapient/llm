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

动手前重读 `../crates/prof/src/build.rs` 与 `buildops.rs` 后，必须把 build.rs 的职责拆成两类：

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