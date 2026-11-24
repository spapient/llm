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

动手前重读 `crates/prof/build.rs` 与 `buildops.rs` 后，必须把 build.rs 的职责拆成两类：

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

## 12. 风险



* 本机约 31 个并行会话进程在飞，工作树可能被周期 reset；每批落地以磁盘现状复核，改动集中收尾再跑门禁。

* 每批保持 `fmt`/`clippy`/`nextest`/`gate`/`testkit`/`jscpd` 绿，不豁免存量棘轮。