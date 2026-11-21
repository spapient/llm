# cutile DSL —— 设备无关的替换式 kernel DSL

> 版本 v1.1 · 2026-11-14 · 状态：P0 纵切面已落地，P1–P4 仍是 §17 的目标（替代 v0.3 的 `tess` 草案）
> 落地位置：前端机制在 `crates/cutile`，执行半边（编译/见证/报告）在 `../optim`，
> 三层数据在 `crates/op`、`crates/prof`、`crates/prim`；
> 本文只定义 cutile，不涉及运行时之外的训练/服务层
> 本文既是本 crate 的 README，也是整条流水线的设计契约：§0 给结论与现状，§3–§12 给机制，
> §13–§16 给预算与判据，附录给行数口径与复现命令；`tess` 仅作为历史代号保留

---

## 0. 一句话

**算子只声明，设备只建档，指令只写片段：`(算子 × 设备 × 变体)` 的笛卡尔积由一张协商函数消掉，而不是由几万行手写内核消掉。**

三条边界先说清，后面所有取舍都由它们推导出来：

1. **不做通用编译器。** 没有中间表示，没有 pass 流水线，没有自动布局推导。只有一条"展开 → 协商 → 替换 → 发出 → 缓存"的 lowering，每一步都能打印出来给人看。
2. **不做通用算子。** 算子闭集今天 36 条声明（`op::REGISTRY` 是唯一事实源，本文不复述清单）、原语闭集 16 个、布局闭集 4 个、调度闭集 4 个、设备档案 ≤ 8 份。申请闭集之外的组合，显式报错并给出最近可行解，**绝不静默变慢**。
3. **设备无关是硬约束，不是卖点。** 同一份算子声明与同一份模板，必须能在 NVIDIA / AMD / Intel / Apple / 昇腾 / MUSA / CPU / 可移植 GPU 上产出可运行代码；任何"只在某一家的硬件上成立"的设计，直接出局。

这三条对应三个已知的失败模式：CubeCL 用通用 IR 换覆盖面，代价是 30 万行；CuTile 用绑定单一硬件换表达力，代价是换不了设备；TileLang 用 pass 推导换易用性，代价是布局不可控。cutile 选的是第四条路：**声明足够窄，窄到可以穷举；穷举足够全，全到覆盖 LLM 的算子面。**

### 0.1 现在能跑什么（2026-11-14 实测）

设计目标是 §13 的 9,520 行 / 12 算子 / 8 设备。**目标与现状在本文分开写**：§13–§16 的预算是目标，
下表是这条流水线今天真正跑起来的部分——读文档时先看这张表，就不会把预算当成已完成的事实。

| 维度 | 本文目标 | 当前实测 | 复核命令 |
|---|---|---|---|
| 算子声明 | 12 | **22** | `ls crates/op/ops/*.toml \| wc -l` |
| 设备档案 | 8 | **6**：nvidia / intel / metal / ascend / musa / cpu | `ls crates/prof/devices/*.toml` |
| 原语闭集 | 16 | 16 | `prof::atom::Prim::ALL` |
| 片段 | 16 × ≤8 | 6 厂商 × 9 段片段 + 见证 | `ls crates/prim/<vendor>/` |
| 布局 / 调度 | 4 / 4 | 4 / 4 | `op::layout::LayoutSpec` / `op::sch::Schedule` |
| 机制代码 | 9,520（预算） | 9,403 行（含内联测试；剔除后 6,455） | `wc -l crates/cutile/src/*.rs` |
| 用例 | — | 149 单测 + 2 doctest | `cargo test -p cutile --all-features` |

"P0 已落地"指 §16 的五项判据都已有可执行答案；四条命令就是它的全部入口：

```bash
cargo run --bin cutile -- doctor --op gemv_i4 --device cpu   # 整条链逐行打印
cargo run --bin cutile -- verify --op gemv_i4 --devices cpu  # 见证 + 对拍 + 协商首档
cargo run --bin cutile -- bench  --op gemv_i4 --repeat 5     # 达成率（未测量时明说，不省略）
cargo run --bin cutile -- cache gc --limit 1073741824        # 容量回收
cargo run --bin cutile -- ledger --out ledger.md            # 跨档案能力台账
```

CPU 后端在本机（macOS / Apple clang）真实编译并执行，`max_rel_err 0.000e0`；GPU 档案的片段与见证
齐全，但本机没有那些设备时 `verify` 以退出码 3 失败而**不是跳过**——这条区分是 §11.2 的判据，
也是"证据分级"与"有档就算有设备"两种做法最容易被混淆的地方。

---

## 1. 参照实现解剖：取什么、弃什么

### 1.1 五个参照 + 旧工程

| 参照 | 核心机制 | 值得取 | 必须弃 |
|---|---|---|---|
| **CubeCL**（Rust，7 后端） | `#[cube]` 过程宏把 Rust 表达式降到自有 IR，各后端再双轨 codegen | Rust 原生 JIT、`Compiler` trait 抽象、产物缓存、autotune 落盘 | 通用 IR + 优化 pass + 每后端两条 codegen 路径（约 4.5 万行的七个后端）；用户可写任意表达式带来的"范围税" |
| **CuTile Rust**（NVlabs，仅 NVIDIA） | 以 tile 为一等值：`Tensor<T, [BM, BN]>`，宿主侧 `partition()` 切出不相交分区，借用检查器保证无数据竞争，JIT 走 Tile IR | **所有权即并发安全**：宿主用 `&mut` 分片声明独占、`&` 声明只读，网格由分区比推导；tile 层面无手工寻址 | 与 Tile IR 13.x / sm_80+ 强绑定；核心四 crate 75,732 行，其中编译器 41,996 行；换设备即全部作废 |
| **CuTe DSL**（CUTLASS 4.x，仅 NVIDIA） | 布局代数 `Layout`/`Swizzle`/`ComposedLayout` + 原子 `TiledCopy`/`TiledMMA`，编译期组合、零开销 | **"模板 + 配置选择"能到峰值**（CUTLASS 3.x 已证明）；"原子"概念：一个原语带形状参数就能覆盖多代指令 | 布局代数的完备性靠专家脑内建模；模板元编程的错误信息与学习曲线；同样只能 NVIDIA |
| **TileLang**（Python + C++） | tile 级算子 `T.copy`/`T.gemm`/`T.Pipelined`，布局与流水由 pass 推导 | tile 粒度的表达习惯：用户写的是搬运/矩阵乘/归约，不是下标算术 | 布局推导藏在 pass 里——它自己都要留 `annotate_layout` / `use_swizzle` 这类逃生舱，说明推导覆盖不全；实现规模 317,510 行，调试要读 TVM IR |
| **FlashAttention-4** | 算法与 kernel 流水协同设计：warp 专门化、异步 MMA 与 softmax 重叠、按架构不对称扩展调整流水 | **三条工程结论**，全部可以变成配置：①非 MMA 工作必须被流水掩盖；②搬运/MMA/归约分工给不同 warp；③软缩放可以延后到真正需要时 | 把结论写成某一代 GPU 的手写 kernel（换硬件即重写） |
| **旧工程 `llm.u`**（Rust，7 真机后端） | 算子不写代码文件：`{op, skeleton, params}` 条目 + 骨架闭集校验；设备指纹 → 能力行协商；精度按策略表择优 | **已证明"配置驱动 + 闭集校验"能跑**；架构分档的能力行（sm70 无 bf16 / sm120 有 fp4）；证据分级台账；roofline 峰值推导 | 每个后端的每个算子仍要手写内核（CUDA 侧 `cuda.c` 4,778 行 / 25 类内核 ≈ 191 行/内核）；能力矩阵有大量"仅编译"格子；缺设备时用例静默跳过 |

### 1.2 规模口径（只算可比层，见附录 A 的命令）

| 工程 | 口径 | 行数 |
|---|---|---|
| CubeCL | 全部 crate 的 `src/**/*.rs` | **301,766** |
| CuTile Rust | 核心四 crate `src`（cutile 14,068 + compiler 41,996 + ir 12,119 + macro 7,549） | **75,732** |
| CuTile Rust | 全部 crate `src`（含 cuda-core / cuda-async 等绑定） | 101,509 |
| TileLang | DSL 表面 `tilelang/language/**/*.py` | 13,808 |
| TileLang | 自身实现（包内 Python 94,994 + `src/` C++ 222,516） | **317,510** |
| 旧工程 `llm.u` | DSL 相关层（config 4,638 + device 28,325 + kernel 1,451 + C 9,522） | **43,936** |
| 旧工程 `llm.u` | 全部 `src`（含引擎/存储/训练/服务） | 85,649 + 9,523 C |
| **cutile（本文预算）** | P3 全量（12 算子 / 8 设备） | **9,520** |

**这张表要横着读**：CuTile 用 75,732 行只买到一台设备；llm.u 用 43,936 行买到 7 台设备，但每个 `(算子, 设备)` 格子都要人手写内核；cutile 的目标是 9,520 行买到 8 台设备，且**新增一格不需要写代码**。规模不是唯一指标，但它是唯一可复现的指标。

### 1.3 明确认输的地方（写在这里，评审时才不会被当成隐藏负债）

| 对手 | 强在哪 | cutile 的取舍 |
|---|---|---|
| CuTile Rust | 借用检查器级别的数据竞争证明；Tile IR 里有 NVIDIA 自己的调度器 | cutile 计划用宿主侧 `partition()` 继承 Rust 借用检查（§4.4，**P0 尚未实现**），但**不接管设备调度**，交给各家原生编译器 |
| TileLang | 算子生态与 Python 上手速度；tile 级 API 覆盖广 | cutile 只覆盖闭集里的算子（今天 36 条），超出即报错；不提供 Python 前端 |
| CubeCL | IR 级优化、后端成熟度、社区 | cutile 放弃 IR，因此**拿不到跨算子融合**，只做算子内融合 |
| CuTe DSL | 布局代数的表达完备性 | cutile 用 4 构造子的闭集换可判定性，非常规布局在闭集外直接拒绝 |
| 旧工程 `llm.u` | 真机覆盖面与性能台账的严谨度 | cutile 继承它的台账方法（证据分级），但用自动生成的见证替代人填 |

需要说明：**本文不合并 `llm.u`**。它是独立仓库、独立版本、独立发布节奏；cutile 只继承它的两条方法（按架构分档的能力协商、证据分级台账）和一组可复现的实测数字，不迁移它的代码。

---

## 2. 设计公理

1. **闭集优先于表达力。** 任何"用户可以写任意 X"的设计，都会把成本转移到维护 pass 上。cutile 的每个可写维度都是有限枚举，枚举外显式失败。
2. **能力必须是参数化的，不是字符串。** `provides = ["mma"]` 这种写法无法表达"我的 MMA 只支持 m=16"；能力签名必须是 `(原语, 形状区间, 数据类型, 存储空间)`，协商做区间包含判断。
3. **布局是契约，不是推导结果。** 布局只能四选一，写错写不出来的布局就写不出来；落到设备上的唯一硬约束（swizzle 粒度 ≤ bank 数）在构建期校验，不合就失败。
4. **调度是枚举，不是搜索空间。** 4 个调度模板覆盖从 elementwise 到 ping-pong attention 的全部需要；用户选模板 + 填槽位，不写流水线。
5. **正确性来自双源。** 每个算子在 `crates/op` 有一份与设备无关的标量参考实现（`op::model::Model::reference`），片段只与参考对拍；片段写错不会"看起来对"。
6. **每个能力声明都要有见证。** 档案里写 `provides` 一律附带证据等级 L0–L3；`verify` 拒绝把 L1 当验证，缺设备时**失败而不是跳过**。
7. **可解释性是产品能力。** 任何一次选择都必须能回答"为什么是这个变体、缓存为什么命中/失效、缺了哪个原子、最近的可行变体是什么"。

---

## 3. 架构：三层声明 + 三组闭集 + 一条 lowering

三层数据分属三个 crate：**声明层**在 `crates/op`（`ops/`）、**档案层**在 `crates/prof`（`devices/`）、
**片段层**在 `crates/prim`（`<vendor>/`），三者都是纯数据、不含机制；解析、协商、拼接、emit 全部在
`crates/cutile`。依赖方向 `cutile → {op, prof, prim}`、`prim → prof`（片段文件名必须落在原语闭集内），
`op` 与另外两者之间没有依赖。

```
        声明层（人写）                     档案层（每设备一份）              片段层（每原子每设备一段）
   ops/<op>.toml   ops/<op>.k          devices/<vendor>.toml           prim/<vendor>/<prim>.<ext>
   骨架+轴+需求+布局   模板文本            指纹/存储/能力/工具链/证据         10–30 行的指令级实现
        │                 │                      │                              │
        └────────┬────────┴──────────┬───────────┴──────────────┬───────────────┘
                 ▼                   ▼                          ▼
           ① expand            ② negotiate                ④ substitute
        轴 → ≤8 个变体      需求 ∩ 供给 → 路径          模板槽位 ← 片段文本
                 │                   │                          │
                 │            ③ select（plan / roofline）        │
                 └───────────────────┴──────────────────────────┘
                                     ▼
                              ⑤ emit + compile + cache
```

五步的中间产物都是**可打印的数据结构**（`TileCombo` / `Path` / `Plan` / 拼接文本 / `Source`），这是"无 IR"的具体含义：没有隐藏的优化阶段，任何结论都能在 `doctor` 里逐行看到。

### 3.1 目录与模块

逐模块的职责判决由本节与各模块的 `//!` 文档给出；数据 / 机制 / 运行时后端的边界见
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) 第一、二节。

```
crates/op/                      # 声明层：纯数据 + 解析，不含机制
├── Cargo.toml
├── README.md                   # 两个文件各写什么、如何新增算子
├── build.rs                    # ops/*.toml → $OUT_DIR/declared.rs（运行期无 TOML）
├── ops/                        # 36 份声明（.toml 声明 + .k 模板成对出现）
│   ├── attn_decode.toml        #   { skeleton, axes, needs, variants, layouts }
│   └── attn_decode.k           #   模板文本（C 风格 + {{插值}} + @原语）
└── src/
    ├── lib.rs                  # REGISTRY / builtin / model；声明表来自构建期生成物
    ├── decl.rs                 # OpDecl：轴展开、变体链、调用点参数校验
    ├── tpl.rs                  # 模板词法：文本 / {{槽位}} / @原语(...)
    ├── sig.rs                  # 能力签名的声明侧（与档案层同一张属性键表）
    ├── sch.rs                  # 布局与调度的闭集（`layout` / `sch` 两个模块，4 + 4）
    ├── model.rs                # 一条求和表达式 + 确定性样本 + 标量参考
    ├── einsum.rs               # flops / bytes 由表达式与操作数位宽推出
    └── <族>.rs                 # 算子族语义（f32_small / unary / elementwise / reduce / attention / gemv / gemm / dequant / scan），与 ops/<op>.k 对应

crates/prof/                    # 档案层：纯数据，不含机制
├── Cargo.toml
├── README.md                   # 一份档案写什么、如何新增设备
├── build.rs                    # devices/*.toml → $OUT_DIR/embedded.rs
├── devices/                    # 6 份档案（目标 8）：ascend / cpu / intel / metal / musa / nvidia
└── src/
    ├── lib.rs                  # BUILTIN 内嵌档案与索引
    ├── prof.rs                 # schema：指纹、存储、能力、证据等级
    └── atom.rs                 # 词汇表：原语 / 存储 / 属性键 / 证据等级

crates/prim/                    # 片段层：纯数据，不含机制
├── Cargo.toml
├── README.md                   # 片段与见证各写什么、如何新增设备
├── build.rs                    # <vendor>/<prim>.<ext> → $OUT_DIR/tables.rs
├── cpu/                        # 9 段片段 + witness.c，文件名即原语名
│   ├── dot.c
│   ├── reduce.c
│   └── witness.c               # 见证源码（L2 证据），装载片段时跳过
├── nvidia/  intel/  metal/  ascend/   # 同构：cu / cl / metal / cpp（musa 同 cu）
└── src/
    ├── lib.rs                  # BUILTIN / WITNESS 查表、Fragment::parse（签名声明）
    ├── device.rs               # 设备镜像：这次构建带上的源码与可选设备侧二进制
    └── tables.rs               # 构建期生成物的 include! 落点（只有数据）

crates/cutile/                  # 前端（无 IR）：装载、协商、替换、发出、缓存
├── Cargo.toml
├── README.md                   # 本文
└── src/
    ├── lib.rs                  # 公开面、re-export、SCHEMA_VER
    ├── builtin.rs              # 三层数据的装载与交叉校验（声明 / 档案 / 片段）
    ├── cap.rs                  # ② negotiate：需求 ∩ 供给、降级链、诊断构造
    ├── ast.rs                  # 模板渲染的可见中间产物（ADR 0017 的 L1）
    ├── substitute.rs           # ④ 槽位与 `@原语` 替换
    ├── emit.rs                 # ⑤ 加头、加前导、拼正文；片段行数判据
    ├── cache.rs                # 缓存存储：身份键（源文本指纹、产物键、逐项差异）、条目、锁、原子落盘与 LRU 回收
    ├── plan.rs                 # tile 取值与 roofline
    ├── reference.rs            # 对拍口径：相对误差 + 按数据类型的阈值
    ├── harness.rs              # 被测内核的 C / CUDA 基准封装生成（两种方言同模块）
    ├── dirs.rs                 # 三个目录的纯数据值对象（默认路径在调用方，本模块不读环境变量）
    ├── probe.rs / space.rs     # 设备探测与形状空间（纯数据；起进程在 `cutile-run`）
    └── cache/                  # 缓存条目、锁与回收的细分模块

crates/cutile-run/              # 执行半边：编译、执行、见证、语言门与四份报告
└── src/
    ├── toolchain.rs            # 编译器/驱动探测与身份（cc/clang/gcc、nvcc、xcrun metal、MSVC cl）
    ├── runner.rs               # 编译、执行与回读原语（可执行文件、Outcome、分位样本、位模式解码）
    ├── build.rs                # ⑤ 指纹 → 查缓存 → 取锁 → 拼接 → 落盘
    ├── device.rs               # 统一设备：按 Emit 查描述符表；共享装载/起进程/回读 + C/CUDA/Metal 三段编排（无 unsafe）
    ├── backend.rs              # 后端缝（Codegen / Driver / Readback）与注册表
    ├── gate.rs / witness.rs    # 语言门与见证执行器
    ├── pipeline.rs             # 四条命令共用的流水线（Run / Prepared / …）
    ├── ledger.rs               # 跨档案能力台账：证据行聚合与原语×后端矩阵格（不走 pipeline）
    └── doctor.rs / verify.rs / bench.rs / execution.rs   # 命令策略与结构化报告
```

前端与执行分处 `crates/cutile` 与 `../optim`，划线判据是**证据从哪来**：
前端只回答"这段源文本长什么样"，跨过 `witness` 之后必须有"这台机器上真的跑出过这个结果"。
执行半边内部**分层不混**：`build` / `runner` / `toolchain` / `device` / `cap` / `cache` / `emit`
是机制（决定"产物长什么样"），`doctor` / `verify` / `bench` / `execution` 是命令策略；`ledger` 跨档案聚合证据行与矩阵格，不经 `pipeline`
（决定"跑哪些形状、跑几次、工作目录怎么命名、达成率按什么口径算"），后者只经由 `pipeline`
调用前者。这条接缝是可检验的：改"怎么审"不需要动任何一行生成逻辑，`pipeline.rs` 是唯一的
共同入口——四条命令各写一遍流水线曾让同一台设备在两个命令里得出不同结论。

闭集与解析不在本 crate：布局构造子、调度模板、模板词法、`OpDecl` 的解析都在 `crates/op`
（`op::layout` / `op::sch` / `op::tpl` / `op::decl`）；`cutile::layout` 与 `cutile::sch` 是
`pub use` 转发路径，模块路径不变，但闭集只有一份。命令行文法同样不在本 crate：它是
`llm-cli::cutile` 的 `CutileCommand`，驱动方是 `llm-cli` 包的 `src/bin/cutile.rs`（L7 入口，只做
解析 → `cutile` → 排版 → 退出码），因此 `cutile` 不再持有第二套命令语法或第二套输出排版。

TOML 解析与运行时配置不在本 crate：它们是与领域无关的基础设施。语料（`ops/*.toml`、`devices/*.toml`）
在构建期由各自的 `build.rs` 解析成 `diagnostic::data` 的表，运行期没有任何 TOML 解析器
（[ADR 0022](../../docs/adr/0022-build-time-corpora.md)）；
配置策略同样不在本 crate：本 crate 只声明"我要哪三个目录"（`crate::dirs::Dirs`，纯数据），默认目录、
`CUTILE_*`、`--set` 与来源追踪都在命令行层 `llm-cli::settings`
（[ADR 0023](../../docs/adr/0023-settings-to-cli.md)）。默认目录是部署策略、
`--set` 是命令行文法，两者都属于调用方；内核因此只依赖 `std::path`，测试给临时目录、服务给容器卷时
都不必改本 crate 一行。它曾在本 crate 的 `settings/` 里，删掉配置文件层（ADR 0022）之后
剩下的三层不足以撑起一套分层机制，于是整块上移。

声明层数据也不在本 crate：它在 `crates/op`（包名 `op`，只依赖 `std`），`cutile` 单向依赖它。
因此新增算子、改声明或改模板都不必动本 crate 的代码，数据与机制不会混成一处。
算子的**语义**同样在那里（`op::model`）：一条求和表达式、确定性样本、标量参考与内核实参——`cutile`
只消费它们，代码里不出现任何算子名或算子专属的流量公式。`flops` 与 `bytes` 不再手写：它们由语义层
`op::einsum` 从表达式与操作数位宽推出（`Cost` 的所有权在那里，`op::model` 只再导出）。
分组轴（`scales[n, g]` 的 `g = m/GROUP`）
声明为派生轴，只计流量、不计运算次数——否则 roofline 的 `flops` 会多乘一个因子。

档案层同理，在 `crates/prof`（包名 `prof`）：新增设备只加 `devices/<name>.toml` 并在
`BUILTIN` 里补一行，`cutile` 侧一行不改。

片段层在 `crates/prim`（包名 `prim`，只依赖 `prof` 的词汇表）：`<vendor>/<prim>.<ext>` 是"这条能力怎么写"。
装载（`prim::builtin` / `prim::load`）随数据一起搬走，文件名不属于原语闭集即报错；`witness.<ext>`
与片段同目录同扩展名，装载时按名字跳过。档案里的 `witness` 因此是**相对片段层根**的路径
（仓库内根是 `crates/prim`，设置 `data.root` 时是 `<data.root>/prim`），`cutile` 不再需要知道自己
的清单目录在哪。

每段片段的第一行是**签名声明** `/* cutile-sign: <prim>/<impl> */`：同一原语在档案里可以有多条
签名（`dot` 的 `dp4a` 与标量 `fma`），片段只说"这段文本写的是哪一条"。`prim::Fragment::parse`
在装载期解析它——缺声明、声明的原语与文件名不一致、`impl` 为空都直接报错；`Catalogue::validate`
再用 `cap::fragment_gaps`（与 `negotiate` 同一张需求表、同一个 `select`）判定"协商会挑中的那条
实现"是否正是"片段声明写的那条"。缺这一段时，"能协商"与"能生成"可以同时为真却指向不同的算法，
错误会晚到构建期甚至不被发现。声明行不进 emit（拼接只取它之后的文本，见 `Fragment::body`），
也不进缓存指纹，所以它是纯元数据：补一行不会改变已生成的源文本。

### 3.2 与 workspace 的接口

`crates/cutile` 已是 workspace 成员，继承 `version/edition/rust-version/license/repository` 与 `[lints] workspace = true`。`missing_docs`、`unsafe_op_in_unsafe_fn`、`clippy::all`+`cargo` 全部生效，因此本设计必须自带三条工程条款：

- **unsafe 边界**：本 crate 不含 `unsafe`。`runner` 通过 `cc` 子进程编译并运行生成的源文件，不与设备 SDK 链接，因此没有需要白名单的 FFI 段；若将来引入真机驱动，`unsafe` 只允许出现在 `device` 编排模块的新增驱动段里，并且每个 `unsafe` 块必须写明安全前提（前条件、后条件、失败行为）。
- **文档豁免**：`substitute.rs` / `emit.rs` 与 `cache.rs` 的私有项允许 `#[allow(missing_docs)]`，公开项不豁免；`crates/prim/**` 目标是 C/CL/MSL 源码，不受 Rust 文档检查约束。
- **配置边界**：依赖方向固定为 `llm → cutile → {op, prof, prim}`（命令策略与机制同处一个 crate；反向边不存在），其中 `prim → prof`（片段靠词汇表校验文件名），`op`/`prof`/`diagnostic` 之间互不依赖，任何一层都不得反向依赖 `cutile`。本 crate 不得自行读环境变量、不得拼相对路径、不得新增配置键：
  - 缓存目录、工作目录、可选数据根（`data.root`）以 `crate::dirs::Dirs` 传入，**纯数据**：内核不读环境变量、不拼默认路径；
  - 新增配置项的入口在 `llm-cli::settings` 的键表（一行一个）；`--set` 里未登记的键由那一层直接报错，本 crate 不认键名；
  - 默认目录取平台用户目录（macOS `~/Library/Caches/cutile/cache`、Linux `$XDG_CACHE_HOME/cutile/cache`、Windows `%LOCALAPPDATA%\cutile\cache`），与仓库布局、启动目录都无关；没有配置文件，需要偏离默认时用 `CUTILE_*` 环境变量或 `--set`。

---

## 4. 声明层：用户写什么

用户只写两个文件，都在 `crates/op/ops/` 下（`ops/*.toml` 在构建期由 `op` 的 `build.rs` 解析成声明表，
`cutile` 只消费解析结果，不解释存放位置）。

### 4.1 算子声明（`crates/op/ops/attn_decode.toml`）

```toml
[op]
name     = "attn_decode"
skeleton = "elementwise"       # 骨架闭集：elementwise | reduce | gemm | scan
summary  = "分页 KV 的单 token 解码注意力（GQA）"

[axes]                        # 有限枚举，编译期展开；乘积 ≤ 8 个变体，超出即拒绝
BM = [16, 32]
BN = [64, 128]
STAGE = [2, 3]

[needs]                       # 能力签名：原语 + 形状区间 + 数据类型 + 存储空间
mma      = { m = "16..64", n = "8..128", k = 16, dt = ["f16", "bf16"], space = "FAST" }
load     = { bytes = "<=512", src = "GM", dst = "FAST" }
exp2     = { dt = ["f16", "f32"] }
reduce   = { axis = "BN", op = ["max", "sum"], dt = ["f32"] }

[layouts]                     # 布局契约：槽位 → 闭集构造子
q = "row"
k = "swz<3,3,3>"
v = "row"
o = "row"

# 有序降级链：order 小者优先；每个变体各自声明增量需求（关键修正）
[[variants]]
name      = "ws_pingpong"
order     = 1
schedule  = "pingpong<2>"
mma_share = 0.82
extra_needs.shuffle    = { width = 32 }
extra_needs.load_async = { bytes = "<=512", src = "GM", dst = "FAST" }

[[variants]]
name      = "pipe3"
order     = 2
schedule  = "pipe<3>"
mma_share = 0.74
extra_needs.load_async = { bytes = "<=512", src = "GM", dst = "FAST" }

[[variants]]
name      = "smem_only"
order     = 3
schedule  = "pipe<2>"
mma_share = 0.61

[[variants]]
name      = "naive"
order     = 4
schedule  = "direct"
mma_share = 0.40
```

要点：

- `needs` 是**算子级公共需求**，`extra_needs` 是**变体级增量需求**，两者必须用同一套能力签名词汇（不是字符串数组）。`naive` 不需要 `shuffle`、也不需要 `load_async`——这一点必须能在数据里表达，否则协商对"降级"无效。
- `mma_share` 是该变体在 roofline 模型里**落在 MMA 上的时间占比**：未掩盖的非 MMA 时间就是 `(1 - mma_share) / stages`。首选变体必须把它压到 20% 以内，否则装载期直接拒绝（判据在 `op::sch::check_masking`，见 §12.2）。
- **`schedule` 只进计划，不改正文**：`pipe<2>` / `pipe<3>` 是**建模**上的流水级数，掩盖账按它算；
  P0 的片段把 `load_async` 降成同步加载（档案据此登记 `stages = ["1"]`，见 §5），所以同一算子的
  `pipe2` 与 `direct` 生成的内核正文相同，差别只在头部的 `variant=` / `schedule=` 和 `plan` 行。
  真做多级预取要新增一条供给签名并升证据等级，而不是在需求的 `stages` 里填个大数字——填了只会让
  「已验证的供给必须有正文」这条闸亮红。
- **调用点是签名的第二半**：模板里 `@原语(key: value)` 的命名参数会原样拼进片段文本，说的就是"这次生成用的是哪条签名"。若 `key` 属于能力属性闭集（`sat` / `op` / `axis` / `kind` / `accuracy` / `group` / `lanes` / `src` / `dst` / `stages` / `exec`），它必须在该变体的 `needs` / `extra_needs` 里声明且取值被允许，否则设备会按声明的值被验证、代码却按模板的值生成；闭集之外的命名参数（如 `along:`）是片段级提示，协商不检查它们，留给片段。判据在装载期（`op::decl::OpDecl::check_call_arguments`），错误带变体名、模板行号与原语名。
- `axes` 的笛卡尔积是**tile 候选**（≤ 8），`[[variants]]` 是**降级链**（≤ 8），两者不是同一维度：前者决定一次编译内的分块，后者决定用哪套调度。P0 只取笛卡尔积的**第一个**组合（轴取值的顺序即优先级，与降级链同一口径）；在候选之间比较需要占用率/分块收益这类可推导常数，P0 的档案里没有，所以不做（§9）。
- 变体数上限 8；`axes` 乘积超过 8 直接拒绝，报错时给出建议剪掉的轴。
- 为什么是 TOML 而不是 `#[op]` 过程宏：声明要可 diff、可被 `verify` 单独校验、不引入 `syn`/`quote` 依赖（见 §18 D3）。Rust 只在**新增骨架**时才写——骨架闭集只有 4 个，写满即止。

### 4.2 模板（`crates/op/ops/attn_decode.k`）

语法刻意保持极小：可解析构造只有 `{{槽位}}` 插值、`@原语(参数)` 调用、`#define` 与 `for`，其余文本原样保留。

```c
// 逻辑一份，槽位极少；设备差异全部在片段里
#define BM  {{BM}}
#define BN  {{BN}}

q = @load(Q, GM -> FAST, {{q}});               // 布局槽位：来自 [layouts]，可被校验
for (kb = 0; kb < NB; ++kb) {
    k = @load(K + kb * BN * D, GM -> FAST, {{k}});
    s = @mma(q, k, {}, tile(BM, BN, D));
    s = @exp2(s - @reduce(s, along: BN, op: max));
    acc += @mma(s, v, acc, tile(BM, D, BN));
}
@store(O, FAST -> GM, {{o}});
```

**模板语法（完整 EBNF 见 §10.1）**：标识符、数字、`{{轴或布局槽位}}`、`@原语(...)`、方向箭头 `GM -> FAST`、布局构造子调用、`#define`、`for`、`//` 注释。没有函数定义、没有用户宏、没有泛型；控制流只有 `for`。

**诊断精度承诺**：解析错误定位到 `(文件, 行, 槽位名)`，不承诺列号。这是省代码的正确做法——先把精度降下来，解析器才能小。

**错误契约**：`cutile`（连同 `op`、`prof`、`prim`、`cutile`）的失败一律是 `diagnostic::Error`，类别取自闭集 `Code`：缺能力 `unsupported`、输入不合法 `invalid`、缺东西 `missing`、读写 `io`、本层不变量被破坏 `internal`。诊断文字把算子名、形参名、行号作为上下文帧挂上，不再手拼前缀；`Result<_, String>` 在这五个 crate 里已经绝迹（`tests/architecture.rs` 会扫）。数据层（`prof` / `prim`）自己给类别：档案写错、片段第一行不是签名声明是 `invalid`，文件不在是 `missing`（`Error::from_io` 是这条判定的唯一来源），`cutile` 只往上挂"哪个厂商、哪一层"的上下文帧，不重新分类；装载期的多条失败用 `Error::also` 逐行追加，首条决定退出码，其余行各自带类别。类别同时是退出码：`Code::exit_code()` 给出 `2 / 3 / 4 / 5 / 70`，所以 `verify` 遇到"本机没有该设备"是 `3`，而 `--shape 3x1` 写错是 `2`。

### 4.3 一份声明的作用域

| 维度 | 谁写 | 数量 | 改动成本 |
|---|---|---|---|
| 算子 | 人（声明 + 模板） | ≤ 12 | 一份 TOML + 一份模板，约 220 行 |
| 设备 | 人（档案） | ≤ 8 | 约 60 行 TOML + 16 段片段，约 210 行 |
| 原语片段 | 人（指令级） | 16 × 8 ≈ 128 段 | 每段 10–30 行 |
| 变体 | 展开器生成 | ≤ 8 / 算子 | 0 行 |
| 布局 | 闭集 4 选 1 + 组合 | 每个槽位一个 | 0 行 |
| 调度 | 闭集 4 选 1 + 槽位 | 每个变体一个 | 0 行 |

### 4.4 宿主侧：分区即网格，所有权即并发安全（P0 未实现）

> 状态：**设计保留，代码尚未提供**。P0 的消费方式是 `cli` 内的测试夹具：内核拿到裸指针与标量，
> 分片与 `grid` 由调用方自己给。原先的 `host.rs` 写了 `Tensor::partition` 与网格推导，
> 但全仓零调用、也没有对外 API 落地（D5 仍未决），已于 2026-11-12 删除；
> 等 `llm` 真的要直接调用 kernel 时按本节重建，并把它接进构建路径而不是另写一份。

```rust
// 借用检查器保证：分片之间不相交，输入只读
let mut y = Tensor::<f32>::zeros(&[4096]).partition([128]); // 32 片 → grid.x = 32
let x = Tensor::<u8>::from_host(&x_host);
let w = Tensor::<u8>::from_host(&w_host);

cutile::ops::gemv_i4(&mut y, &x, &w).sync()?;   // 同步；.async_()? 返回 future
let y_host = y.unpartition().to_host_vec()?;
```

三件事由类型系统保证，而不是由文档保证：

- **独占由 `&mut` 分片表达**：同一分片不能同时被两个 kernel 写（借用检查器直接拒绝编译）。
- **网格不由人写**：`partition([128])` 的分片比就是 `grid.x`，分片数与形状不整除在构造期报错。
- **设备不由人写**：`probe` 选档案、`negotiate` 选变体、`select` 取声明的 tile（P0 无候选可比较，见 §9），调用点只给形状。

这套宿主模型与 CuTile Rust 一致；差别在于**所有权不依赖 CUDA**：同一个 `partition`/`&mut` 模式原样服务 8 台设备。CuTile 的遗憾是它把这套好模型绑在了 Tile IR 上，而不是模型本身有问题。

---

## 5. 能力模型：16 个原语槽位 + 参数化签名

### 5.1 原语闭集

原语是模板唯一能调用的东西，也是片段唯一的实现单位。**16 个，不多不少**；加第 17 个要走设计变更。
这份闭集与档案 schema 的事实源在 `crates/prof`（`src/atom/`、`src/prof.rs`），`cutile` 只 re-export——词汇表与设备数据同住，机制层改不动它们。

| # | 原语 | 语义契约 | 关键参数 | 布局要求 |
|---|---|---|---|---|
| P1 | `mma` | `D = A × B + C`，tile 粒度，累积 | `m/n/k`, `dt`, `space` | A 行主、B 列主/转置两者其一 |
| P2 | `dot` | 整数点积（dp4a/dpas/dot4） | `lanes`, `dt` | 元素连续 |
| P3 | `load` | 全局 → 片上，同步 | `bytes`, `src/dst` | 源连续或跨步 |
| P4 | `load_async` | 异步搬运 + 完成令牌 | `bytes`, `stages` | 同 P3，允许未完成 |
| P5 | `commit_wait` | 异步组提交/等待（cp.async.commit / wait_group） | `group`, `n` | 无 |
| P6 | `store` | 片上 → 全局 | `bytes`, `src/dst` | 目标连续 |
| P7 | `bar` | 块内屏障 / arrive-wait | 无（`scope` 不在属性闭集里，由片段决定） | 无 |
| P8 | `shuffle` | 子组内 lane 交换（宽度 = 子组宽度） | `width`, `dt` | 寄存器级 |
| P9 | `broadcast` | 子组内广播 | `width`, `dt` | 寄存器级 |
| P10 | `reduce` | 沿轴归约（max/sum/min） | `axis`, `op`, `dt` | 归约轴在片内连续 |
| P11 | `scan` | 沿轴包含前缀（inclusive） | `axis`, `op`, `dt` | 同 P10 |
| P12 | `exp2` | 快速 2 的幂（硬件 ex2 或查表） | `dt`, `accuracy` | 元素连续 |
| P13 | `rcp` | 倒数 / rsqrt | `dt`, `kind` | 元素连续 |
| P14 | `cvt` | 数据类型转换（含饱和 f32→i8） | `from`, `to`, `sat` | 元素连续 |
| P15 | `quant` | 缩放/零点应用、int4 打包 | `bits`, `group`, `sym` | 组内连续 |
| P16 | `atomic` | 全局原子（add/max），tile 粒度 | `op`, `dt`, `space` | 目标对齐 |

约定：`P3/P4/P5` 三件套取代了 v0.3 里 `load` 与 `load_async` 并存又互不认账的问题——**只有 `load` 和 `load_async` 两个名字，异步语义由 `commit_wait` 显式表达**，`provides` 里写的是参数而不是别名。

### 5.2 能力签名与兼容性

```rust
pub struct Need {
    pub prim: Prim,               // 闭集枚举，16 个变体
    pub m: Range<u32>,            // 形状区间（其余维度同理）
    pub dt: DTypeSet,             // 闭集位集
    pub space: SpaceSet,          // {GM, FAST, REG}
}

pub struct Provide {
    pub prim: Prim,
    pub m: Range<u32>,            // 设备实际支持的区间
    pub dt: DTypeSet,
    pub space: SpaceSet,
    pub impl_name: &'static str,  // "wgmma" / "mma.sync" / "dpas" / "neon" ...
    pub grade: Grade,             // L0..L3，见 §11.2
}

impl Provide {
    pub fn covers(&self, need: &Need) -> bool {
        self.prim == need.prim
            && self.m.start <= need.m.start && need.m.end <= self.m.end
            && need.dt.is_subset(&self.dt)
            && need.space.is_subset(&self.space)
    }
}
```

**这是与所有参照实现最重要的差别**：能力不是"有没有 mma"，而是"哪种 mma、支持多宽的形状、什么精度、在哪个存储层"。`(m=16) ⟹ (m=64)` 这种包含关系让协商可以被一次区间比较判定，不需要任何启发式。

**签名的参数也属于原语**：`Prim::attr_keys` 把"一个签名能有哪些属性键"压成每原语的闭集
（§5.1 那张表的"关键参数"列，`width` 即键 `lanes`；`exec` 对所有原语成立），`[needs]` 与
`[[provides]]` 用同一张表拒收。两层的解析器还共用同一套写法：主维度接受 `m` / `bytes` /
`extent`，`dt` 与 `space` 接受标量或数组，形状不对即报错。否则同一份签名换个写法就会少一个
约束（`space = "FAST"` 落成空集），"能协商"与"声明过"从此说的不是一回事。

#### 执行单元也是闭集里的一格

`exec` 是属性键闭集（`AttrKey::ALL`）里的一个键，把"这条能力由哪个硬件单元实现"写进数据：

```toml
[units.cube]             # 单元名由档案定义，代码不认识任何具体名字
space = ["FAST", "REG"]  # Cube 只能访问 L1/L0

[[provides]]
prim  = "dot"
impl  = "mmad"
dt    = ["f16", "bf16"]
space = ["FAST", "REG"]
exec  = ["cube"]         # 加载期校验：cube 必须访问得到上面声明的每一层存储
```

于是协商里没有厂商分支：`Need` 写 `exec = ["vector"]` 就是"我要 vector 实现的点积"，
`Provide::covers` 用同一套集合/区间判定回答"有没有"；档案若写 `exec = ["cube"]` 却声明
`space = ["GM"]`，加载期直接报 ``provides[dot].space：`cube` 访问不到 [GM]``，而不是等真机跑错。
新增设备乃至新增一种执行单元都只改 `devices/<name>.toml`，`cutile` 一行不动。

### 5.3 档案（`crates/prof/devices/nvidia.toml`）

```toml
[meta]
vendor   = "nvidia"
id       = "sm100"          # 指纹匹配键，最具体者胜出（继承旧工程的 pick_specific）
emit     = "cuda"           # 输出语言，可切换（P4 验证 cuda ↔ ptx）
tool     = "nvcc"
tool_ver = "13.3"
checked  = "2026-11-11"

[fingerprint]               # 只如实探测，不做判断
match = { cc = "10.0..10.3", name = "contains: B200|GB200" }

[space]                     # 三级存储，语义名固定，物理名可配
GM   = "global"
FAST = "shared"
REG  = "register"

[mem]
FAST = { bytes = 233472, bank = 32 }   # 228 KB smem
REG  = { bytes = 262144 }

[simt]
group      = 32
groups_max = 64

[[provides]]                # 每行一个能力签名，全部带证据等级
prim = "mma"
impl = "tcgen05"
m    = "16..256"
n    = "8..256"
k    = "16..32"
dt   = ["f16", "bf16", "f8e4m3", "f4"]
space = ["FAST"]
grade = "L3"
witness = { file = "nvidia/witness.cu", func = "witness_mma", expect = [1.0] }
```

规则（全部由 `prof.rs` 与 `cutile` 的 `witness` 校验，不是约定）：

- `provides` 里出现闭集外的 `prim` / `dt` / `space` → 加载失败。
- `grade = "L2"` 及以上必须有 `witness = { file, func, expect }`：文件存在、函数在文件里、期望值非空；`verify` 再逐条编译执行并与 `expect` 对拍。
- 同一 `(vendor, id)` 只允许一份档案；指纹多行同等具体 → 加载失败（不猜顺序）。
- 同一原语可以有多条 `provides`（用 `impl` 区分互不覆盖的签名）：协商先挑**能覆盖**需求的那条，再比证据等级；都不覆盖时挑差异最少的那条做诊断。
- 每条 `provides` 的 `impl` 必须在片段层有一段声明它是同一签名的文本（`/* cutile-sign: <prim>/<impl> */`）：`Catalogue::validate` 逐条比对协商会挑中的实现与片段声明的实现，不一致即装载失败。
- 缺 `emit` 或 `tool` → 加载失败。

### 5.4 八份档案一览

"设备无关"必须在数据上成立，所以把 8 份档案的差异一次列清。注意 `FAST` 一列：**同一份模板在五种不同的片上存储名字上跑**，这就是设备无关的具体含义。

| 设备 | 指纹键 | `FAST` 物理映射 | `emit`（可切换） | 工具链 | 见证来源 |
|---|---|---|---|---|---|
| NVIDIA | `cc = 7.0..12.1`（按代分 sm70/80/90/100） | `shared` | `cuda` / `ptx` | nvcc / NVRTC | 真机 |
| AMD | `gfx = gfx9..gfx12` | `lds` | `hip` / `opencl` | hipcc / clang | 真机（RDNA） |
| Intel GPU | `name contains Intel`、`arch = gen12+` | `slm` | `opencl` / `spirv` | IGC / oneAPI | 真机 |
| Apple | `name contains: Apple M`、`gpu_cores 10..80` | `threadgroup` | `metal` | xcrun metal | 真机 |
| 昇腾 | `soc = Ascend 910B..` | `L1`（MTE 搬运） | `ascendc` | ccec（CANN） | 文档（L0/L1） |
| MUSA | `name contains MTT S` | `shared` | `cuda`（源码兼容） | mcc | 文档 + 模拟 |
| CPU | `arch = x86_64 / aarch64`、`features = <arch>+avx2…`（运行期 cpuid，不是编译目标） | `L2`（分块） | `c` | cc / clang | 真机 |
| 可移植 GPU | `api = vulkan / webgpu` | `workgroup` | `spirv` / `wgsl` | glslang / naga | 模拟 |

加第 8 台（可移植 GPU）的理由不是"凑数"，而是**兜底**：在没有厂商 SDK 的环境里，`emit = wgsl` 仍能让同一份模板产出可运行代码；能力不足时协商自动降级，而不是整套不可用。同理，`cuda` 一条 emit 同时服务 NVIDIA 与 MUSA，说明"设备"与"输出语言"是两个维度，不能合并成一个概念。

---

## 6. 布局：4 个构造子 + 声明式

### 6.1 闭集

| # | 构造子 | 语义 | 例 |
|---|---|---|---|
| L1 | `row(shape)` | 行主序紧凑 | `row(BM, BN)` |
| L2 | `col(shape)` | 列主序紧凑 | `col(BN, BM)` |
| L3 | `swz<B,M,S>(base)` | 在 base 上做 XOR swizzle（bank 冲突消解） | `swz<3,3,3>(row(64,64))` |
| L4 | `pad(base, p)` | 在 base 的最后一维插入 `p` 个填充元素 | `pad(row(64,64), 4)` |

四个构造子的**组合是受限的**：`swz` 与 `pad` 只能包裹 `row`/`col`，不能互相嵌套。这条限制把布局的全集压到可枚举（正规形因此在 P3 引入时唯一）。

### 6.2 声明式与判定

`[layouts]` 的取值就是闭集里的构造子本身：`op::layout::LayoutSpec` 负责解析、校验，并渲染回同一串文本
（`row` / `col` / `swz<3,3,3>` / `pad(4)`）。这串文本既是模板插值（`{{w}}`）的内容，也是缓存键里的一项——
布局在构建期的身份是**声明**，不是某个具体形状下的步长。

正规形（`(order, shape, stride, swizzle, pad)` 五元组）需要槽位的两级形状，而 P0 的构建期没有形状：
`tile` 是轴取值（`GMAX=32`），不是 `w[n, m/2]` 这样的槽位形状。用编出来的形状去实例化，只会让缓存键与
诊断看起来覆盖了形状，实际上描述了一次从未发生的编译（审计里原来的 `instantiate(tile, 64)` 就是如此）。
因此 P0 只保留闭集与声明式；等槽位形状与片段声明的布局都到位（P3）时再加回正规形与判定。

**P0 没有"模板给 vs 片段要"的比对。** 片段是 C 源码，参数只有指针与标量，不声明自己要求的布局，
也就没有第二个正规形可比。原先设计的三段判定（相等 / 可 `@repack` 重组 / 不可重组）需要片段
先能声明布局，否则判定的两个输入里有一个永远是空的。与其留一个测试过却从不参与流水线的函数，
不如把它删掉，等片段层真的有布局声明时再加回来——那时才测得出 `@repack` 的频率（D2）。

**设备侧唯一真实的布局约束是 bank 数**，它在构建期执行：`build_source` 对 `[layouts]` 里每个
`swz<B,…>` 调 `space::check_alignment`，`2^B` 超过档案 `mem.fast_bank` 即报错，`B ≥ 32` 也不会
溢出成 panic。声明里写 `swz<40,3,3>` 现在得到一条诊断，而不是运行期崩溃。

**为什么不做布局推导**：CubeCL 的 `cubecl-tile/{layout,space,stage}` 合计 16,666 行、TileLang 的 `tilelang/{layout,transform}` 合计 3,520 行，再加上各自 C++/IR 侧的实现，全部目的都是"给定算子自动求布局"。cutile 把这件事换成"人写一个四选一的属性 + 机器校验"。少掉的不是能力，是**不可解释性**——而且写错的代价从"运行慢 3 倍且没人知道"变成"构建期报错"。

### 6.3 三级存储

`GM / FAST / REG` 是**语义名**，每个档案映射到本设备的物理名（NVIDIA `shared`、Intel `slm`、昇腾 `L1`、Metal `threadgroup`、CPU `L2/栈`）。模板只写语义名，因此同一份模板在昇腾上的 `GM -> FAST` 会自动变成 `GM -> L1` 的 MTE 搬运。

`device.rs` 的 `space` 模块只做四件事：名字映射、容量取值（`capacity`，档案缺项即"不限"）、容量校验（`FAST`/`REG` 声明用量 ≤ 档案字节数）、对齐校验（`swz` 的粒度与 bank 数兼容）。后两件事不再是"留着等人的工具"：`build::contract` 对协商出的变体算一遍声明用量，逐层调容量校验，并对每个 `swz` 槽位调一次对齐校验，不合就在构建期失败。

这个契约**在查缓存之前**执行：声明里的 `bytes` 上界不改变源文本，因此不在缓存键里；若把判定放到命中之后，一份已经落盘的旧文本就成了"声明越界也免检"的通行证。`contract` 返回的就是判定时真正比较过的那些数（原语、层、字节），`doctor` 的 `[存储]` 段直接打印它们与同一份档案容量——"它为什么通过"与"它到底算了什么"因此是同一组值。

### 6.4 闭集变更流程

§8.3 提到"闭集变更"，这里把它写成固定动作，避免闭集被随手扩大：

1. 写一页 ADR，说明新取值要覆盖的现实需求（含至少一个真实算子/设备）。
2. 重算 §13 预算；闭集变大的部分必须写明从哪里让出（总量不许超 10,000）。
3. 补见证：每个新增原语 × 新增设备都要有 L2 见证，否则该能力默认不可用。
4. 升 `schema_ver`，旧 `plan` 表与缓存全部失效（这是刻意的：宁可重编译，不要用旧映射跑新闭集）。

---

## 7. 调度：4 个模板

| # | 模板 | 结构 | 何时用 | 槽位 |
|---|---|---|---|---|
| S1 | `direct` | 单级，无重叠 | elementwise、带宽受限的 GEMV 小形状 | `block`, `vec` |
| S2 | `pipe<N>` | N 级预取 + 双缓冲（有 `load_async` 则异步，否则同步双缓冲） | GEMM、scan、attention 的 K/V 流 | `N`, `commit` 位置, `prefetch_bytes` |
| S3 | `ws<producer, consumer>` | warp 专门化：搬运 warp / MMA warp / 归约 warp 分离 | attention prefill、大 GEMM | `roles`, `barrier` 拓扑, `mailbox` |
| S4 | `pingpong<2>` | 两组交替，用另一组的非 MMA 时间填自己的 MMA 空档 | 非 MMA 占比 > 25% 的算子 | `groups`, `swap` 点, `mma_share` |

**FA-4 的三条经验如何落成配置**（这是"取优点"的具体动作）：

1. **非 MMA 必须被掩盖** → 每个变体声明 `mma_share`；`verify` 用 roofline 检查"未掩盖的非 MMA 时间占比 ≤ 20%"，超了不仅警告，还要求换调度模板。
2. **搬运/MMA/归约分工** → S3 的 `roles` 槽位就是分工表；写不出合法分工时协商失败（而不是悄悄退化成单 warp）。
3. **软缩放延后** → 不是新算子，是 attention 变体的一个开关 `rescale = "lazy"`；变体闭集从 8 里分一个位置给它，收益用 `mma_share` 增量度量。

调度模板本身是代码（`sch.rs` 420 行），但**它生成的是模板文本里的骨架槽位**，不是设备指令。设备差异不进入调度层——这是 S1–S4 能在 8 台设备上成立的原因。

---

## 8. 协商：需求 ∩ 供给，显式降级

### 8.1 算法（类型正确的版本）

```rust
pub fn negotiate(op: &OpDecl, prof: &Profile) -> Result<Path, NoPath> {
    for (idx, var) in op.variants.iter().enumerate() {
        let needs = op.needs.iter().chain(var.extra_needs.iter());
        if !needs.clone().all(|n| prof.satisfies(n)) {
            continue;
        }
        if !var.schedule.supported_by(prof) {
            continue;
        }
        return Ok(Path { variant: idx, schedule: var.schedule.clone() });
    }
    Err(NoPath::new(op, prof))   // 携带逐原子差异，供诊断
}
```

`Path` 只是"哪个变体 + 哪个调度"，不含任何代码生成信息。命中第一个可行变体即停——降级链的顺序就是人在 TOML 里写的优先级，不做自动打分（自动打分的可解释性成本高于收益）。

协商的第二个输入是**这次执行到的证据**（`cap::Proof`，`(原语, 实现名)` 的集合）：档案里的 `grade` 只回答
"作者说它具备"，`negotiate_with` 还要求这条供给出现在见证执行的结论里。于是同一个布尔被拆成三种状态——
参数不匹配、等级不够（`L1 < L2`）、以及"写了 `L2` 但这次没跑"。`negotiate`（按档案自述）保留为解释性口径：
§16 的"首选变体在声明上是否命中"与"本机执行不了的设备"（非 C 档案，`witness::Basis::Declared`）
需要它；任何声称"这台设备能跑"的路径都不许用。

### 8.2 诊断格式（可解释性的落点）

```
error: attn_decode 在 ascend(sm-) 上没有可行变体
  need  mma{m:16..64, n:8..128, k:16, dt:[f16,bf16], space:[FAST]}
  have  mma{m:16, n:16, k:16, dt:[f16], space:[FAST], impl:"cube", grade:L2}
         ↳ 不满足：n 16 < 64；dt 缺 bf16
  need  load_async{bytes<=512}
  have  —（该档案未声明 load_async；Cube/MTE 搬运为同步）
         ↳ 不满足：缺失
  降级链：
    ws_pingpong  ✗ （shuffle 缺失）
    pipe3        ✗ （load_async 缺失）
    smem_only    ✓ （可用，预计 mma_share 0.61，比目标低 0.21）
    naive        ✓
  建议：改用 smem_only（预计 -26% 峰值），或在 crates/prof/devices/ascend.toml 补 load_async 的 L2 见证
```

每条诊断必须包含三件事：**缺什么原子**、**供给方实际是什么**、**最近的可行解与代价**。这三件事是 CubeCL（只报编译错）、TileLang（只报 pass 失败）、CuTile（只报 Tile IR 版本不符）都给不出的。

「建议」行里的变体名与缺项清单必须来自**同一个变体**：建议的是缺得最少的那个可行解（通常是兜底，
所以同时打印档位与它的 `mma_share`），补完它就能用。把首选的变体名配上另一个变体的缺项，会让人
补完证之后发现仍然不可用——这条由 `cap::Unlock`（变体下标与缺项一起记）保证。

### 8.3 闭集外请求

请求 `prim = "tma_bulk"`、`layout = "interleaved"`、`schedule = "rs"` 等闭集外取值时，错误信息固定为：

```
error: `layout = "interleaved"` 不在闭集内
  闭集：row | col | swz<B,M,S> | pad(base,p)
  最近的等价表达：swz<2,4,3>(row(64,128)) 覆盖 4 交错
  若是新需求，需走本文 §6.4 的闭集变更流程（含预算重算）
```

---

## 9. 变体选择与 plan

`plan.rs` 负责"给定形状，给出 tile 与 roofline 估值"。P0 与 P3 交付的东西不同。

**P0（当前实现）**：

- 输入：`(op, prof, path, shape)`。
- tile：取 `[axes]` 笛卡尔积的**第一个**组合。轴取值的顺序就是优先级，与降级链同一口径：TOML 顺序即优先级，**不做自动打分**。
- 估值：`max(flops/peak, bytes/bw) + 未掩盖项`，只用于 `verify` 打印预测下界、`bench` 算达成率。
- 掩盖账：`Plan` 带 `(mma_share, stages, unmasked)`，计划行直接给出阈值结论（§12.2 判据 1），不另算一份。

P0 不做候选比较，原因是**没有可比的量**：档案里能推导占用率或分块收益的常数一个也没有（`[simt]` 只有组宽度），与 tile 相关的只有 `deq[GMAX]` 的栈占用，它不改 flops、不改流量。硬要给出"哪个 tile 更快"，只能在机制层编一个系数；这种假精确比不做更贵。因此 `gemv_i4` 的 `[axes]` 也只声明实现真正需要的那个取值。

**P3（未实现）**：

- 首选：已落盘的 `plan` 表（`plan/<vendor>-<id>-<op>.toml`），键是形状桶（`m` 按 2 的幂分桶，`n/k` 同理）。
- 缺失：用上面的 roofline 估算，在**预测最优的 ±1 档**内取 3 个候选做一次微基准，落盘。
- autotune 上限：每 (op, device) 最多 3 候选 × 5 次重复；这是与 CubeCL 全空间 autotune 的刻意差别——**搜索空间用闭集已经剪过一次，不需要再暴力剪**。
- 失效：`plan` 表带指纹（`schema_ver`、`tool_ver`、`emit`、`dtype` 位集），任一不符即视为缺失，重新估算。

P3 的完成标准：表里出现至少一台设备的两个形状桶，且 `bench` 的达成率高于 P0 的首个取值。在那之前，§12.3 的性能目标只能按 P0 的实现口径读。

---

## 10. lowering：五步，无 IR

### 10.1 模板语法（EBNF 摘要）

```ebnf
template  = { line } ;
line      = directive | stmt ;
directive = "#define" ident expr ;
stmt      = assign | call | for | comment | text ;
assign    = ident "=" expr ";" ;
call      = "@" prim "(" [ arg { "," arg } ] ")" ";" ;
expr      = ident | number | "{{" axis "}}" | call | layout | binop ;
layout    = ("row" | "col") "(" shape ")" | "swz<" n "," n "," n ">" "(" layout ")"
          | "pad" "(" layout "," n ")" ;
arg       = expr | dir | "along:" axis | "op:" ident | ident ":" expr ;
dir       = ("GM" | "FAST" | "REG") "->" ("GM" | "FAST" | "REG") ;
for       = "for" "(" ident "=" expr ";" expr ";" expr ")" block ;
comment   = "//" { any } ;
text      = { any } ;            (* 片段拼接时原样保留 *)
```

解析器只做三件事：切词、识别 `{{}}` 与 `@调用`、保留其余文本原样。词法约 120 行，替换约 200 行，位置信息只记行号。

### 10.2 五步的产物

| 步 | 输入 | 输出 | 失败原因 |
|---|---|---|---|
| ① expand | `op.axes` | `Vec<TileCombo>`（≤8） | 轴乘积 > 8 |
| ② negotiate | `op.needs` + `prof.provides` | `Path` | 无变体可行（§8.2 诊断） |
| ③ select | `(op, prof, shape)` | `Plan { variant, tile }` | 轴展开超过 8 / 档案无 peak 常数 |
| ④ substitute | 模板 + 片段 + 布局 | `Source { lang, text }` | 槽位缺失 / 布局不可重组 |
| ⑤ emit | `Source` + `prof.tool` | `Artifact { path, bytes, reused }` | 编译器报错（原文透出） |

### 10.3 缓存指纹

```rust
pub struct CacheKey {
    schema_ver: u16,          // cutile 自身版本
    emit: Emit,               // cuda|ptx|hip|opencl|spirv|metal|ascendc|c|wgsl
    tool: (String, String),   // ("nvcc", "13.3")
    flags_hash: u64,          // 档案 [compile].flags 的哈希
    arch: String,             // 指纹里的架构键
    template_hash: u64,       // 模板文本哈希
    fragments: Vec<(Prim, u64)>, // 参与拼接的片段名 + 内容哈希
    layouts: Vec<(String, String)>, // 声明式：row / col / swz<…> / pad(p)
    variant: VariantKey,      // 轴取值 + 调度槽位
    dtype: DTypeSet,
}
```

- 目录：`$CUTILE_CACHE/<key[0..2]>/<key>/`，内含 `source`（生成的源文本）与 `meta.txt`（键的逐项快照，一行 `字段=值`）；编译产物落在同一条目的 `bin/<hash>` 下。
- 产物：命中只跳过"重新拼装"是不够的——真正慢的是 `cc`。产物的身份是 `ArtifactKey` = 源文本键 + **实测工具链身份**（`命令@版本`，由 `toolchain::Toolchain::id` 从 `--version` 解析）+ **编译参数**（`ArtifactKey::params`：算子取 `CacheKey::flags_hash`，见证程序取实际那一行 `flags` 的指纹）+ **主函数**（形状、样本、重复次数都在 `main()` 里），因此"命中产物"与"重新编译"必然等价：换形状、换 `--repeat`、换输出精度、换 `-O3`/`-O0` 都换一份产物，而同一份产物在第二次 `verify`/`bench` 里连编译器都不需要（本机没有 `cc` 也照样跑得出已验证的结论）。参数这一项是能力的第三个维度，不是锦上添花：片段与工具链都在 `ArtifactKey` 里，唯独参数此前只进了**算子**的缓存键，见证程序按 `标记;程序文本` 定名，于是改一次 `[compile].flags` 会复用旧参数编出的可执行文件——「这条见证跑通过」是对着一份不再会被编出来的程序成立的。少 `params=` 的旧产物一律算"没有身份"（缓存可重建，重编一次远比把证据挂到未知参数上便宜）。编译在 `work.dir` 下的一次性沙箱里进行，产物拷进缓存后沙箱即删，`work.dir` 不残留第二份可执行文件。代价要说清：样本是编译进可执行文件的 C 数组，产物体积与工作集同阶（`gemv_i4` 的 4096×4096 一份约 10 MiB），所以这笔账由 GC 兜底——一次 `cc` 约 3 s，上限内每多留一份都是净赚。身份写在产物旁边（`bin/<hex>.key`，`ArtifactKey::describe` 原文）：没有编译器的机器不必先解析工具链就能把产物认回来——`build::execute` 的 `tool` 因此是 `Option`，按「源文本 + 编译参数 + 主函数」在条目里找候选，唯一一份就直接复用；多份候选（同一个源文本被不同编译器编过）则报错而不是替用户选，因为选哪一份都替 `bench` 决定了耗时的来历。
- 见证同路：见证程序用 `Cache::put_program` 按程序文本登记成普通条目（条目名只由标签与程序文本决定——编译器是产物身份的一部分，`meta.txt` 记 `program=` 而不是算子指纹），自检因此也只在第一次调用编译器，产物同样计入占用、同样被 LRU 回收——而不是缓存根下一个扫不到也删不掉的角落。
- 并发：`<key>/build.lock` 文件 + `O_EXCL`；另一个进程正在生成同一个 key 时等它落盘并复用其产物（上限 5 s），超时即报错（不静默重复编译）。`O_EXCL` 锁是**文件**而不是 flock：持有者被 `kill` 后没人能释放它，这个键会永久卡在"等待超时"上。因此取锁失败时先看锁是否陈旧（`cache::STALE_LOCK` = 60 s，一次 emit 是毫秒级、等锁上限只有 5 s）：陈旧就删掉重试，活着的锁一律不抢。
- 落盘：写临时文件 → `rename` 原子替换。临时名带进程号与序号：同一份 key 的两个并发构建各自落盘，先 `rename` 的不会把后来者的临时文件搬走（那会让后者拿到 `ENOENT`）。
- 残留：只在 `rename` 之前崩溃会留下没有 `source` 的目录（临时文件 / `meta.txt` / 锁）。它不是"别人的目录"而是**永远不会命中**的记录（`cache::State::Partial`）：旧实现把它排除在扫描之外，于是它既不计入占用、也永远不被回收。现在扫描按目录内容判状态，`cache gc` 把残留与 LRU 删除分两行报告。
- GC：按 LRU + 容量上限（默认 2 GiB，`cache::CACHE_LIMIT`），`cutile cache gc [--limit <字节>]` 手动触发。占用**递归**统计：产物在 `bin/` 里，只数顶层文件会让上限看着很空、实际删不掉真正占空间的东西。LRU 的"使用"由读命中（源文本或产物）更新 `meta.txt` 的时间戳来记——不记的话它就退化成"最近写入"，一份天天命中的缓存反而先被删。残留与上限无关、一律回收（它等到任何一次"超限"都不会发生）；可复用条目按 LRU 删到上限内。仍在用的（窗口内被动过，或还有活着的锁）一律跳过，跳过的条数在结果里如实计数。
- 可复现：`meta.txt` 记录完整键，`cutile doctor --why <key>` 用 `CacheKey::fields` 与它逐项比较，只列出真正不同的字段（含只在一侧出现的项：新版本加的字段与已删掉的字段同样是"为什么没命中"）。字段定义只有一份，`describe`/哈希/差异都从它派生。结论分四类（`CacheState`）：一致 / 不同 / 从未落盘 / **只有半截落盘**——最后一类必须先判，因为它的差异为空，按差异会把它说成"一致"。键里的 `dtype` 是模型的输出精度，所以换精度既换源文本也换产物——这正是"接入第二个精度时缓存键不会碰撞"的判据。

---

## 11. 正确性工程

### 11.1 双源对拍

每个算子在 `crates/op/src/<name>.rs` 有一份标量参考实现（不依赖任何设备、不依赖片段），由 `op::model::Model::reference` 暴露；`cutile/src/reference.rs` 只保留对拍口径——`max|got−want| / max|want|`，阈值按数据类型：整数精确、f16/bf16 ≤ 1e-3、f32 ≤ 1e-5。片段只与参考对拍，**片段之间不互相对拍**（否则一个错会传染成"两个都对"）。

输出类型也只有一份说法：`.k` 的可写输出是什么，回读就按什么解释。`op::sig::Signature::output_type`
把它读出来，`#[annotation::model]` 生成的校验把 `Model::dtype` 与它逐项比较（注解写 `f16` 而内核写
`float *y` 是测试红灯，而不是"阈值看起来不太对"），`harness::harness` 在拼任何 C 之前就拒掉非
`float` 的输出——否则 C 编译器只会吐一句 `incompatible pointer type`，不带算子名与缓冲名。本节开头
那条阈值因此不是注释里的约定，而是三处可执行判据的交点：注解声明的类型、签名的类型、回读的类型。

### 11.2 证据分级（继承旧工程，改为自动生成）

| 等级 | 含义 | 谁产生 | 能否用于发布 |
|---|---|---|---|
| L0 | 档案里声明了 | 人 | 否 |
| L1 | 片段能编译 | CI | 否 |
| L2 | 与参考对拍通过（真机或注入的假指纹 + 模拟器） | CI / 本地 | 是（但仍标记） |
| L3 | 真机性能达到 roofline 的 ≥ 70% | 真机 bench | 是 |

`capability` 台账（`<work.dir>/ledger/<vendor>.capability.md`）由 `cutile verify --emit-ledger` 生成，人不再手填。**缺设备即失败**：`verify` 遇到本机不支持的设备直接返回非零码并列出缺失项，不写"跳过"（这是旧工程 `clblast` 用例静默通过的直接修正）。

**通道是台账的一栏，不是日志**（`Channel`）：同一条 "L2" 可能是三种形态之一——`Parallel`（按 `[grid]` / `[chunk]`
发射并行网格，每线程一元素或每块一线程）、`Harness`（该形态尚无并行实现，回落单线程语义门）、`Reference`
（C 参考通道本身就是串行语义基准：串行是它的定义，不是缺口，并行形态由设备通道证）。三者混记会让
"该形态缺并行实现"这条缺口永久隐身，因此它随结论一起打印。判据不是自述：`Parallel` 只在生产态设备源
真的按声明发射并通过对拍时成立，`verify` 逐算子记录，报告里可以直接数出"36 个算子里有几条不是并行"。

### 11.3 见证

每条 `provides` 附一个可执行的见证（`<vendor>/witness.<ext>`）：一段只调用该原语的最小函数 + 档案里写明的期望值。

```toml
witness = { file = "cpu/witness.c", func = "witness_quant", expect = [-4.0, 3.5] }
```

统一 ABI 是 `int func(float *out)`：返回写入的元素个数（≤ 8），值按 f32 读回后与 `expect` 对拍。
`cutile verify` 会先把**档案里每一条**已声明的见证编成可执行文件并执行一遍（设备在本机时走
设备前端，否则走 host 自检），任何一条不过就整台设备失败；
通过后才跑形状矩阵（最小合法、多行非方形、生产形状）与标量参考对拍。

因此 `grade` 不是作者的自述，而是这次执行的结论。协商看的是**这次有没有证据**：自述模式
（`Basis::Declared`）只接受 `grade ≥ L2`，设备在本机时只接受本次跑通的 `(原语, 实现)`——
**未验证的能力视为不具备**，这是把旧工程 R5 风险（"声明有 TMA 实际没有"）前移成构建期问题的
关键一条。
"执行结论"这条要求对三条命令一致：`doctor` 与 `bench` 也先跑证据（见本节末"证据只算一次"），
只读档案的 `negotiate` 只服务于解释性报告。

`grade < L2` 的供给另有**本机自检**（`witness::self_check`，结论的 `scope = Host`）：见证的 ABI 是
host C，与片段语言无关，所以自检跳过档案的 `[compile].preamble` 与 `flags`，用 host 编译器把
`<vendor>/witness.<ext>` 编成可执行文件并逐条执行。它把“见证写没写、期望值算不算得对”变成
今天就能跑的判据（`nvidia` / `intel` / `ascend` / `metal` / `musa` 的 88 条 L0/L1 见证因此不再是纯文本），但**不**
证明设备具备该能力：自检不改写任何 `grade`。把 host 自检通过写成 `L2`，等于用主机替设备作证。

两个范围**互补**：设备在本机时 `run_all` 跑全部声明，自检一条不跑；设备不在本机时全部声明进
自检。此前的两个坑都在这里收口：只挑 `grade ≥ L2` 等于让没敢写 `L2` 的设备永远拿不到运行结论
（`grade` 是执行的结论，不是执行的前提），而自检只对非 C 设备调用又让 `cpu` 的 `L0`/`L1`
两头落空。现在 `Verified.witnesses`（设备执行）与 `Verified.self_checked`（自检）相加等于该设备
声明的全部见证：一份见证只执行一次，由能给出更强结论的范围执行。

**设备是一张 `const` 描述符表**：执行层只有一台统一设备，按档案的 `Emit` 在 `const SPECS` 表里分发；每一行登记这门语言的探测、程序生成与执行回读三个函数指针，没有 `match`、没有 `dyn`，也不按厂商各写一套 Rust 前端类型。片段本体来自 `prim`，本层只拼接、编译、起进程、回读。每条见证的写出上限以 const 泛型 `OUT` 传进程序生成函数，生成的程序里缓冲区尺寸是字面量。`emit = metal` 因此有了真正的设备执行：MSL →
`metallib` → host 驱动发射到 GPU → 回读；等级仍停在 `L1`，因为 `L2` 要的是**内核**跑通并对拍，
见证执行只完成其中的见证那一段。

`ascendc` 与 `metal` / `cuda` 走同一条路：后端在 `engine`（`AscendBackend`），可用性要求 `ccec` 与
`npu-smi` 都在，生成的是**单源** `.cpp`——`extern "C" __global__ __aicore__` 核函数 + AscendCL 驱动，
显存指针用 `GM_ADDR` / `__gm__`，见证函数加 `__aicore__` 后一条占一个 AI Core。编译参数全取自档案
`[compile].flags`（`-xcce` / `--cce-aicore-arch=dav-c220` 是"哪门设备语言、哪个核芯"的声明，本层不猜），
`-l` 类参数排在源文件之后（ccec 把它们转给链接器，顺序反了就是"undefined reference to aclInit"）。
`verify` 这边它与 `cuda` / `metal` 平级：`Lane::Ascend` 一行把它接进形状矩阵（此前 `ascendc` 在选通道时
就报"内核执行路径尚未接通"，后端注册了也走不到），`AscendBackend::harness` 把 `.k` 内核拼成**单块**
`<<<1, nullptr, stream>>>` 的 AscendC 程序——档案 preamble 里的 `_cutile_sum_f32` 一族同时转成
`__aicore__`（`@reduce` 的 `$func` 落在那里，不给它设备性就是"AI Core 函数调用了宿主函数"），回读与
CUDA 共用同一份位模式口径。没有并行内核时 `parallel` 如实 `Unsupported`，报告里的通道因此写"语义门"
而不是"并行"。仍未接通的是 `lower` / `benchmark_harness`（与 `Source::Pair` 的生产态入口）：P0 片段只到
标量 C 语义，把它编成 AscendC 只会得到"能跑但和 CPU 一样慢"的核函数，`DataCopy` / `Mmad` 的落点属 P3。
等级仍是 `L0`：这条路径还没在任何 910B 上跑过。

**第三种结论是语言门**（`LanguageGate`）：自检只证明“这段代码在 C 里成立”，而
`metal/witness.metal` 曾写成 `float *out`——host `cc` 绿灯、`xcrun metal` 编不过，那份见证从未
在自己声称的语言里存在过。于是设备不在本机时，见证还要交给**它自己语言的解析器**只解析一遍：
`cuda` → `clang -x cuda --cuda-host-only`（只 host 那一半，`__syncthreads()` 在 host 函数里必须
被拒），`opencl` → `clang -x cl -cl-std=CL3.0`（函数指针在 OpenCL C 中被明确禁止），`ascendc`
→ `ccec -xcce --cce-aicore-arch=dav-c220 -fsyntax-only`（`-xcce` 让 ccec 按设备语言解析这份翻译
单元，核芯名跟 `id = "910b"` 走——换了核芯，这一行也要跟着换）。每行声明的解析器**只认那一格**：
`clang` 行指本机的 clang 前端（先 `clang`、再名字里带 clang 的 `cc`），其余按 `[meta].tool` 的写法
原样探测；`ccec` 随 CANN 安装，宿主 `cc` 不认识上面这些开关，拿它去解析只会把「没装 CANN」报成一条
C 语法错误。缺席因此是 `Absent` 并点名缺哪个程序，而不是 `Rejected`。语法前端与执行前端同构：
`const GATES` 表一行一个 `GateSpec`（`emit` / `tool` / `check`），加一门语言就是加一行，没有独立的
前端类型，也没有按语言单态化的函数。门不执行任何一行，所以不改 `grade`；但
`Rejected` 会进 `Evidence::failure`——被自己的语言拒绝是证据缺陷，不是“本机没有这台设备”。

**证据只算一次，三条命令同一结论**：`witness::establish` 把设备执行与自检合成一份 `Evidence`
（依据 `Basis`、设备执行条数、自检条数、失败原因、`Proof`），`doctor` / `verify` / `bench` 都用它协商。
此前只有 `verify` 跑见证，`doctor` 与 `bench` 仍在读档案里的 `grade`——同一台设备可以同时"协商可行"
（`doctor`）与"见证不成立"（`verify`）。现在的口径是：`doctor` 先跑证据再协商，见证不成立时仍然交报告
（把原因写成"L2 但没有这次执行的见证"，并在 `[证据]` 段里给出具体失败），`bench` 则直接拒绝计时
（拿一台没有执行证据的设备计时，测的是一条未经确认的能力）。本机执行不了的设备（非 C 档案）只能按
档案自述协商，报告里的 `Basis::Declared` 就是这句话。

**证据的适用条件是「片段 + 工具链 + 参数」**：前两项分别由见证执行（P0-1）与实测工具链身份（P0-8）
落下，第三项由产物身份里新增的 `ArtifactKey::params` 落下——参数一变就换一份产物，见证随之重编重跑，
不会出现"用 `-O3` 的证据替 `-O0` 的代码生成作证"。`Evidence` 因此分别记设备执行的参数与自检的参数
（后者固定是 host 的一套），`doctor` 的 `[证据]` 段把两者并排写出；`verify` 的结论行与 `bench` 的产物行
也带上参数，读者不必从哈希反推"这份证据是哪次编译的"。

**一条签名一条见证**：`witness.{file, func}` 不能被两条签名共用。见证回答的是"这条签名写的
是什么算法"，共用会让"每条 provides 都有见证"退化成"每条都指向同一份证据"，执行通过的结论
也无法归属（`mmad` 与 `scalar_fma` 是两种算法）。`Profile::validate` 在装载期拒绝共用。

**见证矩阵**：闭集的 16 个原语在**每台设备**上都要有带见证的签名——五台加速器（`nvidia` 17
条、`intel` 17 条、`ascend` 20 条、`metal` 17 条、`musa` 17 条）加 `cpu` 14 条，共 102 条。加速器除 P0 用到的
9 个之外，补的是设备自称有的能力：张量核 `mma`、lane 交换/广播/前缀扫描、`exp2`/`rcp`，以及
`intel` 原先缺的 `atomic`；这些签名没有片段文本（P0 模板不调用），等级停在 `L0`——矩阵补的
是覆盖，不是等级。唯一到过 `L2` 的是 `musa`：那 17 条在 S4000 上真机执行过，其中 9 段片段对应的
算子与参考对拍通过。`cpu` 补的是它真能算的四条（标量分块 `mma`、`scan`、`exp2`、`rcp`）与原先
无见证的 `atomic`，并把它从 `L1` 降为 `L0`：单线程只有读改写，没有原子语义。

空白格必须能被数据解释：`shuffle`/`broadcast` 要求 `[simt].group > 1`，`cpu` 的
`[simt].group = 1` 意味着没有第二条 lane 可交换，声明它们只会造出自欺的证据。
`Profile::validate` 把这条做成装载期判据，且只对**写了** `[simt]` 的档案生效——`ascend` 没有
这一段（它的 lane 交换由向量单元 `vgather`/`vbroadcast` 承担），默认值 1 不代表没有并行。

**证据带工具链身份**：档案里的 `[meta].tool` / `tool_ver` 是**登记值**——人写下"这条证据是在哪个
工具链上取得的"，它和 `checked` 一样是出处，不是这台机器的事实。实测值由 `toolchain::probe`
（跑一次 `--version`，按 clang / gcc / nvcc 的共同形状取第一段“数字与点”）给出，
`Toolchain::id()` 是 `命令@版本`，`verify` 的报告与台账都打印两者：
`产物 3 份（复用 0，cc@21.0.0） | 工具链 cc@21.0.0（档案登记 clang-17，不一致）`。
两者不必相等（证据可以在别的机器上取得），所以不一致**不改等级**；但它必须是可观测的差异，
否则"换了一台机器、换了一个编译器"会静悄悄地发生。产物身份用的是实测值：升一次编译器就是
另一份代码生成，旧产物不能再冒充新编译器的产物；缓存键里则**没有**工具链——源文本与编译器
无关，换编译器只换产物，不换文本与它的指纹。

---

## 12. 性能方法论

### 12.1 roofline 是判定器，不是口号

档案必须提供可推导的常数，每个值都带自己的 `src`（推导式，而不是抄来的数字），
且 `src` 必须是**本层可求值**的表达式：只允许数值字面量、`+ - * /` 与括号。
装载期会把求值结果与 `value` 比对，不一致即失败；量纲与物理含义写在同一行的注释里，
数字因此只写一次。

```toml
[peak]
# 8 核 × 2.6 GHz × 8 条 f32 流水 × 2 FLOP/cycle
flops_f32 = { value = 332.8e9,  src = "8*2.6e9*8*2" }
# 2 通道 × 8 字节 × 2666 MHz
bw        = { value = 42.656e9, src = "2*8*2666e6" }
```

带变量名与单位的公式（`sm_count*64*2*clock`）无法在没有变量表的情况下核验，因此不被接受：
这类公式的可靠性只能靠人眼，与"每个值都可求值"的目标冲突。

`verify` 用 `max(flops/peak_flops, bytes/peak_bw) + 未掩盖的非 MMA 时间` 给出预测下界；`bench` 打印达成率。目标数字（见 §12.3）全部是**目标**，不是实测，验收必须附命令与机器。

`bench` 的计时口径是固定的：`CLOCK_MONOTONIC` 墙钟、warmup 不计入、每个样本只包一次内核调用
（亚微秒内核自动批量，样本按批量折算），报告 `min` 与 `p50`；达成率按 `p50` 计算（对单次抖动与
CPU 频率状态更稳），`min` 只表示最好情况。它同时打印输入工作集与档案的 `FAST` 容量，并把达成率
分成五类：`> 100%` → 口径无效（数据在缓存中）；工作集 ≤ `FAST` → 口径无效（测到的是缓存带宽，
与主存峰值不可比）；其余才按 70% 门槛判 `Reached` / `Below`；没有峰值则是 `Unknown`。口径无效的
数字不参与 L3 判定——这正是 §12.3 "任何未附口径的达成率视为未测量"的执行版本。五类与那个门槛
都收在 `bench::Performance`（`is_measured` / `needs_specialisation`），`bench` 每轮都打一行结论：
口径无效或档案没声明峰值时打印“未测量，不参与判定”，而不是什么都不说——只在失败时出声的判据，
读者会把沉默当成通过。

### 12.2 三个可测的构建期判据

1. **流水掩盖**：`(1 - mma_share) / stages ≤ 20%`，否则要求换 S3/S4。判据在**装载期**落到声明上（`op::sch::check_masking`）：只判降级链的首选变体，不达标即拒绝解析——首选都盖不住，只能加级数或承认这条路径不走 MMA（`mma_share = 0` 的变体没有可掩盖的对象，判据不适用）。兜底变体不判：它们存在的理由正是“设备缺原语时仍能跑”，用质量判据把它们判死等于取消显式降级；它们的账照样进报告——`Plan` 携带 `(mma_share, stages, unmasked)`，`doctor`/`bench` 的计划行直接给结论（`未掩盖 19%（阈值 20%，达标）` / `未掩盖 65%（阈值 20%，未达标：降级兜底）`），判定的数与打印的数是同一组。
2. **存储契约**：声明的搬运量（`bytes` 上界）必须落在它声明的每个存储层容量内，`swz` 粒度必须 ≤ 档案 bank 数，否则构建期失败（§6.2）。这条只看数据，因此对尚未有真机的设备也生效；判定先于缓存命中执行，报告里的容量与用量就是判定比较过的那两个数（`space::capacity` + `build::contract` 的返回值），不另算一份。
3. **变体收益单调**：降级链上 `mma_share` 必须单调不增；违反说明 `extra_needs` 写反了。

### 12.3 目标（非实测）

| 算子类 | 目标（相对各设备原生库的最佳实现） |
|---|---|
| GEMM（prefill） | 90–95% |
| 分页 attention（decode） | 85–93% |
| 量化 GEMV | ≥ 95%（带宽受限，贴 roofline） |
| SSM 扫描 | ≥ 90% |
| norm / elementwise | ≥ 95% |

这些数字的用途只有一个：`P0` 之后每加一台设备、一个算子，都用同一把尺子量。**任何未附 `bench` 输出与机器型号的达成率，视为未测量。**

---

## 13. 预算

### 13.1 P0（1 算子 / 3 设备 / 10 原语 / 2 调度）

| 模块 | 行数 | 说明 |
|---|---:|---|
| `prof` schema + 3 份档案 + 校验 | 240 | nvidia / intel / cpu |
| `cap` 协商 + 诊断 | 280 | 判定与需求差异；§8.2 的文本只在 `NoPath` 的 `Display` |
| `layout` 4 构造子 + 声明式（在 `op`） | 200 | swz 只用一种参数；正规形等槽位形状（P3） |
| `space` | 120 | 三级映射 + 容量取值/校验 + 对齐校验 |
| `prim` 片段 | 240 | 10 原语 × 3 设备 × 8 行 |
| `sch`（在 `op`） | 140 | S1 + S2 |
| `ops` 声明 + 模板 + 语义 + **解析** | 440 | `gemv_i4` 一份：声明/模板 + 代价/样本/参考 + `decl`/`tpl` |
| `substitute` + `emit` | 300 | 替换 + 成文（词法在 `op`） |
| `cache` | 140 | 指纹 + 锁 + 落盘 + LRU 回收 |
| `probe` | 120 | 三个后端 + 假指纹注入 |
| `plan` | 80 | 取声明的 tile + roofline 估算 |
| `ref` | 60 | 对拍口径（相对误差 + 阈值） |
| `test` | 180 | 见证 + 对拍用例（执行器在 `cutile`） |
| **合计** | **2,540** | 不含 `cutile`：它现在是一门独立成员，见下 |

> **审计并入内核（2026-11-12）**：`doctor` / `verify` / `bench` 的编排（形状矩阵、重复次数、工作目录命名、达成率口径）与见证执行器搬到 `crates/cutile/src`（包名 `cutile`，L3 的第二名成员），本 crate 同时持有机制与命令策略：`build`（⑤ 的指纹 / 锁 / 拼接 / 落盘）与 `backend`（编译执行）。原先 `audit = 100` 那行预算随代码一起迁出，P0 合计从 2,640 降到 2,540。实测（剔除内联测试的 6,455 行，含测试 9,403 行）：`device` 757、`backend` 635、`cache` 578、`verify` 545、`witness` 520、`cap` 518、`build` 473、`bench` 335、`doctor` 328、`builtin` 242、`gate` 222、`pipeline` 175、`execution` 170、`substitute` 141、`emit` 128、`plan` 125、`probe` 125、`lib` 109、`reference` 103、`space` 87、`ast` 61、`dirs` 58、`gc` 20。`cache` 从预算的 140 涨到 807，涨的是**存量管理**——扫描、状态判定、LRU 回收、残留清理、`--why` 的逐字段比较（P0-6 再加编译产物：`ArtifactKey`、`bin/<hash>`、递归占用统计、见证程序登记）：§10.3 一直许诺这些事，只是预算表当初没把它们算进去。这一页换到的是"第二次 `verify` 不调用编译器"——实测 4.6 s → 1.55 s，且产物命中时连 `cc` 是否在 `PATH` 上都不影响结论。拆分的判据不是行数，而是"改命令策略要不要动内核"：此前 `cutile` 一个文件 806 行、八倍于它自己的预算，说明预算表量错了对象——命令策略本来就不是内核的一部分。再修订（2026-11-13）：`../../.config/report` 这层目录已删除——它不产出“报告”（排版在驱动方），又让四条命令各写一遍同一条流水线。现在命令策略是 crate 根的 `bench` / `doctor` / `verify` / `execution`，共用一条 `pipeline`（`Run` / `Prepared`）；见证执行是 `witness` + `device` + `gate`，`Fragment` 与 §16 的两个门槛归 `emit`，缓存命令策略在 `gc`。

**预算与实测的差**（2026-11-14）：P0 预算 2,540 行，实测 6,455 行（剔除内联测试），2.5 倍。超出不是
"顺手写复杂了"，而是四块**证据采集基础设施**当初没进预算：`cache` 807（存量管理：产物身份、锁、
LRU 回收、残留清理）、`device` 791 + `backend` 615（三种前端的编译与执行，含 CUDA / Metal 的命令行
与结果解码）、`witness` 577（每条已声明的供给都要真跑一次）。它们支撑的正是"第二次 `verify`
不调用编译器"与"没跑通的供给不算具备"这两条判据；砍掉它们，§11.2 的证据分级就退回作者自述。
**因此这张表按"哪一块超了、超在哪条判据上"读，不要按合计读**；P3 的 9,520 仍是目标，不是承诺。

### 13.2 P3（12 算子 / 8 设备 / 16 原语 / 4 调度）

| 模块 | P0 | P3 | 增量说明 |
|---|---:|---:|---|
| `prof` | 240 | 600 | 8 份档案 + 证据台账生成 + 指纹家族 |
| `cap` | 280 | 520 | 参数化区间匹配 + 三档诊断 + 闭集变更检查 |
| `layout`（在 `op`） | 200 | 460 | 4 构造子全组合 + 正规形 + repack 计费 |
| `space` | 120 | 240 | 昇腾 L1 / Metal threadgroup / CPU 栈 |
| `prim` | 240 | 1,240 | 16 原语 × ≤8 设备，约 128 段 × 9.7 行 |
| `sch`（在 `op`） | 140 | 420 | S3 warp 专门化 + S4 ping-pong |
| `ops` | 440 | 3,300 | 12 算子 × 220 行（含各自的代价/样本/参考） |
| `substitute` + `emit` | 300 | 680 | 语法检查 + 位置信息 + 多 emit |
| `cache` | 140 | 300 | 跨版本迁移 + 更细的占用统计 |
| `probe` | 120 | 280 | 8 后端 + 系统版本约束（Metal 需 macOS 版本） |
| `plan` | 80 | 380 | 形状桶 + 3 候选微基准 + 失效 |
| `ref` | 60 | 160 | 对拍口径：阈值分档 + 随机形状 |
| `test` | 180 | 700 | 见证矩阵 + 假指纹回归 |
| `cutile`（在 `cutile`） | 100 | 240 | `doctor` / `verify` / `bench` 的结构化报告 + `--emit-ledger` 台账（排版在驱动方） |
| **合计** | **2,540 + 100**（`cutile` 在 `cutile`） | **9,520** | |

**这张表按 12 个算子写，实际闭集是 36 条**（`op::REGISTRY`）：`ops` 一行若按今天的规模重算就是 36 × 220 ≈ 7,920 行，
P3 合计随之按同比例上调（表里的 9,520 是 12 条口径，不是今天的事实）。差额不是「预算失效」，而是 D1（§18 的 12 个算子清单）在落地时被更早的需求顶开——
本表保留原口径是为了让每一笔增量都对得上，而不是拿新总数掩盖旧估算。

`ops` 一行的量计在预算里，但它现在不在 `cutile`：声明、模板与算子语义（代价/样本/标量参考）都住在
`crates/op`，连同**解析本身**（`op::decl`）与三张闭集（`op::layout` / `op::sch` / `op::tpl`）；
`cutile` 侧只剩装载与交叉校验（`builtin.rs`）、拼接与 emit，以及对拍口径。P3 增加算子
只增数据与一页语义，不动机制。
`prof` 同理：档案（`devices/*.toml`）住在 `crates/prof`，`cutile` 侧只剩 `prof.rs` 的 schema 与校验，
新增设备同样是"只加数据"。

### 13.3 为什么这 9,520 不是幻觉：三笔消元

| 消元 | 朴素做法 | cutile | 比值 |
|---|---|---|---|
| **笛卡尔积** | 12 算子 × 8 设备 × 191 行手写内核 = 18,240 行（191 取自旧工程 `cuda.c` 4,778 行 / 25 类内核；按今天的 22 算子则是 33,440 行） | 声明 2,640 + 片段 1,240 = 3,880 行 | **4.7×**（按 22 算子 8.6×） |
| **布局代数** | CubeCL `cubecl-tile/{layout,space,stage}` 16,666 行（唯一可核实的口径；TileLang 对应层只有 3,520 行，其余在 C++ 侧） | 460 行闭集 + 2 种合法动作（改模板或失败） | 按 CubeCL 口径 36× |
| **变体组合** | 每设备每形状一份手调参数 | 8 变体上限 + plan 表 + 3 候选搜索 | 搜索空间从 `2ⁿ` 降到 ≤ 24 |

并且：**新增一格 `(算子, 设备)` 的边际成本是 0 行**——只要骨架与原语都在闭集内。

---

## 14. 与四家参照 + 旧工程的量化对比

### 14.1 成本

| 指标 | cutile | CubeCL | CuTile Rust | TileLang | 旧工程 llm.u |
|---|---:|---:|---:|---:|---:|
| 可比层行数 | **9,520** | 301,766 | 75,732 | 317,510 | 43,936 |
| 相对 cutile | 1.0× | 31.7× | 8.0× | 33.4× | 4.6× |
| 覆盖设备 | **8** | 7 | 1 | 6（目录数，验证深度不一） | 7（真机台账） |
| 新增一台设备 | ≤ 300 行 | 3,134–11,925 行（实测每后端 src） | 不支持 | 需改 C++ codegen | 2–4k Rust + 0.2–4.8k C |
| 新增一个算子 | ≤ 270 行，0 行设备代码 | 100–300 行 + 各后端验证 | 50–150 行（仅 NVIDIA） | 30–80 行（限形状/后端） | 骨架 + 每后端约 191 行 C |
| 新增一格 (算子×设备) | **0 行** | 需该后端 codegen 支持 | 不适用 | 需 codegen 支持 | 约 191 行 |

### 14.2 质量属性

| 属性 | cutile | CubeCL | CuTile Rust | TileLang | llm.u |
|---|---|---|---|---|---|
| 布局是否可解释 | **闭集 + 构建期校验（bank 粒度 / 搬运量 vs 容量）** | IR pass，需读 IR | 框架接管 | pass 推导，留逃生舱 | 手写死 |
| 缺能力时行为 | **构建期失败 + 最近可行解** | 编译错 | Tile IR 版本错 | pass 失败 | 启动期拒绝（部分格子仅编译） |
| 并发安全 | 宿主 `partition` 借用检查 | 用户自管 | **借用检查（最强）** | 无 | 用户自管 |
| 能力声明可信度 | **见证 + L0–L3，未验证视为不具备** | 无矩阵 | 架构表硬编码 | 隐式 | 人填台账（有"仅编译"格） |
| 决策可追溯 | **`doctor --why` 全链** | autotune 日志 | 无 | IR dump | 日志 |

**结论口径**：cutile 在"每单位覆盖面积的代码量"和"决策可解释性"上领先；在**并发安全的证明强度**上不如 CuTile Rust，在**算子丰富度与生态**上不如 TileLang，在**IR 级优化与成熟度**上不如 CubeCL，在**真机覆盖广度**上尚不及旧工程。这四项是明确的、可接受的落后。

---

## 15. 风险

| # | 风险 | 后果 | 应对 |
|---|---|---|---|
| R1 | 片段写错但"很像对" | 所有用它的算子一起错 | §11 双源对拍 + 见证；片段之间不互相对拍 |
| R2 | 原语粒度选错（太粗→差异漏进模板；太细→片段爆炸） | 模板变复杂或片段数超预算 | P0 用 10 原语 × 3 设备量行数，> 400 行即判定粒度错 |
| R3 | 布局闭集不够用（重度 swizzle 的算子写不出） | 某些算子只能退到 naive | 先 4 构造子，超出显式报错；P2 再评估是否加第 5 个 |
| R4 | 昇腾 / 可移植 GPU 无真机 | 能力只能 L0/L1 | 能力不足视为不具备，协商自动跳过；台账标 L1 |
| R5 | `mma_share` 是估计值而非实测 | §12.2 判据失真 | `bench` 从实测反填 `mma_share`，估计值只用于首次选择 |
| R6 | 新增 emit 语言导致模板需要分叉 | 模板退化成每设备一份 | emit 只能改片段，不能改模板；P4 的 hip↔opencl 切换就是这条的验收 |
| R7 | 依赖引入失控（`syn`/`serde`/`toml`） | 供应链无治理 | 依赖白名单 + `cargo deny`；本轮先不加依赖，档案用手写解析验证可行性 |

---

## 16. P0：Go / No-Go

| 项 | 内容 |
|---|---|
| 算子 | `gemv_i4`（weight-only int4 解码 GEMV，带宽受限、无布局推导、好度量） |
| 设备 | NVIDIA（`emit = cuda`）／ Intel（`emit = opencl`）／ CPU（`emit = c`） |
| 原语 | 10 个：`dot` `load` `load_async` `commit_wait` `store` `bar` `cvt` `quant` `atomic` `reduce`（GEMV 不需要 tile MMA） |
| 调度 | S1 + S2 |
| 不搭的 | S3/S4、plan 落盘、多 emit 切换、昇腾/MUSA/可移植 GPU |

判据与失败含义：

| 结果 | 结论 |
|---|---|
| 片段 ≤ 400 行（内部目标 240）、roofline 达成率 ≥ 70%、协商 3/3 命中首档、`doctor` 能打印完整链 | **方案成立**，进 P1 |
| 片段 > 400 行 | 原语粒度错，先合并原语 |
| 某设备 < 70% | 该设备承认需要特化代码，把它从"配置化"降级为"手写 + 片段复用" |
| 协商命中率 < 3/3 | `needs` 声明粒度错（这条是 v0.3 保留下来最关键的判据） |
| 某设备的声明超出档案容量或 bank 数 | 闭集/档案取错，或声明写错：先看诊断里的 need/have 差异 |

「协商 3/3 命中首档」由 `verify` 结尾那段给出（`协商首档 1/3` + 逐设备一行）：分数与原因都在报告里
（`verify::Negotiation`：首档 / 降级到第 k 档 / 链上无可行变体（列出首选缺的原语）/ 无档案）。
这段与执行结果正交：协商只读声明与档案，**本机没有该设备时照样有答案**——缺设备是执行层面的结论
（退出码 3），不该把“这台设备本来怎么协商”一起吞掉。降到第 2 档不是失败，而是能力差的证据；
分数下降只在 `needs` 写得比设备能给的更宽时才算判据不达标。

「片段 ≤ 400 行」由 `emit::Fragment` 判定：`doctor` 的 `[emit]` 与 `verify` 的逐设备结论行都打印
`片段 53 行（上限 400：达标；内部目标 240：达标）`。上限管成败（超过说明原语粒度错，先合并原语），
内部目标只是余量、未达不算失败；两处共用同一句排版，打印的数与判定的数是同一个（`within_limit`
就是行数与 `FRAGMENT_LIMIT` 比），所以不会出现“打印 53 行却说超标”。

「性能」判据的代码模型是 roofline：`bench::Performance` 把“测到多少”与“这个数算不算数”一起给
（`is_measured` / `needs_specialisation`），门槛为 70%。`Emit::C` 继续走可缓存的 host harness；
Metal 的 `bench` 接受单段 `Pointwise` / 网格 `Loop`；CUDA/MUSA 另对 `gemv_i4` 使用 32-lane 协作式解包与
规约。路径直接编译并执行生产态 kernel，用 CUDA/MUSA event 或 Metal command-buffer GPU 时间戳采样
（每个样本批量派发 16 次），并将回读结果与标量参考对拍。设备产物目前不缓存；多段归约/扫描、专用 tile
与未覆盖的 lowering 明确拒绝，不能降级到串行 host 路径。`bench` 输出会标明工具链、是否缓存与对拍误差；
工作集装得进 FAST 时仍不把缓存带宽
冒充主存达成率。分母是档案声明峰值，不是“相对原生库的 85%”；仓库没有同口径原生库配对实现，85–95%
仍只是 §12.3 目标。判据只说明测量口径是否有效，不能证明“远超 Burn”；该结论还需要同机、同形状、
同精度、同统计口径的 Burn 实测。与 `docs/lld.md` 的 E7（带宽利用率 > 85%，Must）也不同：E7 要求校准后的
持续带宽、瓶颈分类与置信区间下界，两者不能互相顶替。

执行命令（P0 交付物：语法在 `llm-cli::cutile`，计算在 `cutile`，排版在 `llm-cli` 包的
`src/bin/cutile.rs`；`verify` 在缺设备时返回非零码）：

```bash
cargo run --bin cutile -- doctor   --op gemv_i4 --device auto
cargo run --bin cutile -- doctor   --op gemv_i4 --device cpu --why 83a3ecdb31cd1225
cargo run --bin cutile -- verify   --op gemv_i4 --devices cpu,intel,nvidia
cargo run --bin cutile -- bench    --op gemv_i4 --repeat 50   # 形状缺省取算子的生产形状
cargo run --bin cutile -- cache gc --limit 1073741824         # 缺省上限 2 GiB
```

---

## 17. 里程碑

| 阶段 | 交付 | 判据 |
|---|---|---|
| P0 | Go/No-Go 报告 | §16 五项判据 |
| P1 | `prof` + `cap` + `layout` + `prim`(3 设备) + `substitute`/`emit`，跑通 1 算子 | 同一份 `.k` 三设备可运行；与 `ref` 对拍通过；协商命中；布局校验零失败 |
| P2 | `ops` 扩到 6 + `plan` + autotune + S3 | 形状变化自动换变体，达成率差 < 5%；`doctor --why` 可解释每次选择 |
| P3 | 设备扩到 8（含昇腾/MUSA/可移植 GPU）+ `ops` 扩到 12 + S4 | 新增设备 ≤ 300 行；总量 ≤ 10,000；台账 L2 覆盖 ≥ 90% 的 `provides` |
| P4 | emit 切换验证（cuda↔ptx / hip↔opencl / wgsl↔spirv） | 换 emit 不改模板、不改算子声明 |

---

## 18. 待决事项

| # | 待决 | 影响 | 建议决策点 |
|---|---|---|---|
| D1 | ~~12 个算子的最终清单~~ **已按需求落地**：闭集今天 36 条声明（`op::REGISTRY` 是唯一事实源），§4 的 12 条候选是当时的预算口径 | `ops/` 一行按 36 × 220 ≈ 7,920 行重算（§13.2 的 9,520 是 12 条口径） | 再扩一条算子须先改 `REGISTRY` 与 `ops/<op>.{toml,k}`，并按 §13.2 重算预算 |
| D2 | 布局闭集取哪 4 个构造子（本文取 row/col/swz/pad） | 决定是否需要 `@repack` | P1 用 row + swz 两种跑通，P2 补 col/pad；片段能声明布局之后再补 `@repack` 计费 |
| D3 | `cutile` 是否直接依赖 `syn`/`serde`/`toml` | 影响供应链治理与 `substitute`/`emit` 行数 | P0 用手写解析 + 手写序列化，证明可行性后再评估 |
| D4 | 昇腾与可移植 GPU 的 `provides` 来源 | 只能 L0/L1 | 先标 `grade = L1`，协商自动降级；有真机再补见证 |
| D5 | 与 `llm` 入口二进制（`llm-cli` 包）的接口形态 | 决定内核是否需要跑在宿主进程里 | P1 前定：入口通过子进程执行 `cc` 产物，还是引入真机驱动并对外暴露 launch |
| D6 | 是否保留 `tess` 代号 | 命名一致性 | 建议弃用；本文起统一为 `cutile DSL` |

---

## 附录 A：行数口径与复现命令

所有数字均在本机 2026-11-11 实测，命令如下（`--exclude` 掉 `target`）：

```bash
# CubeCL：全部 crate 的 src
cd /Users/fei/code/cubecl && find crates -path "*/src/*" -name "*.rs" | xargs wc -l | tail -1
# → 301766

# CuTile Rust：核心四 crate src（cutile / compiler / ir / macro）
cd /Users/fei/code/cutile-rs && for c in cutile cutile-compiler cutile-ir cutile-macro; do \
  find $c/src -name "*.rs" | xargs wc -l | tail -1; done
# → 14068 / 41996 / 12119 / 7549，合计 75732；全部 crate src 合计 101509

# TileLang：表面与实现分开量
cd /Users/fei/code/tilelang && find tilelang/language -name "*.py" | xargs wc -l | tail -1   # → 13808
find tilelang -name "*.py" | xargs wc -l | tail -1                                            # → 94994
find src \( -name "*.cc" -o -name "*.h" -o -name "*.cpp" -o -name "*.cu" \) | xargs wc -l | tail -1  # → 222516

# 旧工程 llm.u：DSL 相关层 = config + device + kernel + C
cd /Users/fei/code/llm.u && for d in config device kernel; do find crates/$d/src -name "*.rs" | xargs wc -l | tail -1; done
wc -l crates/device/c/*                                                                       # → 9523
```

不参与比较的：CuTe DSL（本机无 CUTLASS/CuTe 源码，无法按同口径实测，故只作定性比较）、FlashAttention-4（论文与 kernel 发布，不做规模比较）。

---

## 附录 B：术语

| 术语 | 含义 |
|---|---|
| 算子（op） | 声明层的最小单位，对应 `crates/op/ops/<name>.toml` |
| 档案（archive） | 档案层的最小单位，对应 `crates/prof/devices/<name>.toml`：一台设备的能力与证据 |
| 骨架（skeleton） | 算子的结构类别，闭集 4 个；决定用哪个模板基座 |
| 原语（prim） | 模板唯一可调用的指令级操作，闭集 16 个 |
| 片段（fragment） | 某原语在某设备上的实现，`crates/prim/<vendor>/<prim>.<ext>` |
| 变体（variant） | 轴取值 + 调度 + 布局的组合，每算子 ≤ 8 |
| 能力签名 | `(prim, 形状区间, 位集, 存储集, 实现名, 证据等级)` |
| 见证（witness） | 证明某条 `provides` 为真的一段最小 kernel + 期望值 |
| 正规形 | 布局的五元组 `(order, shape, stride, swizzle, pad)`；P0 不实现（构建期没有槽位形状） |
| 路径（path） | 协商结果：可行的变体 + 调度 |
| 台账 | 由 `verify` 自动生成的设备能力证据表 |

---

## 附录 C：参照与引用

| 主题 | 来源 |
|---|---|
| CubeCL 架构与后端 | `/Users/fei/code/cubecl`（本地，Apache-2.0 / MIT 双许可） |
| CuTile Rust（tile 值 + 所有权 + Tile IR） | `/Users/fei/code/cutile-rs`（Apache-2.0，NVIDIA），README 中的 Tile IR 13.2+/sm_80+ 约束表 |
| CuTe DSL 布局代数与原子 | CUTLASS 4.x 文档（`Layout`/`Swizzle`/`ComposedLayout`、`TiledCopy`/`TiledMMA`） |
| TileLang 的 tile 级 API 与布局逃生舱 | `/Users/fei/code/tilelang` v0.1.15（MIT），`tilelang/language/annotations.py` 的 `annotate_layout`/`use_swizzle` |
| FA-4 的流水与重叠结论 | 《FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling》，论文口径，非本项目实测 |
| 配置驱动 + 能力协商的工程先例 | `/Users/fei/code/llm.u`：`config/device.toml`、`crates/config/src/negotiate.rs`、`docs/capability-matrix.md`、`docs/device-capability-config.md` |

**许可与归属**：本文只借鉴上述实现的机制与公开结论，不复制其源码。若后续片段实现参考了某一具体实现（例如 CubeCL 的 wgmma 序列），必须在 `crates/prim/<vendor>/NOTICE` 中保留原许可与出处；本项目自身维持 `MIT OR Apache-2.0` 双许可。
