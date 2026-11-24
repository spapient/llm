# 0056：prof 真内核与 cutile DSL/IR——10 分标尺裁决、可闭合切片与冻结边界

- 状态：已采纳（已闭合 `einsum` 减码拆分切片 + cutile typed IR 表驱动一元闭集与发射模块拆分；
  typed IR 元素档贯通、跨算子融合、设备执行与 CUDA 真机证据列开放项 / PARTIAL，不以未测能力签发满分）
- 日期：2026-11-24
- 相关：[0017](0017-kernel-ast-and-ir-freeze.md)（单 kernel AST 与运行时张量 IR 冻结）、
  [0054](0054-operator-coverage-spp.md)（骨架元素档 `K` 分层与贯通硬点）、
  [0005](0005-annotations-as-contracts.md)（注解即契约）、
  [`docs/COMPARISON-llm-llmu-burn.md`](../COMPARISON-llm-llmu-burn.md) §16.3 维度 17、
  [`docs/GATE-RUBRIC-2026-11-24.md`](../GATE-RUBRIC-2026-11-24.md) 八维标尺、
  [`docs/PROMPT-S++-AUDIT.md`](../PROMPT-S++-AUDIT.md) N3 / N7

## 背景

用户目标：真内核落在 `crates/prof`，`crates/cutile` 的 DSL / IR 做到 10 分、对标并超越 Burn，
要求泛型、注解、配置化、优先路径，能力 / 性能 / 质量三提升，代码与代码文件大量减少。

对照基线（`COMPARISON` §16.3 维度 17「内核 DSL 扩展」）：**llm 6 / llm.u 2 / Burn 9**。
Burn 的 9 分来自 CubeCL：设备无关内核语言 + cubek IR，JIT lowering 到 CUDA/HIP/Metal/Vulkan/
WebGPU/wgpu/CPU，配算子融合与 autotune；`burn-cubecl` 25,240 行、`burn-cubecl-fusion` 14,771 行，
后端壳层极薄（`burn-cpu` 61、`burn-cuda` 58），重内核在 IR 侧。

本仓两条已签发约束决定了本轮不能照搬该路线：

1. **ADR 0017 冻结 L2**：跨算子融合、运行时 JIT、动态形状重写不在本批；解冻须同时满足单 kernel
   真机对拍达 L2、性能达 roofline 70%（L3）、roofline 证据证明存在跨算子融合收益，且先过新 ADR。
2. **ADR 0054 §110 贯通硬点**：完整元素档 `K` 贯通卡在三处——① `Model.sample`/`reference` 是裸
   `fn` 指针槽（erased 装配槽，存不下泛型结果）；② `Values` 档位闭集仅 `F32(Vec<f32>)`/
   `U8(Vec<u8>)`；③ 对拍下游（`need.rs`/`schema.rs`/cutile `reference.rs`）按 f32 归一，阈值与
   桥接宽度未按档分档。一次贯通牵动整条取值通道与 52 个算子 reference、verify 阈值与桥接宽度，
   属跨批次大重构。

## 取证基线（当前 revision 回链）

### prof（词汇 / 契约 / 档案 / 主机微核）

- 词汇层泛型与闭集基础在位：`atom.rs` 闭集宏 `vocabulary!`/`inherent_names!`、编译期位集
  `Set<E: Closed, const WORDS>`、`Prim` 16 原语 + `AttrKey` 闭集、`Span` 区间代数（exact /
  at_most / at_least / range / covers）、`KernelKey` 配置登记键（prim / dtype / space / lanes /
  arch / block / impl_name）。
- 类型单一事实源：`atom::DType` 直接复用 L1 `dtype::DType`（U4/U8/U16/U32/I4/I8/I32/F16/BF16/F32
  等 23 变体 + MX 族），prof 不另造类型枚举。
- 优先路径已配置驱动：`cap.rs` 协商按 TOML 降级链序命中第一个可行变体即停（`Path::is_head` 标识
  首选），不做自动打分；`plan.rs` 按声明顺序取 tile，roofline 只给估值。
- 主机真内核隔离子岛 `native/`：f32 NEON/AVX2 8×8 `gemm8x8`、i8 dotprod 4×4 `sdot`、Apple SME
  16×64 ZA 外积；`gemm_negotiate` 经编译期配置表 `GEMM_KERNELS` 择优，arch 门 / 特性探测 / unsafe
  全收在岛内。
- 存量越线：`einsum.rs` 1054 行，超单文件 600 硬线（N3 一票否决项）。

### cutile（设备无关 kernel DSL 与编译前端）

- 旧模板链路五步：expand → negotiate → select → substitute → emit + cache；中间产物都是可打印、
  可哈希的数据结构。
- typed IR（`kernel.rs` / `validate.rs` / `fuse.rs` / `pass.rs`）现状边界：
  - 类型档：`ScalarType { F32, Usize }`；缓冲元素 `buf_ok` 仅 F32。
  - 表达式族：`F32` 字面量（IEEE bits）/ `Scalar` / `Load`（隐式循环索引）/ `Neg` /
    `Unary(UnaryFn)`（表驱动十族：`Abs`/`Sqrt`/`Exp`/`Ln`/`Sin`/`Cos`/`Ceil`/`Floor`/
    `Round`/`Trunc`）/ `Binary(+,-,*,/)` / `Fma` / `Min` / `Max`；无比较 / 选择 / 控制流，
    无任意下标。
  - 一元闭集：`UnaryFn::c_name`（C `<math.h>` 发射名）与 `UnaryFn::apply`（Rust 主机 f32 求值）
    各登记一行；校验闸、遍历、字面量折叠、解释执行、双目标发射全部从表取，不为单变体各写 match 臂。
  - 写回：逐点 `Map(Expr)`、归约 `Reduce { at, value }`。
  - 校验六道闸：名字 → 形参 → 生成名 → extent → 输出 → 表达式；类型一致闸按 f32 判定。
  - pass 框架：`Simplify`（代数定律 + 常量折叠）/ `EliminateDeadParams` / `FuseFma`，扫到不动点，
    每 pass 后校验，轮次上限取初始 cost，震荡报错；`standard`（位精确）与 `contracted`（opt-in FMA
    收缩，改舍入）两通道。
  - 双目标发射：可移植 C（配对长度 + 守卫 + 返回 0）与设备 `.k` 正文（无长度配对 / 守卫，交 vendor
    骨架）；目标分叉收敛进 `Target` 枚举，正文只写一遍。
  - 融合：仅逐点-逐点；归约链、缺失链接、可写链接、extent 不一致一律拒绝。
- 前端：`#[dsl]` 过程宏把一个 `for idx in 0..extent` 循环 + 一次写回翻译成 `Kernel`。当前显式拒绝
  泛型函数（`generics.params` 非空即拒），参数档仅 `f32` / `usize` / `&[f32]` / `&mut [f32]`，
  注解不接配置参数（`#[dsl]` takes no arguments）。
- **未接设备执行**：typed IR 尚未替换旧模板链路；`Image` 与布局推导未接；`#[dsl]` 无生产挂载点，
  仅 annotation 测试 fixture。

## 10 分标尺拆解（套 GATE-RUBRIC 八维，逐项给当前档位）

标尺（`COMPARISON` §16.1）：9–10 成熟完备生产可用；7–8 主体完整；5–6 端口 / 骨架；3–4 最小可用；
1–2 仅声明；0 缺失。状态：● 可跑 / ◐ 骨架部分 / ○ 缺失。

| 维度 | 10 分判据 | 当前证据 | 状态 | 档位 | 待校准 / 前置动作 |
| --- | --- | --- | --- | --- | --- |
| A IR 类型覆盖面 | 元素档对齐 L1 `DType` 闭集，标量 / 缓冲 / 累加档可表达 | IR 仅 F32/Usize；基座 23 变体在 prof/dtype 侧齐备，未下放到 IR | ◐ | 5 | 元素档贯通，先解 ADR 0054 三处硬点（见开放项 G1） |
| B 表达式 / 控制流 | 比较 / 选择 / 数学函数 / 取整 / 必要控制流成族，逐族有正反 oracle | 算术 / FMA / min/max/neg；一元数学 + 取整十族表驱动（Abs/Sqrt/Exp/Ln/Sin/Cos/Ceil/Floor/Round/Trunc），含发射断言、校验反向用例与位级族逐位对拍；比较 / 选择 / 控制流仍缺 | ◐ | 6 | 比较 / 选择出布尔值牵动类型系统，另案；超越族容差对拍见 G6 |
| C 配置化 / 闭集安全 | 档位 / 通道 / 优先级全声明化，拼错键构建期拒绝 | prof 词汇 / 链序 / tile 选择已配置驱动；IR pipeline 通道两档在位但 emit 固定走 standard；`#[dsl]` 零配置参数 | ●/◐ | 7 | IR 通道选择与 `#[dsl]` 配置位待有真实消费者后接入（见 G4，不造无消费者抽象） |
| D 泛型 / 声明化密度 | 同构 ≥3 份全部泛型 / 宏 / 表驱动 | pass / Target / 词汇表 / `UnaryFn` 闭集均表驱动；新增一元族只登记一行（C 名 + 主机求值），校验/遍历/折叠/解释/发射从表取 | ● | 8 | 维持；新增同构先判体量再抽（N7 门槛 ≥3 份 / ≥5 行 / ≥80 token） |
| E 规模 / 密度 | 无超线文件；公开项密度受控 | prof `einsum/` 拆分后最大 `device.rs` 570；cutile 发射侧拆入 `printer.rs`（242），`kernel.rs` 431，最大 `cap.rs` 599，均 <600 | ● | 8 | 拆分后 jscpd 零克隆；净删生产行随后续批次实测 |
| F IR / 发射器性能 | 生成与优化有耗时基线与回归门；热点复杂度可证 | pass 为不动点线性扫描；无 micro-bench 基线；非热路径 | ○ | 5 | 先取单 kernel 优化 / 发射耗时基线，再决定是否上区间归属 |
| G 设备执行 / 证据完备 | 多 runtime lowering 可跑，真机对拍 + roofline 达标 | typed IR 未接设备执行；本机仅 macOS Metal 逐点族真机对拍；win（RTX A3000）CUDA 硬件 + nvcc13.3 + MSVC cl 工具链闭环探针已跑通，但仓内 CUDA 后端对拍套件尚未同步到 win 运行；Ascend/ROCm/Intel/MUSA 无证据 | ○ | 3 | 见开放项 G2/G3；仓内后端真机对拍跑通前不签 CUDA 执行结论 |
| H 测试分层 | 每族反向注入用例；夹具复用；测试文件 ≤800 | IR 有 ir/pass/exec 三类测试；一元族独立 `unary_tests.rs`（发射断言 + 非 f32 反向用例），位级族进 `ir_exec` 逐位对拍；`ir_tests.rs` 760 行线内 | ◐ | 7 | 逐表达式族补「非法节点 → 校验红」反向用例 |

**等权参考分（算术平均，仅基线锚点，不作交付结论）**：(5+6+7+8+8+5+3+7)/8 ≈ **6.1**，
基线维度 17 的 llm 6。Burn 9 分的差额主要在 G（多 runtime 真机执行 + 融合 + autotune）与
A（类型覆盖面），这两块均受 ADR 0017 / 0054 冻结，不在本批闭合。

## 本轮落地（证据闭合、可逆、不改既有语义）

### S1 — `einsum.rs` 按职责轴拆分回 600 硬线内（质量 / 减码）

旧单文件 1054 行（N3 超线）。按职责轴拆成 `crates/prof/src/einsum/` 八个子模块，入口
`einsum/mod.rs` 聚合重导出，对外路径 `prof::einsum::*` 逐字不变：

| 子模块 | 职责 |
| --- | --- |
| `bits.rs` | `Bits` 位宽闭集 + 名字 / 位宽表 + `Closed` 实现 |
| `cost.rs` | `Cost`（flops / read / write）分量与派生合计 |
| `expr.rs` | `Expr` 显式模式文法解析 + 输出 / 重复字母校验；`is_label`/`NOT_LABEL` 提升 `pub(crate)` 共享 |
| `extents.rs` | `Extents` 轴长度表 + 分组轴来历 + 整除校验 |
| `operand.rs` | `Access` / `OperandSpec` / `Operand` + f32 标准操作数常量 |
| `axis.rs` | `AxisRole` / `Axis` / `Axes` 轴分解与 flops 口径；共享 `multiply`/`overflow` 溢出算术 |
| `formula.rs` | `Formula` 配对、轴分解、流量推导与同名 / 单出口 / 写回输出轴校验 |
| `cache.rs` | `FormulaCache` OnceLock 表达式缓存 |

- 拆分后最大文件 `formula.rs` 262 行、`expr.rs` 164、`extents.rs` 157，全部 <600。
- 语义不变：`prof` lib 164 单测通过；对外引用（`nn` 骨架、`tpl.rs`、`model.rs` 再导出、
  `tests/spec/{einsum,cost}_tests.rs`）路径不变。
- `Axis`/`Axes` 构造字段改 `pub(crate)` 供 `formula.rs` 同 crate 子模块构造，不向 crate 外泄漏。

### S2 — cutile typed IR 表驱动一元闭集 + 发射模块拆分（能力 / 减码 / 质量）

- **闭集表驱动**：原独立 `Expr::Abs` 变体改为 `Expr::Unary { op: UnaryFn, child }`，十族
  `Abs/Sqrt/Exp/Ln/Sin/Cos/Ceil/Floor/Round/Trunc`。`UnaryFn::c_name`（C `<math.h>` 发射名）与
  `UnaryFn::apply`（Rust 主机 f32 求值）各只登记一行；校验闸、`children` 遍历、字面量折叠、
  解释执行、双目标发射全部从这张表取，不为单变体各写一份 match 臂。非有限结果不折叠，沿用既有
  `finite` 闸。
- **发射侧职责拆分**：原混在 `kernel.rs` 的 `Target` 枚举与全部 `write_*` 渲染函数拆入
  `printer.rs`（242 行）；`kernel.rs`（431 行）只留类型定义与 `Kernel` 骨架，`emit_c`/`emit_k`
  委托 `printer::render`。`ValueType::c_type`/`Arith::c_symbol`/`buf_length_name` 提 `pub(crate)`
  供跨模块调用，不向 crate 外泄漏。
- **前端闭集同步**：`#[dsl]` 方法表 `METHODS` 12 项（10 一元 + `min`/`max` 二元）；未登记的方法名
  在展开期打回，一元落 `Expr::Unary`，`min`/`max` 落二元 `Min`/`Max`。
- **测试 oracle**：`unary_tests.rs` 表驱动逐族断言发射对应 math 库调用 + `<math.h>`，并补「子项套
  `usize` 标量 → 校验红」反向用例；`ir_exec_tests.rs` 对位级五族（Abs/Ceil/Floor/Round/Trunc）走
  `cc -O2 -ffp-contract=off` 编译 emit_c 输出逐位对拍。超越五族（Sqrt/Exp/Ln/Sin/Cos）存在 libm
  ulp 差，**不进逐位通道**，容差对拍另案（PARTIAL）。
- **命名 / 行数合规**：枚举去禁用 `Op` 后缀取 `UnaryFn`；发射模块取单词名 `printer.rs`；一元专属测试
  拆到独立 `unary_tests.rs`，`ir_tests.rs` 回 760 行（测试门 800 线内）。

## 本轮不做的事（冻结边界与反模式）

- **不打通元素档 IR**：`ScalarType` 维持 F32/Usize。整数 / 半精度档缺下游取值通道与对拍归一，
  单独下放类型枚举会造出半吊子；贯通须先解 ADR 0054 三处硬点（G1）。
- **不加无消费者的 `#[dsl]` 配置位 / IR 通道开关**：`#[dsl]` 当前无生产挂载点。配置位在没有真实
  消费者前接入，属 `PROMPT-S++-AUDIT.md` §9 反模式（平铺冒充高内聚 / 提前抽象）；待 IR 替换模板
  链路、前端有生产核后再按 N7 体量判定。
- **不解冻跨算子融合 / JIT / 动态形状**：遵守 ADR 0017；不新增跨 kernel 重写模块。
- **不抽统一 Finding / 元素泛型层**：字段与语义不同，强抽丢信息；tools L1 不反向依赖 L2 prof 宏。
- **不放宽任何阈值**：阈值只收紧；新增档位必须进闭集并经构建期校验。

## 开放项（PARTIAL，前置条件满足后另案）

- **G1 元素档贯通（跨批次大重构）**：依次解 ① sample/reference 泛型槽、② `Values` 档位闭集扩档、
  ③ 对拍下游按档分阈值与桥接宽度；IR `ScalarType`/`Expr`/校验闸/发射拼写/前端参数档同批下放到
  i32/u32/i8/u8/f16/bf16。前置：ADR 0054 修订或新 ADR；每档附主机参考对拍，设备发射另验。
- **G2 设备执行闭环**：typed IR 替换旧模板链路，接 `Image` 与布局推导；`.k` 正文经 `shard::ir`
  桥进 vendor `Codegen::lower` 后，IR 核才能宣称设备执行。
- **G3 多 runtime 真机对拍 + roofline**：CUDA/Ascend/ROCm/Intel/MUSA 真机证据；性能达 roofline
  70%（L3）是 ADR 0017 解冻 L2 的条件之一。CUDA 第一站 = win（`win`@192.168.0.65，A3000）：
  先把当前仓同步到该机（现有副本均为旧代次快照、非 git 仓），再跑 `llm-device` 的 CUDA 生成 / 编译 /
  回读对拍，核 `nvcc_ready()` 在 win 判真；Ascend/ROCm/Intel/MUSA 设备通道本轮未探测。
- **G4 注解 / 通道配置化**：IR standard/contracted 通道选择、`#[dsl(contracted)]` 闭集配置位，
  待 G2 让前端有生产消费者后接入；走声明表，不写第二套按名分派。
- **G5 autotune**：变体择优从链序首命中升级到实测计时选优；依赖 G3 真机计时通道。
- **G6 表达式族扩充**：本轮已闭合一元数学 + 取整十族（见 S2）。剩余：比较 / 选择出布尔值（牵动 IR
  类型系统，须另案定布尔档）；`sign`/`recip`/`log1p`/`exp2`/`rsqrt` 等未纳入本轮闭集；超越五族的
  容差对拍通道（非逐位）待补。

## 未知与未测

- **CUDA / Windows 硬件与工具链通道已实测可达（confirmed），仓内后端对拍仍未在 win 跑过（PARTIAL）**：
  前一轮把"执行环境注册表内无 Windows 实例"当成"无 Windows 通道"，漏扫了 `~/.ssh/config` 里的局域网主机
  `win → 192.168.0.65`（用户 `fei`，密钥 `id_ed25519_mac_local`）。本轮经 SSH 实测该机：
  - GPU：NVIDIA RTX A3000 12GB，驱动 621.14，UMD CUDA 13.5；`nvidia-smi` 正常。
  - 工具链：`nvcc` 13.3 V13.3.33（`CUDA_PATH` 在位、nvcc 在 PATH）；MSVC `cl.exe` 14.51.36231
    （VS18 BuildTools，vswhere 定位，与仓内 `platform::find_cl` 同一路径）；Rust `rustc`/`cargo`
    1.100.0-nightly。
  - 闭环探针：一条 8 线程平方 kernel 经 `nvcc -ccbin <cl.exe>` 编译并在 A3000 上运行，回读
    `0 1 4 9 16 25 36 49`、`cudaGetLastError = 0`；编译执行链路打通（探针临时目录已清理）。
- **仓内 CUDA 套件尚未在 win 跑（PARTIAL，缺口是代码同步，不是无设备）**：win 上现有源码副本
  （`llm-cuda`/`llm-verify`/`llm_clean`/`llm_b`）均为旧代次快照——无当前 `device` crate，cutile 无
  typed IR（`kernel/expr/pass/validate/printer`），且都不是 git 仓；当前仓在本机 MBP。把当前仓同步到
  win 后，方可 `cargo test -p llm-device` 跑仓内 CUDA 生成 / 编译 / 回读对拍，并核 `nvcc_ready()`
  （`detect_nvcc` + `find_cl`）在 win 判真。linux 云主机仍无 GPU / `nvcc` / `cargo` / 仓库副本，不能
  作为 CUDA 通道。
- **性能数字待校准**：IR 优化 / 发射无耗时基线；roofline 达标率、跨算子融合收益均未实测，不在落地
  前写提速承诺。
- **超越函数精度口径未定档**：Sqrt/Exp/Ln/Sin/Cos 的 Rust `f32` 方法与 C `<math.h>` 库实现存在 ulp
  级差异，本轮只做发射文本断言，未建立容差对拍通道；逐位通道仅覆盖位级五族（Abs/Ceil/Floor/Round/Trunc）。
- **净减生产行 / 文件数**：本轮为巨石拆分（1 文件 → 8 子模块），生产代码路径未增；净删行数目标随后续批次实测，不在此写未测降幅。einsum 新文件 jscpd 零克隆已实测。
- Burn `burn-cubecl` 内部实现深度未逐行读；对照以 `burn-cubecl`/`burn-cubecl-fusion` 模块清单与
  CubeCL registry 0.12.0-pre.2 源码目录为准。

## 门禁状态（本批实测）

- `llm-prof`：lib **164** + 集成 **56** + doctest **3** passed（1 ignored）；`llm-nn` **528** passed
  （4 ignored）、`cutile` **142** libtest + **2** doctest passed、`llm-annotation` **48 + 6** passed
  （2 ignored doctest）。下游 `nn`/`cutile` 经 `cargo check` 编译通过，einsum 拆分与一元闭集路径无回归。
- `cargo clippy -p cutile -p llm-annotation --all-targets -- -D warnings` 零 error/warning；
  `cargo fmt --check -p cutile -p llm-annotation` 干净。
- 静态门：einsum 八个子模块与 cutile 本批文件全部过行数 / CCN / NLOC 闸；`every_source_passes_the_static_gate`
  的红项仅为 `train/session.rs`、`train/kernel.rs`、`train/backprop.rs`、`train/geometry.rs` 四项并发
  在途存量越线，非本批引入，本批不越界拆分适配层。
- jscpd（threshold 0）：本批 cutile `printer.rs`/`unary_tests.rs` 与 annotation 新代码 **0** 克隆；全仓
  5 处存量克隆（`nn/tests` 2、`shard/tests` 2、`weights/src/closed.rs` 1），重复行 0.03%，均不在本批
  认领边界内。
- 拆分口径：旧 `einsum.rs` 1054 行（>600 一票否决）→ 八子模块合计 1096 行，最大 `formula.rs` 263；
  多出的 42 行为模块文档与 `pub(crate)` 可见性样板，生产代码路径未增。本轮减码性质是**巨石拆分回门内
  + 职责轴内聚**，不是净删生产行；净删行数 / 文件数目标随后续批次实测，不在此写未测降幅。

## 回滚

- `einsum` 拆分纯为模块搬迁，无状态、无持久数据、无在途写路径；回滚 = 还原旧 `einsum.rs` 单文件并
  删除 `einsum/` 目录，编译期路径由 `pub use` 重导出兜底，下游无需改动。
- S2 一元闭集与发射拆分同为纯类型 / 模块搬迁：回滚 = `Expr::Unary` 退回独立 `Abs` 变体、删
  `printer.rs`/`unary_tests.rs` 并把渲染函数并回 `kernel.rs`，前端 `METHODS` 回退到 3 项；无持久状态、
  无设备写路径。
- 本批未触碰并发在途文件（shard plan/schedule、train session/backprop、tensor、weights gguf 等）；
  提交前按 `git status --short` 划边界，只认领 cutile/annotation 本批文件与本 ADR。
