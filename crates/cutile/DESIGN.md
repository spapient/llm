# cutile DSL —— 设备无关的替换式 kernel DSL

> 版本 v1.0 · 2026-11-11 · 状态：待评审（替代 v0.3 的 `tess` 草案）
> 落地位置：`crates/cutile`；本文只定义 cutile，不涉及运行时之外的训练/服务层
> 参照实现与行数口径见附录 A，所有数字可在本机复现；`tess` 仅作为历史代号保留

---

## 0. 一句话

**算子只声明，设备只建档，指令只写片段：`(算子 × 设备 × 变体)` 的笛卡尔积由一张协商函数消掉，而不是由几万行手写内核消掉。**

三条边界先说清，后面所有取舍都由它们推导出来：

1. **不做通用编译器。** 没有中间表示，没有 pass 流水线，没有自动布局推导。只有一条"展开 → 协商 → 替换 → 发出 → 缓存"的 lowering，每一步都能打印出来给人看。
2. **不做通用算子。** 算子闭集 12 个、原语闭集 16 个、布局闭集 4 个、调度闭集 4 个、设备档案 ≤ 8 份。申请闭集之外的组合，显式报错并给出最近可行解，**绝不静默变慢**。
3. **设备无关是硬约束，不是卖点。** 同一份算子声明与同一份模板，必须能在 NVIDIA / AMD / Intel / Apple / 昇腾 / MUSA / CPU / 可移植 GPU 上产出可运行代码；任何"只在某一家的硬件上成立"的设计，直接出局。

这三条对应三个已知的失败模式：CubeCL 用通用 IR 换覆盖面，代价是 30 万行；CuTile 用绑定单一硬件换表达力，代价是换不了设备；TileLang 用 pass 推导换易用性，代价是布局不可控。cutile 选的是第四条路：**声明足够窄，窄到可以穷举；穷举足够全，全到覆盖 LLM 的算子面。**

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
| **cutile（本文预算）** | P3 全量（12 算子 / 8 设备） | **9,780** |

**这张表要横着读**：CuTile 用 75,732 行只买到一台设备；llm.u 用 43,936 行买到 7 台设备，但每个 `(算子, 设备)` 格子都要人手写内核；cutile 的目标是 9,780 行买到 8 台设备，且**新增一格不需要写代码**。规模不是唯一指标，但它是唯一可复现的指标。

### 1.3 明确认输的地方（写在这里，评审时才不会被当成隐藏负债）

| 对手 | 强在哪 | cutile 的取舍 |
|---|---|---|
| CuTile Rust | 借用检查器级别的数据竞争证明；Tile IR 里有 NVIDIA 自己的调度器 | cutile 用宿主侧 `partition()` 继承 Rust 借用检查（成本约 260 行），但**不接管设备调度**，交给各家原生编译器 |
| TileLang | 算子生态与 Python 上手速度；tile 级 API 覆盖广 | cutile 只覆盖 12 个算子，超出即报错；不提供 Python 前端 |
| CubeCL | IR 级优化、后端成熟度、社区 | cutile 放弃 IR，因此**拿不到跨算子融合**，只做算子内融合 |
| CuTe DSL | 布局代数的表达完备性 | cutile 用 4 构造子的闭集换可判定性，非常规布局必须显式 `@repack` 并计入成本 |
| 旧工程 `llm.u` | 真机覆盖面与性能台账的严谨度 | cutile 继承它的台账方法（证据分级），但用自动生成的见证替代人填 |

需要说明：**本文不合并 `llm.u`**。它是独立仓库、独立版本、独立发布节奏；cutile 只继承它的两条方法（按架构分档的能力协商、证据分级台账）和一组可复现的实测数字，不迁移它的代码。

---

## 2. 设计公理

1. **闭集优先于表达力。** 任何"用户可以写任意 X"的设计，都会把成本转移到维护 pass 上。cutile 的每个可写维度都是有限枚举，枚举外显式失败。
2. **能力必须是参数化的，不是字符串。** `provides = ["mma"]` 这种写法无法表达"我的 MMA 只支持 m=16"；能力签名必须是 `(原语, 形状区间, 数据类型, 存储空间)`，协商做区间包含判断。
3. **布局是契约，不是推导结果。** 每个片段声明"我要求输入是什么布局、输出是什么布局"；模板声明自己给的是什么。两者不符只有三种合法动作：改模板、插 `@repack`（计成本）、失败。
4. **调度是枚举，不是搜索空间。** 4 个调度模板覆盖从 elementwise 到 ping-pong attention 的全部需要；用户选模板 + 填槽位，不写流水线。
5. **正确性来自双源。** 每个算子都有一份与设备无关的标量参考实现（`ref/`），片段只与参考对拍；片段写错不会"看起来对"。
6. **每个能力声明都要有见证。** 档案里写 `provides` 一律附带证据等级 L0–L3；`verify` 拒绝把 L1 当验证，缺设备时**失败而不是跳过**。
7. **可解释性是产品能力。** 任何一次选择都必须能回答"为什么是这个变体、缓存为什么命中/失效、缺了哪个原子、最近的可行变体是什么"。

---

## 3. 架构：三层声明 + 三组闭集 + 一条 lowering

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

五步的中间产物都是**可打印的数据结构**（`Variant` / `Path` / `Plan` / `Source` / `Artifact`），这是"无 IR"的具体含义：没有隐藏的优化阶段，任何结论都能在 `doctor` 里逐行看到。

### 3.1 目录与模块

```
crates/cutile/
├── DESIGN.md
├── Cargo.toml
├── ops/                        # 声明层：12 个算子（闭集）
│   ├── attn_decode.toml        #   { skeleton, axes, needs, variants, layouts }
│   └── attn_decode.k           #   模板文本（C 风格 + {{插值}} + @原语）
├── prim/                       # 片段层：16 原语 × ≤8 设备
│   ├── nvidia/mma.c
│   └── cpu/reduce.c
├── devices/                    # 档案层：≤8 份
│   ├── nvidia.toml
│   └── cpu.toml
└── src/
    ├── lib.rs
    ├── backend.rs   # FFI / 驱动调用 / launch（unsafe 白名单）
    ├── atom.rs      # 原语槽位 + 参数化能力签名 + 兼容性判定
    ├── prof.rs      # 档案 schema、指纹匹配、证据等级、字段校验
    ├── cap.rs       # 协商、降级链、诊断构造
    ├── layout.rs    # 4 构造子、正规形、契约校验、@repack 计费
    ├── space.rs     # GM/FAST/REG 三级存储与寻址
    ├── sch.rs       # 4 个调度模板与其槽位
    ├── lower/       # expand / substitute / emit
    ├── cache.rs     # 指纹、锁、原子落盘、GC
    ├── probe.rs     # 设备探测（可注入假指纹）
    ├── plan.rs      # 形状 → 变体、autotune、落盘
    ├── host.rs      # 宿主侧 API：partition / 所有权 / 网格推导
    ├── reference.rs # 标量参考解释器（正确性基准）
    ├── toml.rs      # 手写受限 TOML 解析（D3：本轮不引入 serde/toml）
    ├── main.rs      # bin：doctor / verify / bench 入口
    └── cli.rs       # doctor / verify / bench / doctor --why
```

### 3.2 与 workspace 的接口

`crates/cutile` 已是 workspace 成员，继承 `version/edition/rust-version/license/repository` 与 `[lints] workspace = true`。`missing_docs`、`unsafe_op_in_unsafe_fn`、`clippy::all`+`cargo` 全部生效，因此本设计必须自带两条工程条款：

- **unsafe 边界**：`unsafe` 只允许出现在 `backend/`（FFI 与驱动调用）与 `host.rs` 的 launch 段；其余模块 `#![deny(unsafe_code)]`。每个 `unsafe` 块必须写明安全前提（前条件、后条件、失败行为）。
- **文档豁免**：`lower/` 与 `cache.rs` 的私有项允许 `#[allow(missing_docs)]`，公开项不豁免；`prim/**` 目标是 C/CL/MSL 源码，不受 Rust 文档检查约束。

---

## 4. 声明层：用户写什么

用户只写两个文件，都在 `ops/` 下。

### 4.1 算子声明（`ops/attn_decode.toml`）

```toml
[op]
name     = "attn_decode"
skeleton = "attention"        # 骨架闭集：elementwise | reduce | gemm | attention | scan
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
- `mma_share` 是该变体在 roofline 模型里的非 MMA 占比上界，用于 `verify` 的流水掩盖检查（见 §12）。
- `axes` 的笛卡尔积是**tile 候选**（≤ 8），`[[variants]]` 是**降级链**（≤ 8），两者不是同一维度：前者决定一次编译内的分块，后者决定用哪套调度。
- 变体数上限 8；`axes` 乘积超过 8 直接拒绝，报错时给出建议剪掉的轴。
- 为什么是 TOML 而不是 `#[op]` 过程宏：声明要可 diff、可被 `verify` 单独校验、不引入 `syn`/`quote` 依赖（见 §18 D3）。Rust 只在**新增骨架**时才写——骨架闭集只有 5 个，写满即止。

### 4.2 模板（`ops/attn_decode.k`）

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

### 4.3 一份声明的作用域

| 维度 | 谁写 | 数量 | 改动成本 |
|---|---|---|---|
| 算子 | 人（声明 + 模板） | ≤ 12 | 一份 TOML + 一份模板，约 220 行 |
| 设备 | 人（档案） | ≤ 8 | 约 60 行 TOML + 16 段片段，约 210 行 |
| 原语片段 | 人（指令级） | 16 × 8 ≈ 128 段 | 每段 10–30 行 |
| 变体 | 展开器生成 | ≤ 8 / 算子 | 0 行 |
| 布局 | 闭集 4 选 1 + 组合 | 每个槽位一个 | 0 行 |
| 调度 | 闭集 4 选 1 + 槽位 | 每个变体一个 | 0 行 |

### 4.4 宿主侧：分区即网格，所有权即并发安全

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
- **设备不由人写**：`probe` 选档案、`negotiate` 选变体、`select` 选 tile，调用点只给形状。

这套宿主模型与 CuTile Rust 一致；差别在于**所有权不依赖 CUDA**：同一个 `partition`/`&mut` 模式原样服务 8 台设备。CuTile 的遗憾是它把这套好模型绑在了 Tile IR 上，而不是模型本身有问题。

---

## 5. 能力模型：16 个原语槽位 + 参数化签名

### 5.1 原语闭集

原语是模板唯一能调用的东西，也是片段唯一的实现单位。**16 个，不多不少**；加第 17 个要走设计变更。

| # | 原语 | 语义契约 | 关键参数 | 布局要求 |
|---|---|---|---|---|
| P1 | `mma` | `D = A × B + C`，tile 粒度，累积 | `m/n/k`, `dt`, `space` | A 行主、B 列主/转置两者其一 |
| P2 | `dot` | 整数点积（dp4a/dpas/dot4） | `lanes`, `dt` | 元素连续 |
| P3 | `load` | 全局 → 片上，同步 | `bytes`, `src/dst` | 源连续或跨步 |
| P4 | `load_async` | 异步搬运 + 完成令牌 | `bytes`, `stages` | 同 P3，允许未完成 |
| P5 | `commit_wait` | 异步组提交/等待（cp.async.commit / wait_group） | `group`, `n` | 无 |
| P6 | `store` | 片上 → 全局 | `bytes`, `src/dst` | 目标连续 |
| P7 | `bar` | 块内屏障 / arrive-wait | `scope` | 无 |
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

### 5.3 档案（`devices/nvidia.toml`）

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
witness = "prim/nvidia/mma_witness"
```

规则（全部由 `prof.rs` 校验，不是约定）：

- `provides` 里出现闭集外的 `prim` / `dt` / `space` → 加载失败。
- `grade = "L2"` 及以上必须有 `witness` 字段且对应文件存在。
- 同一 `(vendor, id)` 只允许一份档案；指纹多行同等具体 → 加载失败（不猜顺序）。
- 缺 `emit` 或 `tool` → 加载失败。

### 5.4 八份档案一览

"设备无关"必须在数据上成立，所以把 8 份档案的差异一次列清。注意 `FAST` 一列：**同一份模板在五种不同的片上存储名字上跑**，这就是设备无关的具体含义。

| 设备 | 指纹键 | `FAST` 物理映射 | `emit`（可切换） | 工具链 | 见证来源 |
|---|---|---|---|---|---|
| NVIDIA | `cc = 7.0..12.1`（按代分 sm70/80/90/100） | `shared` | `cuda` / `ptx` | nvcc / NVRTC | 真机 |
| AMD | `gfx = gfx9..gfx12` | `lds` | `hip` / `opencl` | hipcc / clang | 真机（RDNA） |
| Intel GPU | `name contains Intel`、`arch = gen12+` | `slm` | `opencl` / `spirv` | IGC / oneAPI | 真机 |
| Apple | `name contains Apple M`、`os >= 26` | `threadgroup` | `metal` | xcrun metal | 真机 |
| 昇腾 | `soc = Ascend 910B..` | `L1`（MTE 搬运） | `ascendc` | CANN | 文档（L0/L1） |
| MUSA | `name contains MTT S` | `shared` | `cuda`（源码兼容） | mcc | 文档 + 模拟 |
| CPU | `isa = x86_64{avx2,avx512} / aarch64{neon,sve}` | `L2`（分块） | `c` | cc / clang | 真机 |
| 可移植 GPU | `api = vulkan / webgpu` | `workgroup` | `spirv` / `wgsl` | glslang / naga | 模拟 |

加第 8 台（可移植 GPU）的理由不是"凑数"，而是**兜底**：在没有厂商 SDK 的环境里，`emit = wgsl` 仍能让同一份模板产出可运行代码；能力不足时协商自动降级，而不是整套不可用。同理，`cuda` 一条 emit 同时服务 NVIDIA 与 MUSA，说明"设备"与"输出语言"是两个维度，不能合并成一个概念。

---

## 6. 布局：4 个构造子 + 正规形 + 契约

### 6.1 闭集

| # | 构造子 | 语义 | 例 |
|---|---|---|---|
| L1 | `row(shape)` | 行主序紧凑 | `row(BM, BN)` |
| L2 | `col(shape)` | 列主序紧凑 | `col(BN, BM)` |
| L3 | `swz<B,M,S>(base)` | 在 base 上做 XOR swizzle（bank 冲突消解） | `swz<3,3,3>(row(64,64))` |
| L4 | `pad(base, p)` | 在 base 的最后一维插入 `p` 个填充元素 | `pad(row(64,64), 4)` |

四个构造子的**组合是受限的**：`swz` 与 `pad` 只能包裹 `row`/`col`，不能互相嵌套。这条限制把布局的全集压到可枚举，也让"正规形"唯一。

### 6.2 正规形与判定

`layout.rs` 把任意合法布局归约成 `(order, shape, stride, swizzle, pad)` 五元组；两个布局相等当且仅当五元组相等。归约函数是纯函数、无搜索，约 90 行。

契约校验的输入是三个：模板声明的布局、片段要求的布局、设备档案里的 `mem.bank`。判定结果只有三种：

1. **相等** → 直通。
2. **可重组**（同一 order，stride/swizzle 不同）→ 插入一次 `@repack`，成本计入 roofline 的字节数。
3. **不可重组**（order 不同，或 pad 后无法对齐）→ 失败，报"模板给 row、片段要 col，且转置会破坏 P1 的 B 操作数约定"。

**为什么不做布局推导**：CubeCL 的 `cubecl-tile/{layout,space,stage}` 合计 16,666 行、TileLang 的 `tilelang/{layout,transform}` 合计 3,520 行，再加上各自 C++/IR 侧的实现，全部目的都是"给定算子自动求布局"。cutile 把这件事换成"人写一个四选一的属性 + 机器校验 + 不匹配时显式计费"。少掉的不是能力，是**不可解释性**——而且写错的代价从"运行慢 3 倍且没人知道"变成"构建期报错"。

### 6.3 三级存储

`GM / FAST / REG` 是**语义名**，每个档案映射到本设备的物理名（NVIDIA `shared`、Intel `slm`、昇腾 `L1`、Metal `threadgroup`、CPU `L2/栈`）。模板只写语义名，因此同一份模板在昇腾上的 `GM -> FAST` 会自动变成 `GM -> L1` 的 MTE 搬运。

`space.rs` 只做三件事：名字映射、容量校验（`FAST` 用量 ≤ 档案字节数）、对齐校验（`swz` 的粒度与 bank 数兼容）。共 240 行。

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
  建议：改用 smem_only（预计 -26% 峰值），或在 devices/ascend.toml 补 load_async 的 L2 见证
```

每条诊断必须包含三件事：**缺什么原子**、**供给方实际是什么**、**最近的可行解与代价**。这三件事是 CubeCL（只报编译错）、TileLang（只报 pass 失败）、CuTile（只报 Tile IR 版本不符）都给不出的。

### 8.3 闭集外请求

请求 `prim = "tma_bulk"`、`layout = "interleaved"`、`schedule = "rs"` 等闭集外取值时，错误信息固定为：

```
error: `layout = "interleaved"` 不在闭集内
  闭集：row | col | swz<B,M,S> | pad(base,p)
  最近的等价表达：swz<2,4,3>(row(64,128)) 覆盖 4 交错
  若是新需求，需走 DESIGN.md §6 的闭集变更流程（含预算重算）
```

---

## 9. 变体选择与 plan

`plan.rs` 负责"给定形状，选哪个变体、多大 tile"。

- 输入：`(op, prof, shape, dtype, policy)`。
- 首选：已落盘的 `plan` 表（`plan/<vendor>-<id>-<op>.toml`），键是形状桶（`m` 按 2 的幂分桶，`n/k` 同理）。
- 缺失：用 roofline 估算 `max(flops/peak, bytes/bw) + 未掩盖项`，在**预测最优的 ±1 档**内取 3 个候选做一次微基准，落盘。
- autotune 上限：每 (op, device) 最多 3 候选 × 5 次重复；这是与 CubeCL 全空间 autotune 的刻意差别——**搜索空间用闭集已经剪过一次，不需要再暴力剪**。
- 失效：`plan` 表带指纹（`schema_ver`、`tool_ver`、`emit`、`dtype` 位集），任一不符即视为缺失，重新估算。

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
| ① expand | `op.axes` | `Vec<Variant>`（≤8） | 轴乘积 > 8 |
| ② negotiate | `op.needs` + `prof.provides` | `Path` | 无变体可行（§8.2 诊断） |
| ③ select | `(op, prof, shape)` | `Plan { variant, tile }` | 形状超出轴范围 |
| ④ substitute | 模板 + 片段 + 布局 | `Source { lang, text }` | 槽位缺失 / 布局不可重组 |
| ⑤ emit | `Source` + `prof.tool` | `Artifact { bytes, key }` | 编译器报错（原文透出） |

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
    layout_nf: Vec<(String, LayoutNF)>, // 正规形
    variant: VariantKey,      // 轴取值 + 调度槽位
    dtype: DTypeSet,
}
```

- 目录：`$CUTILE_CACHE/<key[0..2]>/<key>/`，内含 `artifact`、`source.<ext>`、`meta.toml`。
- 并发：`<key>.lock` 文件 + `O_EXCL`，等待上限 120 s，超时即报错（不静默重复编译）。
- 落盘：写临时目录 → `rename` 原子替换。
- GC：按 LRU + 容量上限（默认 2 GiB），`cutile cache gc` 可手动触发。
- 可复现：`meta.toml` 记录完整 `CacheKey` 与工具版本，`cutile doctor --why <key>` 能解释命中/失效原因。

---

## 11. 正确性工程

### 11.1 双源对拍

每个算子在 `src/reference.rs` 有一份标量参考实现（不依赖任何设备、不依赖片段）。对拍口径固定为 `max|got−want| / max|want|`，阈值按数据类型：整数精确、f16/bf16 ≤ 1e-3、f32 ≤ 1e-5。片段只与参考对拍，**片段之间不互相对拍**（否则一个错会传染成"两个都对"）。

### 11.2 证据分级（继承旧工程，改为自动生成）

| 等级 | 含义 | 谁产生 | 能否用于发布 |
|---|---|---|---|
| L0 | 档案里声明了 | 人 | 否 |
| L1 | 片段能编译 | CI | 否 |
| L2 | 与参考对拍通过（真机或注入的假指纹 + 模拟器） | CI / 本地 | 是（但仍标记） |
| L3 | 真机性能达到 roofline 的 ≥ 70% | 真机 bench | 是 |

`capability` 台账（`devices/<vendor>.capability.md`）由 `cutile verify --emit-ledger` 生成，人不再手填。**缺设备即失败**：`verify` 遇到本机不支持的设备直接返回非零码并列出缺失项，不写"跳过"（这是旧工程 `clblast` 用例静默通过的直接修正）。

### 11.3 见证

每条 `provides` 附一个最小见证（`prim/<vendor>/<prim>_witness`）：一段只调用该原语的迷你 kernel + 期望值。CI 在能跑的设备上执行见证，并把等级写回台账。协商只接受 `grade ≥ L2` 的供给——**未验证的能力视为不具备**，这是把旧工程 R5 风险（"声明有 TMA 实际没有"）前移成构建期问题的关键一条。

---

## 12. 性能方法论

### 12.1 roofline 是判定器，不是口号

档案必须提供可推导的常数，每个值都带自己的 `src`（推导式，而不是抄来的数字）：

```toml
[peak]
flops_f32 = { value = 1.709e12, src = "sm_count*64*2*clock" }   # 例：T600 口径
bw        = { value = 160.0e9,  src = "2*mem_clock*bus_bytes" }
```

`verify` 用 `max(flops/peak_flops, bytes/peak_bw) + 未掩盖的非 MMA 时间` 给出预测下界；`bench` 打印达成率。目标数字（见 §12.3）全部是**目标**，不是实测，验收必须附命令与机器。

### 12.2 三个可测的构建期判据

1. **流水掩盖**：`unmasked_non_mma / total ≤ 20%`，否则要求换 S3/S4。
2. **repack 预算**：`@repack` 引入的额外字节 ≤ 该算子总流量的 10%，否则布局设计错误。
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
| `cap` 协商 + 诊断 | 280 | 含 §8.2 的格式化 |
| `layout` 4 构造子 + 正规形 | 200 | swz 只用一种参数 |
| `space` | 120 | 三级映射 + 容量/对齐校验 |
| `prim` 片段 | 240 | 10 原语 × 3 设备 × 8 行 |
| `sch` | 140 | S1 + S2 |
| `ops` 声明 + 模板 | 260 | `gemv_i4` 一份 |
| `lower` | 300 | 词法 + 替换 + emit |
| `cache` | 140 | 指纹 + 锁 + 落盘 |
| `probe` | 120 | 三个后端 + 假指纹注入 |
| `plan` | 80 | 只做 roofline 估算 |
| `host` | 120 | partition + 所有权 + 网格推导 |
| `ref` | 240 | 标量参考 + 对拍框架 |
| `test` | 180 | 见证 + 对拍用例 |
| `cli` | 100 | `doctor` / `verify` / `bench` |
| **合计** | **2,760** | |

### 13.2 P3（12 算子 / 8 设备 / 16 原语 / 4 调度）

| 模块 | P0 | P3 | 增量说明 |
|---|---:|---:|---|
| `prof` | 240 | 600 | 8 份档案 + 证据台账生成 + 指纹家族 |
| `cap` | 280 | 520 | 参数化区间匹配 + 三档诊断 + 闭集变更检查 |
| `layout` | 200 | 460 | 4 构造子全组合 + 正规形 + repack 计费 |
| `space` | 120 | 240 | 昇腾 L1 / Metal threadgroup / CPU 栈 |
| `prim` | 240 | 1,240 | 16 原语 × ≤8 设备，约 128 段 × 9.7 行 |
| `sch` | 140 | 420 | S3 warp 专门化 + S4 ping-pong |
| `ops` | 260 | 2,640 | 12 算子 × 220 行 |
| `lower` | 300 | 680 | 语法检查 + 位置信息 + 多 emit |
| `cache` | 140 | 300 | GC + 并发 + 迁移 |
| `probe` | 120 | 280 | 8 后端 + 系统版本约束（Metal 需 macOS 版本） |
| `plan` | 80 | 380 | 形状桶 + 3 候选微基准 + 失效 |
| `host` | 120 | 260 | async + 图捕获 + 错误传播 |
| `ref` | 240 | 820 | 12 份标量参考 + 阈值分档 + 随机形状 |
| `test` | 180 | 700 | 见证矩阵 + 假指纹回归 |
| `cli` | 100 | 240 | `doctor --why` + `cache gc` + `verify --emit-ledger` |
| **合计** | **2,760** | **9,780** | |

### 13.3 为什么这 9,780 不是幻觉：三笔消元

| 消元 | 朴素做法 | cutile | 比值 |
|---|---|---|---|
| **笛卡尔积** | 12 算子 × 8 设备 × 191 行手写内核 = 18,240 行（191 取自旧工程 `cuda.c` 4,778 行 / 25 类内核） | 声明 2,640 + 片段 1,240 = 3,880 行 | **4.7×** |
| **布局代数** | CubeCL `cubecl-tile/{layout,space,stage}` 16,666 行（唯一可核实的口径；TileLang 对应层只有 3,520 行，其余在 C++ 侧） | 460 行闭集 + 3 种合法动作 | 按 CubeCL 口径 36× |
| **变体组合** | 每设备每形状一份手调参数 | 8 变体上限 + plan 表 + 3 候选搜索 | 搜索空间从 `2ⁿ` 降到 ≤ 24 |

并且：**新增一格 `(算子, 设备)` 的边际成本是 0 行**——只要骨架与原语都在闭集内。

---

## 14. 与四家参照 + 旧工程的量化对比

### 14.1 成本

| 指标 | cutile | CubeCL | CuTile Rust | TileLang | 旧工程 llm.u |
|---|---:|---:|---:|---:|---:|
| 可比层行数 | **9,780** | 301,766 | 75,732 | 317,510 | 43,936 |
| 相对 cutile | 1.0× | 30.9× | 7.7× | 32.5× | 4.5× |
| 覆盖设备 | **8** | 7 | 1 | 6（目录数，验证深度不一） | 7（真机台账） |
| 新增一台设备 | ≤ 300 行 | 3,134–11,925 行（实测每后端 src） | 不支持 | 需改 C++ codegen | 2–4k Rust + 0.2–4.8k C |
| 新增一个算子 | ≤ 270 行，0 行设备代码 | 100–300 行 + 各后端验证 | 50–150 行（仅 NVIDIA） | 30–80 行（限形状/后端） | 骨架 + 每后端约 191 行 C |
| 新增一格 (算子×设备) | **0 行** | 需该后端 codegen 支持 | 不适用 | 需 codegen 支持 | 约 191 行 |

### 14.2 质量属性

| 属性 | cutile | CubeCL | CuTile Rust | TileLang | llm.u |
|---|---|---|---|---|---|
| 布局是否可解释 | **构建期契约 + 显式 repack 计费** | IR pass，需读 IR | 框架接管 | pass 推导，留逃生舱 | 手写死 |
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
| 片段 ≤ 400 行（内部目标 240）、性能 ≥ 原生 85%、协商 3/3 命中首档、`doctor` 能打印完整链 | **方案成立**，进 P1 |
| 片段 > 400 行 | 原语粒度错，先合并原语 |
| 某设备 < 70% | 该设备承认需要特化代码，把它从"配置化"降级为"手写 + 片段复用" |
| 协商命中率 < 3/3 | `needs` 声明粒度错（这条是 v0.3 保留下来最关键的判据） |
| 布局校验出现 > 2 次不可重组 | 闭集取错，需重新选 4 个构造子 |

执行命令（P0 交付物，已在 `crates/cutile` 实现；`verify` 在缺设备时返回非零码）：

```bash
cargo run -p cutile -- doctor  --op gemv_i4 --device auto
cargo run -p cutile -- verify  --op gemv_i4 --devices cpu,opencl,cuda
cargo run -p cutile -- bench   --op gemv_i4 --shape 4096x4096 --repeat 50
```

---

## 17. 里程碑

| 阶段 | 交付 | 判据 |
|---|---|---|
| P0 | Go/No-Go 报告 | §16 五项判据 |
| P1 | `prof` + `cap` + `layout` + `prim`(3 设备) + `lower` + `host`，跑通 1 算子 | 同一份 `.k` 三设备可运行；与 `ref` 对拍通过；协商命中；布局校验零失败 |
| P2 | `ops` 扩到 6 + `plan` + autotune + S3 | 形状变化自动换变体，达成率差 < 5%；`doctor --why` 可解释每次选择 |
| P3 | 设备扩到 8（含昇腾/MUSA/可移植 GPU）+ `ops` 扩到 12 + S4 | 新增设备 ≤ 300 行；总量 ≤ 10,000；台账 L2 覆盖 ≥ 90% 的 `provides` |
| P4 | emit 切换验证（cuda↔ptx / hip↔opencl / wgsl↔spirv） | 换 emit 不改模板、不改算子声明 |

---

## 18. 待决事项

| # | 待决 | 影响 | 建议决策点 |
|---|---|---|---|
| D1 | 12 个算子的最终清单（本文 §4 已给候选：`tok_embed` / `rms_norm_res` / `rope` / `gemv_i4` / `gemm_bf16` / `mlp_gate_up` / `attn_decode` / `attn_prefill` / `kv_store` / `ssm_scan` / `sampler` / `adamw`） | 决定 `ops/` 是 2,640 行还是更多 | P1 开始前定稿；超出 12 个必须按 §13.2 重算并重定闭集 |
| D2 | 布局闭集取哪 4 个构造子（本文取 row/col/swz/pad） | 决定 `@repack` 频率 | P1 用 row + swz 两种跑通，P2 补 col/pad |
| D3 | `cutile` 是否直接依赖 `syn`/`serde`/`toml` | 影响供应链治理与 `lower` 行数 | P0 用手写解析 + 手写序列化，证明可行性后再评估 |
| D4 | 昇腾与可移植 GPU 的 `provides` 来源 | 只能 L0/L1 | 先标 `grade = L1`，协商自动降级；有真机再补见证 |
| D5 | 与 `llm`（根包）的接口形态 | 决定 `host` 是否需要对外暴露 | P1 前定：`llm` 通过 `cutile::host::launch` 调用，还是只共享 `plan` 表 |
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
| 算子（op） | 声明层的最小单位，对应 `ops/<name>.toml` |
| 骨架（skeleton） | 算子的结构类别，闭集 5 个；决定用哪个模板基座 |
| 原语（prim） | 模板唯一可调用的指令级操作，闭集 16 个 |
| 片段（fragment） | 某原语在某设备上的实现，`prim/<vendor>/<prim>` |
| 变体（variant） | 轴取值 + 调度 + 布局的组合，每算子 ≤ 8 |
| 能力签名 | `(prim, 形状区间, 位集, 存储集, 实现名, 证据等级)` |
| 见证（witness） | 证明某条 `provides` 为真的一段最小 kernel + 期望值 |
| 正规形 | 布局的五元组 `(order, shape, stride, swizzle, pad)` |
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

**许可与归属**：本文只借鉴上述实现的机制与公开结论，不复制其源码。若后续片段实现参考了某一具体实现（例如 CubeCL 的 wgmma 序列），必须在 `prim/<vendor>/NOTICE` 中保留原许可与出处；本项目自身维持 `MIT OR Apache-2.0` 双许可。
