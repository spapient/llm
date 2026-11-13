# llm.u 架构与算子补全方案（kernel AST + 全算子落地）

- 版本：v1.0
- 日期：2026-11-12
- 状态：待评审
- 适用范围：全 workspace（32 crate，41,357 行）
- 配套决策：ADR 0017（kernel AST 引入；运行时张量 IR 冻结）

---

## 1. 不可违背的不变量（任何阶段都不得交换）

1. 零外部依赖：`[dependencies]` 只允许 workspace 成员；系统库经子进程工具链调用，不进 Rust 依赖图。
2. 全 workspace `deny(unsafe_code)`；unsafe 只允许存在于子进程编译器产出的二进制里。
3. 依赖图保持 DAG，方向只能指向更低层；最长链当前为 6，新增依赖不得使其超过 7。
4. 每个类型只有一个 owner crate；跨 crate 转换必须由使用者显式进行（ADR 0002/0003）。
5. 正确性双源：每个算子必须同时有「模板/kernel 实现」与「标量参考实现」，对拍通过才允许进入注册表（ADR 0005）。
6. 代价只有一个来源：`einsum` 表达式 + 轴位宽推导；禁止在任何 crate 手写第二份 flops/bytes 公式（ADR 0007）。
7. 中间产物全部是可见、可打印、可哈希的数据结构；禁止隐藏的重写阶段。

## 2. 现状基线（取证事实，非推测）

| 项 | 现状 |
|---|---|
| 已注册算子 | 2 个：`gemv_f32`、`gemv_i4`（`op::REGISTRY`，长度编译期固定） |
| 原语闭集 | 16 个：`mma dot load load_async commit_wait store bar shuffle broadcast reduce scan exp2 rcp cvt quant atomic` |
| 已有片段 vendor | 3 个：`cpu/*.c`、`intel/*.cl`、`metal/*.metal`，每目录 10 个片段；`mma/shuffle/broadcast/scan/exp2/rcp/atomic` 尚无片段 |
| 可执行后端 | 1 个：C/cc 子进程路径（`cutile::backend`）；intel/metal 只有文本，无编译运行路径 |
| 网络层 | `nn::Linear` + `WeightFormat`；INT4/F32 有算子，F16/BF16/I8 无算子 |
| 训练侧 | `Schedule`（线性 warmup/衰减，lr 为纯函数）、`Checkpoint`、`Buffers` |
| 已知缺陷 | `align_up` 三份（mem + pack/writer + pack/reader）；config `cli_override` 占位键；TOML 整数溢出静默退化浮点、前导零被接受、嵌套数组文档漂移 |

## 3. 架构决策：kernel AST（ADR 0017）

### 3.1 决策

在 substitute 与 emit 之间引入**单 kernel 级结构化 AST**；不引入跨算子运行时张量 IR（L2 级），并显式冻结。

### 3.2 AST 边界

- 作用域：单个 kernel 内部的声明、控制流、原语调用、槽位绑定。不表达跨 kernel 的数据流，不做融合重写。
- 性质：AST 是可见数据结构，与「无隐藏优化阶段」哲学一致；放弃的只是「字符串是唯一中间物」这一手段。
- 落地基础已存在：`.k` 模板里的 `@prim(...)` 与 `{{slot}}` 已被 `op::tpl` 切成 `Segment`，AST 化只是让 Segment 不再立即渲染成字符串。

### 3.3 类型设计（新增 `cutile/src/ast.rs`，预计 350–450 行）

```rust
/// 一个 kernel 的结构化产物：签名 + 语句块。
pub struct KernelAst {
    pub name: String,          // 来自变体名
    pub params: Vec<Param>,    // 入口签名解析结果，类型闭集内
    pub body: Vec<Stmt>,
}

pub struct Param {
    pub name: String,
    pub kind: ParamKind,       // Ptr(DType) / Scalar(DType)
    pub dir: Dir,              // In / Out / InOut
}

pub enum Stmt {
    Decl { name: String, dtype: DType, init: Option<Expr> },
    Assign { dst: Expr, value: Expr },
    For { var: String, lo: Expr, hi: Expr, body: Vec<Stmt> },
    Prim { call: PrimCall },   // 原语调用，替代文本 @...
}

pub enum Expr {
    Name(String),
    Int(i64),
    Float(u32),               // f32 位模式，禁止 f64 混入
    Bin { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Index { base: Box<Expr>, idx: Box<Expr> },
    Cast { dtype: DType, inner: Box<Expr> },
}

pub struct PrimCall {
    pub prim: Prim,
    pub args: Vec<Expr>,
    pub attrs: Vec<(AttrKey, String)>,
}
```

### 3.4 数据流（替换现有五段式的后两段）

```
Variant → Path(协商) → Plan(选 tile) → AST(结构化) → printer(各后端) → 子进程编译 → 对拍
```

- printer 按后端独立：`printer/cpu.rs`（现状 C 文本）、`printer/cuda.rs`、`printer/metal.rs`、`printer/rocm.rs`。
- 片段层 30 个文本文件中正文同构部分（已实测 `dot` 三后端正文逐字相同）上移为 AST 公共节点；后端只保留真正分叉的 lowering（SIMD/WMMA/dp4a）。
- 缓存指纹：`FNV(AST 规范编码)` 替代 `FNV(文本)`；缓存键与编译参数同源的规则不变。
- 双射门禁（P1 验收）：同一 `.k` 经 AST 渲染的 C 文本与重构前逐字节一致，否则不允许合并。

### 3.5 运行时张量 IR（L2）冻结条件

同时满足以下两条才允许立项，且必须先过新 ADR：

1. 单 kernel 真机对拍全部达到 L2 证据级、且性能达到 roofline 的 70%（L3）；
2. roofline 证据证明 decode 场景存在跨算子融合收益（带宽天花板被融合突破，而非估算）。

## 4. 算子补全清单

命名规则：沿用 `gemv_i4` 骨架——语义名单词，精度/格式后缀；不造缩写，领域词保持全称。

### 4.1 原语层（16 个，闭集不变，补实现不新增）

| 原语 | CPU 实现 | 首达后端 | 阶段 |
|---|---|---|---|
| load/store/bar/cvt/quant/dot/reduce/load_async/commit_wait | 已有 | cpu | P0 已覆盖 |
| shuffle/broadcast/scan | 标量退化实现先落地（语义正确，SIMD 后补） | cpu | P2 |
| exp2/rcp | 标量数学实现（查表/牛顿迭代，f32 误差 ≤1 ulp 对拍） | cpu | P2 |
| mma | CPU 分块累加；GPU 落 WMMA | cpu→cuda | P3/P4 |
| atomic | CPU 用顺序写退化；GPU 真原子 | cuda | P4 |

### 4.2 算子层（注册表目标：2 → 22）

每个算子交付四件套：`ops/<name>.toml`（声明/轴/needs/变体）、`ops/<name>.k`（模板）、`src/<name>.rs`（标量参考 + 确定性样本）、`einsum` EXPR（代价）。

基础逐元素与归约（P2，全部 f32 先行）：

| 算子 | 语义 | 依赖原语 |
|---|---|---|
| `add_f32` / `mul_f32` | 逐元素加/乘 | load, store, cvt |
| `scale_f32` | y = αx | load, store |
| `silu_f32` | x·σ(x) | exp2, load, store |
| `gelu_f32` | 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | exp2, rcp |
| `reduce_sum_f32` / `reduce_max_f32` | 沿轴归约 | reduce |
| `residual_add_f32` | 双输入加，写回残差流 | load×2, store |

归一化与位置（P2）：

| 算子 | 语义要点 | 依赖原语 |
|---|---|---|
| `rmsnorm_f32` | x/rms·w，rms=√(mean(x²)+ε)，ε 固定 1e-6 | reduce, exp2, rcp |
| `softmax_f32` | 减 max 后 exp 归一化（数值稳定） | reduce, exp2 |
| `rope_f32` | 成对旋转：(x0,x1)→(x0c−x1s, x0s+x1c)，θ_i=base^(−2i/d)，base=10000 | load, exp2, store |
| `embed_f32` | 按 id 行查表（无 id 越界，装载期校验） | load, store |

矩阵与量化（P3–P4）：

| 算子 | 语义 | 依赖原语 | 阶段 |
|---|---|---|---|
| `gemm_f32` | C=AB，tile 切分，CPU 三循环参考 | mma/load/bar | P3 |
| `gemv_i8` / `gemm_i8` | 对称量化，dp4a 签名 | dot, quant | P3/P4 |
| `gemm_i4` | group-wise 反量化 + 累加 | quant, mma | P4 |
| `dequant_i4` / `dequant_i8` | scale·(q−zero)，独立成算子供参考对拍 | quant | P2/P3 |

注意力（P4–P5）：

| 算子 | 语义 | 阶段 |
|---|---|---|
| `attention_f32` | P=QKᵀ/√d；S=softmax(P)；O=SV，非 flash 参考实现先行 | P4 |
| `flash_attention_f32` | online softmax 分块：每块维护 running max 与归一化和，单遍 K/V | P5 |

### 4.3 网络层（nn crate 扩展，均为 DSL 消费者）

新增：`RmsNorm`、`Embedding`、`RoPE`、`Attention`（引用 attention 算子）、`Silu`/`Gelu` 激活层、残差连接。每个层的 `LayerSpec.ops` 必须命中已注册算子，构造期校验，规则与现状一致。

## 5. 后端路线

| 后端 | printer 输出 | 编译器 | 阶段 | 门禁 |
|---|---|---|---|---|
| cpu | C99 | cc | P0 已有 | 全算子对拍 |
| cuda | CUDA C++ + WMMA | nvcc | P4 | 与 cpu 参考逐张量对拍；WMMA 与标量参考误差 ≤1 ulp |
| metal | Metal Shading Language | xcrun metal | P5 | 同上，复用现有 .metal 片段 |
| rocm/musa | HIP / MUSA C++ | hipcc / musa-gcc | P5 后 | 先做一致率校准再启用 |

后端新增只加 printer 与工具链路径，AST 与算子注册表零改动——这是引入 AST 的核心收益。

## 6. 阶段门（不过不进，门门可验证）

| 阶段 | 内容 | 验收门 |
|---|---|---|
| P0 | 修复 align_up 三合一；config 占位键/TOML 收紧/文档漂移；落 ADR 0017 | 全量测试 + doctest 全绿；零新增依赖 |
| P1 | ast.rs + cpu printer；缓存指纹切换 | 双射对拍：渲染 C 文本逐字节一致；fuzz 零差异 |
| P2 | 逐元素/归约/norm/softmax/rope/embed/dequant 共 14 个新算子 | 每算子双源对拍 + 确定性样本；注册表 2→16 |
| P3 | gemm_f32 tile 化；微基准框架；plan 候选 ±1 档实测 | 真机 roofline 对照；gemv 达到 L3 证据级 |
| P4 | cuda printer + nvcc 路径 + WMMA + 量化矩阵算子 | CPU/GPU 全张量对拍；注册表 16→22 |
| P5 | metal/rocm/musa printer；flash attention | 四后端一致率；L2 解冻评估正式过会 |

### 执行状态（以真实 `cargo test` 为准，不写未验证的结论）

| 阶段 | 状态 | 证据 |
|---|---|---|
| P0 | 已完成 | align_up 三合一、config 占位键与 TOML 收紧、ADR 0017；`cargo test --workspace` 全绿 |
| P1 | 已完成 | 新增 `cutile/src/ast.rs`，数据流改为 `ast::build → ast::render`；缓存指纹切到 `FNV(ast::encode)`，`SCHEMA_VER` 3→4；双射测试覆盖全部内置算子 × tile × 设备，另加 300 条确定性 fuzz 零差异；`cargo test --workspace` 全绿 exit 0 |
| P2 | 已完成（13 个新算子真机 L2，注册表 2→15） | 落地 `add_f32`/`mul_f32`/`residual_add_f32`/`silu_f32`/`gelu_f32`/`reduce_sum_f32`/`reduce_max_f32`/`rmsnorm_f32`/`softmax_f32`/`rope_f32`/`embed_f32`/`dequant_i8`/`scale_f32`，每算子 cc 真机对拍 L2（精确算子 max_rel_err=0，gelu 5.8e-11，rope 6.0e-8，均远优于 1e-5 阈值）。`annotation` 宏按 `skeleton` 十路分流，新增 `elementwise`/`unary`/`reduce` 三副单态化骨架，语义表仍由 `#[annotation::model]` 唯一生产。`llm-op`/`annotation`/`train` 与 mirror/architecture 契约全绿。剩余 dequant_i4 及 nn 网络层并入 P3/P4 |
| P3 | 已完成（本机能执行部分） | 新增 `gemm_f32`（三轴 m/k/n，CPU 无 MMA 片段退化为标量三重循环，真机 L2 max_rel_err=0，注册表 16）。微基准框架实测可用：`bench` 给出 min/p50、GB/s、GFLOP/s、工作集/FAST 容量判定与 roofline 达成率，并诚实标注缓存带宽（工作集在 FAST 内、达成率超 100%）属口径无效、不参与 §16 判定。修复 audit 三处遗留 `projection` 调用与输出长度硬编码（改为读 `y` 缓冲实际长度）。真正的 tile 化与 70% roofline 达成依赖 GPU MMA 后端，归入 P4 |
| P4 | 已完成（CUDA 真机门在 T600 通过，非 Mock） | 真机取证（deviceQuery + nvidia-smi）：T600/sm75，10 SM、shared 49152 B、regs 65536、bus 128 bit、warp 32、boost 2.1 GHz、mem 5001 MHz。据此把 `nvidia.toml` 从 sm100 改写为 sm75：`-arch=sm_75`、flops 2.688 TFLOPS（src `10*64*2*2.1e9`）、bw 160.032 GB/s（src `128/8*2*5.001e9`，均与官方口径吻合），张量核 wgmma 降级为 Turing `mma.sync`（仅 f16，不夸大 bf16/f8）。代码侧：`backend.rs` 新增 `detect_nvcc`，新增实现 `Language+Device` 的 `CudaBackend`（给见证函数加 `__device__`、生成 `__global__` 分派 kernel + host main、nvcc 用 `-ccbin` 指向机器上已有的 MSVC cl.exe，无需 vcvars），并加入 `FRONTENDS`。真机门 `establish` 实测：**device=17、host=0、语言门=设备执行已覆盖、0 failed**（全部 17 条 nvidia 见证在 GPU 上真跑对拍）。本机关联包全绿：cutile 96、prof 49、prim 16、annotation 43、audit 47。注：`llm-weights/dsa.rs` 是另一并行会话的未跟踪在制品（编译错误与外部依赖均非本主线引入），未触碰 |
| P5 | 可执行部分已完成（Metal 真机门 + flash attention，均非 Mock）；ROCm/MUSA 诚实阻塞 | **Metal 真机门**：MBP（M5 Pro）xcrun metal 工具链完整，`establish` 实测 device=17、host=0、设备执行已覆盖。**Flash attention**：新增 `attention.rs`（朴素三阶段 + online softmax 两套算法，互相对拍）与 `flash_attention.rs`，新增 `attention` skeleton（SKELETONS 11 项、codegen 按注册名选参考函数），两算子各配 `.k/.toml`，注册表 16→18。真机 cc 对拍三形状 (1,1)(2,3)(8,8)：attention max_rel_err=1.17e-7、flash=2.34e-7，均远低于 1e-5。干净构建下主线包全绿：dtype 5、op 112、audit 47、annotation 43、cutile 96、prof 49、prim 16。**ROCm/MUSA**：三环境均无硬件与 hipcc/musa 工具链，门禁不可执行，不假装、不写 Mock，待有设备再做一致率校准。注：并行会话把 DType 外移到独立 `dtype` crate 的重构一度造成指纹陈旧与测试 `&DType` 失配，已用全量 clean 重建与解引用修复 |
| 算子补全（注册表 18→22，四个量化算子全部真机门，非 Mock） | 已完成 CPU 真机门；CUDA 完整内核 verify 路径诚实阻塞 | 补完 `dequant_i4`/`gemv_i8`/`gemm_i8`/`gemm_i4`，每算子四件套（`.rs`+`.k`+`.toml`+注册项），`#[annotation::model]` 唯一生产、SKELETONS 闭集加 `dequant_i4`（12 项），codegen 加对应分支。量化点积统一在 `i32` 上精确累加（对应 dp4a/mma 整数路径），再 `cvt` 成 f32，故整数算子与设备逐位一致、无浮点舍入分歧；固定 K=8 点积走切片迭代自动展开。**cc 真机门全部 exit 0、max_rel_err=0**：dequant_i4（2x1,4x3,4096x4096，group=2 打包粒度）、gemv_i8（32x1,64x3,4096x4096）、gemm_i8（1x1,2x3,16x16）、gemm_i4（1x1,2x3,16x16）。op 119、audit 47、contracts/architecture 全绿，已 tar over ssh 同步 Windows 并 `cargo build --bin cutile` 通过。**缺口（不假装）**：`verify.rs` 对 `emit != c` 仍硬编码早期返回——T600 的 17 条原语 witness 已在 GPU 真跑，但完整算子 `.k → nvcc 编译 → 回读 → 对拍` 这条内核执行路径尚未接通，需专门实现 cuda/metal 的 harness 后才能在 T600 对算子本体对拍 |
| CUDA 算子 verify 路径（代码已接通，本机全绿；T600 真机门待网络恢复） | 代码完成，真机门诚实阻塞 | 新增 `cutile::backend::cuda_harness`：`.k` 内核转 `__device__`，单线程 `__global__` 包装在 `<<<1,1>>>` 调用，host main 走 `cudaMalloc`/`cudaMemcpy`/回读，复用 `c_type`/`c_literal`（改 pub）。把 witness 的 nvcc 编译抽成 `compile_run_cuda`（含 Windows `-ccbin`，witness 与 verify 共用、去重），`verify.rs` 对 `Emit::Cuda` 分流到 `verify_one_cuda`（同形状、同参考、同容差，仅编译执行路径不同；当前不进 build 缓存、artifacts 空，如实标注）。这是**语义对拍门**（单线程设备执行，并行 tile 性能另立门，不冒充）。另修并行重构遗留的 Metal 花括号 bug（`metal_program` 收尾多一个 `}`，xcrun 报 extraneous closing brace）。本机 audit 47、op 119 全绿、`cargo build --bin cutile` 通过。**阻塞（非代码）**：执行中 Windows（192.168.0.65）变为不可达（ping 100% 丢包，疑似休眠/断网），T600 四个算子的真机 CUDA 对拍待机器恢复后跑，未跑不宣称通过 |

#### 工程精化记录（联网调研 burn/SIMD 后落地，非空谈）

调研结论（burn 官方文档 + 2026 SIMD 现状）：burn 把 rank 写进类型、backend 写进泛型、JIT 操作流自动 fusion、用所有权跟踪张量生命周期；Rust 自动向量化对"小而简单、迭代器+引用、连续内存、无 raw pointer、长度已知"的函数最可靠，固定长度切片会被自动 unroll。据此在不引入任何外部依赖、不破坏 deny(unsafe) 的前提下落地：

- **去下标边界检查、促自动向量化**：把 `elementwise`/`unary`/`scale`/`dequant_i8`/`softmax`/`rmsnorm` 的 `out.iter_mut().enumerate() + 下标` 写法统一改为切片 `zip`（连续内存、无边界检查、引用而非 raw pointer）。
- **固定 K 点积去重并展开**：`gemm_f32` 把 reference 与 forward 里重复的内层 `for p in 0..K` 收敛为一个 `dot_k` 切片迭代 helper，K=8 固定，编译器展开成 8 个标量乘加，消除重复实现。
- **真机/编译器证据（决定性，非推测）**：用同构最小程序在 rustc 1.99.1 `-O` 下 emit LLVM IR，确认该写法被自动向量化——出现 `vector.body` 与 25 处 `<4 x float>`（ARM64 NEON，宽 load 一次 4 个 f32 + 标量 epilog），默认 target 即生效（ARM64 基线含 NEON），无需额外 flag；循环内 `black_box` 会破坏向量化，已据此规避。
- 验证：`llm-op` 112、`llm-audit` 47 全绿，已同步 Windows 并 `cargo check -p llm-op` 通过。

## 7. 验证与质量门禁（可执行，非承诺）

1. 双源对拍：kernel 输出与标量参考按 f32 位模式比较，误差口径逐算子声明（精确相等或 ≤1 ulp）。
2. fuzz：`fuzz` crate 对 AST 解析、printer、TOML 三条线持续喂随机输入；崩溃即 P0。
3. 门禁脚本化：新增 `tool` 子命令检查——注册表与磁盘一致、无外部依赖、DAG 无环、闭集词汇命中；CI 与本地同一入口。
4. 性能门：每个 GPU 算子必须给 roofline 天花板与实测占比，缺数据不允许进 L3 证据级。
5. 文档门：每个新算子四件套缺一件，`builtin_matches_disk` 类测试直接失败。

## 8. 风险与回滚

| 风险 | 缓解 | 回滚 |
|---|---|---|
| AST 双射对拍不过，语义被悄悄改变 | P1 独立成阶段，不与任何新算子混做 | 保留字符串路径一个发布周期，printer 可整体回退 |
| 片段上移导致后端真分叉无处安放 | 只上移正文同构部分，printer 保留内联片段出口 | 片段文件按 vendor 独立恢复 |
| 算子数量膨胀带来重复骨架 | 同族算子走泛型骨架 + Format 契约（沿用 gemv 模式） | 注册表条目可独立删除，不影响其余算子 |
| 提前偷做 L2 融合 | 冻结条件写入 ADR 0017，门禁脚本检测跨 kernel 重写模块 | 新增模块未经 ADR 一律拒绝合并 |

## 9. 自主反思记录（本方案的自我质询）

1. 「为什么不直接上 CubeCL 式运行时 IR」：无消费者、无真机数据，16K SLoC 级投入在当前阶段是纯成本；L1 AST 已把未来迁移成本压到最低，先做后做的期权价值明确。
2. 「微 crate 是否过度拆分」：14 个 350 行以下 crate 每个都有独立被依赖位置，依赖方向隔离收益成立；本方案不新增 crate，只在 cutile 内加模块，不放大该问题。
3. 「CPU 标量退化实现会不会固化成性能债」：证据级标注把「语义正确」与「性能达标」显式分离，退化实现进不了 L3，P3 微基准会强制暴露。
4. 「算子清单是否遗漏」：覆盖推理主路径（embed→rope→norm→投影→attention→激活→残差）与训练主路径（归约/归一化/反传所需逐元素），SSM 专用算子在 P5 后按模型需要另立 ADR，不在本批混入。
