# 0058：native 微内核单一事实源、`gemm_micro.k` 与静态安全桥

- 状态：**P0／P1／P2／P3 已完成，P4 收口进行中**。FMA／sdot／SME 均已切换生产，
  旧三份算法文件与空 `native/` 目录已删除；不得据此宣称最终 Q3／A3 或全仓完成。
- 证据：P2 的 30 轮统计与 Windows AVX2 复验见 `summary-p2.json`、`win-p2.log`；P3 的 stable
  真机对拍、30 轮旧／新统计和反汇编见 `sme-p3-final-release.log`、
  `sme-performance-p3-final.json`、`sme-p3-disassembly.log`；P4 最新同源复测见
  `performance-p4-final.json`。两个纯转发文件已删除；最终行数预算与全仓门禁仍未签收。
- 日期：2026-11-25。
- 范围：`crates/prof/src/native.rs`、`crates/prof/src/native/`、`crates/cutile/src/native.rs` 及其生产调用面。
- 相关：[0005](0005-annotations-as-contracts.md)、[0022](0022-build-time-corpora.md)、
  [0035](0035-unsafe-ffi-boundary.md)、[0049](0049-refactor-summary-2026-11.md)、
  [0056](0056-cutile-prof-dsl-ir-spp.md)、[0057](0057-test-conventions-naming-and-assertions.md)。

## 1. 决策摘要

把 `prof::native` 的三族主机微内核收敛为：

1. **一个 `crates/prof/src/ops/gemm_micro.k`**：承载 f32 FMA、i8 sdot、Apple SME 三族算法体及共享探针核心。
2. **`crates/prof/src/ops/gemm_micro.h`**：SME 段 FFI 原型的唯一 C 签名事实源；只声明 `cutile_sme_*` 原型，几何常量不抄写进头，由 `kernels.toml` 经构建脚本投影。
3. **现有 `crates/prof/src/runtime.rs` 中的安全桥**：只负责切片校验、运行时特性探测、手写 `extern "C"` ABI 声明、线程／面板调度和公开路径兼容。
4. **现有 `crates/prof/src/ops/kernels.toml`**：统一登记瓦片几何、具名展开和 SME 规模门槛，不新增配置文件。
5. **现有 `crates/prof/build.rs` / `buildops.rs`**：完成模板替换、注解校验和静态 C 库构建，不新增平行生成器。

删除 `prof/src/native.rs`、`native/gemm8x8.rs`、`native/i8.rs`、`native/sme.rs` 和空目录；
同时删除纯转发文件 `cutile/src/native.rs`，在 `cutile/src/lib.rs` 用一行 `pub use prof::native as native;`
保持原公开路径。该行替换原模块声明，不增加转发层或有效行数。

公开路径 `prof::native::*`、`cutile::native::*` 及既有生产调用不改变。
`native` 仅作为兼容命名保留；新文件、配置、注解和内部符号一律按算法命名，不再新增泛化的 `native_*`。

本方案不是把 Rust 文本机械改名为 `.k`。生产环境使用构建期静态链接；现有 C／CUDA harness 继续承担验证，
二者必须调用同一份算法体，禁止出现“测试走 C、生产仍走旧 Rust”的双真相。

## 2. 取证基线

盘点命令：

```bash
wc -l crates/prof/src/native.rs crates/prof/src/native/*.rs
wc -c crates/prof/src/native.rs crates/prof/src/native/*.rs
wc -l crates/prof/src/runtime.rs crates/cutile/src/native.rs
```

| 文件 | 行数 | 字节 | 主要职责 |
|---|---:|---:|---|
| `prof/src/native.rs` | 138 | 6,794 | 面板缓存、线程配置、对齐、几何常量、SME 协商表 |
| `prof/src/native/gemm8x8.rs` | 319 | 11,932 | NEON／AVX2／标量 8×8 微内核、FMA 探针 |
| `prof/src/native/i8.rs` | 244 | 11,351 | dotprod 4×4 微内核、sdot 探针 |
| `prof/src/native/sme.rs` | 376 | 13,922 | SME 整铺调度、ZA 外积、SME 探针 |
| **native 小计** | **1,077** | **43,999** | **4 个文件** |
| `cutile/src/native.rs` | 27 | 不纳入右侧字节小计 | 纯转发，无算法 |

`runtime.rs` 当前 12 行，只是运行时模块入口；最终验收必须把它增长后的实际行数计入，不能把搬家伪装成删除。

其他关联文件：

- `nn/src/kernel/fast.rs` 及 `fast/pack.rs`、`fast/pool.rs` 是打包、扫描、线程池骨架，不是本批可直接删除的微内核文件。
- 既有 `nn` 测试与基准文件继续复用，不新增平行测试套件。

### 2.1 现有优势，必须保留

- NEON lane-indexed FMA、AVX2+FMA、dotprod `sdot`、SME `fmopa` 均已有明确指令形状。
- 生产与探针已强调同源，不能退化为两套算术。
- `first=false` 续算、K 尾段、非整铺、零维、线程重入与面板缓存都有既有语义。
- SME 的规模门槛和整铺限制是实测决策，不能为了“统一”强行让所有形状进入 SME。
- `nn` 调用方当前没有架构门；迁移后仍不得把 `cfg(target_arch)` 或 intrinsic 泄漏回 `nn`。

### 2.2 当前缺口

1. **生产与验证之间没有 C ABI 桥**：仅新增 `.k` 文件不会改变 `nn` 的生产执行路径。
2. **几何常量仍靠 Rust 常量转发**：配置层必须成为唯一事实源，而不是再抄进 C。
3. **Rust nightly／`std::simd` 限制 SME 可见性**：C 的函数级 target attribute 提供候选解法，但必须真机验证。
4. **探针与生产容易漂移**：纯端口、取数受限探针必须复用模板里的同一核心，并核对反汇编。
5. **目录与平铺规则正在并发演进**：实施时以文件内容、生成表和 ABI 测试为准，不依赖瞬时目录名。
6. **原提案仍有命名和规则缺口**：`native.k`／`cutile-native` 语义过宽，多族展开与匿名单锚点冲突，
   纯转发文件仍被保留；这些在本文修订为下面的正式契约。

## 3. 命名与文件契约

| 层 | 正式名称 | 理由 |
|---|---|---|
| 模板正文 | `ops/gemm_micro.k` | 表明 GEMM 微内核族，不再以来源 `native` 命名 |
| 配置族 | `[gemm_micro.fma]`、`[gemm_micro.sdot]`、`[gemm_micro.sme]` | 与模板名同源，族名是闭集 |
| 生产导出注解 | `/* cutile-export: fma */` | 明确表示生成一个 C ABI 导出，不暗示实现语言 |
| 探针注解 | `/* cutile-probe: fma_pure, core=fma_tile */` | 显式绑定生产核心，禁止探针算法漂移 |
| 具名展开 | `/* cutile-unroll: fma_neon */` | 一份多族模板可分别配置，且不与旧匿名写法混淆 |
| 生成 C 正文 | `OUT_DIR/gemm_micro.c` | 与源模板同名，便于诊断和反汇编归属 |
| 静态库 | `cutile_gemm_micro` | 链接名与算法族一致 |
| 内部算法核心 | `fma_tile`、`sdot_tile`、`sme_tile` | 三个生产／探针共享核心使用同一后缀 |

既有公开名字（如 `port_run`、`port_fma`、`micro_packed`）为兼容而冻结；新增内部符号不得再引入
含义不明的 `run`、`do`、`helper`、`impl` 作为唯一语义。`native` 只保留在兼容模块路径和本文标题中。

## 4. 目标架构

```text
ops/kernels.toml
  │  几何 / 门槛 / 具名展开配置
  ▼
ops/gemm_micro.k
  │  cutile-export / cutile-probe / cutile-unroll:<family>[_isa]
  │  fma_tile / sdot_tile / sme_tile 同一算法核心
  ▼
build.rs + buildops.rs
  │  配置校验 → 模板替换 → OUT_DIR/gemm_micro.c → libcutile_gemm_micro.a
  ▼
runtime.rs 安全桥
  │  长度 / alias 校验 → 运行时特性门 → C ABI
  ├──────────────► prof::native::*（原路径）
  └──────────────► cutile::native::*（原路径）──► nn 生产调用

同一份 OUT_DIR 生成正文 ──► 现有 C／CUDA harness、对拍与基准
```

### 4.1 为什么是单文件而不是三份同构模板

三族内核共享同一语义族：输入打包面板、寄存器累加、尾部处理和写回。它们需要的架构指令不同，
但配置、注解、ABI 和探针机制完全相同。一个 `gemm_micro.k` 可以消除三份头部、入口、探针和构建登记。

单文件仍受 600 行硬线约束；若最终超过线，说明抽象或注释失控，应减码，而不是直接拆分文件掩盖问题。
禁止压缩空行、合并多条语句或删除关键证据注释来凑行数。

### 4.2 为什么不新建 `native/` 配置目录或运行时模块

本变更目标是减少文件而不是换目录。配置写入现有 `ops/kernels.toml`，安全桥写入现有 `runtime.rs`，
公开兼容层由两个现有 `lib.rs` 的同名重导出完成。目标是：

- native 原四文件 → 一个 `ops/gemm_micro.k`，**减少 3 个文件（75%）**；
- 再删除 `cutile/src/native.rs`，native＋纯转发文件是 **5 → 1，净减 4 个文件（80%）**；
- 不新增 TOML、Rust 运行时或测试文件；
- 不新增 GPU 后端、JIT 或动态库加载器。

## 5. 配置化与注解契约

### 5.1 配置唯一事实源

在现有 `ops/kernels.toml` 增加受闭集校验的 `[gemm_micro]` 配置族，示意：

```toml
[gemm_micro.fma]
mr = 8
nr = 8
kb = 256
unroll = 4

[gemm_micro.sdot]
mr = 4
nr = 4
unroll = 4

[gemm_micro.sme]
mr = 16
nr = 64
min_elems = 1_048_576
unroll = 1
```

要求：

- 配置键只接受 `fma` / `sdot` / `sme`；每个字段有类型、正整数和合法性校验。
- 未知族、未知键、重复键、零值、负数、超 `usize` 均在构建期失败。
- `min_elems` 表示形状规模门槛，不参与数值语义；其余几何同时决定打包布局和边界检查。
- `.k` 中只能引用生成值，如 `{{FMA_MR}}`；Rust、C、测试和探针不得再写一份数字。
- `unroll=1` 是显式不展开；各族的 `unroll` 只登记在 `[gemm_micro.<family>]`，既有 `[unroll]`
  不再为同一族复制第二份数字。
- `kernels.toml` 在扫描器中始终是调优配置，不得被当成算子声明。

### 5.2 注解最小闭集与具名展开

复用现有文本扫描基础设施，生产与测量标记如下：

```c
/* cutile-export: fma */
/* cutile-probe: fma_pure, core=fma_tile */
/* cutile-unroll: fma_neon */
```

- `cutile-export` 标识生产 C ABI 导出；一个名字在全表只能登记一次。
- `cutile-probe` 标识测量入口，并显式绑定生产核心；`core` 必须指向一个已登记核心。
- `cutile-unroll:<family>[_<isa>]` 标识该族的 K／收缩循环；实际闭集为 `fma_neon`、`fma_avx2`、
  `fma_scalar`、`sdot`、`sme`。FMA 的三个目标共享 `[gemm_micro.fma].unroll`，但注解名不同，
  这样条件编译分支不会被误判成重复。每个名字至多出现一次，配置与标记必须同源。
- 现有算子文件的匿名 `/* cutile-unroll */` 保持原语义；只有 `gemm_micro.k` 使用具名展开，
  不要求批量改写其他模板。
- 未闭合、未知参数、重复入口、缺失核心、探针引用错误核心、无名／重名展开都在构建期失败。
- 不新增 `#[native]` 等第二套 Rust 过程宏，也不复制 `annotation` 的算子语义系统。

### 5.3 条件编译与指令安全

- NEON 是 aarch64 基线路径；AVX2+FMA 使用函数级 `target("avx2,fma")`；SME 使用函数级 `target("sme")`。
- C 侧不自行猜测操作系统特性。运行时探测集中在 Rust 安全桥：
  AVX2+FMA、dotprod、SME 分别经现有探测能力判定后才允许进入对应函数。
- 非目标架构只生成标量或明确的不可用入口，不能因链接进静态库就在启动时执行非法指令。
- 若本机 C 工具链不接受 SME target attribute 或生成错误指令，SME 切片不得签发完成，保留旧路径并停止最终删除。

## 6. 生产静态安全桥

### 6.1 构建与链接

- 使用工作区锁文件中已有的 `cc` 构建依赖；不增加运行时依赖。
- `build.rs` 将校验、替换后的正文写到 `OUT_DIR/gemm_micro.c`，由 `cc` 针对 Cargo target 编译为
  `cutile_gemm_micro` 静态库。
- 构建期由 `diagnostic::scan::ffi_align` 核对三方：`gemm_micro.h` 头原型 ↔ `.k` 注解登记符号集
  ↔ `runtime.rs` 手写 extern 的符号名集与参数宽类（const/mut 指针、整数宽度）；任一侧漂移即终止构建。
  通用对齐内核在 `diagnostic`，prof 侧只保留本仓词法映射；不引入 bindgen。
- 不读取用户环境里的另一份模板，不在运行时编译，不使用 `dlopen`。
- 交叉编译沿用 Cargo target；禁止用构建宿主架构替代目标架构。

### 6.2 安全边界

`runtime.rs` 是唯一新增 unsafe 的区域：

- 公开接口继续接收 Rust slice／既有 `Tile`、`Gram` 和几何类型。
- 调用前验证输入输出长度、形状乘法溢出、对齐和 `first`／尾部条件。
- 默认拒绝输出与输入重叠；若 C 使用 `restrict`，安全桥必须先证明不重叠。
- C 符号保持私有；公开 Rust 调用面不暴露裸指针。
- 运行时特性未命中时走原标量／既有拒绝路径，不执行目标指令。
- 面板缓存的线程私有、只增不减、重入退回临时缓冲语义原样保留。

### 6.3 API 与生产路径

必须保持：

- `prof::native::{gemm_negotiate, with_panels, threads_override, align_up}`；
- `prof::native::{GEMM_KERNELS, Gemm, Native}`；
- `prof::native::{f32, i8_geo, sme_geo}` 的几何常量；
- `prof::native::{gemm8x8, i8, sme}` 及其既有公开类型、函数和特性门；
- `cutile::native` 的现有公开路径；
- `nn` 的生产调用不需要架构分支，也不直接写 extern C。

允许将实现模块内联到 `runtime.rs` 后重导出，但不允许仅删除公开项来降低行数。

## 7. 三族迁移细则

### 7.1 FMA

- NEON 保留 16 个向量累加器与 lane-indexed FMA；AVX2 保留每行一个 8-lane 累加器。
- 标量路径作为其他架构和功能探测失败的完整参考，不得缺失。
- `first=true/false`、K 分块续算、非整铺尾块和 Tn/Nt 布局必须逐项对拍。
- 纯 FMA 与取数受限探针调用同一 `fma_tile`；原 `fma_step` 仅作为迁移前名字记录，
  新内部核心统一为 `fma_tile`，不得复制 16 条算术语句。

### 7.2 sdot

- 保持 4×4 瓦片、K16 步进、水平求和和标量尾段。
- 整数结果要求精确相等；同时覆盖 K<16、K=16、K 非 16 倍数、多行尾块和列尾块。
- 无 dotprod 时必须明确返回 `false`，由既有标量路径接管。
- 探针地址必须随循环推进，防止 load 被整体提升到循环外。

### 7.3 SME

- 保留 16×64、规模门槛、整铺限制、线程上限和 `smstart`／`smstop` 配对。
- SME 的调度仍是安全桥职责；C 核只承担打包相关算法和 ZA 外积，避免把第二套线程池写入 C。
- 目标是从 C 工具链获得稳定的 SME 实现候选，从而移除 Rust nightly 限制；未经 M 系列真机验证不得宣称完成。
- 纯外积与取数混合探针复用同一 `sme_tile` 核心，保留现有“实测否证 2×2 更快”的注释结论。

## 8. 分阶段实施与回退

### P0：冻结基线

- 记录 native、纯转发文件和 `runtime.rs` 的行数、公开项、测试数及发布模式基准。
- 固定 FMA／sdot／SME 的形状表、线程表和探针指令形状。
- 在同一台机器、同一会话保存旧实现二进制，避免只靠旧日志比较。

### P1：生成与安全桥，不切生产

- 加 `[gemm_micro]` 配置、具名展开、注解校验、模板替换和静态库构建。
- 加入长度／溢出／特性门及旧匿名展开的反向测试。
- 先证明 C 核与 Rust 核对拍一致，生产仍走旧实现。

### P2：FMA＋sdot

- 切生产到安全桥，跑 NEON、AVX2、标量与探测失败路径。
- 性能达标后删除对应 Rust 实现；否则回退该切片，不带病进入 SME。

### P3：SME

- 在本机 Apple M5 Pro 上先确认运行时 SME 可见性和 C 工具链支持。
- 跑整铺、非整铺、规模门槛、线程缩放和探针对拍。
- 若 nightly 移除导致能力下降或性能回退，保留旧 SME，整体不得签最终完成。

### P4：收口

- 删除四个旧 native 文件与空目录，并删除 `cutile/src/native.rs`。
- `prof/src/lib.rs` 用同名重导出保持 `prof::native`；`cutile/src/lib.rs` 用一行
  `pub use prof::native as native;` 替换原模块声明。
- 运行全仓测试、格式、clippy、重复率和性能门。
- 实测文件与行数达到预算后才接受“减文件／减码”结论。

每个切片必须可独立回退；禁止留下生产旧实现与新模板长期并存的中间态。

## 9. 质量与架构“两档”验收标尺

该档位只用于本变更，不与其他 ADR 的评分混算。

| 档次 | 质量判据 | 架构判据 |
|---|---|---|
| Q1／A1（当前基线） | 指令与 unsafe 有隔离，但生产桥、配置和探针仍有分散事实源 | 独立模块清晰，但目录、转发和常量层次较多 |
| Q2／A2 | 单一 `gemm_micro.k`、静态安全桥、配置闭集、公开 API 不变，全部常规测试通过 | 文件平铺、一个生成器、生产和验证同正文 |
| Q3／A3（目标） | Q2 全部成立，且各 ISA 真机、反汇编、并发、探针和性能门全部通过 | Q2 全部成立，且文件／行数预算实测达成，无双实现残留 |

**“质量提升两档、架构提升两档”定义为 Q1→Q3、A1→A3。只有最终证据齐全才签发；
写完 `gemm_micro.k` 或本地单架构通过最多算 Q2／A2。**

## 10. 数量与性能硬门

### 10.1 文件与代码

- native 文件：**4 → 1，净减 3（75%）**。
- 加上删除的 `cutile/src/native.rs`，native＋纯转发文件文件：**5 → 1，净减 4（80%）**。
- 新 TOML、运行时文件、测试文件：**0**。
- `gemm_micro.k ≤ 400 行`，`runtime.rs` 最终总行数 ≤200；两者合计 ≤600 行。
- 相对 native 当前 1,077 行，净减少至少 **477 行（44.3%）**；额外删除的 27 行纯转发文件
  不计入该百分比。`runtime.rs` 现有 12 行也不能在对比中重复计算。
- 单文件仍遵守 600 行生产硬线；测试文件不新增、不合并既有断言来凑数。
- 重复扫描不出现三份同构入口、探针算术或几何常量。

### 10.2 数值与能力

- 全量 `llm-prof`、`cutile`、`llm-nn` 测试通过；既有测试数不下降。
- FMA／sdot 标量与 SIMD 对拍通过；整数路径精确，浮点路径沿用既有 IEEE 容差。
- 零维、K 尾、非整铺、续算、溢出、输出 alias、无特性、线程重入均有用例。
- SME 无设备／无工具链时明确保持未验证；不以“编译过”签发 SME 能力。
- 公开项和调用路径逐项对齐，禁止以删 API 的方式完成减码。

### 10.3 性能与指令

- 使用 release，同机同会话旧／新二进制对测；至少 30 次，报告 min／p50／p95。
- 形状覆盖生产形、尾块、线程 1／多线程、SME 整铺大形状。
- **p50 回退不得超过 3%**，且 p95 不得出现无法解释的新离群；否则不合并该切片。
- 反汇编分别确认 `vfmaq`、`vfmadd`、`vdotq`、`fmopa` 存在且目标函数边界正确。
- 取数探针确认 load 未被提升到循环外；纯探针确认环内无 load／store。
- 不以“代码更少”推断“性能更快”，也不以单一最快样本替代中位数。

## 11. 验收命令与命名门禁

```bash
cargo fmt --all -- --check
cargo test -p llm-prof
cargo test -p cutile
cargo test -p llm-nn
cargo test -p llm-nn model_reference_reads_the_contraction_axis_from_the_sample
cargo clippy -p llm-prof --all-targets -- -D warnings
cargo test -p llm-nn --release kernel_fast -- --nocapture
cargo test -p llm-nn --release bench_throughput -- --ignored --nocapture

test ! -e crates/prof/src/native.rs
test ! -d crates/prof/src/native
test ! -e crates/cutile/src/native.rs
test -f crates/prof/src/ops/gemm_micro.k
```

另外必须保存：

1. 文件与行数前后 `wc` 原始输出及最终 `git diff --numstat`。
2. 公开 API 清单和测试数量前后对比。
3. 各 ISA 的发布模式性能原始样本。
4. 目标函数反汇编及探针 load 位置检查。
5. SME 设备、编译器与运行时特性探测结果；缺证据保持 OPEN。
6. 注解名、配置族名、生成 C 文件名和静态库名的一致性检查，禁止残留旧 `native.k`／
   `native.c`／`cutile-native` 契约。

## 12. 非目标与禁止项

- 不重写整个 `nn/fast` 打包与线程池骨架，不把 Rust 调度机械翻译成另一套 C 线程池。
- 不引入 JIT、动态加载、第二套算子 DSL 或新的运行时配置系统。
- 不改变 NVIDIA／MUSA 现有 cuBLAS 与设备内核路径。
- 不用宏隐藏所有循环、压行、删测试、删公开 API 或删必要安全注释来制造减量。
- 不在没有目标硬件时签发“能力最好、性能最优、质量最优”。

## 13. 最终完成定义

只有同时满足以下条件，本提案才可转为“已采纳并完成”：

- 四份旧 native 文件和纯转发文件已删除，只剩约定的一个 `gemm_micro.k`，且预算实测达标；
- 配置、注解、静态桥和生产调用全部贯通，具名展开与旧匿名规则互不回归；
- 所有公开路径、数值语义、尾部与并发行为保持不变；
- FMA、sdot、SME 分别有所需硬件证据，缺项明确阻止最终签字；
- 性能不回退，探针与生产同源且有反汇编证据；
- 全仓格式、测试、clippy、重复率和文档链接门禁通过。
