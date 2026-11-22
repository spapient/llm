# Changelog

本文件的格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]
### Added

- **`crates/rocm`（`ascend`）与 `crates/opencl`（`intel`）的片段层开始有测试：两边从 0 条到 9 + 10 条**。
  此前两个 crate 都是 `autotests = false` 且没有任何测试目标，`cargo test -p llm-rocm -p llm-opencl`
  跑出 0 条——片段正文、见证入口与档案里的 `[[provides]]` 三者之间没有任何机器判据。现在一台设备的
  三张生成表（`FRAGMENTS` / `PRIMS` / `DEVICE`）与它的档案摊到同一份契约上逐条对账：P0 九条不多不少、
  每条签名都是带见证的供给且入口名与 `PRIMS` 一致、每个见证函数被恰好一条签名认领、目录是
  `<prim>.<ext>` + `witness.<ext>` 的闭集、期望值不超回读上限、见证自足（没有引号包含）、每条供给
  至少有正文或见证之一——最后由 `testkit::witness` 把见证真的编成可执行文件跑一遍，与档案里的
  `expect` 对拍（`ascend` 20 条、`intel` 17 条，一条不漏）。判据本身在 `testkit::device`，六台设备共用
  一套，新增设备只加一张 `P0` 清单，判据一行不改；`intel` 另有语言门，用 `clang -x cl -cl-std=CL2.0`
  解析 `witness.cl`（宿主 `cc` 编得过它，只因为它写的是标量 C）。两台设备的用例正文逐字相同，所以
  九条用例只写一次：`testkit::device_suite!(P0)` 把它们摊成 `#[test]` 函数，设备 crate 的
  `tests/fragments.rs` 只剩 crate 文档、`P0` 清单与一行调用（`intel` 多一条前端用例）。
  判据本身只**产出缺陷**（`Result<(), String>`），断言留在用例里——`src/` 上的静态门禁不给
  `panic`/`expect` 留预算，夹具自己 panic 就是它在替调用方决定"什么算错"。
  **验证**：`cargo test -p llm-rocm`（9 + doctest 1）、`cargo test -p llm-opencl`（10 + doctest 1）、
  `cargo test -p llm-testkit --lib -- device:: evidence:: witness::`（13 条自检：`device` 7、
  `evidence` 3、`witness` 3）全绿；`npx --yes jscpd` 对这两个 crate 零 clone。

### Changed

- **不再需要 nightly：默认工具链换成已发布的 stable，clone 即能构建**。`crates/gemm` 的两处不稳定
  能力改为 `#[cfg(nightly)]` 挂起，各配一条语义等价的 stable 退路，由本 crate 新增的 `build.rs`
  （一次 `rustc --version` 判定，产出 `cfg(nightly)`）选择：
  - `gemv::simd` 的 16 车道解包点积 → stable 走新增的 `gemv/simd_scalar.rs`。它**保持相同的分段
    形状**（`WIDE` 个元素一段、`CHAINS` 条独立累加链轮转、段内逐条 `mul_add`），因此每条链累加的
    元素序列与向量版逐车道一一对应，LLVM 在 stable 上也能把这份源码自动向量化；打包格式的
    `admit`/`forward_batch` 在 stable 上恒不可达（`Option<Infallible>`），调用方无改动地落到本来
    就存在的标量叶 `forward_rows_scaled`。两个模块同名同签名，`gemv/f32.rs`、`gemv/i8.rs`、
    `gemv/packed.rs` 一行没动。
  - `fast::sme` 的 ZA 外积 → 整块只在 `nightly + aarch64` 存在，stable 上 `gemm_f32` 直接走 NEON
    通道。模块门控之后块内 26 条 `#[cfg(target_arch = "aarch64")]` 全部冗余，一并删除，
    `sme::gemm` 里那条"非 aarch64 恒假"的分支也随之下线，文件净减 28 行。
  - `benchmarks` 的 SME 端口探针（`port_run`/`port_mix`）随之门控；`sdot_peak` 从 `sme_peak`
    里提出来单独调用，i8 峰值口径在 stable 上照常测量。
  - `rust-toolchain.toml` 的 `channel` 从 `nightly-2026-09-01` 改为 `stable`，工作区
    `rust-version` 从 `1.100`（当时尚未发布）改为 `1.99`，`clippy.toml` 的 `msrv` 同步回 `1.99`
    （否则 clippy 自报"两处 MSRV 不一致"）；CI 的工具链引用除 `fmt` 一格外全部换成 `1.99.1`。
  - 新增 CI job `nightly channel`（macOS / aarch64）跑 `cargo test -p llm-gemm --lib`：默认工具链
    已是 stable，没有这条守卫，上面两条 `cfg(nightly)` 通道会静默腐坏。
  - 构建/测试/clippy/deny/coverage 全部落到 stable。`crates/gemm` 为此净 **+52 行**
    （新增 `gemv/simd_scalar.rs` 80 行、`fast/sme.rs` 411→383 减 28 行——后者是纯减法，
    模块被门控后块内 26 条 `#[cfg(target_arch = "aarch64")]` 全成冗余），已在
    `scripts/loc-budget.sh` 按"（一）本次 +52 /（二）并发会话 +237"两笔分开登记，
    `SRC_MAX` 93800 → 94200（实测 94,089）。
  - **取证**：`cargo test -p llm-gemm --lib` → stable **242 passed**、nightly **254 passed**；
    差的 12 条正是 `fast::sme::tests::*`（5）与 `gemv::simd::tests::*`（7），即 `cfg(nightly)`
    被置上与未置上的恰是设计里那两条通道，而不是"两边都编译、退化无人发现"。
  - **`fmt` 门单独留在 nightly**：rustfmt 的排版随版本漂移，仓库的格式基线是 nightly 的 rustfmt 产出的。
    实测用 stable 1.99 的 rustfmt 重排会把 `device/tests/metal/correctness.rs` 从 800 行展到 807
    （`FILE_MAX` 一票否决）、把 `fast/i8.rs` 的多行布尔条件解析成 `Ccn=7`——是门禁变红而不是更整齐，
    且要为此改三个与工具链无关的文件。故格式门与构建工具链解耦，`fmt` 固定 nightly，
    README 的命令表同步改成 `cargo +nightly fmt --all`。

- **`testkit` 的 L3 跨层对拍按"一个文件答一个问题"拆开，手写 LCG 删除**。`crates/testkit/src/l3.rs`
  （242 行）此前同时住着一个自造 `Lcg`、六个夹具与四类判据，文件头还声称随机源是本 crate 的 `Rng`——
  实际是第二份生成器，与"唯一事实来源"直接冲突。现在 `l3.rs` 只剩引言与三个声明；`l3/measure.rs` 是
  唯一夹具（样本直接取 `crate::sample`、`ε` 由 `weights::quantize` 打包再解回现算、`‖W‖₂ ≤ ‖W‖_F`
  当 `ρ`、前向走 `op::matmul::forward_tn_tiled`）；`l3/bound.rs` 只答"界是否罩住同一条内核的实测
  误差"（单层与整批 `√rows`，参考与压缩共用 `two_layer`，不再各自重跑一遍 ReLU 段）；`l3/ruling.rs`
  只答"真实容器的逐位往返是否为 `Lossless` 的第二项先决条件"。两处纯代数判据（`ε > 0` 必 `Refuse`、
  缺往返证据先于覆盖缺口）与 `weights::l3` 自带单测逐字重复，后者已在
  `a_segment_outside_the_theorem_keeps_the_certificate_partial` 补上 `ruling(false) == Unverified`
  这一行，`testkit` 侧不再重抄。**验证**：`cargo test -p llm-testkit -p llm-weights` 全绿（`l3::` 3 条、
  `weights::l3` 14 条）。

- **`prim` 收成纯机制，六台设备的片段各自成 crate；`prim/src/lib.rs` 440 → 65 行的薄入口**。
  `crates/prim` 原先既是机制又是六个厂家的语料仓库：`musa/` `metal/` `intel/` `ascend/` `cpu/` `nvidia/`
  六个目录、54 段片段、6 份见证、6 个 `build.rs` 全挤在一个 crate 里，`lib.rs` 里躺着装载、布局、内嵌表、
  片段类型四件事。现在一台设备一个成员 crate（`crates/{cpu,cuda,metal,musa,opencl,rocm}`，各自持有片段、
  见证与 `ROOT`/`FRAGMENTS`/`DEVICE` 三个 `pub const`，只依赖 `prof`）；`prim` 里只剩三个文件——
  `fragment.rs`（片段是什么）、`layout.rs`（片段从哪来）、`builtin.rs`（编译进来了什么，吸收原 `tables.rs`），
  外加 65 行的 `lib.rs`：只有 crate 文档、`mod` 声明与 `pub use`，一行逻辑都没有。**公开路径一字未变**：
  消费方照旧写 `prim::Layout` / `prim::FragmentSet` / `prim::builtin`，扁平再导出替它们兜住。顺带删掉
  已无消费者的 `llm-musa` 工具 crate。`vendor` 与目录名不再绑定（`crates/rocm` 装 `ascend`、`crates/opencl`
  装 `intel`、`crates/cuda` 装 `nvidia`），派生的 `devices()` / `device()` / `dir_of()` 仍按 vendor 查。
  **验证**：`cargo test -p llm-prim -p llm-prof`（18 + 57 条 + doctest）、六个设备 crate 的 `clippy
  --all-targets -D warnings`、`cargo build --workspace --all-targets` 全绿；架构门禁 12 道（成员集合、
  边集合与方向、所有权表、错误边界、unsafe 白名单）逐条通过。

- **片段生成器从 `prim` 搬进 `prof::codegen`，`include!` 另一 crate 源文本的修法随之消失**。删
  `crates/prim/generate.rs`（552 行）与 `crates/prim/src/signature.rs`（26 行）：生成器要读 `prof::Device`、
  要校验档案，而六个设备 crate 都依赖 `prof`——让生成器依赖 `prim` 会绕成环，故新家是
  `crates/prof/src/codegen`（`prof` 的 `build` feature 之后，与 `diagnostic::buildgen` 同一套做法），
  签名规则独立成 `prof::sign`（36 行）。`codegen` 按概念分四个文件：`mod.rs` 13 行只做声明、`scan.rs`
  291 行（扫描 + 四条构建期校验）、`table.rs` 100 行（三张表的输出）、`image.rs` 204 行（设备镜像管线）。
  六个 `build.rs` 从"抄一遍扫描逻辑"变成一行 `fn main() { prof::codegen::device("<vendor>"); }`，
  生成结果与旧生成器**逐字节相同**（`FRAGMENTS` / `PRIMS` / `DEVICE` 三张表，只有表头注释指向新位置）。
  顺手修掉一处真实告警：生成器在 `const` 里写 `&'static str`，`clippy::redundant-static-lifetimes` 在
  `-D warnings` 下会红——改在输出侧（`&str`），而不是 `#[allow]`。**验证**：`llm-prof` 57 条 + `--features build`
  的 `clippy -D warnings`、六个设备 crate 重新生成的表与旧产物逐字节对拍、架构门禁的"错误边界不得退回
  `Result<_, String>`"（`Reason` newtype 承载 `Status::Skipped` 的原因文本，不做诊断）。

- **审计门禁的剔除器去掉递归，改成 16 字节一步的位并行扫描（实测 5.0×）**。`crates/annotation/tests/audit.rs`
  的 `code_only` 要把注释与字面量剔掉，否则"被生成的代码"会被当成"生成器的实现"；此前实现是"找到下一个
  注释 / 字面量开头 → 追加它之前那段代码 → 对剩余部分自递归"，栈深等于注释 + 字面量的个数，而判据只是
  一条线性扫描。现在是一条单趟字节游标，每圈 `find` 一次比 16 字节：每个字节道里放一个 `0x01` 走借位链，
  `word ^ 目标` 之后只有"该道为 0"会留下 `0x80`；三个目标（`"` `'` `/`）的命中位图先 OR 再取最低命中道，
  误命中只可能出现在真命中右侧（借位从低往高传），所以最低那一位永远是真的。裸 `/`（除号）与生命周期
  `'a` 这类假候选**就地清位**、在同一窗口里续找，不回外层重开——否则每个除号都要重读一整块，除号密集
  的语料实测慢 4 倍。定性（`Lexeme::of`，只看开头两字节）与量长（注释 / 字符串正文扫描）刻意分成两层：
  把正文扫描内联进窗口循环会抬高寄存器压力，真实语料反而慢三成。**栈深与输入长度无关**，未收尾的注释 /
  字面量只吞到末尾，行注释的换行仍留在代码里（剔除后行数与原文件一一对应）。**实测**（macOS aarch64，
  release，5 次取最小）：20.5 MiB 语料 20.7 ms → **4.1 ms（5295 MiB/s，5.0×）**；真实仓库源码 5.12 MiB
  5.16 ms → **0.98 ms（5.3×）**；合成的最坏形态全部不退化——除号密集 7.29 → 5.01、生命周期密集
  3.90 → 1.44、纯代码 2.99 → 0.55。语料重复到 20 MiB 时递归版要 512 MiB 栈才跑得完，新实现不需要。
  **验证**：五种实现（递归 / 逐字节 `position` / 逐目标取最小 / 切片重开窗 / 分层）在同一份语料上输出
  **逐字节相同**；新增两条用例——十万段字面量（栈深不再随输入增长，旧实现此处是 abort 而不是断言失败）
  与三种未收尾形态（`/*`、普通串、raw string 都只吞到末尾，游标不越界）；`llm-annotation` 43 + 6 条全绿，
  `fmt --check` 与 `clippy --all-targets` 无告警。

- **prefill 投影改走线性层布局的 8×8 内核，实测快 33.5×**：`serve::kernels::project_rows` 在 `rows ≥ op::fast::MIN_ROWS` 时直接调 `op::fast::gemm_f32_tn`（权重按输出行驻留的布局，打包阶段做一次「8 行 × 一段 k」的转置，微内核与「收缩轴在前」的 `gemm_f32` 同一份），不足一个微内核行块时仍走按列 tile（转置成本摊不回来），单行仍走 `gemv`。**实测**（macOS aarch64，release，`rows = 256, n = k = 4096` 即服务端投影几何，5 次迭代均值）：`matmul::forward_tn_tiled` 303.9 ms → **9.07 ms/iter（33.5×）**，最大相对误差 2.0e-7（同为 f32 累加，只是寄存器分块与线程不同）。**验证**：`llm-serve` 196 条（含行批跨 `MIN_ROWS` 的分派与逐行 `gemv` 参考对拍）、`llm-op` 404 条（含 `tn_matches_transposed_nt_bit_patterns`：TN 与「转置后喂 NT 入口」逐位相同）全绿。

- **打包面板不再每次调用重分配 67 MB**：`op::fast` 的 B 面板是 `npad · k` 个 f32，服务端的投影几何（`n = k = 4096`）下正好 67 MB，而 decode 与 prefill 是同一线程反复调同一个几何——这块内存每步都用得上，却被每步丢弃一次。新增 `crates/op/src/fast/scratch.rs`：线程私有的 `RefCell<Vec<f32>>` 只增不减，`with_panels(len, f)` 借出前 `len` 个元素，扩容时只清零新增段（面板的补零列依赖它），借用期间再次借用则退化成一次性 `Vec` 而不 panic。五处分配全部改成写入它：`gemm_f32` 的常驻池路径、`scoped_f32` 兜底、`gemm_f32_single`，以及 `gemm_f32_tn` 的池路径与兜底；`pack_b_f32` / `pack_b_f32_par` / `pack_tn` 随之改成 `*_into(bpack, …)` 只写调用方缓冲、不再返回 `Vec`（并行打包仍按 8 列组切线程）。安全性只靠两条：池里的 worker 拿裸指针写的是各自分片，收回发生在 `with_panels` 返回之前；常驻量等于本线程见过的最大面板，与「每次分配再释放」的峰值一致。**实测**（`m = 256, n = k = 4096`，release，10 次迭代均值）：多线程 13.07 → **11.75 ms/iter**（−10%），单线程 91.1 → 95.0 ms（差异落在噪声内——`vec![0.0; len]` 走的是惰性零页，首次触页成本与复用写一次相同，省下的是 allocator 往返）。**验证**：`llm-op` 404 条与 `llm-serve` 196 条全绿（含 `tn_matches_transposed_nt_bit_patterns`、行批跨 `MIN_ROWS` 的分派对拍），`fast/scratch.rs` 自带三条（两次借用同址、扩容清零新增段、重入换块且不 panic），`cargo fmt --check` 与 `clippy` 无新增告警。

- **`crates/tools` 回到"一文件一概念"：`gate` 拆成阈值与四个分析器，`card` 只留判定，统计与 JSON 转义各收成唯一出处**。
  原先 `gate.rs` 一个文件里躺着阈值（`Gate` / `Metric` / `Over`）、禁止调用形态扫描、markdown 相对链接扫描与
  文件行数棘轮四件事；`card.rs` 里混着均值/标准差/最近秩分位/重采样的实现和一张手写 JSON 转义表，而
  `bench/json.rs` 还各有一份同样的转义。现在 `call.rs`（禁止调用形态）、`docs.rs`（文档漂移）、`dup.rs`
  （单文件行数棘轮）各自独立，`gate.rs` 只剩判据与棘轮；`stats.rs` 收拢「均值 / 标准差 / 最近秩分位 / 对任意
  统计量重采样」，统计量以闭包传入，区间与统计量同坐标；`escape.rs` 是 `card` 与 `bench::json` 共用的唯一
  `quote`。**API 未变**：`llm_tools::{Gate, analyze, parse, strip_tests}` 与
  `llm_tools::card::{Card, Protocol, Tally, Statistic, Standing, Shortfall}` 的签名一个没动，变化的只是
  `gate::call` / `gate::docs` / `gate::dup` 三个子模块上移为顶层模块。`gate.rs` 762 → 459 行，顶层模块由 7 个
  平铺成 12 个，嵌套只有 `bench/json.rs` 一处（写出侧改用共享的 `quote`，删掉自己那份转义表）。
  **验证**：`cargo test -p llm-tools`（74 单测 + 2 doctest + 仓库静态门禁 2 条）与 `cargo test -p llm-testkit`
  （含 `architecture` 12 条）全绿，`cargo clippy -p llm-tools --all-targets` 0 告警。

- **MUSA 测量通道从脚本目录收成成员 `crates/musa`（`llm-musa`），根 `tools/` 随之删除**。原先 `tools/musa-harness/`
  是"生成器（仓库外的独立 crate）+ `build.sh` + `bench_patch.py` + `check_bits.py` + 14 个 `t_*.cu` 实验片段"，
  四段靠人手串、中间产物落固定路径；现在四段各是一个子命令——`gen`（声明 → 协商 → 规划 → 设备源 + 标量参考）、
  `build`（`mcc` 一次编译，转发头随源落地）、`bench`（夹住「首发射 → 末回读」注入 `cudaEvent` 计时）、
  `verify`（逐元素相对误差，一个超阈即退出码 2），`run` 把四段按数据流串起来并把产物落临时目录、跑完删除。
  **生成器不再按算子名分派**：旧实现自己判"逐点 / 归一化 / 张量核特化"三条路再拼 host 驱动，现在唯一形态来源是
  `cutile::execution::launch(op, sample)`，聚合走 `cutile::build::source`，装配走 `engine::program::cuda`——
  这与 `cutile` 的 `[chunk]` / `[grid]` 声明同源，不会再在两处各自漂移。**新增 11 条单测**钉住四件事：
  生成只依赖声明（`Proof::declared`，不需要本机跑过验证）、参考值读回逐位相等、计时注入把发射夹在循环里且
  非整除守卫不被破坏、对拍在长度不等/空输入/超阈/非有限值上的判定。**`tools` 成员因此回到单职责**：只剩
  `lex` / `ccn` / `dup` / `call` / `docs` / `strip` / `gate` / `regress` / `bench` / `card` 这些静态分析与判定，
  不再依赖 `cutile` / `engine` / `op` / `report` 四条运行时边。工具链三条坑（`__MUSA_QY2_MMA__` 宏、mp_22、
  完整头的路径）、精度口径（tf32 是 `1e-3` 级、3xTF32 长 K 超线性）、三路性能对照与探针表迁进
  `docs/MUSA.md`，`docs/README.md` 与根 `README.md` 的文档表各加一行。`ARCHITECTURE.md` 的成员表与边集
  记为工具成员 `musa`（`→ cutile, diagnostic, engine, op, report`），`tools` 的边去掉 `report`。

- **对拍工具（`testkit::compare`）收成一条判据：读数只算一次，分母变成显式策略**。此前同一件事实有三处
  各自实现：`max_rel_err` 与 `assert_within_rel` 各算一遍分母，`engine::metal_tests` 里还带着一份
  per-element 判定——而它们要的其实只是**分母选哪条**。现在唯一实现是
  `rel_err_with(got, want, Scale) -> RelErr { err, scale, worst }`：`RelErr` 只答「整批最大绝对误差 / 分母」
  与这个分母自己的值，`Worst { index, got, want }` 只答「读数落在哪个元素」，`Scale` 只答
  「分母按整批最大幅度（`Batch`）还是每元素留 1 的绝对地板（`Unit`）取」；`max_rel_err` / `within_ratio` /
  `within_rel` / `assert_rel_with` 全部退成它的转发，比较仍只靠 `ratio()`。**阈值没有新增第二张表**：
  `Scale::Batch` 仍与 `cutile::reference::max_rel_err` 同口径、由 `crates/testkit/tests/oracle.rs` 逐位钉住
  （含全零参考、空批、次正规、三类非有限值），引擎真机基准仍用自己的 `1e-3`。**为什么需要两档**：
  `engine::metal_tests` 的输出量级差四个数量级——GEMM 元素是上千项乘积之和（`1e1`–`1e4`），注意力输出是
  softmax 权重乘 V 的和（`1e-4`）；整批分母会把注意力输出上 `1e-5` 的绝对差放大成 `1e0` 的相对误差，
  那不是内核算错而是分母选错（本机真机复现过：两个注意力基准因此变红）。`Unit` 的每元素地板正是原来
  引擎侧一直在用的口径，切回后 `metal_tiled_attention_s512` 与 `metal_flash_attention_long` 恢复绿。
  **顺带两处收口**：`testkit` 里那份与 `cutile::reference` 逐字相同的 `nonfinite_match` 删除——非有限值
  配对改走本文件已有的零容差相等判定（`close_scalar`），全仓复制粘贴复核时它正是唯一的克隆；
  `rel_err_with` 与失败消息按「长度 → 分母 → 单元素读数 → 累加」拆成六个各答一问的短函数，静态质量门里
  本文件零条目（原先 11 的圈复杂度与三处「直接 panic / 取空值」写法一并消失，失败文案逐字未变）。
  **验证**：`llm-testkit` 56 条中 55 条通过（含 `the_two_denominators_differ_only_in_the_floor`：同一绝对差在
  `Batch` 下放大两个数量级、在幅度大于 1 处反过来更严，钉住「两条是不同的问题而不是松紧两档」），唯一红灯
  `every_public_type_is_listed_or_baselined` 指向并行会话新加的 `op` 类型（`Conv1d` / `Conv2d` / `Gru` /
  `Lstm` / `Rnn` / 两族池化 / `SplitMix64`），不在本文件；对拍相关的 21 条单独复跑全绿。
  `llm-engine` 的 `#[ignore]` 设备基准**真机 9/9**（含两个 int8 GEMM 与两个注意力基准）；全仓 `jscpd` 在本文件
  收口后为 **0 克隆**（复跑时新出现的 8 处全在并行会话的 `op/src/{conv,rnn}.rs`，不属本轮）；
  `docs/ARCHITECTURE.md` 的类型表同步为 `RelErr` / `Worst` / `Scale` 三行合一的说法。

### Added

- **脉冲臂补上反传与量化感知训练（`lld.md` 第二十二步之后的执行侧收口）**。此前 `nn::spike` 只有
  前向：`step` / `drive` 能算出脉冲，但发放在阈值处是阶跃、导数在别处为零，照它反传什么也学不到。
  现在三件事各只有一份定义：`Membrane` 顺手记下**发放前**膜电位 `pre`（发放一提交电位就被写成
  `reset`，那个值只能由前向留下，`settle` 因此拆成 `peek` + `commit`），`Surrogate` 给出代理导数
  （`Rect` 面积 1 峰 1/(2w) / `Triangle` 面积 1 峰 1/w，`NaN` 与恰好压在边上都在窗外），
  `Spiking::backward` 沿链给出三条梯度——输出电流（交给上一层）、逐行尺度（`g·acc`）、权重
  （`g·scale·spike`）。**QAT** 由 `Master` 承担：训练留浮点主权重，`view` 只在 `round().clamp(±127)`
  之后交给前向层，`descend` 一步 SGD 只动浮点侧，因此「梯度落在浮点、码位下一拍再取」是结构性的而
  不是巧合。失败一律在首次写入前拒绝：脉冲长度/非三元、输出梯度长度/非有限、缓冲与参数不同形、
  浮点权重非有限（`NaN` 经 `round().clamp()` 会变成一个看着正常的 `i8`）、学习率非正、更新把某条
  尺度推成非正——最后一条在写权重之前验完，避免留下「权重已降、尺度是负的」的半截状态。
  **验证**：`spike::spike_tests` **27 条**（含 2 条 `#[ignore]` 实测探针：既有的 `spike_arm_costs`
  与本轮的 `qat_costs`），覆盖发放前电位、第二步
  换成第二步的值、两种窗的单位面积与窗外为零、解析链梯度、静默输入只让电流有值而尺度与权重为零、
  窗外整层零梯度、六类入口拒绝、取整夹码、训练一步只动浮点权重而码位下一拍才变、走穿量化区间整步
  回滚且一格未写、非有限主权重与零尺度在量化前被拒、非正/非有限学习率被拒。**实测**（`qat_costs`，
  `#[ignore]` 探针：8 维目标行、尺度冻结 1、梯度 `2·(w − target)`、步长 0.25）：量化重建的 MSE 从
  初值 **2.0225** 降到 **0.0725**，16 步即停在这个值上不再下降——那不是没学，是已经贴到 `round()`
  的半格底噪（对称 `i8` 一格为 1，理论底噪约 1/12 ≈ 0.083）。

### Removed

- **一次性的 Burn 运行时对照探针（仓库根 `benchmarks/`）在结论并入 ADR 后删除**。`benchmarks/burn-runtime`
  是为 ADR 0053「与 Burn 的对比口径」临时立的对照夹具：本仓 `llm-side`（路径依赖 `train` / `autograd` /
  `tensor` / `op`）、Burn Flex `burn-side` 与 Burn Metal `burn-side-metal`（crates.io `0.23.0-pre.2`），
  三个包各自带 `[workspace]`，因此 Burn 从不进入本仓依赖图、CI 与 `cargo test --workspace`。四张表
  （CPU 前向、Metal matmul、注意力、量化 GEMM）连同读表所需的环境与方法口径（机器与工具链、Burn 的
  feature 组合、预热 2 轮 + 11 次采样取中位、每轮内层重复次数、两侧各 3 次取中位、输入生成式与
  `head_sum` / `full_sum` 校验口径）已落进 ADR 0053 正文；逐轮原始输出与 2.4 GB 的 `target/` 产物不再
  随代码存放——按 `docs/README.md` 写作规则 3（"删除也是维护"），一次性报告的寿命是当轮评审，结论
  要么进代码与测试、要么进 ADR。ADR 里三处指向该目录的证据指针同步改写（原写作
  `crates/benchmarks/burn-runtime/`，路径本就不对），改成"三套一次性探针"的口径并写明重跑所需的
  输入式与 `llm-engine` 的 `#[ignore]` 设备用例。**验证**：`cargo machete` 此前对
  `benchmarks/burn-runtime/llm-side` 报的未使用依赖随目录消失（现已零告警）；`cargo fmt --all --check`、
  `nextest -p llm-tool -p llm-testkit` 110 条（含静态门与所有权表）全绿。工作区源码一行未动；该目录本就不在
  `jscpd` 的扫描路径（`crates` + `tools`）内，重复率不受影响。

### Fixed

- **`op::fast` 的 i8 常驻线程池在落地同轮要过三处门禁，就地收口**。三处改动都不改一条计算：字段
  `gen` 与 2024 edition 的保留字冲突（整仓 `cargo check` 直接解析失败），改名 `generation`；`part` 原本一位参与者兼做「打包自己那批列组」与「算自己那段行」，拆成 `pack_part` +
  `compute_part`（Barrier 仍夹在中间，两个阶段看到的仍是同一份 `bpack`）；`gemm_i8` 的「单线程」
  与「池不可用时退回 scope」两条路径各自成函数（`single_i8` / `scoped_i8`），于是它的圈复杂度由 7
  降到 3，而 `gemm_i8_single` 与 `single_i8` 不再各写一遍「打包一次、算一整块」。**验证**：静态门
  `every_source_passes_the_static_gate` 2/2 通过、对 `fast.rs` 零条目；`clippy -D warnings` 零告警
  （`spin` 的嵌套 `if` 折成 let-chain、`run_pooled` 的九个散参数收成 `Shape` + 参与数）；`llm-op`
  **336 条**全过（含 i8 分块与多线程的对拍）。

- **七个函数的圈复杂度回到上限之内：一个函数只答一个问题，静态门不再报它们，也没有一条走基线登记**。
  这些函数在合入的分支里就已超标（上一轮如实记为「登记缺口」）：`engine::codegen::metal_benchmark_driver`、
  `engine::cuda_codegen::cuda_benchmark_driver`、`engine::backends::benchmark_grid` / `cuda_bench_geometry`、
  `cutile::bench::run_sample` / `production_lower`、`serve::http::content_length`，病根同型——一个函数里
  同时做「取几何、拼模板、摊读数」。落点：`cuda_bench_geometry` 只按 lowering 分派，三种形态各由
  `pointwise_wrap` / `loop_wrap` / `warp_wrap` 拼自己的参数与网格；`benchmark_grid` 把「网格标量 → 正整数
  维长」交给 `grid_extent`；`metal_benchmark_driver` 的缓冲绑定与回读取自 `bench_output` / `bench_wiring`；
  `cuda_benchmark_driver` 退成「缓冲 → 计时 → 回读 → 释放」四段拼装；`run_sample` 拆成 `prepare_sample` /
  `measure` / `readings`；`production_lower` 先认生产态特例再落 `skeleton_lower`（逐点走 `pointwise_or_grid`、
  其余走 `grid_lower`）；`content_length` 的头行折叠与收尾交给 `apply_header` 与 `finish`。**边界**：生成的
  MSL / CUDA 文本、报错文案与判定次序一行未改。**验证**：静态门 `every_source_passes_the_static_gate` 通过、
  对这七个函数零条目；`cargo nextest run -p llm-engine -p cutile -p llm-serve --all-features` **528 通过 /
  9 跳过**（跳过的是 `#[ignore]` 设备基准）。

- **合入后重新抬头的重复收成唯一出处，全仓复制粘贴回到 0**：`engine::backends` 的 CUDA 与昇腾 host 驱动
  各抄了一遍「逐见证打印位模式 + `witness N`」的回读段，现由 `witness_readback(max)` 生成一份（两段的
  拼接次序、缩进与字面逐字不变）；`engine::metal_tests` 的两个 int8 GEMM 基准（本地生成 / cutile 生产入口）
  共用 `Int8GemmBench`（码表、笔数、A/B 上传、Y 缓冲与 CPU 参考一次给齐），形状常量、标签与断言仍留在
  各自用例里。`backends_tests` 的花括号配平用例顺手守住了这类拆分：收尾段不再经过 `format!`，转义写法
  `}}` 必须还原成单个 `}`。**验证**：`npx jscpd`（`crates` + `tools`，`minLines=5` / `minTokens=80`，阈值 0）
  **0 克隆 / 0.00%**；`cargo fmt -p llm-engine --check`、`cargo clippy -p llm-engine --all-targets
  --all-features -D warnings` 干净，上面那条 528 条用例同时覆盖这两处收口。

- **合入分支后回归的三处入口配置补齐，`cargo nextest run --workspace` 与静态门恢复可跑**。
  `crates/fuzz/tests/fuzz_http_head.rs`（HTTP 头解析 harness）未登记进 `.config/nextest.toml` 的
  `default-filter`，而它的 libFuzzer 主函数不接受 `--list --format terse`，于是 nextest 在**列举测试**阶段
  整体失败（`The required directory "terse" does not exist`），全仓用例一条都跑不起来；`crates/tools`
  里指向已删除临时探针的 `[[test]] zz_probe_tmp` 让 `cargo test -p llm-tool` 直接因缺文件编不过；
  `engine::cuda_codegen` 的 `cuda_benchmark_driver` 文档在合并时把一节 `# Errors` 串进了四段模板清单中间，
  整理回「摘要 → 四段清单 → `# Errors`」的顺序。**验证**：`cargo nextest run --workspace --all-features --locked`
  **2213 通过 / 17 跳过 / 0 失败**（含静态门 `every_source_passes_the_static_gate`）；`cargo clippy --workspace
  --all-targets --all-features -D warnings`、`cargo fmt --all --check`、`cargo test --doc --workspace`、
  `RUSTDOCFLAGS="-D warnings" cargo doc`、`cargo machete` 全通过；`npx jscpd` **0 克隆**（296 文件 / 126 277 行）；
  Metal 侧 22 条设备用例（含 6 个 `#[ignore]` 基准）真机全绿。

### Added

- **36 个算子全部换成并行网格核：Metal 的分块族补齐「多段发射 + 主机折叠」，六族声明补 `[grid]`，真机重跑 36/36
  走并行通道**。此前 Metal 只有单源逐点 / 循环注入两族有驱动，分块族（归约 / RMSNorm / Softmax / 扫描）只生成了
  设备段、没有 host 串联，于是只能回落单线程语义门——**算得对但形态没证**。本轮新增 `engine::metal_host`
  （`xcrun metal` → `metallib` → host 驱动 → 回读，从 1301 行的 `backends` 拆出）与 `engine::metal_pipeline`
  （四族的分段装配与 f64 折叠：`sqrt(Σx²/N + 1e-6)`、`max`/`exp` 求和、`C_{g+1} = a^len·C_g + 尾态`，`a^len` 用
  平方乘），`codegen` 增加多段驱动生成器并把段入口名参数化（`kernel_partial` / `kernel_expsum` / `kernel_norm` /
  `kernel_scan_*`，同一份 MSL 里不能再有第二个 `kernel_main`）。声明侧给 `gemm_i2/i4/i8`、`matmul_i8`、`embed_f32`、
  `rope_f32`、`dequant_i2/i4` 补上 `[grid]`，`rope` / `dequant` 的循环头改成「上界就是标量 + 守卫」，因为网格注入只
  能绑这种循环，表达式上界会让每个线程各跑一整遍把结果写坏（`dequant` 的 `b >= N/cpb` 与 `rope` 的 `2*i+1 >= N`
  守卫把多出来的线程直接送回，奇数长度的尾元素只由 `2*i == N-1` 那一个线程搬运）。**`verify` 新增
  `Channel::Reference`**：C 参考通道的串行是语义的定义，不该被读成「回落」；台账因此分「并行 / 回落单线程 /
  语义基准」三栏，「该形态缺并行实现」不再隐身。**实测**（Metal）：36 个算子全部 `L2` 且通道全为并行，
  `bench --op gemv_i4 --device metal --repeat 50` 连跑三次为 p50 0.049–0.059 ms、177–213 GB/s、带宽达成率
  **57.6%–69.3%**（峰值 307.2 GB/s）——**仍低于 70% 门槛，按 §16 承认这台设备需要特化代码**；单次采样曾到
  0.044 ms / 77.6%，波动说明那是偏乐观样本，不写进结论。`cpu` 36/36 记为语义基准。
  MUSA 那台机器本轮**没有设备**，`verify --devices musa` 按「缺设备即失败」如实缺席，不记为通过；上一轮的
  36/36 真机 `L2`（`max_rel_err ≤ 3.88e-7`）作为历史前提保留在设备表里。
  **验证**：`cargo test --workspace` 全绿（含静态门、复杂度门与架构契约），
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 退出码 0、零 error（`weights::ledger`
  的 `Certified::left/right` 有一条 rustc `dead_code` 告警，属并行改动尚未接线，不在本轮范围）；文档相对链接无断链。

- **验证有了设备通道：声明里降得出并行网格就真上设备跑，降不出才回落语义门，且回落是显式记录而不是
  静默通过**。此前 `Backend` 只有一个单源入口，`verify` 的设备车道实际跑的仍是「一个串行 harness」，
  声明的 `[grid]` / `[chunk]` 在设备上从未按格发射——于是「按行拆线程」「每块一个工作项」这类声明
  只能靠模板文本断言，跑起来还是单线程。本轮把三件事分开：`cutile::execution::launch` 是唯一分派点
  （形态与派发几何都从声明与样本推出，按算子名在装配侧挑驱动会形成第二份分派，声明一改就漂移）；
  `Launch` 把「工作项数」`items`（一维逐点/循环是元素数、分块族是块数、`[grid]` 是各维取值）与
  「每点线程数」`group` 拆成两个字段（此前一个数同时被当作两者，于是 `dispatchThreads` 拿不到总线程数）；
  `Backend` 新增 `parallel`（默认返回 `Unsupported`，即「该后端没有并行实现」）。设备侧实现收进
  `engine::program`：CUDA 按形态装配段数（逐点/循环单段 + 主机侧无折叠、扫描两段、分块归约与 softmax
  带主机折叠）、Metal 按形态给「设备源 + host 驱动」；形态列表仍有 tile / flash / 波内特化——它们各有
  生产态入口，这里**报「没有实现」而不是拿去按逐点 lowering 试一遍**，因为后者会先在签名校验上撞
  `Invalid`，把「缺实现」说成「签名错」，调用方据此无法回落；Metal 侧同一条判据反过来也用得上——零维网格
  没有可发射的线程形状（`dispatchThreads` 收的就是线程数，给 0 既不执行也不报错），报 `Unsupported` 才能
  让调用方回落到语义门，而不是拿一份不发射任何线程的驱动当通过。**验证**：`cutile` 163 条（`execution`
  8 条：一维网格逐维取值、分块族一块一项且不分组、逐点按长度轴一项一元素、串行回落恒 1 项，另有形状与
  见证回读的老用例随文件一并迁出）、`llm-engine` 160 条（`program` 8 条：CUDA 段数与折叠算子、Metal
  线程形状、两族的缺实现回落）。

- **谱隙证书自报口径：`ε` 用哪种方法算的、这张界是 L0 诊断还是 L0' 候选、补偿项误差有没有重复计入**。
  三件事此前只写在 `lld.md` §4.2.1 / §4.3.1 / §4.8 的正文里，证书本身不记，于是「幂迭代只够排序」与
  「候选界不当证书」都只是注释级别的约定。`ledger::Note` 因此新增三个字段：`method`（`EpsMethod`
  闭集 `verified_backward_error` / `verified_residual` / `candidate_only`，`certify` 产出的是中间那条）、
  `bound`（`Bound` 闭集 `l0_diagnostic` / `l0prime_candidate_open` / `none`）、`comp`（只在残差补偿
  模式存在的 `E_comp`，`certify` 的残差以 `f32` 原样落盘故为 0）；合成误差只剩一处算式
  `Note::synth() = E_decomp + E_comp`。`check` 新增四条可执行约束——候选方法签不出 `Pass`、L0' 候选界
  签不出无损、用了秩选择就必须声明界类型、逐位往返则补偿项必须为零（同一份误差记两遍即被拒）；
  `l3::Rung::from_note` 同步拒绝候选口径的证书，它的 `ε` 只够排序、不够当上界；读端 `ledger::audit`
  同样要求落盘证书的方法必须是实测残差、界必须是 L0 诊断、补偿误差必须为零——伪造或手写一个别的
  口径，载会被这条载荷自己拒掉（`the_payload_audit_rejects_a_certificate_that_names_another_measurement`）。`rank` 载荷魔数
  `AX02` → `AX03`（多两码与一个 f64），旧布局不再被当作合法载荷。**验证**：`llm-weights` 199 条
  （`ledger` 16、`l3` 14）全绿，两个新枚举的短名与码位各有用例钉住。

- **L3 合成律落地：`weights::l3` 把 LLD §4.5 的逐段误差合成从「每个算子各自声明」收成网络级可签发
  的上界与适用域**。`Chain::bracket` 一次走两套递归（`a_i = L_i·ρ̂_i·a_{i-1} + b_i`、
  `e_i ≤ L_i·(ρ_i·e_{i-1} + ε_i·a_{i-1})`，后者把 `ε_i` 摊到**未扰动激活**上）给出 `Bracket`；`Ruling`
  的判定次序是「有非零 `ε` 就 `Refuse` → 缺产物逐位证据就 `Unverified` → 有段在定理外就 `Partial` →
  三条都过才 `Lossless`」，于是「压缩有损」与「能否签发整网无损证书」永远是两个问题。`ε` 不再由调用方
  拍：`Rung::from_note` 直接吃谱隙证书的 `Note::eps`（同一次分解的后验残差）；`L` **没有默认值**（1 不是
  "差不多"的默认，没有 Lipschitz 证据的段必须显式登记，否则静默把误差算小）；`ρ̂ ≠ ρ` 分开记，压缩把
  范数放大的那一段不会因为只乘左边而漏掉；闭式只在各级「放大或平移为正」时给出，否则只报递归界。
  **跨层对拍**在 `crates/testkit/src/l3/`（3 条）：4bit 量化后的真实误差（走
  `op::matmul::forward_tn_tiled`，与 `serve`/`train` 同一条内核）确实落在上界内且非零、整批界按
  `√rows` 而非 `rows` 放大、真实容器的逐位往返是 `Lossless` 的第二项先决条件（同样 `ε = 0`，往返不
  逐位即 `Unverified`）。`llm-weights` 199 条（`l3` 14 条）、
  `llm-testkit` 新增 3 条跨层用例。

- **模型存储有了逐张量沉降账本：`weights::settle` 把 LLD §4.7 的"每个张量落在哪个体积、为什么、
  是否无损"从散落的度量字段收成一份由**落盘字节**现算的账，`llm settle --in <容器> [--out <账本>]`
  是它的唯一出口**。`Settlement::settle` 只走已落盘的描述符与解码往返：解码一次取熵与值域、按描述符
  给出实际位宽与原因（引用去重 / 小段 / 轴字典 / 不可压），量化条目按既有 `Quant` 语义判为有损，
  未识别 codec 不计无损——这就是 `Archive::metrics` 里那条 `lossless` 判定的同一份实现
  （`archive::lossless_entry`），容器的两个视图因此不可能给出两种"无损"。账本 JSON 只有一个文本来源
  （`Settlement::render`）：`--out` 写盘与 stdout 是同一份字节（进程级用例钉住逐字节相等），不给
  `--out` 就把 JSON 打到 stdout，本层不猜产物目录旁的默认文件名。12 条 `settle` 用例覆盖体积算式、
  原因闭集、无损口径与坏字节拒绝；`llm settle` 补两条解析用例与一条端到端用例。**同一份口径也接到导入
  侧**：`llm import` 落盘后把刚写出的字节重新读一遍现算账本（`archive::ledger_of` 是两处唯一入口），
  报告里的体积比与 `lossless` 因此是"读出来的"而不是"自以为写进去的"——进程级用例从 `import` 的输出就
  开始核对这两项，`settle` 与 `import` 不可能对同一份字节给出两种结论。**实测**（`.u` 容器
  含两条权重段）：全量体积比与逐张量体积可复核，账本与 `inspect` 报告的无损结论一致。

- **装配层从"一层"走到"一栈"：`nn::hybrid` 新增 `Tower` / `State` / `Kind` / `MemoryStep`，把 K3 的
  3:1 配比（每三个通道记忆层配一个混合层）落成构造条件而不是注释**。`Tower::order` 逐格给出
  `Memory`/`Attend` 的序列，配比不对、没有混合层、记忆层读出宽与主路不等、混合层之间流条数或宽度不同源
  一律在装配期报 `Code::Invalid`。两处结构决定：**记忆状态与 KV 缓存彻底分开**（`State` 只随序列长累积、
  `Latent` 按页账走，于是 `Tower::step` 不需要页号、`write_kv` 不需要记忆几何）；**一次推进要么整栈走一步
  要么一格都不动**（流与记忆都在副本上算、末尾一次性提交，第 5 个记忆层参数错了不会留下前 4 层已写、
  后 2 层未写的半截状态）。记忆层按 `Kda` 的口径**写进去立刻读得出来**，读数只加回主路——同一份增量加给
  每条流会让多流退化成单流。19 条 `hybrid` 用例（一层 7 + 一栈 12）钉住：零记忆且不写时整栈与单独跑混合层
  逐位相同、写一次后当步与下一格都与"只遗忘"分叉、三层各持自己写入的那一行、缓存 `seq` 与元素数不动、
  失败整栈回滚、门宽/份数/几何错全部在第一次写入前拒绝。`ARCHITECTURE.md` 所有权表登记该行。

- **MUSA 后端（MTT S4000 / `mp_22`）从"只有档案"走到真机 `L2`：新增 `crates/prof/devices/musa.toml` 与
  `crates/prim/musa/` 的 9 段片段、17 条见证，36 个算子（全部声明）在真机上编成、执行并与 `op` 的标量参考对拍
  （`max_rel_err ≤ 3.878e-7`，多数为 `0`）**。它是第一台"档案与 `nvidia` 同构、工具链不同"的加速器：`[meta].emit`
  仍是 `cuda`（同一门输出语言），`[meta].tool = "mcc"` 决定命令名与运行时头——`engine` 的 CUDA 后端
  不再写死 `nvcc`，编译命令与 `<cuda_runtime.h>` / `<cuda_fp16.h>` 的转发垫片都从档案派生，于是同一份
  `.cu` 换一台机器只需换档案。峰值不凭记忆：`flops_f32` 与 `bw` 写成可求值 `src`
  （`56*128*2*1.791e9`、`384/8*2*8.012e9`），由装载期判据与 `value` 对齐。**波宽 128 而不是 32**：
  档案 `[simt].group = 128`，`reduce` / `scan` / `shuffle` 的偏移一律从 `threadIdx` 派生，按 32 写死在这张卡上会算错。
  **同时修掉协商的一处真实缺陷**：`cap::Offer::satisfied` 原本要求 `grade ≥ L2` **且**本次有执行证据，
  这让"档案诚实写 `L1`、设备就在本机"永远升不上来（要 `L2` 才能协商，却要协商过才能跑出 `L2`）；
  现在自述模式只认 `L2`、执行模式只认本次跑通的 `(原语, 实现)`，两条各挡一类错。
  **如实报告**：`bench --op gemv_i4 --device musa` 报 p50 14.97 ms/次、0.7 GB/s、2.2 GFLOP/s，带宽达成率 0.1%，按 §16 判“需要特化代码”——P0 片段是单线程 C 语义（`for (n…)` 串行、无线程索引），并行化的真正落点在 engine 的网格核；档案峰值未改小来让比率好看。
  文档同批更新：根 `README` 设备表与 `crates/{prim,prof,cutile}/README.md` 的档案数、见证矩阵、等级口径。

### Changed

- **分支合并带来的重复复归零：`jscpd` 296 文件 / 125 470 行上 0 克隆 / 0.00%，三处同源文本各收成唯一出处**。
  合并进来的代码让重复重新抬头：`engine/src/backends.rs` 的见证驱动与 `engine/src/codegen.rs` 的基准驱动
  各写了一份 Metal host 前导，`engine/src/program.rs` 的 gemv 夹具又与 `testkit` 的共享夹具各写了一份
  `cutile_x_kernel` 声明与样本，`serve/src/http.rs` 的 `exchange` / `exchange_with_timeout` /
  `exchange_closed_write` 是三份只有两个开关不同的往返骨架。落点：**Metal host 前导与一轮调度**收成
  `codegen::host_prologue(entry)` 与 `codegen::METAL_ROUND`，见证驱动只剩 head / tail 两段字面，
  基准驱动的预热与计时共用同一份前导与轮次；**gemv 夹具**由 `engine::program` 以 `#[path]` 取用
  `crates/testkit/tests/support/cutile_x_kernel.rs`，并新增 `SOURCE` 常量让 `Signature::parse` 与待降级的
  内核文本不再各抄一份；**服务端往返**收成 `exchange_with(timeout, half_close, request)`，三个用例只在这
  两个开关上分岔（绑定方式随 `timeout`，半关闭发生在写完请求之后）。行为面未动：生成的 Metal 宿主文本、
  错误文本与返回码、夹具取值（`m = 4`、`n = 2`、`x` 4×1.0、`w` `[0x98, 0x76]`、`scales` 2×0.5、`y` 2×0.0）
  以及三个往返用例的断言逐字保留。顺带修两处存量 lint：`serve::http` 的常量断言改用 `const { assert!(..) }`，
  `cli::lib` 的选项表用 `type PathRow` 替掉内联元组类型。**验证**：`jscpd` 0 克隆，
  `cargo fmt --all --check`、`clippy --workspace --all-targets --all-features -D warnings`、
  `cargo test --doc`、`cargo machete`、`cargo bench --no-run` 全绿；`nextest --workspace` 2188/2189 通过，
  唯一红灯是静态质量门对**合并进来的** `metal_benchmark_driver` / `cuda_benchmark_driver` /
  `benchmark_grid` / `cuda_bench_geometry` / `run_sample` / `production_lower` / `content_length` 七个函数的
  复杂度登记缺口（这些函数在 HEAD 里就已超标、本轮一行未动）。

- **批调度改由 IFB 打包拍板：`serve::Completion` 一轮跑哪些行、跑几行，现在是 `engine::ifb::pack`
  的结论而不是"按准入次序全跑"**。`State::round` 先把本层一次提交的粒度（`PREFILL_CHUNK`）收进
  `Budget`，再把 `Verdict::Packed` 的行原样交给下面两层：上下文候选各算自己分到的那一块、生成行一次
  整批提交。于是"这一轮算几行"只有一个来源——`chunk` 给块尾、`pack` 给配额，推进时取小的那个，不会
  出现"打包说 8 行、推进算 12 行"。三处配套硬约束：**准入按 `min(B_max, T_max)` 挡人**（只判 `B_max`
  会放进"进得来、却永远排不进一轮"的序列）；**提示段没算完就不解码**（历史不完整时解码读到的是零
  填充的"过去"）；**打包拒绝本轮不静默空转**（`OverSeqs`/`OverTokens` 上抛 `Internal`——准入已经挡过
  一遍，走到那里说明预算与准入用了两套上限）。3 条新用例钉住：`T_max = 8` 时 20 行提示按 8+8+4 分
  三轮算完、每个位置只算一次且按序、之后才解码；`B_max` 还空着而 `T_max` 已满时新请求同样 503；
  预算只够一行时一轮只推进一条、另一条留在批里历史不变。

- **可复制片段清零：重复率 0.34% → 0.00%（40 处克隆 → 0），`jscpd` 硬门从 0.48 收到 0**。此前剩下的
  40 处集中在两类：**同文件的测试脚手架**（`engine/src/metal_tests.rs` 5 处、`train/tests/backprop_tests.rs`
  4 处、`autograd/src/replay.rs` 3 处、`cli/tests/run_tests.rs` 3 处、`op/src/attention.rs` 2 处、
  `serve/tests/model_tests.rs` 2 处，以及 `nn/src/kda.rs`、`train/src/optimizer.rs`、`engine/src/tiled.rs`、
  `engine/src/batch.rs`、`weights/src/archive.rs`、`nn/src/hybrid.rs` 各 1 处）与**跨 crate 抄了同一份规则**
  （`tool`↔`weights` 的 JSON 读写、`cutile`↔`engine` 的 `cutile_x_kernel` 夹具、`op`↔`prof` 的标量渲染、
  `serve`↔`tokenizer` 的词表样例、bench↔`train::store` 的链构造、bench 三目标之间的样例张量与合成模型）。
  收口分三层：
  - **规则上移到被依赖的一侧**：字节读取骨架落 `diagnostic::scan::Cursor`（`tool` 与 `weights` 都只依赖
    `diagnostic`，方向单一、无环）；标量渲染落 `prof::prof::scalar_texts`（`op` 依赖 `prof`）；词表样例落
    `tokenizer::fixtures`（`serve` 依赖 `tokenizer`）；`op::same_bytes` 成为逐字节比对的唯一出处，
    `operator::same_str` 那份同体副本删除。
  - **测试夹具只留一份源文件**：`crates/benchmarks` 由四个 target 以 `#[path]`
    取用（这两份此前不在版本控制里，干净检出会因取不到而编不过，现已一并入库）；`crates/train/tests/support.rs`
    由 `src/lib.rs` 以 `#[cfg(test)]` 挂载，测试、`learner` 与 bench 共用同一份建图；
    `crates/testkit/tests/support/cutile_x_kernel.rs` 同时喂 `cutile` 单测与 `engine` 的签名快照用例
    （夹具取值逐字未变）。
  - **同文件内参数化或抽助手**：`engine::norm` 的两副驱动合成一副 `driver(..)` + `Stage` 骨架（RMSNorm
    原先手写的求和改调 `fold_sum`，Softmax 仍是 `partial_pass` / `fold_max` / `fold_sum` /
    `normalized_pass` 的次序）；`metal_tests.rs` 的基准开头收成 `bench_device` / `catalogue_and_dirs` /
    `grid_pipeline` / `AttentionBench` 等助手（无设备时的静默跳过语义与 `eprintln` 文案逐字保留）；
    `optimizer` 的 `advance_columns` / `advance_rows` 合成 `advance(.., Side)`（累加顺序与
    `gram × value` 的乘法次序未改）。
  **边界**：全部是装配、夹具与脚手架的合并，MSL/CUDA 生成文本、数值累加顺序与断言口径一行未动；
  唯一的行为面改动是 `crates/cutile/src/runner.rs` 里一个指向私有模块的文档链接降级为代码体，以让
  `cargo doc -D warnings` 通过。**验证**：`nextest` 2105 通过 / 14 跳过，`fmt --check`、
  `clippy -D warnings`、`cargo test --doc`、`RUSTDOCFLAGS="-D warnings" cargo doc`、`cargo deny check`、
  `cargo machete`、`cargo bench --no-run` 全绿；`jscpd` **0 克隆 / 0.00%**，硬门阈值同步改为 `0`，扫描路径从
  `crates` 扩到 `crates` + `tools`（287 个文件）——`tools/musa-harness/gen-cuda` 那个仓库外独立 crate 的
  生成器此前不在门禁视野内，它与 `cli` 的 CUDA 归约用例各写了一份「分块轴长度 = 签名里唯一非缓冲标量」
  的推导，现由该测试文件内的 `scalar_len` 一处承担。

- **GEMV 的 forward 骨架只剩一副：打包与不打包共用同一份守卫、行切片与遍历，`packed` 退回
  「解包点积 + 两层薄适配」（净减 59 行；`gemv/packed.rs` 146 → 84 行）**。此前 `forward_scaled_rows` /
  `forward_batch_rows`（不打包）与 `packed::{forward_rows, forward_batch}`（打包）是两副并行的骨架：
  同一个「分组推导 → 粒度倍数 → 权重长度 → 标度有限」四连守卫写了两遍（`batch_rows_context` 与
  `batch_packed_context` 各自一份），同一个「输出行在外、输入行在内」的双层遍历也写了两遍，差别只有
  **行宽 `m` vs `m / cpb`** 与一份逐行点积算术。**合流方式**：骨架收一个 `cpb` 参数——「一个权重单位
  里有多少个元素」，打包格式给每字节码数（`2` / `4`），不打包的逐元素格式给 `1`，此时行宽就是 `m`
  个元素、"分组必须是 `cpb` 的倍数"恒真，于是两条格式走同一条路径、按元素类型单态化；四条守卫收成
  `scaled_row_context` 一处，单行与行批只差先判几何。`packed` 只留 `row_dot(cpb, dequant)` 这层适配
  （骨架给 `group`，解包点积还要 `group / cpb` 与码表）与两个转发。**报错次序归一为「先判结构、再判
  取值」**：权重长度先于标度有限，两处都是 `Code::Invalid`，且原打包路径本就是这一次序。
  **边界**：解包位域、分组推进、累加顺序与行批嵌套一行未动，批结果仍与逐行调用逐位相同；
  `op` 326 条 + `train` + `testkit` 共 548 条用例、clippy `-D warnings`、doc 全绿。

- **`op` 的 GEMV 量化路径收成一条格式契约：`gemv.rs` 与 `gemv_i8` 的公共常量只剩一份，三档量化的
  「非有限标度」与行批判据各由一条宏生成（净减 43 行；重复率 0.36% → 0.34%，`gemv` 相关克隆清零）**。
  此前 `Packed` 与 `SymI8` 各自手抄同一张操作数表与同一组粒度常量（`GRAIN` / `DERIVED` / `SCALES` /
  `STEP`），两份 `DERIVED` 的除数与 `GRAIN` 靠人工保持一致——正是 `derived_matches_grain` 要防的那种
  漂移（代价按一个分组算、样本按另一个分组给，对拍比的却是两种语义）。**差异收成一个函数 + 一次展开**：
  新增 `gemv::scaled_operands(bits)` 给操作数表（权重按位宽读，输入/标度/输出恒 32 位），
  `scaled_format!(bits, grain)` 展开那五个常量，两条格式只传「权重位宽 + 粒度」两个数。**测试侧**：
  `non_finite_scale_test!(bits)` 与 `packed_gemv_batch_tests!(…; cpb = 1)` 让 i2/i4/i8 共用同一份
  守卫与行批判据（`i8` 的行批用例本是宏展开结果的手抄版），`Rng::signed_unit` 收掉 `op` 与 `testkit`
  各写一遍的 `[-1, 1)` 取值闭包。**边界**：纯装配与夹具收敛，解包、分组、累加顺序一行未动；
  `op` 326 条 + `testkit` 37 条用例、clippy `-D warnings`、doc、重复率硬门全绿。

- **六平台执行合同落成裁决层（`engine::platform`，ADR 0052）+ `llm platform` 子命令**。此前「六平台」只
  在文档里是一张表，代码里找不到「这台机器能不能跑这一行」的答案：`backends` 探的是编译器在不在、
  `metal` 探的是设备在不在，两件事分头回答，于是「没有设备」与「没有工具链」在调用方看起来一样。
  - `Row` 是六行闭集（`Cuda` / `RocmHip` / `Metal` / `Gpu(Vulkan|Sycl)` / `X86-64` / `Arm64`，设备行在前），
    一行只有**集齐设备与编译器两件证据**才算 `Covered`，否则 `Open` 且必带理由——少一行会让「未覆盖」
    与「根本没问」变成同一种输出，所以 `audit` 恒出六行，`require` 只对点名的行报 `E-PLAT`。
    `Why::{NoDevice, NoToolchain, Uncompiled, NoRow}` 是闭集：换机器与装编译器是两个不同的修法，不合成
    一句自由文本。判定只看**这一行能不能跑**，不代替 `cutile::verify`（算得对不对）与 `tool::card`（够不够 Must）。
  - `Machine` 是唯一的主机事实源（`Arch`/`Abi` 编译期常量，`Metal`/`Cuda`/`Rocm`/`Vulkan`/`Sycl` 运行期探针，
    逐项缓存）；`Host` 在运行期提供同样的六行，于是「探针看到的」与「文档说的」不能再漂移。
  - CLI 新增零选项命令 `platform`（`cli::platform` 只排版，判定全在 `engine::platform`；多给一个词即报错）；
    `cli::entry` 顺手抽出 `entry_compute`，把「读权重但不写状态」的 `status`/`run` 归到一处。
  - **实测**（macOS aarch64，`cargo run -p llm-cli --bin llm -- platform`）：`metal covered`（`xcrun metal@32023.922`
    · simdgroup matrix + 同构 ABI）、`arm covered`（aarch64 · NEON/SME 有序点积）、`cuda`/`hip`/`gpu`/`x86`
    全部 `open` 带 `no-device`——本机没有的**照实写没有**，不拿「未测」冒充「不支持」。
  - 文档：新建 `docs/adr/0052-execution-platform-contract.md`；`docs/ARCHITECTURE.md` 所有权表新增该行、边表
    `engine → prof` 补齐，同时从 `train` 行去掉只活在注释里的 `annotation` 边（110 条，架构门按 Cargo.toml
    重算后一致）；补 `crates/{engine,nn,serve}/README.md`——静态质量门要数文档条数（下限 30），此前树里
    只有 27 份，这三份既是门禁口径也是三个千行级 crate 的入口。
  - **同轮收口**：训练侧夹具 `tests/support.rs` 改由 crate 根按 `#[path]` 挂一次（内联测试模块的 `#[path]`
    基准是虚拟目录，跨级上跳编不过）；`train::store` 的 `QuantChain` 只在测试构建里存在——零生产消费者，
    四个只被自身测试读取的转发访问器一并删除，一个不进发布二进制的夹具不该占公开面；`diagnostic::scan`
    的游标 doctest 改为真的走到末尾（原例的 `"  {}"` 里 eat `}` 本会成功）；`lints/vocabulary.toml` 删掉
    过期的 `wire` 所有权基线条目、`train` 的公开面换成实际存在的 `OperatorSetOps`。工作区 2142 条 +
    doctest 44 条全绿。

- **`op` 的收缩轴可变 GEMM 收成一副骨架：`matmul.rs` 与 `matmul_i8.rs` 两份同构分块代码合成一份
  （净减 85 行、少 1 个文件；重复率 0.41% → 0.36%，`matmul` 相关克隆从 3 处 58 行清零）**。此前
  `f32` 与对称 `int8` 两档各写一遍 `matmul_scalar` / `matmul_tiled` / `accumulate_tile` / `check_dims` /
  `cost` / `sample` 与两条 `forward`，jscpd 报出 25 + 23 + 10 行逐字重复，差别只有「元素类型、累加器、
  一次乘加」。**差异收成一个契约**：新增 `matmul::Precision`（`type Acc`、`const NAME`、`fn mac`），
  `i8` 档的 `i32` 精确累加就落在那一次乘加上——遍历次序怎么换都不改结果，于是两档共用同一份标量参考、
  tile 遍历与尺寸校验，按元素类型单态化成两个入口；加一档精度只需再写这三件事。**文件数**：
  `op/src/matmul_i8.rs`（317 行）作为 `pub mod matmul_i8` 并进 `matmul.rs`，模块路径
  `op::matmul_i8::*` 不变（注册表与注解本就按模块扫描），`matmul.rs` 落 886 行，低于
  `file_max_lines = 1200`。**边界**：纯合并，算术与调度一行未动；`op` 与 `train` 511 条用例、
  `testkit::contracts` 6 条、clippy `-D warnings`、doc、重复率硬门全绿。

- **ADR 从 49 个文件（48 个编号）收敛到 15 条（`docs/adr` 278 KB → 100 KB，净删 34 个文件）**。判据只有一条：
  这条决策的结论**是否已被代码或门禁接管**。沿用旧链接就能查到的"显式层表""一个概念一个主人"
  "代价来自求和表达式""层下界由语义推出"等，已经被 `architecture.rs`（成员 / 边 / 所有权三向比对）、
  `contracts.rs`（闭集 / 位集 / 注解镜像）与各 crate 的 `//!` 文档逐条守住，写第二遍只会漂移；
  其余两类本来就不该占一个编号：只承载历史的重构纪事（`0049` 之类），以及把当轮评审塞进 ADR 目录的
  非决策文件（`0035-review.md`）。**留下的是仍然会被人问"为什么"的那几条**：`0005` 注解即契约
  （无运行期注册表）、`0017` AST 冻结、`0018` 精度单一事实源、`0022`/`0023`/`0048` 配置层归属、
  `0025`/`0027`/`0035`/`0050`/`0051` 边界与基座、`0043`–`0046` 运行时形状。**引用方同批改完**：
  `crates/serve`、`crates/nn`、`crates/op`、
  `crates/train`、`crates/weights`、`crates/cli`、`crates/storage`、`crates/tensor`、`crates/cutile`
  文档与 `tools/musa-harness` 里指向退役编号的链接改为指向 `docs/ARCHITECTURE.md` 的对应节或删去
  括号注；`docs/README.md` 的索引只列现存 15 条，并写明旧编号只作历史前提出现。

- **文档收敛为四类活文档：`README` 与 `docs/ARCHITECTURE.md` 重写、文档地图新增、一次性报告退场
  （`docs/ARCHITECTURE.md` 94.8 KB → 32.2 KB、根 `README` 13.4 KB → 9.3 KB，净减约 2/3；写作规则从"评审清单"变成代码判据）**。此前同一份事实
  有三处副本：根 `README` 复述层表与算子数（"22 条"已过期）、`ARCHITECTURE` 把所有权表按"新增轮次"拆成
  两张（P0 与运行时各一张，读者要跳读）、`docs/` 里散落着当轮评审报告与任务分解。**一份事实只写一次**：
  成员数、边数、算子数只在 `ARCHITECTURE` 出现，其余文档引用不复述；所有权表按主人分组、一张表到底，
  合并同名概念的不同轮次条目（`engine::Executor` 与 `LocalExecutor` 基座、`train::Batch` 与 `engine::Batch`
  等相邻项各自注明"同名不同物"）。**门禁先于文字**：新表把漏登记的 `nn::Spiking` / `nn::Membrane`
  （`src/spike.rs`，脉冲前向执行侧）补进表，`lints/vocabulary.toml` 里因此过期的 `diagnostic:Value` /
  `weights:Value` 基线条目一并删除——`architecture.rs` 的 8 条与 `contracts.rs` 的 6 条用例全绿。
  **新增 `docs/README.md`**：文档地图（谁回答"是什么 / 允许什么 / 怎么实现"）、五条写作规则（能守住的写进
  测试、守不住的写进 ADR；一次性报告不长期存放；删文档同一次提交改完引用方）与 48 条 ADR 的主题索引。
  **路径**：`README.md`、`docs/README.md`、`docs/ARCHITECTURE.md`、`.config/lints/vocabulary.toml`；
  `docs/lld.md` 与 `docs/adr/` 未动——前者是冻结规格、章号与行号被代码与外部计划按引用锚定，后者只追加。

- **`op` 的算子契约只剩一个生成点：9 份 `impl Operator` 收成一处，`train` 适配层的 11 处手写转发收成 3 条装配规则，
  `operator_tests.rs` 并回模块（净减 149 行、少 1 个文件；重复率 0.43% → 0.41%）**。此前同一个「名字 + 生产形状 +
  代价/样本/参考」的 impl 被抄了 9 遍：`unary_f32` / `binary_f32` / `reduce_f32` / `dequant_packed` / `gemm_variant` /
  `gemv_packed` 六族装配宏各内嵌一份，`gemv_i8` / `gemv_f32` / `dequant_i8` 又各手写一份，彼此只差模块前缀。
  **契约一份**：`operator::operator_via!` 成为唯一生成点，四条行为由调用方给四个入口路径（族骨架取 `$crate::$skel::*`
  的泛型入口，模块自带入口取 `self::*`），粒度按 `grained` 形态从 `Format::GRAIN` 求或显式给；`operator_here!`
  退化成一次转发。加精度变体不再有第二处语义表要同步——`gemv_f32` / `gemv_i8` 那两份手写 impl（23 行逐字相同，
  重复率硬门原本就报了这一处）由此消失。
  **适配一条规则**：`train::kernel` 的 `gemv_i2/i4/i8`、`dequant_i2/i4/i8`、`attention_f32` + `flash_attention_f32`、
  `decode_attention_f32` + `decode_flash_attention_f32` 是同一段「拆视图 → 转发」抄三遍/两遍，收成 `quant_gemv!`、
  `dequant!` 两条宏与 `attention_paths` / `decode_paths` 两个公共几何入口——两个注意力版本只差一个函数指针。
  **文件数**：`op/src/operator_tests.rs` 并回 `operator.rs`（265 行）。测试外移的动因是 `file_max_lines = 1200`，
  把一个百行文件的测试切出去只是多一个要打开的碎文件，不是这里要防的东西。
  **边界**：纯装配，算术与调度一行未动；`nextest` 功能用例全绿，clippy `-D warnings`、doc-test、重复率硬门同绿。

- **SpikingBrain 2.0 的脉冲臂落地：int8 权重 × 三元脉冲的整数累加 + SSE 事件路径，两条路径逐位相同
  （`nn` 13 条新用例 + 1 条 `#[ignore]` 实测探针）**。此前这一款只有神经元原语（`mix::Lif`）与事件编码
  （`mix::Encoder` / `mix::Events`），缺的是「一层脉冲线性怎么算」——于是十一款标杆里这一款只有零件、
  没有执行路径。
  **一层脉冲线性**：新增 `nn::Spiking`（参数：i8 权重 `[outputs, inputs]`、逐行反量化尺度、LIF 动力学）
  与 `nn::Membrane`（状态：每通道膜电位 + 复用的整数累加缓冲）。一步 = 三元脉冲累加进 `i32` → 按行反
  量化 → 交给 `Lif` 推进，发放判据仍只由 `Lif` 定义，这里不写第二个阈值。缓冲留在状态里而不是每步
  `vec!`：`drive` 的稀疏性不该被一次分配抹平。
  **两条路径共用一份算术**：`step` 吃稠密三元脉冲、`drive` 吃 `Events`（SSE 只记发放位置）——后者取事件
  码的符号后落进同一个 `fire`。`i8 × {−1,0,+1}` 是整数加法，所以「编码后 `drive`」与「解码取符号后
  `step`」逐位相同（用例逐位比对膜电位与输出，不是近似相等）。差别只在索引：`step` 按宽度逐个看、
  `drive` 只走发放记录，因此「省了多少乘加」由 `Events::sparsity` 给，「省了多少遍历」才是墙钟之差。
  **溢出是可证的而不是靠饱和分支**：`|w| ≤ 127` 且构造期要求 `inputs ≤ i32::MAX / 127`，每行至多累加
  `127·inputs` 次、恒在 `i32` 内；放行更宽的输入会让「结果对不对」取决于宽度，所以在入口就拒。
  码位门槛决定稀疏度、与 `scale` 解耦，换步长不会顺手换掉稀疏度。
  **状态不可跨层**：`Membrane` 只能由 `Spiking::membrane` 构造，输出通道数不同的层在入口被拒——与
  `Kda::memory`、`Mamba::Carry` 同一口径。
  **实测**（`cargo test -p llm-nn --lib --release -- spike_arm_costs --ignored --nocapture`；1024 → 1024，
  固定种子权重与激活，码位门槛 3）：稀疏度 **S = 0.6182**（active 391/1024），一次前向的乘加
  **1,048,576 → 400,384（0.3818）**；同一份整数算术下，事件路径的墙钟是稠密路径的 **1/3 左右**
  （两轮 2.95× / 3.53×，机器相关）。这与 SpB2.0 公开的 64.31% 稀疏同量级，但**不是同装置对照**——
  本层实测只证明「稀疏度由码位门槛决定且可测」，E14 的判决仍归协议卡；E13（能效 ≥ 6×）要目标设备。
  **边界**：STE 代理梯度 / 脉冲反传不在本层（`train` 侧无此路径），层配方、批调度与设备后端也未接；
  `docs/ARCHITECTURE.md` 所有权表新增该行，写明与 `Calib` / `Lif` / `Encoder` / `Events` 那一行的分工。

- **`engine` 侧三处结构性收口：生成器不再逐个透传派发几何，两个超长源文件的测试段移出（1302 → 525 行、
  1215 → 808 行），静态质量门与文档门重新转绿**。此前 `tiled::flash_attention_msl` / `flash_acc_update`
  各自接 4 个几何标量，每个生成函数都重算一遍 `div_ceil` 与「除不除得尽」；`metal.rs` / `backends.rs`
  连同内联测试一起超过 `file_max_lines = 1200` 的上限，`tiled.rs` 又把 `"}"` 当字符串 `push_str`。
  **生成器**：派发几何收成 `Flash { seq, dim, threads, block }` 并带 `lanes`（每线程寄存器输出维数）与
  `tail`（键块除不尽序列才需尾块守卫），四个显式参数不再沿生成函数逐个传递，`push_str("}")` 改
  `push('}')`；同时修掉 `[32,1024]` 被当成 intra-doc 链接导致的 `-D warnings` 文档门失败（改
  `` `32..=1024` ``）。
  **测试归属**：`metal` / `backends` 的 `#[cfg(test)] mod tests` 按仓库既有惯例移成同目录
  `metal_tests.rs`（782 行）/ `backends_tests.rs`（407 行），两个源文件回到 525 / 808 行——门槛要的是
  「源文件别无限长」，把设备 FFI 与生成器的正文与它们的用例分开读，比给阈值加豁免更贴原意。
  **边界**：纯搬迁与去重，无行为变化；Metal 设备用例仍按「无设备则跳过而非失败」跑，E1–E15 无新增实测数。

- **服务三轴收口：排批预算真正决定每轮跑哪些行，投机解码补上草稿器（`engine` 4 条 + `serve` 8 条新用例）**。
  此前 `ifb::pack` 只做「这一轮塞得下几行」的纯函数判定，`Batch` 仍按准入顺序整批推进；投机只有
  「接受几个」的判定，草稿从哪来没有答案。
  **排批**：新增 `Batch::round(budget)`，把批内序列按自身事实翻成候选——`prefilled < prompt_len` 即
  上下文（块大小由 `ifb::chunk` 按页对齐给出）、否则生成（一轮一行）——再交给 `ifb::pack`。于是
  「先跑谁」不再是调度参数而是一条算式：挤压时只挤上下文、生成行一条不丢；`chunk` 返回 0（剩余
  不足一整页）的候选不进结果，调用方不必再判一次「这行要不要跑」。两类装不下分开返回：序列数超了
  可等一条结束（`OverSeqs`）、生成行自己超预算重试无用（`OverTokens`）。
  **草稿**：`serve::speculate::Drafter` 取已算完历史里**最近一次**出现的相同后缀、续写其后最多 `max`
  个 token；后缀没出现过就给空草稿，而不是拿最后两个 token 抄一遍——空草稿只让这一步退化成常规
  前向，抄出来的草稿要花一次真实前向去否掉。`Verdict::produced` 给出这一步真正走出的 token（被接受
  的前缀 + 第一处不一致的补位），`Drafter::walk` 把「草稿 → 校验 → 产出」串成一步，行数与草稿对不上
  时按 `Code::Invalid` 拒绝。`order < 2`、`max = 0` 在构造期拒绝。
  **边界**：本步给的是排批口径与草稿来源，不训草稿模型；E3 ≥ 12× 仍无实测数。

- **谱隙证书随载荷落盘，读端用载荷自己复核它（`AX02`；`weights` 172 + 16 + 3 条全绿）**。此前
  `ledger::certify` 的判定只活到写端为止：秩被写进 `AX01` 的 `rank` 段，而「凭什么这么压」那张
  `Note` 写完即丢——于是「可证压缩」只剩写端的一句声称，读端无法判别一张载荷是过了门还是绕过了门。
  **布局**：`rank` 段改为 `note_len | note | low_len | low | rest`，`Note` 与两段残差同受一条 CRC
  保护（魔数 `AX01` → `AX02`，旧的裸秩布局不再被当作合法载荷）；秩不再是载荷里一个孤立数字，它
  与模式、稳定性、谱隙、扰动、截断误差、逐位往返同处一张证书。
  **读端复核**：新增 `ledger::audit`（与 `certify` 同一文件、同一组字段），把能从载荷重算的断言
  全部重算——`decomp` 必须等于 `rest` 的 Frobenius 范数、`used` / `Residual` / `Pass` 三条同时
  成立、秩落在 `1 <= r < min(m,n)`、两段等长；容差按 `f32` 往返上界取
  `1e-6 · (|decomp| + ‖W‖_F)`，不把舍入当篡改。写端伪造一个更大的秩就得同时伪造谱隙，而谱隙那
  一条在读端会被重算；`equal` / `lossless` 不在复核范围（读端没有原始矩阵，那两条只能由写端的逐位
  验证承担），这条边界写进了文档而不是留给读者猜。
  **只读通道**：新增 `axes::certificate`，不解码权重也能拿到同一张 `Note`（与 `decode` 读的是同一
  个 `rank_head`，所以“证书长什么样”只有一种解析）；`decode` / `search` 的 `AxisCfg::decode` 改为
  接收形状而不只是元素数——复核「秩落在可判域」要用原始 `(m,n)`，元素数已经把它丢了。
  **测试**：新增三条——证书随载荷可取且与写端那一张相等；按字段改写 `decomp` 或把 `stable` 翻成
  `Fail` 都被拒（这两处单看证书自身的 `check` 查不出，正是“两方对质”才成立的判据）；旧裸秩布局
  不再可读。**E8 仍 OPEN、Must 不降**：本步补的是证书账本这一段，`scripts/train_own.py` 退役与
  `config/train/e8_own.json` 删除导致的不可复跑没有因此恢复。

- **条件记忆层接进服务路径 + 2–8bit 自适应组量化（E5 的「自适应策略」有了可复跑探针：30–50 dB
  预算全组落到 2bit、bpw 2.000、权重空间 SNR 61.91 dB）**。此前 `nn::engram::MemoryBlock` 只有契约
  与纯函数，`weights::ptq` 也只有固定档位——「层能装配吗」与「每组该用几 bit」两件事都停在机制层。
  **条件记忆层是第七个层族**：`serve::plan` 新增 `LayerKind::Memory` 与 `MemorySpec`（表条目、阶、
  buckets、gate 在规划期就定死），`assembly::NodeSink::push_memory` 出三节点旁路（查表
  `[槽表, 驻留表] → [width]` → 门控缩放 → 加回主链），`MemoryBinding` 记该层的槽表与门控两张**批
  表**；`model_host::fill_memory` 每步按历史（与 `nn::engram::slot_of` 同一份槽映射）覆写这两张表而
  图结构不变，权重复用与其它层同一份 `Params` 台账。门是**乘**不是选路，γ = 0 时加回零向量、主链
  逐位不变，与 `Bypass` 返回「这一行没有记忆」同一结果；三条用例分别钉住「记忆层不能开栈」「宽度
  不符在装配期被拒」「历史达到阶之后门控行只加一次（此前一步加两次也能对上宽度）」。
  **2–8bit 自适应组量化**：`weights::ptq::adaptive` 新增 `Adaptive` / `AdaptiveCompensated` /
  `adaptive_quantize` / `compensate_adaptive`。候选位宽闭集 `{2,3,4,8}` 只是对 `Scheme` 的索引表，
  码位与打包仍只走父模块那两份 `quantize` / `dequantize`；降档是**误差驱动贪心**（所有组从 `I8`
  起，每步把「每省 1 bit 增误最少」的组降一档，直到再降就越 Frobenius 预算），产物是逐组位宽图 +
  平均码位 + 是否达标 + 重构。`met = false` 时位宽图照样给出（调用方据此拒压，而不是拿到空结果）；
  误差是解析估计 `n·scale²/12`，所以**零预算下没有一档算达标**——把它读成「8bit 无损」会把「停在
  最宽档」与「达标」混为一谈。补偿形态与 `compensate` 同一条低秩分解（`decompose` / `signal_rank`
  已收成共用实现），差别只在残差走自适应位宽。
  **E5 探针**：`ptq::adaptive::e5_width_map`（`#[ignore]`，命令写在用例文档里）在 512×512、rank-8
  加噪声的**合成**矩阵上（E5 明确禁止拿合成矩阵代替模型级结果，故只作机制探针、不出判决）实测：
  30 / 40 / 50 dB 预算下 8192 组**全部**停在 2bit、bpw **2.000**、权重空间 SNR **61.91 dB**；60 dB
  预算下 418 组升到 3bit、bpw **2.051**、SNR **62.45 dB**，秩 8、截断处谱隙 2.089e2、`sigma1`
  312.3——低秩项之后残差本就很淡，所以 2bit 就够，这条单调关系与预算方向一致。
  **规范**：静态门四条圈复杂度红灯按「一个助手只答一个问题」拆开（`compensate_adaptive` →
  `validate` / `adaptive_for` / `assemble`，`plan::view_plan` → `match`，`assembly::push_memory` 8 与
  `advance_layer` 7 → `memory_slots` / `emit_memory_embed` / `planned_bypass` / `memory_entry` /
  `attach` / `push_single`）；四个大件的测试按仓库布局外移成同目录文件（`annotation/codegen_tests.rs`
  403 行、`cutile/ast_tests.rs` 131 行、`op/{softmax_f32,rmsnorm_f32,embed_f32,rope_f32}_tests.rs`
  374 行，嵌套 inline 模块里改用 `include!` 相对源文件解析）；`train` / `serve` 两份逐字相同的
  `kernel_table!` 收口到 `op::kernel_table!`（宏定义 54 → 53 处，`Entry` 按调用点解析，字段契约
  不变），各表仍各留自己的名字/顺序断言。

- **能效臂补上事件侧，并把 E10 / E12 从「待测」变成可复跑的探针（实测 CV 1.23%、查表 0.35 ns
  与历史长度无关）**。此前三条能效维（E13 / E14 / E15）只有 `Lif` 这个神经元原语——γ 通道的
  非恒等形态一直没有对应的编码算子，「事件驱动省在哪」没有任何可测物。
  **事件臂（E14 / E15）**：`nn::mix` 新增 `Encoder` / `Events` / `Audit` / `steepest` /
  `accumulate`。`Encoder` 只持步长与**码位**门槛——门槛按码位而不按浮点值，于是换一个步长不会
  顺手把稀疏度换成另一个数（门槛取 0 会让每个位置都发放，事件臂退化成"带量化误差的稠密臂"，
  所以在入口就拒绝）；`Events` 只记发放位置与整数码，**连还原步长一起走**：事件与"按什么步长
  还原"分开存放时，拿另一层的 `Events` 去解会静默解出另一组数，内积照算却不是这一层的值，因此
  `decode` 不需要编码器；`Audit::measure` 给出 E14 的冻结定义 `S = 1 − N_active/N_total` 与
  E15 的相对重构损失（`‖y‖ = 0` 时改用绝对误差——相对误差在零输入上是 `0/0`，记成 0 会把
  "没测到"与"完美重构"混为一谈）；`steepest` 报的是给定样本对上的放大比，`accumulate` 只遍历
  发放位置做 `i32` 累加，与"先铺成稠密码表再整行累加"**逐位相同**（累加次序仍是 `i` 升序）。
  **不拥有判定语汇**：PASS/FAIL 仍只由协议卡出，量化器也不是真 Lipschitz，`L_e` 必须与样本集
  一起冻结。
  **量化补偿**：`weights::ptq` 新增 `compensate` / `Compensated`——一次 f64 SVD 取秩-r 低秩项，
  显式残差 `C = W − L_r` 再走同一个 `quantize` / `dequantize`，编解码仍只有一份定义；结果另记
  `rank`、`sigma1` 与**截断处谱隙 `gap`**（这次分解的产物，不是调用方的假设）。
  **E10 探针**：`op::matmul` 的 `e10_predictability` 预热后对 120 个窗口计时，实测 **CV = 1.23%**
  （mean 214.2 ms/window，P50 213.6 / P95 216.2 / P99 232.5，512×512×512 tile）——低于 §10.10
  的 5% 门槛。**E12 探针**：`engine::kv` 的 `e12_random_access` 在冻结 4096 页 × 16 token
  工作集（65536 历史）上做 10 万次确定性随机访问，recall **1.000000**；`Table::slot` 每次
  **0.351 ns**，页数 1024 → 65536（历史 16384 → 1048576）逐档不变——地址计算 O(1) 的经验佐证，
  最远端 token 也不比最近端慢。两条探针都是 `#[ignore]`，命令写在用例文档里。
  静态质量门本轮报了两处圈复杂度（新增的 `mix::pair_gain`、`ptq::compensate`），分别把样本对
  契约与"分解 / 打包"两段拆成独立助手后转绿，行为逐位不变。

- **补上十一款里唯一整款缺失的机制：Mamba-3 的指数梯形状态空间（`nn::Mamba`）**。此前 `nn` 的
  状态空间只有 `kda` 的近亲形态——通道级 delta 遗忘记忆，写入是「纠错替换」、一步只留最新那个键；
  本条是**累加**型：同一个输入写两次是叠加而不是覆盖，因此两者判据正交，没有合并成一个模块。
  `Mamba`（宽度 × 状态维 × 秩 + 逐维 `λ`、逐通道直连）一步做三件事：双抽头梯形离散化、低秩
  MIMO 选择写入、更新后读数；`Carry` 持状态与上一个抽头，所以梯形不需要调用方额外传参。
  **复值状态不引入复数类型**：`λ = λ_re + i·ω` 的每一步是 `op::scan::PlaneStep` 的一次施加
  （2×2 实值块 = 旋转 + 衰减），每个状态维占两个实值分量，「状态宽减半、同困惑度」那条实证红利
  于是有可测的实现形态。**两条离散化按算式取名**：`op::scan` 的 `exp_zero_order`（步内常数，
  `scan_f32` 用它）与 `exp_trapezoid`（`x_{t−1}` / `x_t` 双抽头，本层用它）——`ω = 0` 时前者与
  后者数值仍不同（0.2745 对 0.2753，从零起跳时 0.2745 对 0.15），用例把这条差异钉住，防止有人
  把其中一个当成另一个的等价物。
  **失败不写半截**：几何（任一维为零、秩超状态维、尺寸溢出）、`λ`（非有限、实部 > 0 每步放大）、
  长度、步长非正、输入非有限都在第一次写入之前拒绝，校验按「形状 / 参数 / 谱 / 有限性 / 步长」拆成
  各自只答一个问题的助手函数（收集成表再判，避免为每条长度各写一个 `?` 顶穿圈复杂度门限）。
  新增 13 条用例：状态尺寸与序列长无关、零步长与正实部被拒、秩超状态维被拒、叠加写、旋转保模长、
  虚部在已有状态上可见、静默秩（第二组选择量为零）退化成单秩、`f32` 与 `f64` 同算术、因果隔离
  （未来 token 不影响过去、过去一改就影响）。`docs/ARCHITECTURE.md` 所有权表补 `Mamba` / `Carry`
  一行并把 `PlaneStep` 登记进 `op::scan` 那一行。

- **量化 GEMM 不再逐元素重解位流（实测 i2 9.8× / i4 9.6× / i8 2.8×）**。`gemm_i2` / `gemm_i4` /
  `gemm_i8` 的执行路径原本是 `matmul::<G>`——逐 `(i, j)` 走 `Gemm::dot`，于是移位、掩码与零点平移
  这些**解包成本**在每个输出元素上重做一遍，共 `m·n·K` 次；而真正需要的乘加只有 `m·n·K` 次。
  **一次解包**：`decode_stream` 把 A（`m·K` 个码）与 B（`K·n` 个码）各解一遍铺成 `i32`，解包次数
  降到 `m·K + K·n`（256×256、K=8 时从 524288 次降到 4096 次）。
  **列块整数累加**：`accumulate_tiled` 按 `TILE_N = 64` 走列块，块内 `p` 在外、列在内——每个输出元素
  仍按 `p` 升序累加，故与标量参考**逐位相等**；列在内让 B 的列块段与累加器都是连续读写。
  **格式契约收口**：`Gemm::mul` 补成无默认实现的方法（默认返回全零会让「忘了覆盖」静默通过），
  打包格式走 `packed_tiled`、i8 逐元素路径走 `signed_codes` + `accumulate_tiled`，三条入口共用
  `decode_stream` 与 `accumulate_tiled` 各一份。
  **稠密 f32 实测反向、故不分块**：`K = 8` 时一行 A 的权重就在寄存器里，列块（64 宽）与
  `matmul_tiled` 的三级块实测分别是 37 µs / 51 µs，都比顺序遍历的 17 µs 慢 2~3 倍——`gemm_f32`
  台架保留「执行入口 vs 标量入口」两档，把这条结论做成可复现的证据而非注释承诺。
  **实测**（macOS aarch64，criterion，256×256、K=8，同一次采样内对照）：i2 494.6 µs → **50.7 µs
  （9.8×）**；i4 483.1 µs → **50.4 µs（9.6×）**；i8 133.1 µs → **48.3 µs（2.8×）**（8 位每字节只有
  一个码，解包本来就便宜，收益相应小）；稠密 f32 17.3 µs 对 16.8 µs（同一份算术）。新增
  `gemm_quant` 基准（标量 / tile 各三档）并把三档 tile 登记进 `regress` 基线（预算 10%），
  对拍形状矩阵补 `n = 64 / 65 / 130` 三档跨列块接缝。

- **注意力两条内核都真正走分块 tile（实测 1.80× / 4.69×）**。上一轮把 tile 的*点积*收成多链归约，
  但执行面只到一半：`attention_tiled` 只有 Flash 一条，执行侧仍有一处走朴素路径；且两条 tile 内核的
  加权段按「定头扫列」取 `V`，每读一个元素跨 `d` 个位置，向量化不了——稠密 tile 实测只比朴素快 1.57×。
  **稠密分块 tile**：`attention_tiled::forward` 补上「K/V 列块 + 查询行块 + 按缓存预算收窄分数轨」的
  实现，行块分数只驻留 `rows × s`，不落地 `s×s`；与 `attention::forward` 逐位一致，由
  `tiled_dense_matches_naive_bitwise` 在六组 `(s, d)` 上钉住（含非整数倍与大序列）。
  **加权段改为 AXPY**：列在外、头在内——`V` 的每一列连续读、`out` 连续写，而每头沿列仍升序累加，
  于是逐位不变、内层可向量化；同时省掉每头一次重复除法（`prob / sum` 每列只除一次）。
  **Flash 跳过恒等乘**：块内 `reweight` 只在最大值被本列抬高时偏离 1，乘 `1.0` 是精确恒等，走
  `l += prob` / `acc += prob·v` 的支路不改变任何一位，只省掉每列 `d` 次乘法。
  **执行侧接线**：`serve::kernels::attention_f32`、`train::kernel::{attention_f32, flash_attention_f32}`
  都指向 `attention_tiled`（decode 仍走单行的 `decode_flash_attention_f32`，那里没有可复用的行块）。
  **实测**（macOS aarch64，criterion，`s=1024, d=64`）：稠密朴素 43.89ms → 稠密 tile
  **9.35ms（4.69×）**；online 17.83ms → Flash tile **9.89ms（1.80×）**（此前 11.70/17.94 = 1.53×）。
  基准新增 `attention_dense_f32` 对照组（`naive_1024x64` / `tiled_1024x64`）。
- **训练闭环补齐通用矩阵乘参考内核（编译期断言修复）**：`op::REGISTRY` 已是 36 项，而 `train::kernel`
  的参考表停在 34 项，长度断言 `TRAINED == REGISTRY` 直接让整仓编译失败。本轮补上 `matmul_f32` /
  `matmul_i8` 两条参考路径——收缩轴取生产档 `K`、行数由操作数长度推出、几何推导与 `u8`→`i8` 码转换各
  收成一处（稠密版共用），训练侧因此能直接拿 prefill 主链路的形状做对拍，int8 一侧因累加器是 `i32`
  而对拍逐位相等。同时 `op::matmul` 与 `op::matmul_i8` 的语义表改由 `#[annotation::model]` 生成：
  注册名、形状、输出类型与参考函数只声明一次，`matmul` 骨架与 `matmul_i8` 的模块覆盖登记在注解层的
  两张闭集表里，不再有「注册名改了、语义表没改」这种能编译却绑错算子的状态。
- **性能：矩阵乘与注意力全部走分块内核，实测各提速 1.8× / 1.7×**。此前"走 tile"只到调用点——
  线性层 `matmul_tn_tiled` 的收缩块仅 32 宽、内层是单链标量归约，实测（256×256、k=4096）**比逐行 `gemv`
  慢 8.9%**（36.8ms vs 33.8ms）；注意力两条内核也各自写着单链点积。
  **线性层**：收缩块 `TILE_K` 32 → `TN_K` 256（累加器"读出→加→写回"次数从 `k/32` 降到 `k/256`）、
  列块独立成 `TN_COLS = 32`、行块 `TN_ROWS = 16`；**注意力**：`attention::dot` 与 `attention_tiled`
  的内层归约改调同一份多链点积。
  **单一来源**：多链点积只有 `matmul::dot_lanes` 一份（`LANES = 8` 条独立累加链，尾数并入标量尾部），
  矩阵乘 tile 与两条注意力内核共用——同一个归约循环不再被写成四份、向量化程度也不再四份不一样。
  恒等变换类的改动保持输出语义：线性层 tile 与逐行 `gemv` 逐位一致、注意力两条内核的逐位对拍用例不变。
  **实测**（macOS aarch64，criterion，256×256、k=4096，3s 测量）：线性层逐行 `gemv` 33.25ms →
  **分块 19.15ms（1.74×）**；`[m,k]×[k,n]` 通用 tile 18.49ms（标量参考 304.55ms，16.5×）；注意力
  prefill tiled 11.70ms 对 online 17.94ms（1.53×）；int8 tile 19.98ms（标量参考 143.35ms，7.2×）。
  基准见 `crates/benchmarks`
  （线性层新增"逐行 vs tile"对照组）。
- **量化内核行批入口**：`op::{gemv_i2,gemv_i4,gemv_i8}::forward_batch` 让一次调用算 `rows` 行，
  权重按行块只解包一遍（此前 prefill 里逐行调用会把同一份打包权重解包 `rows` 次）。服务侧
  `gemv_quant` 跳板按批行数分流——一行仍走单行入口（`m` 即权重行宽），多行才走行批；两条路径累加
  顺序相同，行批结果与逐行调用逐位一致（`packed_gemv_batch_tests!` 对 i2/i4/i8 三档各钉一次）。

- **全仓深度重构（ADR 0048/0049）**：src 文件 255 → 195（-23.5%）、外部测试文件 49 → 38（-22.4%），
  全部为纯移动内联 + `pub use` 保持公开路径，公开项计数逐 crate 核对一致，测试数 1,779 不跌。
  配置文件层恢复（`cutile.toml` 四层优先级：内置默认 < 配置文件 < `CUTILE_*` < `--set`，`toml` 为
  L7 唯一外部依赖例外）；`settings` 键值泛型化（`SettingValue`）；测试合并硬门（测试函数零改名零删除）。
  五起回退裁决（gemv/nn 宏化、snapshot/cursor、无消费配置键、`#[backward]`）落档 ADR 0049。
- **Windows cfg 修复**：`run_tests.rs` 的 Metal 专属 helper 补 `#[cfg(target_os = "macos")]`、CUDA
  对拍补 nvidia 档案定义——Windows 原生全仓 check 通过、CUDA 真机对拍（T600/nvcc）2/2 全绿。
- **`#[spec]` 运行时形态**：属性值支持 `$` 参数洞（与静态路径共用解析框架），Embedding/Attention/
  MatMul 三层统一到注解路径；annotation 测试 +7 条。

### Added

- **Mamba-3 指数梯形状态空间（`lld.md` 第十二步）**。十一款标杆里这一款此前**整款没有实现对
  应物**——`kda` 是通道级 delta 规则（纠错替换），不是指数梯形。本步把 §5.6 的
  `exponential_trapezoid` 一格落成 `nn::mamba`，并复用 `op::scan` 的离散化而不另写一份数学。
  **离散化先分成两条式子**：`op::scan` 原本只有一个 `exp_ssm`，算式是 `exp(λΔ)` 乘当前输入——
  那是零阶保持，却被两处文档当成「指数梯形」引用。现在它按算式改名 `exp_zero_order`（`scan_f32`
  算子仍用它），`exp_trapezoid` 是新加的**双抽头**版本（`x_{t−1}` 与 `x_t` 的平均），顺带补上
  `Scalar::{sin, cos}` 与 2×2 旋转块 `PlaneStep`——名字与算式对上，才谈得上「哪条是梯形」。
  **状态是复值的、存成 2×2 实值块**：`λ = λ_re + i·ω` 时一步就是 `PlaneStep` 的一次施加，
  每个状态维占 2 个实值分量，于是「状态宽减半、同困惑度」那条红利的实现形态可测而不必引入复数类型。
  **选择是低秩 MIMO**：`B`/`C` 各 `rank` 组、各长 `pairs`，更新与读数都对秩求和，`rank = 1` 即经典
  单输入单输出；秩不得超过状态维（行多于列必有线性相关）。读数取**本 token 更新之后**的状态。
  **与 `kda` 的判据正交、故各一份实现**：累加写两次是叠加（用例钉住「第二次写后状态更大」），
  纠错写两次是覆盖；两行都在 `docs/ARCHITECTURE.md` 里写明「规则 0.11 禁混」。
  **入口硬条件**：衰减实部 `≤ 0`（正实部每步放大，几步到 `inf`）、步长 `> 0`、维度非零、
  `rank ≤ pairs`、三组参数长度精确——全部排在**第一次写入之前**，失败的那一步不留半截状态
  （用例对状态做逐位比对）。
  **实测**：`cargo test -p llm-nn --all-features` **254 条 + 2 doctest**（`mamba` 13 条：几何与稳定性五拒、
  状态尺寸与 128 个 token 无关、失败不改状态、读数用更新后状态（闭式对照）、改未来不影响过去、
  叠加写、旋转保模长、虚部改变读数、静默秩与单秩等价、`f32` 同算术）；`op::scan` 侧 8 条新用例
  钉住「λ=0 时零阶保持是恒等／零阶保持对得上闭式／恒等块不动状态／2×2 块就是复乘／
  旋转保模长／ω=0 退化成实衰减／梯形与零阶保持在常输入下**不相等**／梯形真的取了两点平均」。

- **潜在栈：patch 级因果注意力与字节回读（`lld.md` 第十一步）**。第 11 款标杆的三段此前两端与循环各有主人，
  中间那两段数值层还是空的——文档里一律写作「属装配侧」。本步把它们落成 `nn::latent`，并复用已有的块表与
  算术，不新开第二份实现。
  **全局侧**（`Global`）：输入是 `[patch, width]` 的列，一遍的**行数**就是 patch 数，于是「按 patch 前向」不是
  约定而是循环上界（平均 patch 拉长，全局遍数与总成本一起降，而输出投影宽度不变）；因果来自复用 `csa::BlockTable`
  的两级寻址——整块落在 `t` 之前的压缩块加 `[t+1−window, t]` 的原始行，不另写一份掩码。top-k 仍是阈值掩码而非
  重排（权重随后乘 `V`，换位会让分数与值错配）。
  **解码侧**（`Backread`）：字节回读自己所在 patch 及更早的表示，可见集直接取自 `Patches` 的区间（切分只有一份
  来源）；层数是闭集 `LOCAL_MIN..=LOCAL_MAX`（6–9）。`Layer` 只是 q/k/v/o 四张行主序方阵，宽度由**栈**校验，
  于是「层说宽 4、栈按宽 8 用」这种只在乘法里才暴露的状态不存在；残差让零权重成为恒等，「这一层读了什么」因此
  可以用差分测出来。点积、softmax、加权累加、块表全部复用 `csa` 那一份；`codec` / `emit` 里「local decoder 与
  全局 transformer 属装配侧」的说法同步改为指向这两个模块。
  **实测**：`cargo test -p llm-nn --all-features` **241 条 + 2 doctest**（`latent` 13 条：改未来过去逐位不变
  （四组块因子与滑窗）、单行只见自己、零权重是恒等、遍数即 patch 数、空列、全局几何五拒、字节读不到更晚的 patch、
  字节读得到自己的 patch、零查询等权、层数闭集、字节/表示行数不符被拒、`f64` 同算术）；工作区全门禁通过。
  **收口**：`Backread::forward` 的几何校验抽成 `check`（`glyph_stride` 只求一次，报错文案不再重算同一份算式），
  `op::matmul::matmul_tiled` 的块内累加抽成 `accumulate_tile`（三条区间包成 `Tile`，块内不必知道自己是第几块），
  两处分属静态质量门的圈复杂度与代码行上限；`policy` 里指向已不存在的 `crate::synthesize` 的链接改为说明
  `mix::synthesize` 是唯一入口，`RUSTDOCFLAGS=-D warnings cargo doc -p llm-nn -p llm-op` 因此转绿。
- **逐字节自回归生成循环与 CUDA 驱动骨架合并（`lld.md` 第十步）**。第 11 款标杆的三段此前只到
  「切分 + 两端」，缺的是把它们串成**一次生成**的那一层；本步补上，并把三处「同一件事写两遍」的
  CUDA 驱动样板收成一份。
  **生成侧**（`nn::emit` 的 `Emitter` / `Emission` / `Halt`）：边生成边判边界（分数就是该字节的编码代价
  `−ln p`，`p` 出自 `ByteDecoder` 的同一份分布，不另立熵模型）、已收口 patch 的表示不重算、一步只走一个
  字节。条件只取**已收口** patch 的表示加上上一个字节的 N-gram 向量：仍在长的 patch 表示会随字节变化
  （池化以边界字节为查询，而「断在哪」是收口那一刻才定的），拿它当条件等于让同一个字节的分布在切分定下来
  之前与之后不一样，也就把「边界只看过去」丢在了生成侧；历史为空时条件是全零，与 `Bypass` 同一条判据的
  两个方向。`Halt` 是闭集（`Ceiling` 预算用完 / `Stop` 产出停止字节）而不用 `Option`——`None` 说不清
  「截断」与「空结果」，而两者上游处置不同；预算为零就不生成并记 `Ceiling`，没有上限的循环不叫生成。
  `Emission` 只装这一次新产出的字节，停止字节留在里面（它是模型真的产出的那一个）；两侧宽度不等、预算与
  种子非法在入口报 `Code::Invalid`。
  **同一语义只留一份**：`engine::codegen` 收出逐点 / 循环 / WMMA 三个驱动共用的骨架
  （`cuda_head` / `cuda_launch` / `cuda_release` / `buffer_args` / `Driver`），`norm` 与 `scan` 各自那份
  「遍历缓冲拼 `cudaFree`」的清单和启动实参随之删除，`wmma` 的 host 驱动改成同一骨架的参数化调用；
  `backends::cuda_lower` 拆成「多段族一层 + 单段族一层」，不再用一条兜底臂回答两种问题；`cutile` 的
  扫描程序抽出 `scan_pair`，「块大小 + 衰减 → `Lower` → 降源 → 两段」只剩一处。
  **实测**：`cargo test -p llm-nn --all-features` **228 条 + 2 doctest**（`emit` 16 条：预算为零不生成、
  停止字节在产出它的那一步收口、已收口 patch 只编码一次、前缀永不重写、增量切分等于整段切分、空历史以零为
  条件、更紧的阈值切得不比更松的少、分数就是被选中字节的自信息、`f64` 跑同一循环、装配期宽度不符与非法种子
  被拒）；工作区 **1,986 条 / 0 失败**、doctest **43 条 / 0 失败**；
  架构门 `llm-testkit --test architecture` 8/8（所有权表新增 `Emission` / `Emitter` / `Halt` 一行）、
  静态质量门 `llm-tool --test gate` 2/2、`clippy --workspace --all-targets --all-features -D warnings`、
  `fmt --check`、`cargo machete`、`cargo deny check` 四类全部通过；`jscpd` 重复率 0.04%（2 处克隆）。
  仍未接线的部分：全局 transformer（`reprs` 那一列上的 `Attention`，每 patch 一次）与 local decoder 的其余层
  属装配侧，六平台执行合同仍未开始，E1–E15 依旧没有新增实测数。

- **混合 delta 记忆、跨块选择性读取与矩阵级正交更新（`lld.md` 第九步）**。三处都落在「已有公式但没有执行侧」
  的那一栏，且都是为了**替代**既有做法而不是叠加：`nn::kda` 顶掉指数衰减的线性注意力，`nn::attnres` 给残差一条
  「读」的路，`train::Muon` 换掉对角预条件的二阶矩。
  **KDA**（`Kda` / `Memory`）：状态是固定 `[d_v, d_k]` 的矩阵，大小与序列长无关。规则 0.11 明令「delta ≠ 指数梯形」，
  两者的差别被钉成用例——同一个单位键写两次，delta 读回来是**后一次的值**（投影 `I − β k kᵀ` 是纠错不是累加），
  指数衰减核读回来会是加权和。门 `α ∈ [0,1]^{d_k}` **逐通道**，记忆第 `j` 列乘 `α[j]`，写两步的顺序不能换
  （先写后忘会把刚写进去的纠错一起衰减掉）。`Memory` 无公开构造函数，只能由 `Kda::memory` 给出，状态维度与
  层维度因此不可能不一致；键在入口按 L2 归一（投影性质要求 `‖k‖ = 1`），β ∉ [0,1]、α 出界、非有限值一律 `Code::Invalid`。
  **AttnRes**（`AttnRes` / `Sources` / `splice`）：残差不再只有「加」，还可以在「嵌入 + 前面各块的输出」里按学到的
  权重**读**——`s_i = ⟨q, h_i⟩/√width`、`w = softmax(s)`、输出 `Σ w_i h_i`，于是深处不必沿加法链一路继承。与 mHC 的
  分工写在一处：mHC 混的是**同一时刻的多条流**（双随机、非扩张），AttnRes 读的是**不同深度**（凸组合、选择性），
  规则 0.11 禁止互换。`Sources` 按**层号**记簿：严格递增、不与主路重复——重复的层会被算两次，等于偷偷改权重；`splice` 把「混合值以系数并回主路」收成唯一口径（`main + α·(mixed − main)`），`α ∈ [0,1]` 时是凸组合、非扩张。
  **Muon**（`train::Muon`）：动量照旧，方向走 Newton–Schulz 五次迭代做近似极分解，`O·Oᵀ ≈ I` 使更新量与梯度尺度
  无关（用例：整体乘 1000，一步走多远不变）。形状**从图上的登记形状派生**——`Slot` 新增 `shape`、`ParamSlot` 新增 `matrix()`，`Session`
  把它随槽传给优化器，不从元素数反推（`4` 既可以是 `2 × 2` 也可能是 `4 × 1`）。非矩阵槽**拒绝**而不是悄悄跳过
  （跳过是「这一组这一步没更新」，跑一万步就是某个组从未被训练过）；声明的行列积与长度不符、
  梯度含非有限值都在**任何一个参数被写回之前**拒绝；解耦权重衰减仍按角色豁免（偏置/归一化不动）。
  **实测**：`cargo test -p llm-nn` **212 条 + 2 doctest**（`kda` 15 条、`attnres` 14 条）、`cargo test -p llm-train`
  **185 条 + 6 doctest**（`Muon` 10 条），全绿；架构门 `llm-testkit --test architecture` 8/8（所有权表新增
  `Kda`/`Memory`、`AttnRes`/`Sources`/`splice`、`Muon` 三行）、静态质量门 `llm-tool --test gate` 2/2、
  `clippy --all-targets --all-features -D warnings` 干净。
  仍未接线的部分：KDA 的 3:1 块排布、AttnRes 的逐层训练侧、MoE/Muon 的分布式分片属于装配侧；Muon / KDA / AttnRes
  之外，第 11 款标杆的逐字节自回归循环与六平台执行合同仍未开始，E1–E15 依旧没有新增实测数。

- **专家轴与残差轴落地：稀疏路由（MoE）与双随机多流混合（mHC）（`lld.md` 第八步）**。
  `nn::moe` 与 `nn::mhc` 把 §5.8.2 的另两根柱子接到执行路径上：前者管「这一次前向算哪一部分 θ」，
  后者管「多条残差流怎么在层间重组」。
  **MoE**：`K3_EXPERTS` / `K3_ACTIVE` = 896 取 16；路由**不新开决策逻辑**，分数 → 分布仍走唯一端口
  `policy::Calib`（`DecisionHead`、专家与层间路由是同一个候选集合），top-k 只在该分布上取，选中集
  **重新归一**——不归一的话丢掉的质量会让输出整体缩小，`k` 一改就换了尺度。共享专家恒参与、不占 top-k
  名额、不被路由权重缩放，`forward` 因此是「共享输出 + 选中专家的加权重归一」；专家这一层只是一张
  `[width, width]` 映射，真实 FFN 的两层与非线性属于外层 `Contract`。
  **mHC**：`MHC_STREAMS = 4`（Engram-27B / V4），`B_l` 由 `sinkhorn` 归一（与 CSA 块权重同一份实现）、
  `‖B‖₂` 由 `spectral_norm` 实测后留在 `sigma` 上，于是「非扩张」是可复核的测量值。**非负性是入口硬条件**，
  两条理由都是实测出来的：带负号的双随机阵也能放大信号（`[[1.5, −0.5], [−0.5, 1.5]]` 真谱范数 2）；
  且 `spectral_norm` 的幂法从全 1 起步，全 1 正是那个矩阵特征值 1 的特征方向，实测只报 `1.0`——「实测 ≤ 1」
  会假通过，而非负阵由 Perron–Frobenius 保证主特征向量同号、不会被骗。几何（流条数、各流宽度）、取值
  （非有限、负项、全零行列）一律在入口报 `Code::Invalid`。
  **实测**：`cargo test -p llm-nn --all-features` **183 条 + 2 doctest 全绿**（`moe` 与 `mhc` 共新增 23 条：
  单流空操作、选中集权重和为 1、`k = 1` 只留最优、`forward` 等于手算加权和、改未选中专家不动输出、
  共享专家贡献恰为 `1·(1·x)`、全专家路由两两不同、几何在入口失败；单位混合不改流、单流归一成 `[[1]]`、
  输出落在输入凸包内、`‖B⊗I‖₁ = 1` 不放大、Sinkhorn 结果与 CSA 同一份、实测谱范数 ≤ 1、置换阵只换位置、
  负项/全零/非有限项一律拒、几何在入口失败）；架构门 `llm-testkit --test architecture` 8/8（所有权表新增
  `Moe` / `Mhc` 两行）、静态质量门 `llm-tool --test gate` 2/2、`RUSTDOCFLAGS="-D warnings" cargo doc -p llm-nn`
  干净。
  专家内部的 FFN 形状、门控标量（`mix` 的一步合成）与多流残差的写侧训练仍不在这一层；Muon / KDA / AttnRes
  由第九步落地，六平台执行合同仍未开始，E1–E15 也无新增实测数。

- **byte-latent 的两端落地：N-gram 表示的编码侧与 256-way 解码侧（`lld.md` 第七步）**。
  `nn::codec` 新增 `ByteEncoder` / `ByteDecoder`，把 §5.17 第 11 款标杆除切分外的两层接到执行路径上：
  一次 byte-latent 前向是 **字节 → 动态 patch → latent transformer → 逐字节解码**，而 `patch` 只回答第一个问题。
  `ByteEncoder` 每字节取 `n = 3..=8` 的哈希 N-gram 桶向量（阶闭集 `ORDERS`，`seed ^ 阶` 当种子），
  **桶表复用 `engram::Table`**——表长取素数、`O(1)` 槽寻址、哈希与查表都不另写一份；窗口取**已有后缀**、
  不补零也不补 BOS 符号（补零会让首字节的表示与"前面真有一个 0 字节"混成同一份输入，与 `Bypass`
  "历史不足就不加、不拿零向量补位"同一条判据）。于是窗口是回看的、一个字节的影响是**前向的**：
  它只进自己与之后至多 `WIDEST − 1` 个字节的窗口。patch 内以**边界字节的向量**为查询做注意力池化
  （边界字节就是这条切分的决策点，不需要再引入一份可学习查询；键值只取本 patch，于是"一个 patch 一行"
  是寻址本身而不是约定），一 patch 一行、行数即全局 transformer 的遍数。
  `ByteDecoder` 是 local decoder 的输出头：一次 `[BYTES, dim]` 行主序投影给出 256-way 分布与贪心字节，
  行下标就是字节值（不需要码本把通道映射回字节）。`byte` 走**分数**而不是分布（softmax 单调、argmax 相同，
  少一次指数），并列取**最小字节值**——同一份表示不会在不同运行里给出不同字节。
  切分覆盖的字节数与输入不等、投影长度不是 `BYTES × width`、表示宽度不符，一律在入口报 `Code::Invalid`
  （宽了是把别的层的数据喂进来，窄了内积照算不误、结果却是另一个数）。
  顺手把 `Patches` 的遍历收成一处：新增 `Patches::ranges`，`get` 与它共用私有 `span`，
  于是"第 i 个区间"与"逐个区间"只有一份算式，编码侧按 `ranges` 给每个 patch 分一行。
  **实测**：`cargo test -p llm-nn --all-features` **160 条 + 2 doctest 全绿**（`codec` 新增 12 条：
  窗口只取已有后缀（六个窗口都以当前字节结尾）、每 patch 一行且行数少于字节数、空输入零行、
  前缀稳定（24 字节的每个前缀都比一遍，已完成 patch 的表示逐位不变）、局部性（改第 4 个字节只动第一个 patch）、
  池化是注意力加权和而非取边界字节、均匀 patch 池化回它自己、切分来自另一次输入即拒、256-way 分布归一且
  贪心与之一致、并列取最小字节值、三处几何在入口失败、`f64` 与 `f32` 同一份算术）；`cargo test --workspace`
  **1908 条**、`cargo test --doc --workspace --all-features` **43 条**全绿；静态质量门
  `llm-tool --test gate` 2/2（新增函数的圈复杂度 ≤ 5）、架构门 `llm-testkit --test architecture` 8/8
  （所有权表新增 `ByteEncoder` / `ByteDecoder` 一行）、`clippy -D warnings`、`fmt --check`、`cargo machete`、
  `RUSTDOCFLAGS="-D warnings" cargo doc -p llm-nn` 逐项通过。
  逐字节自回归循环、local decoder 的其余层（逆向 cross-attention、6–9 层）与全局 transformer
  （`Attention` / `BlockTable`）仍不在这一层，E1–E15 也无新增实测数。

- **byte-latent 的编码侧落地：字节按信息量切成动态 patch，两套边界规则都只看过去（`lld.md` 第六步）**。
  `nn::patch` 新增 `Boundary` / `Patches`：第 11 款标杆不是换 tokenizer 而是换前向——**字节 → 动态 patch →
  latent transformer → 逐字节解码**，"哪些字节该被同一次全局前向一起看"必须先由一条因果规则决定，本模块只做这一步。
  `Boundary` 是边界规则闭集，**参数随变体走**（熵规则持 `θ_g` / `θ_r`、编码率规则持 `window` / `top`），
  于是"规则是熵却带着 window"在类型上就不存在，不是靠校验挡住的。两套规则都来自 §5.17：BLT 的
  `H_i > θ_g` 或 `H_i − H_{i−1} > θ_r`（严格大于，取等不算越界）；ByteFlow 的 Top-K 落成**尾窗内的秩判定**
  （最近 `window` 个分数里排进前 `top` 名）——真实 ByteFlow 的 K 在整块上算，那要用到还没到的字节，
  而 §5.17 把**增量性**列为第一性质（同一前缀在生成的每一步必须仍是同一个切分，否则 KV 与 patch 一起重算），
  所以同一句话只能读过去的字节。
  `Patches` 只存起点与总字节数，于是"区间首尾相接、无空 patch"由**构造**保证、不是要维护的断言；
  `len()` 就是全局 transformer 的遍数（每 patch 一次，不是每字节一次），`mean_len()` 是 §5.17
  "patch 平均 8 字节"那条记账的左边；空输入没有 patch 且 `mean_len` 记 `0` 而不是 `NaN`
  （没有 patch 时平均值不存在，写成 `NaN` 会让上游的求和与比较全部静默传播）。
  非有限分数与退化的 `window` / `top` 在入口报 `Code::Invalid`：任何比较对 `NaN` 都是 `false`，
  放行会把一整段字节静默并进同一个 patch，出错的地方离原因很远。
  **实测**：`cargo test -p llm-nn --all-features` **148 条 + 2 doctest 全绿**（`patch` 新增 12 条：
  覆盖完整与末段收口、前缀稳定（对 16 字节的每个前缀都比一遍，两套规则各一轮）、全局阈值、突增阈值、
  取等不断、尾窗秩判定、噪声段切得比平坦段细、空输入、平均长度算式、参数与分数六类拒绝、规则短名、
  `f64` 与 `f32` 切在同一处）；架构门 `llm-testkit --test architecture` 8/8（所有权表新增
  `Boundary` / `Patches` 一行），`RUSTDOCFLAGS="-D warnings" cargo doc -p llm-nn` 干净。
  前缀稳定这条测试的初版按整段区间比较而失败：落在前缀末尾的 patch 只是**被截短**（终点改为前缀长），
  并不会被撤销，"能撤销"才是边界看了未来的证据——判据因此改为比较起点。全局 transformer 与逐字节解码器
  仍不在这一层（复用 `Attention` / `BlockTable`，接线在装配侧），E1–E15 也无新增实测数。

- **`tools/musa-harness/`**：MUSA S4000（MUSA 5.2 / muBLAS 5.2）真机测量通道——WMMA wave128 正确形态
  （10.60/13.24/10.18/8.61 TF）、FP32 SIMT 八变体、3xTF32、精度对拍（长 K 口径）与全部教训。
  muBLAS 5.2 基线 6.55/14.40/17.77/15.64 TF @ FP32 级精度（1.56e-6）；cube 1024³ 领先 62%。

- **MLA 潜在 KV 有了执行侧：缓存只存 `(c^KV, k^R)`，吸收式与升维参考逐元素对拍（`lld.md` 第五步）**。
  `nn::mla` 新增 `Mla` / `Latent` / `Path`：`Mla` 持超参 `(n_h, d_h, d_c, d_h^R)` 与两条低秩投影
  `W^{UK}` / `W^{UV}`（行主序 `[n_h · d_h, d_c]`，第 `h` 头一块，两头共用同一个 `c` 不必切子矩阵）；
  `Latent` 是缓存本身，每 token 只有 `c^KV`（`d_c`）与解耦 RoPE 键 `k^R`（`d_h^R`）两段，
  **没有** `k^{no-nope}` 也没有 `V`——`k_h = W^{UK}_h c`、`v_h = W^{UV}_h c` 都能现算，存它们等于把
  低秩投影的结果又存一遍。这是 §5.8.1 那条 Must 的可核对形态：`Latent::cache_elements() = seq · (d_c + d_h^R)`。
  同一次注意力给两条**数学等价**的算法：`Path::Upref` 每 token 把 `c` 还原成 `k_h` / `v_h` 再走标准注意力
  （参考式，对拍用）；`Path::Absorb` 用 `q^T k_h = (W^{UK}_h{}^T q)^T c` 把 `W^{UK}` 吸到查询侧、
  `W^{UV}` 吸到输出侧（解码真跑的那条，每 token 只做一次 `d_c` 点积与一次 `d_h^R` 点积，`V` 始终不必还原）。
  两者按 `1e-5` 逐元素对拍、`f64` 下按 `1e-12`——这是"潜在缓存到底等不等价"唯一能验证的形态，
  缺了它"只缓存潜在量"就只是记账而不是证据。因果性来自寻址本身（只累加 `0..=t`，`t` 之后的行根本不读），
  单测用"未来行改 7 倍不影响位置 2、位置 0 改 3 倍就影响位置 2"钉住两条路径。行宽记账走模块级函数
  `row_width` / `dense_width` / `ratio`（与 `csa::effective_len` 同形）：V3 是 `576 / 32768 = 1.76%`、
  61 层 × 576 × 2 字节 = **70272 字节/token**，与 §5.8.1 的记账一致；这些量算得出来就不该先分配 8M 个权重。
  `csa` 的 `dot` / `accumulate` / `softmax_row` 改为 `pub(crate)` 复用，不给 softmax 第二个定义。
  **实测**：`cargo test -p llm-nn --all-features` **136 条 + 2 doctest 全绿**（`mla` 新增 9 条：
  V3 行宽记账、缓存只有两段、吸收式对升维参考（f32/f64）、未来不可达过去、RoPE 查询直接进分数、
  位置与查询宽度越界、构造期五处长度校验、失败的追加不推进一半缓存、路径短名稳定）；
  架构门 `llm-testkit --test architecture` 8/8（所有权表新增 `Mla`/`Latent`/`Path` 一行），
  `RUSTDOCFLAGS="-D warnings" cargo doc -p llm-nn` 干净。与 `BlockTable`（压序列）、`Attention`（压头数）
  不叠加：一次前向只选一条（规则 0.9）。E1–E15 仍无实测数：本步给的是结构证据（每 token 每层缓存宽度），
  不是"KV 显存降到 x%"那样的实测卡——后者要等设备后端按这个行宽真跑长上下文。

- **实测卡：`lld` §10.10 的协议卡落成三态判定，真实 decode 采样第一次判出结论（`lld.md` 第四步）**。
  `tool::card` 新增 `Protocol` / `Statistic` / `Tally` / `Standing` / `Shortfall` / `Card`：一份协议卡收
  「判哪个量 + 怎么聚 + 门槛与方向 + 样本下限 + 主机闭集」，`Card::judge` 把一列样本聚成摘要再落成
  `pass` / `fail` / `open` 三态。口径三条：区间是对**统计量**（均值 / 变异系数）重采样得到的 95% 区间，
  不是对样本取分位——要判的是「这个统计量过不过门」，区间必须与统计量同坐标；`open` 必带理由
  （`host` / `samples` / `degenerate` / `straddle`），因为 `lld` 规则 0.3 写死 **OPEN ≠ 降 Must**，
  把「没测出来」写成 PASS 或 FAIL 都会让下一轮拿不到该补的那一格；判定次序是装置闭集 → 样本量 →
  统计量定义 → 区间与门槛的关系（闭集外机器上的数样本再多也不是这份协议的证据）。重采样种子固定，
  同一列样本只得到一份卡，复核可逐位对拍；样本里混进非有限数报 `Invalid` 而不是静默跳过——那是采集坏了，
  不是「这一格没测出来」。
  `crates/benchmarks` 用合成栈真跑 32 步 decode（4 步热身不计入）取每步墙钟，按 `lld` E10
  判「变异系数 95% 区间上界 < 5%」并归档到 `target/bench-report/<host>/card-E10.json`。
  **实测**（归档 `target/bench-report/macos-aarch64/card-E10.json`）：`macos-aarch64` 上这条合成栈
  （2 层、`d = 8`）一步 decode 的 32 个窗口是均值 **9410 ns**、`σ = 1781 ns`、**变异系数 0.189**，
  95% 区间 **[0.097, 0.267]**，整段高于门槛 0.05 → 结论 **`fail`**；同一份协议早一轮跑出的是
  `open (straddle)`（点估计落在门槛附近、区间跨着 5%）。两份结论都按规则给，测试不断言判成哪个——
  这台机器当时的抖动是**测量结果**，不是用例前提；卡片的价值正在于它既不会把 19% 的抖动写成 PASS，
  也不会把「差一点」写成 FAIL。注意这条数探的是**本机噪声底**（合成小栈的绝对 ns 不代表模型），
  `lld` E10 的 Must 仍要看预注册目标装置上的那组窗口。
  `cargo test -p llm-tool` **71 条**（`card` 新增 14 条：样本不足 / 闭集外 / 稳定列 PASS / 抖动列 FAIL /
  区间跨门槛 / 变异系数无定义 / 非有限样本拒绝 / 构造器三处校验 / 命名稳定 / JSON 形状与转义 /
  同列样本摘要只算一次），`cargo test -p llm-benches --test cards` **3 条**（真实采样归档、
  同列样本换门槛分别 PASS 与 FAIL、差一个窗口与闭集外都是 OPEN）。设备侧本步无新增：`engine::backends`
  的子进程 Metal/CUDA 与 `engine::metal` 的进程内执行器在 HEAD 已能出达成率与回退判定，这一步补的是
  「测出来的数怎么被判成三态」。E1–E15 仍无实测数：卡片只把「有一列样本」变成「一条可复核结论」，
  不会凭空造数。

- **服务三轴收口：一轮的可行域（IFB 三约束）落成独立模型，投机接受判定与贪心共用同一处 argmax（`lld.md` 第三步）**。
  `engine::ifb` 新增 `Budget` / `Phase` / `Candidate` / `Verdict` 与唯一算式 `pack` / `chunk`：`B ≤ B_max` 与
  `Σ T_i ≤ T_max` 在这一层判，第三条 `N_need ≤ N_free` 仍由页账给——`Scheduler::admit` 回答"这条序列这辈子跑不跑得完"，
  `pack` 回答"**这一轮**塞得下几行"，把两者混成一个错误就会让"这轮挤不下"退回客户端，而它下一轮会自动好。
  口径钉三条：packed 无 padding（一行的 token 数只由它自己决定，补位要算进 `T_max`）、生成优先上下文在后
  （上下文行可整体推到下一轮，已生成 token 回不去）、`OverSeqs` 与 `OverTokens` 分开（前者是"批满了"、后者是
  "生成行本身超预算，重试没用"）。分块预填充的对齐也收到这一层：`ifb::chunk` 是唯一算式
  （除末块外 `C ≡ 0 (mod P)`），端点不再自己 `min` 一次；`Completion` 因此多了 `with_budget`，且 `B_max` 撞上
  走 503（与"页不够"同口径，可重试）而不是 400。
  `serve::speculate` 新增 `Verdict` / `accept`：草稿一次猜 `K` 个 token，目标模型按同一批位置出分布取最长一致前缀，
  第一个不一致的位置由目标模型补一个——接受数只影响这一步走了多远，不影响序列是不是目标模型的序列；argmax 与
  `Greedy` 同源（共用 `generate::best`），草稿与校验不会各按一套判。本层只判"接受几个"、不写 KV：
  "接受了 3 个却写了 4 行"这类错因此没有从这一层漏出的路径。
  **实测**：`cargo test -p llm-serve` **173 条 + 3 doctest 全绿**（`speculate` 新增 5 条：全接受、首位不一致即回退、
  中途不一致只留一致前缀、长度不一致报 `Invalid`、不可用 logits 报错；`ifb` 新增 7 条：生成先于上下文、紧预算只挤上下文、
  序列上限、生成行超预算不是挤压、空候选、整页对齐与末块不补齐、零预算不读成不限）；`cargo test -p llm-testkit
  --test architecture` 8/8（所有权表新增 `Budget`/`Candidate`/`Phase`/`Verdict`、`Verdict`/`accept` 两行）。
  服务三轴现状：radix（`kv::Radix` 满块前缀树 + 跨请求页复用）与前缀页账已在 HEAD；chunked prefill 由 `ifb::chunk`
  给对齐、`advance_prompt` 逐块推进；投机只到"接受判定"这一层，草稿模型本身仍缺（EAGLE-3/MTP 未实现），
  因此 E3 ≥ 12× 依旧无实测数。

- **分块预填充把提示段算进页账，前缀页跨请求复用（`lld.md` 第三步：服务侧 radix + chunked prefill）**。
  此前解码只算当前位置、历史从页里读，提示段的 K/V 永远停在零填充——长提示与短提示的解码看到的历史
  其实一样多。现在 `Predict::prefill` 逐位把提示段算进它该在的行，`Serve` 端点在准入后按
  `PREFILL_CHUNK` 分块推进，每块之后 `Batch::mark_prefilled` 只把**已算到的位置**记账（写是提前一格的：
  第 `at` 步写第 `at + 1` 行，登记若跟着写走就会把还没算的块当成算好的）。
  `engine::kv::Radix` 是满块前缀树，它自己就是页的一个持有者：块写满时 `publish` 对它那页多持一个引用，
  请求结束只归还自己那份，页仍活着，下一个同前缀请求经 `longest` 直接采纳——「源序列一结束、页一归还，
  就再没有第二个请求能命中」这条旧限制因此消失。树的每条边恰是一个块，`evict` 只摘 LRU 叶，不会让仍被
  引用的深层块失联；`publish` 要求父块先在树上，不为一辆已被驱逐的父块补一个没有页的中间节点。
  页复用与位置账是同一份：`Scheduler::admit` 现在返回**接走的前缀 token 数**，采纳的页写进新页表的同时
  `Sequence::prefilled` 推进到该前缀长度，`Batch` 让 `prefilled_ready()` 未满足的序列不参与解码——只共享页
  不改位置账，第二个请求会把同一段提示重算一遍，而它写的正是它刚打算去读的页。
  页紧不再等于请求非法：`Predict::prefills()` 让端点**先问能力再要页**（不支持的桩模型不该先花一笔页预算
  才发现自己没有这条路径），`Scheduler::reserve` / `Batch::reserve` 返回「要到了没有」，`false` 时这一轮
  先不算、块位置一个都不推进（已算的前缀仍然有效，下一轮接着算）。
  **实测**：`cargo test -p llm-engine -p llm-serve` **112 + 14 + 167 条测试与 3 + 3 条文档测试全绿**；新增用例
  `a_second_request_with_the_same_prompt_skips_the_prefilled_prefix`（4 token 提示、页宽 4）记录预填充位置：
  首跑 `[0, 1, 2, 3]`，第二个同提示请求**一个位置都不重算**且生成字节与首跑逐位相同；前缀树自身的
  「满块才共享、命中刷新时序戳、只摘 LRU 叶、父块不在就不登记」各有单测。架构门 `llm-testkit --test
  architecture` 8/8（所有权表新增 `Radix` 一行）。

- **谱隙证书接进压缩链：秩轴现在先过 L0 门，秩由证书给出而不是调用方拍（`lld.md` 第二步）**。
  `weights::ledger` 从「三个函数各算一遍 SVD」收成唯一入口 `certify(w, m, n) -> Result<Option<Certified>>`：
  一次 f64 分解同时给出受认证的秩 `r`、低秩项 `L`、显式残差 `C = W − L` 与自证的 `Note`（`llm.cert.v1`），
  返回 `Ok(None)` 就是**拒压**本身——没有谱隙（`σ_r − σ_{r+1} > 2ε` 全不成立）或 `min(m,n) < 2` 时不颁发证书。
  `ε` 取**实测后验残差** `‖A − UΣVᵀ‖_F`（`ε_obs = 0`：被分解的就是落盘那一份），不拿机器精度冒充严格界；
  无损由 `low + rest` 的逐位往返判定，不由模式推断；`Stable` / `Mode` 收成闭集，判定等号一律 `Fail`。
  `axes` 的 `rank` 轴随之改为「证书不给秩就不压」：原先秩硬取 `min(m,n)/2`、证书只能事后记录，现在
  判定与产出同源——「证书不过就拒压回退」从文档口号变成 `encode` 返回 `None` 的那条分支。
  **实测**：`cargo test -p llm-weights` 160 + 16 + 3 条全绿（`ledger` 的 8 条用例按新入口重写：严格阈值
  与等号、可判秩域的边界、`diag(3,2,1,0.5)` 认证出 `r = 3`、精确秩 2 矩阵停在 `r = 2`、全零/单行拒压、
  编码往返、不变量违例拒绝；`axes` 新增 2 条：认证通过时载荷逐位往返、无谱隙时秩轴整轴不参与）；
  架构门 8/8（所有权表新增 `Certified`/`Note`/`Stable`/`Mode` 一行，基线里同步删掉这四项与
  已不存在的 `Spectral`/`Trunc`），`llm-weights` 的 `RUSTDOCFLAGS="-D warnings" cargo doc` 干净。

- **LLD 三根差异化柱子从"只有纯函数"走到"有执行入口"：块表注意力、条件记忆旁路、门控路由**。
  `nn::csa` 新增 `BlockTable`：K/V 按块因子压成块表（复用既有 `compress`），再按「整块（整块必须落在
  查询之前）+ 滑窗原始行」两级因果寻址算注意力输出，`attend(q, t, topk)` 给出位置 `t` 的输出。
  `m = win = 1` 时可见槽恰为 `0..=t`，输出与稠密因果注意力逐位一致；`t` 之后的行怎么改都不影响 `t` 之前
  的输出、`t` 之前的一改就影响——因果性由单测钉住，不靠注释声明。top-k 只做阈值掩码、不改槽序：权重随后
  要乘 `V`，重排会让分数与值错配（这正是不能直接复用 `dsa_topk` 的原因）。
  `nn::engram` 新增 `Bypass`：`LmHead` 前的条件记忆旁路，取尾部 N-gram、查一个槽、按 `gate` 加进隐藏向量；
  历史不足 `order` 时**不加**而不是拿零向量补位（零向量在残差里是一个真实偏移）。
  `nn::policy` 新增 `Channel` / `Gating` / `MissRate`：α/β/γ 通道闭集、一步的 π/γ 与单位剖分系数、
  `ρ_t` 的最近 `W` 步因果窗口（窗口空时 `ρ = 0`，没有证据就不因未命中抬 π）。通道只取系数的 argmax，
  只决定"下一步该准备哪条路的输入"；合成仍走 `mix::synthesize` 这一个入口，不因选了通道就少算一项。
  三者都不持参数、不注册算子，几何与查询越界一律在入口报 `Code::Invalid`。
  **实测**：`cargo test -p llm-nn --all-features` 127 条 + 2 doctest 全绿（新增 13 条：块表的块数与几何、
  稠密对拍、因果隔离、top-1 只留最优槽、越界报错；旁路的短历史、`gate` 缩放、阶与宽度校验；门控的单位
  剖分、argmax 与等值次序、窗口滑动、按特征取通道、合成与 `synthesize` 一致、通道短名稳定）；
  架构门 `llm-testkit --test architecture` 8/8，所有权表已登记这 5 个类型。

- **`#[model]` 支持 `const = "NAME"` 命名语义常量，flash 两条注意力算子不再各占一个包装模块**。
  `annotation` 的 `#[model]` 此前每个模块硬编码生成一个 `pub const MODEL`，逼得
  `flash_attention_f32` / `decode_flash_attention_f32` 各起一个只装"标记类型 + 别名"的包装模块
  （`flash_attention.rs` / `decode_flash_attention.rs`），只为各放一个 `MODEL`。现在 `MODEL` 键表加
  第 5 个可选键 `const = "NAME"`（缺省仍 `MODEL`），生成器按名出 `pub const NAME` 与同名独立的
  `__model_check_<name>` 测试模块；flash 两个 `#[model]` 于是并入 `attention.rs` / `decode_attention.rs`
  （`FLASH_MODEL` / `DECODE_FLASH_MODEL`），删掉两个包装模块。`op` 的 `registry!` 第四列从"语义模块"
  改成"`MODEL` 常量完整路径"，32 条注册仍是一行一算子，flash 两条写 `attention::FLASH_MODEL` /
  `decode_attention::DECODE_FLASH_MODEL`。净删 2 个文件；`cargo test -p llm-op`（287 + 2 doctest）与
  `llm-annotation`（35 + 2）全绿，clippy `-D warnings` 与 `fmt --check` 通过。

- **`engine` 从"页号簿记 + 记账执行器"推进到"页里真的有 KV、提交真的只入队"**。
  `kv` 新增 `Pager`：页池加上它背后的 `f32` 缓冲（一页 = `page_size` 行 × `width` 个元素，整池构造时一次分配，
  借出与归还都不再碰分配器），`row` / `row_mut` / `write` 只经 `Slot` 寻址并先验"页还活着、偏移在页内"，
  `release` 是唯一清零点——归还即清零，因此"复用的页带出上一条序列的字节"没有发生的路径（此前 `Cache`
  只发页号，页里根本没有数据）。`Scheduler` / `Batch` 因此改持 `Pager`，并补 `write_kv` / `kv`：写 KV 只经
  页表，既写不出超前于页表的位置、也写不进一张已归还的页。
  `Sequence` 新增 `tokens`（只读）与 `push_token`（仅同 crate 可见），`Batch::emit` 把"记账 + 推进"收成
  一个调用——`Blocked` 时不多记一个 token，重试因此不会看见一个并不存在的输出；`Predict` 于是能从序列读到
  自己的历史，服务端点不再维护一份平行列表。
  `exec` 的 `CpuExecutor` 改成真正的"提交入队、`sync` 才执行"：`submit` 只按图序入队（跨流交错保持提交
  顺序），`sync` 只排指定流、成功一条才出队一条（失败的那条留在 `pending` 里），与 `Executor` 契约里
  "提交不等于完成"一致——此前 `submit` 会同步算完，让"按流写出依赖顺序"这件事在上层根本没法表达。
  随之调整的对外签名只有一处：`Completion::new` 改收 `Pager` 而不是 `Cache`——行宽
  （`2 · kv_heads · head_dim` 这类数）只有装载模型的一侧知道，服务层不替它猜。
  **实测**：`cargo test -p llm-engine -p llm-serve --all-features` 61 + 9 + 65 + 8 条测试与 3 + 3 条文档测试
  全绿（新增 13 条：页池写读/清零/失效槽位/零行宽/多层同行/跨页不共享、`emit` 记账与 `Blocked` 不记账、
  KV 经页表读写与复用页不串数据、执行器按流排水与失败留队、模型能读到已生成历史）；
  `cargo test -p llm-tool --test gate`、`cargo test -p llm --test architecture --test contracts
  --test cross_layer --test cli`（8 / 9 / 5 / 4）、`cargo clippy --workspace --all-targets --all-features
  -- -D warnings`、`RUSTDOCFLAGS="-D warnings" cargo doc` 与 `cargo run --bin llm -- status`（占 1 页、
  响应 200）逐项通过，新增函数全部在 ccn ≤ 5 以内、未新增债务。

- **引擎、分片与服务从"契约/簿记"推进到可执行实现，成员数与依赖方向一条未变**。
  `engine`：`exec` 新增 `Program`（按名字登记的内核体）、`Bindings`（编号 ↔ 平坦 `f32` 缓冲，按编号排序）
  与 `CpuExecutor`——按图序查内核、把操作数从表里**借出**、执行、再放回（失败路径也放回）；借出/放回
  而不是同时取 `&` 与 `&mut`，因为"两组编号不重叠"是运行期事实、借用检查器证明不了，这样不必装 `unsafe`；
  `kv` 新增 `Slot`（页号 + 页内偏移，`Table::slot` 是寻址的唯一一次除法）与 `Table::truncate`
  （摘页与还页成对，`Scheduler::finish` 因此不会再漏最后一页），`Scheduler` 补 `page_size` / `slot`，
  `Sequence::stop` 让"采到停止符"与"到 `max_new`"都只改状态；新增 `Batch` + `Admission` 把页账与
  正在跑的序列收在一处——准入不够时把序列原样退回（服务层据此报可重试的 503），而不是把"现在满"和
  "请求不成立"混成同一个错误。
  `shard`：`Partition` 的口径固定为**恰好覆盖**（余数摊给前几份，流水并行的层分配因此合法），
  `Reduce` 补上 `identity` / `apply` / `fold`（极值以无穷为单位元、NaN 传播、空输入返回 `None`），
  rank 的正解、反解与进程组统一走 `dp·(tp·pp) + pp·tp + tp` 一条布局规则；新增 `Sharding` / `Tiling`
  （按列切不通信、按行切要 `all_reduce(Sum)`；不整除在构造期失败——各 rank 形状必须一致）与进程内集合
  通信 `Local`，让 TP/DP 的数值语义在没有 NCCL 的机器上也能被 `cargo test` 验证。
  `serve`：新增 `Request`（`id` / `prompt` / `max_new` 行文法，未知键、重复键、缺字段、坏数字都是 400）、
  `Predict` / `Sampling` / `Greedy`（确定性贪心：等值取最小下标、非有限值报错）、`Generation` + `Ending`、
  端点 `Completion`（解析 → 编词 → 准入 → 逐步前向 + 采样 → 归还页 → 包成帧；客户端错误 400、页不够 503、
  只有模型或采样失败才 `Err`，**失败路径同样归还页**）与 `Router`（方法大小写不敏感、重复路由拒绝、
  未命中 404，`Status` 因此多一个 `NotFound`）。于是"一条请求能不能跑完"今天就能在进程内验证，传输层
  落地时只需调用 `Router::call`。
  **实测**：`cargo test -p llm-engine -p llm-serve -p llm-shard --all-features` **160 条单元与集成测试
  + 8 条文档测试全部通过**；`cargo test -p llm --test architecture --test contracts --test cross_layer
  --test cli` 8 / 9 / 5 / 4 全绿；`cargo clippy --workspace --all-targets --all-features -- -D warnings`
  与 `cargo test -p llm-tool --test gate` 零告警零失败（新增函数全部在 ccn ≤ 5 以内，未登记新债务）。
  文档同步：`docs/ARCHITECTURE.md` §三 增一张"运行时实现新增的类型"表（`Program`/`Bindings`/`CpuExecutor`、
  `Slot`、`Batch`/`Admission`、`Sharding`/`Tiling`、`Local`、`Status`/`Request`、生成协作者、`Completion`/`Router`），
  并把在途新增的 `benches → nn` dev 边补进边集（边 95/31 → **96/32**）。

- **基准归档补上 roofline 定位：上界、达成率与瓶颈侧，进报告也进门禁**。`crates/benchmarks` 新增
  `criterion/mem_stream.rs`：16 KiB / 1 MiB / 64 MiB 三档工作集、读 / 写分开测的搬往上界——读侧用
  无符号整数异或归约（浮点求和不可重排，会把循环串行化成加法延迟），写侧用 `fill`。**不用厂商峰值**：
  `docs/lld.md` E6 把「仅有未验证的峰值规格」判为不可计算。`tests/regress.rs` 因此多三样东西：
  ① `HOSTS` 主机档案（机型 / 系统 / 工具链，随报告归档；这一轮同时修掉了注释里把 Apple M5 Pro
  写成 M4 Pro 的旧错）；② `CEILINGS` 上界表（6 条，与基线一样由 `tool::regress` 判回退）；
  ③ 报告新增 `read_bytes` / `write_bytes` / `flops` / `working_set_bytes` / `cache_tier` /
  `roofline_ns` / `bottleneck_side` / `achieved_percent` / `mflops` / `bottleneck`，分子由
  `nn::Bounds` 从算子语义推出（不写第二份流量公式），分母取同机同档位的实测值。后端标识也从常量
  改成 `SCENARIOS` 的一列：新增后端是加一行数据，不是改报告代码。定位结论：七条基准没有一条达到 85% 带宽门槛（`gemv` 36% 最近、
  `attention` 0.15% 最远），「四算子内存型」只在「算术强度 <1 flop/字节」的意义上成立，
  在「实测卡在带宽上」的意义上不被数据支持。

- **`cutile` 整体分析落档，本轮零代码改动**：逐块判完 25 个模块
  （约 7000 行：唯一机制 12 / 命令策略 5 / 设备编排 5 / 纯数据 3），取证 CPU 从 `.k` 模板 → `prim/cpu` 片段 →
  `cpu.toml` 前导 → 见证 → 子进程对拍的完整链路，并给出五设备 × 十六原语矩阵与 Metal 缺口的三类拆分；
  边界裁决：设备数据归 L2（`prim` / `prof`），
  机制留 L3（`cutile`），真设备运行时后端不进 `cutile`；显式记录本轮偏离 roadmap §5/P3「独立后端边界 crate」
  （不新建成员），落点与触发条件随 P3 动工前的边界 ADR 复核。`crates/**` 无 `.rs` / `.toml` 改动。

### Changed

- **`data` 成员并入 `train`：成员 25 → 24、边 112 → 110，样本与训练循环回到同一个 crate（ADR 0038 §修订）**。
  `Batch` / `Dataset` / `Chunks` 的语义就是训练循环的输入端口——`Learner` 收 `&mut dyn Dataset` 取批、
  `Checkpoint` 记 `Batch` 几何——而 `data` 唯一的出边 `data → diagnostic` 正是 `train` 已有的依赖，
  并入后依赖面增量为 0；消失的是 `train → data` 与 `data → diagnostic` 两条边，不新增任何边。
  代码整块搬进 `crates/train/src/samples.rs`（原 `mod batch` / `mod dataset` 两个内联模块合成一个文件、
  一套测试），对外路径改为 `llm_train::{Batch, Chunks, Dataset}`；`crates/data` 目录与它的 `Cargo.lock`
  条目一起删除，`docs/ARCHITECTURE.md` 的层表把"样本"从 L5 移到 L6 的 `train`，所有权表、边清单与
  吸收史（第十一次）同步。这也是 ADR 0038 §4「剩余单生产消费者叶子全部跨层，不合并」的逐条取证修订：
  跨层叶子改为按"语义归属 + 依赖面增量"取证，`rng` / `tokenizer`（L1/L4 通用基础件，语义不属于 `serve`）、
  `autograd`（消费者还有 `benches`）、`shard`（独立集合通信契约）的结论不变。**实测**：`llm-train`
  175 条 + 6 条 doctest 全绿，架构门 8 条（110 边、所有权表 `train` 持三件套）通过，`clippy -D warnings`
  与 `fmt --check` 通过。

- **`data` 第三轮减法：删掉 `Chunks` 上与 `Dataset` 一一对应的三个固有访问器，计数只在 trait 上定义一次**。
  `Chunks::{cols, batches, remaining_batches}` 三个固有方法此前只是 `Dataset` 实现的转发目标：trait 方法
  调固有方法、固有方法读字段，同一语义两份定义，而全仓没有任何调用方走固有那一层（`train` 经
  `&mut dyn Dataset` 取批，其余调用点都在本文件的测试里）。现在 trait 实现直接读字段，固有方法区只剩
  `Chunks::new`，`Iterator::size_hint` 经 `Dataset::remaining_batches` 取值，「还剩几批」仍只有一个来源。
  473 → 457 行。**实测**：`cargo test -p llm-data` 12 条 + 3 条 doctest、`llm-train` 163 条、
  `clippy -D warnings`、`fmt --check` 与架构门 8 条通过。路线 C 到此收口：`data` 只剩 `Batch` / `Dataset` /
  `Chunks` 三件套、单依赖、单消费者，成员边界按 ADR 0038 继续不动。

- **`data` 第二轮减法：两个构造函数共用一份几何校验，删掉三个零消费者的同义访问器**。
  `Batch::new` 与 `Chunks::new` 此前各写一套「行列非零、乘积不溢出」校验，同一对非法行列在两条路径上
  措辞不同；现在共用文件级 `stride(rows, cols, site)`，三种失败（行数为 0 / 列宽为 0 / 乘积溢出）
  只有一份来源。`Chunks::{rows, stride, remaining}` 除自身单测外全仓零调用——`Dataset` 只暴露
  `cols` / `batches` / `remaining_batches` 三个计数，每批行数与每批 token 数分别由 `Batch::rows` 与
  `rows * cols` 给出——按「无消费者不预埋」（ADR 0004；键表同一判据见 ADR 0048 §5）删除。
  484 → 473 行（生产 144 行）。**实测**：`cargo test -p llm-data` 12 条 + 3 条 doctest、
  `clippy -D warnings`、`fmt --check` 通过。路线 C 的第三步（减法后回看 D4）落进 ADR 0038 §修订：
  `data` + `shard` 同层合并只值 -1 成员 / -2 边 / 0 编译时间，按 §2 的实证要求与内聚性理由否决。

- **`data` 做减法：删掉零消费者的训练/验证集切分与批形状访问器，`data → tensor` 这条依赖边整条消失**。
  `Split` 除自身的四条单测与两条 doctest 外全仓无人使用（`train` 只用 `Dataset`/`Chunks`/`Batch`），
  `Batch::shape()` 同样只有测试与 doctest 调用，而它是 `tensor` 依赖的唯一用途，删它就同时把这条边删了。
  现在 `crates/data` 只剩 `Batch` / `Dataset` / `Chunks` 三件套：652 → 485 行、依赖从 2 条降到 1 条，
  切分改由调用方按行区间自行选择（`Chunks` 只认 `(rows, cols)` 几何，不替调用方决定切多少行）。
  `docs/ARCHITECTURE.md` 的边总数 113 → 112，`data` 行与所有权表同步。**实测**：`cargo test -p llm-data`
  12 条 + 3 条 doctest、`llm-train` 163 条、架构门 8 条全绿，`fmt --check` 与 `clippy -D warnings` 通过。

- **L7 入口内聚拆分与文档链接收口（对外接口契约不变）**。`llm` 根包 `main.rs` 837 → 78 行：
  `status` / `run` / `train` 命令的装配、输入解析与汇报分别归位到根包内同层子模块
  （`status.rs` / `run.rs` / `train.rs`），`main.rs` 只留入口、分派与退出码；公共命令行行为与
  输出口径未变。`diagnostic::Error` 的 `impl std::error::Error` 补显式 `source() -> None` 与取舍注释：
  保留可克隆、可比较的值类型错误模型，不引入 `Box<dyn Error>` 链（会破 `Eq`/`Clone`）。`serve`
  采样温度默认值提为具名常量 `DEFAULT_TEMPERATURE`。同时修正一批 rustdoc 内链：失效的 `Pager::share`
  路径、冗余显式链接目标、指向私有模块/条目/字段的链接，以及 `bench`/`model` 同名歧义；
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 全绿。

- **存量债务清零：`[[gate.debt]]` 203 条 → 0 条**。上一轮结清 `crates/cutile` 的 45 条后，剩下的 203 条分散在
  16 个成员里（`weights` 83、`op` 34、`tool` 20、`prof` 19、`annotation` 12、`train` 11、`autograd` 7，
  其余 9 个成员 17），这一轮全部改到限值以内，表也随之清空——`[gate]` 现在没有任何豁免，`Gate::stale`
  再也找不到可留的条目，新增超标项直接失败。三类做法：① **巨型函数按职责拆开**——`weights/bin.rs` 的
  `execute`（ccn 76）拆成读取分派 + `frame`/`placeholder`/`storage_of`/`offset_of`/`shape_of`，`op/decl.rs`
  的 `parse`（ccn 50）按「段 → 键 → 取值文法」拆开，`annotation/syntax.rs` 三个读键 `match`（ccn 40/34/21）
  收成一张 `KeySpec` 表 + `collect`/`read_one`/`key_index`，`prof.rs` 的 `validate`/`parse_provide`/`parse`
  （ccn 38/47/34）按判据拆成 30 余个单一职责函数，`train/store.rs` 的 22 个内核适配器收成
  `unary!`/`binary!`/`gemm!` 三个宏；② **文件压回 899 行**——`train/store.rs` 1477 → 786（测试按仓库既有约定
  搬到 `store/tests.rs`，与 `snapshot/tests.rs` 同形）、`annotation/syntax.rs` 1228 → 817（测试搬到
  `syntax/tests.rs`）、`cutile/backend.rs` 945 → 898、`prof.rs` 1216 行的单文件拆成 `prof.rs` 442 +
  `prof/parse.rs` 563 + `prof/check.rs` 253（公开路径 `prof::Profile` 等一字未变）；③ **裸调用清零**——
  `tool`/`prim`/`weights`/`annotation` 里生产代码的 `unwrap`/`expect`/`panic!` 全部换成带上下文的诊断或
  `let-else` 提前返回（`CALLS` 归零）。
  **判据没有被放宽**：错误文案、错误类别、退出码、生成文本逐字比对——`annotation` 用 98 条解析对拍
  （`Ok(Debug)` 与 `Err((位置, 文案))` 全等）+ 81 条 codegen 输出对拍，`cutile` 用缓存目录 `diff -r` 证明
  harness 源文本逐字节相同，`autograd` 用冻结的 `saved/release/replayed/peak` 数值表（13 个图形 × 预算）
  锁住显存计划。实测：`cargo test --workspace --all-features` 全绿、`cargo test -p llm-tool --test gate` 在
  **空基线**上通过、`gate.rs` 的两个单元测试改为「项目配置没有豁免」与「登记值由本地表给出」（它们原先借
  项目基线里的真实条目做样本，基线清空后必须自带 fixture）。

- **`crates/cutile` 的 45 条存量债务清零：`[[gate.debt]]` 249 → 204，代码 9,526 → 9,403 行**。
  本轮不加规则、不抬基线、不新增目录，只把"曾经超线所以被豁免"的地方改回正常尺寸：九类卡点函数按职责拆分
  （`cap` 的协商与诊断排版、`witness` 的建立与比对、`backend`/`device` 的三种前端程序拼装与执行、
  `verify`/`bench`/`doctor`/`execution` 的"跑→判→报"三段），`cache.rs` 从 1,245 行的历史棘轮降到 **900 行**
  （门限 899，整文件含测试），`builtin.rs` 的两处裸 `expect` 换成带上下文的诊断。同一轮里把几处"先收集再遍历"
  改成一次扫描：`ast::encode`/`escape` 逐字段写出而不是先拼 `char` 迭代器，`substitute::splice` 用
  `find('$')` 定位占位符而不是 `chars().peekable()` 逐字符推进，`pipeline::parse_shape`、`probe::select`
  （`max_by_key` 一次挑出最具体的档案）、`plan::select`（tile 取值与 roofline 估值分开）同理。
  **判据没有被放宽**：`probe::select` 的"同等具体算配置错误"只在**最优档**比较——两份更宽的档案撞在一起
  不影响"选哪一份"，那是 `Catalogue::validate` 在装载期回答的问题；错误文案、退出类别、报告字段逐字未变。
  实测：`cargo test -p cutile --all-features` **149 单测 + 2 doctest**（条数与断言逐条保留，压缩的是注释与重复段，
  不是覆盖点）；`crates/cutile/src` 含测试 9,526 → **9,403** 行、剔除内联测试 6,468 → **6,455** 行；
  `cache`/`device`/`backend`/`verify`/`witness` 五个最大文件逐一下降；`cargo test -p llm-tool --test gate` 绿且
  报告"已达标，删除基线条目"的 45 条全部结清（`cutile` 成员不再有任何复杂度/行长/裸调用豁免）。

- **质量门成为真正的棘轮**。`crates/tools` 的静态质量门在实现里违反了自己的
  三条声明，本轮全部收口：① 限值只有 `Gate::limit(Metric)` 一个出口，阈值字段转私有，`Over` 不再携带
  调用方可任意填的 `limit`，测量方（`tests/gate.rs`）只报"测到多少"；② 棘轮按（身份, 度量）结清——
  `Gate::stale` 改收 `BTreeMap<String, BTreeSet<Metric>>`，一个函数改好 `ccn` 却还超标 `nloc` 时不再能
  永久保留已达标的那条；③ "只准降"落地为"登记值必须等于实测值"，登记表从预算变成快照。接上后第一次运行
  即抓到 **31 个陈旧登记值**与 **3 个过期豁免**，全部按实测下调（`[[gate.debt]]` 249 → 248 条，0 新增、
  0 抬高；例如 `verify_one` 的 `nloc 105 → 77`、`atom.rs` 的 `lines 1264 → 1199`）。同时删除从未被读到的
  `[gate] simd_nloc = 96` 档位（Rust 侧无 SIMD 微核，向量代码在 `prim` 的 C 片段里、由 `cc` 编译），
  `[gate]` 改为闭集、拼错的键直接报错；`unreachable!` 纳入禁止形态（它是 panic 路径且无 lint 兜底）。
  `crates/tools` 因此从"既测量又判定"退化为纯测量。

- **`cutile` 的文档与代码收敛到同一份事实源：`DESIGN.md` 改名 `README.md`，且每一处旧引用都改到了它**。
  这份 1,179 行的文档既是 crate 的 README，也是整条流水线的设计契约，于是它过期就不再是文风问题、而是错误信息。
  本轮先修最容易被证伪的三处：① **§0.1「现在能跑什么」**是新增的实测表——算子 22、设备档案 5（nvidia / intel /
  metal / ascend / cpu）、片段 5×9 + 见证、布局/调度各 4、机制代码 9,526 行（含内联测试；剔除后 6,468）、
  用例 146 单测 + 2 doctest，每行都附复核命令；② **§3.1 目录树重建**：原文在 `crates/cutile` 下又列了一遍
  `crates/cutile/src`（`Cargo.toml`、`README.md` 与 `lib.rs` + `witness.rs`），并把已删除的 `DESIGN.md` 当成员列出，
  现在按 `op` / `prof` / `prim` / `cutile` 四个真实目录逐模块写职责，并说明「机制（`build`/`backend`/`cap`/`cache`）
  与命令策略（`doctor`/`verify`/`bench`/`execution`/`gc`）同处一个 crate 但只经 `pipeline` 相接」；③ **§13.1 的
  预算说明刷新**：原来那串「本 crate 2,813 行」早已不成立，改为按当前实测逐模块列出，并新增一段「预算 2,540 对实测
  6,468、2.5 倍」，把超出归到 `cache` / `device` / `backend` / `witness` 四块**证据采集基础设施**与它们各自支撑的
  判据，而不是笼统记成"写多了"。同时把全仓约 40 处 `DESIGN §x` / `DESIGN.md` 引用（`crates/{cutile,op,prof}` 源码与
  片段注释、两份 crate README、`docs/ARCHITECTURE.md`、`src/bin/cutile.rs` 打印给用户的提示）改为 `README §x` /
  `cutile README §x`，于是 `cargo run --bin cutile -- verify` 的提示里也不再出现一份不存在的文档。
  代码侧同步：`cutile/src/lib.rs` 新增「模块地图」（五步流水线 → 具体模块，并写明 `device`/`gate`/`pipeline` 为何是
  私有接缝），`bench`/`doctor`/`verify`/`execution`/`gate`/`device` 六个模块的文档从一行扩成"证明什么、不证明什么"
  （例如 `gate` 只证明"见证在自己声称的语言里语法成立"，不改 `grade`；`doctor` 不执行内核，`verify` 的失败是错误
  而不是跳过）。**门禁扩面**：`crates/tools` 的链接漂移检查原先只扫 `docs/` 顶层，现在覆盖根 `README.md`、
  `docs/**/*.md`（含全部 ADR）与每个成员的 `README.md`，并加「至少 30 份文档」的防假绿下限——crate README 曾经不在
  门里，这正是上面那棵过期目录树能长期存在的原因；扩面后全仓零失效链接。
  验证：`cargo test -p cutile --all-features` **146 通过 + 2 doctest**、`cargo test -p llm-tool --test gate` 绿
  （`[[gate.debt]]` 250 条不变，未新增未抬高）、`cargo clippy -p cutile -p llm-tool -p llm-op -p llm-prof
  --all-targets --all-features -- -D warnings` 与 `RUSTDOCFLAGS="-D warnings" cargo doc -p cutile --no-deps` 各 0 告警；
  README 里新写的四条命令逐条实跑：`doctor` 打印整条链、`verify --devices nvidia` 在无卡时**失败**（退出码 3）并报
  "见证自检 17 条 + 语言门通过"、`bench --repeat 5` 给出 min/p50 与工作集口径、`cache gc --limit …` 正常回收。

- **配置策略上移到命令行层，`cutile` 只收三个目录（ADR 0023）**。删掉第四层（配置文件）之后，
  `crates/cutile/src/settings/` 只剩「默认值 + `CUTILE_*` + `--set`」，而这三件事没有一件是内核机制该决定的：
  默认目录是**部署策略**（同一个 crate 被测试/服务嵌入时应当能换），`--set` 是**命令行文法**。
  于是 4 文件 924 行整块上移：内核只声明输入形状 `crates/cutile/src/dirs.rs::Dirs`（64 行纯数据，
  不读环境变量、不拼平台路径），策略、来源追踪与两个二进制里原来那段 `take_overrides` 都进
  `crates/cli/src/settings.rs`（660 行，含 13 条测试），键表改成一行一个的 `keys!` 声明宏
  （`cache.dir` / `work.dir` / `data.root`，含 `CUTILE_CACHE` 兼容名），`describe` 与"键未登记"的
  清单都从同一张表派生；键表不再借 `prof::atom::Closed`（`cli` 本就不依赖 `prof`，三个键不值得拉一套位集机制）。
  12 个 `cutile` 模块与两个二进制的参数由 `&Settings` 改为 `&Dirs`，两个 bin 各写一个 3 行
  `kernel_dirs` 投影——刻意不引入 `cli → cutile` 依赖，因此**成员 25、依赖边 75（`+dev` 11）一条未增**。
  搬迁自身净 −200 行 / −3 文件（`cutile/src/settings/**` −924、`dirs.rs` +64、`cli/src/settings.rs` +660）；
  `[[gate.debt]]` 251 → 250（删 `crates/cutile/src/settings/source.rs:cli_override#0`，未新增未抬高）；
  用例 59 目标 / 1 074 → **1 068 通过 / 0 失败 / 1 忽略**（19 → 13 条，6 条只覆盖随第四层消失的行为）。
  用户可见行为逐字不变：三层标题、`← 内置默认值` / `← CUTILE_WORK_DIR（env）` / `← --set …（cli）`
  三行排版、兼容名、空值与未登记键拒绝、`invalid` → 2。

- **两张 22 条的注册表各收成"一行一对"**（§8.2 第二梯队第 7 项）。`op::REGISTRY` 原本逐条手写 `Registration { name, template: include_str!(..), model: .. }`（110 行），现在是一张 `registry! { "名字" => 语义模块 }`（22 行 + 15 行宏），数组长度由表本身数出（`<[()]>::len(&[..])`），不再另写一个会漂移的 `22`；`train::KERNELS` 原本每行把同一个名字写两遍（`Entry { op: "gemv_i4", run: gemv_i4 }`，88 行），现在只有一列标识符，名字由 `stringify!` 派生——判据没有省掉：`const _: () = same_order(&KERNELS, &op::REGISTRY)` 仍在，本轮把两行对调验证过，编译期报"名字或顺序漂移"。`op/src/lib.rs` 479 → 431、`train/src/store.rs` 1531 → 1479、`crates/*/src` 55 138 → 55 086 行。`tests/contracts.rs` 的"注册表与注解互为镜像"门禁随之改为解析新形状（仍检查注解→注册表、注册表→注解、`MODEL` 必须由注解生成三个方向，断言一条未减）。新增 1 条用例（注册表与生成表逐条互查）。**"22 个 `forward` shim 宏化"这一半经实测撤回**：24 个 `pub fn forward` 里真正单行委托的只有 9 个，宏化只省 18 行却把 9 个公开签名藏进宏；`train` 的 22 个逐算子转接段属"内容差异"不属"结构重复"，一并保留。`[[gate.debt]]` 仍 250 条，`store.rs` 的 `lines` 基线 1529 → 1479（只降）。

- **`#[model]` 的十二段同形分支改成一张骨架表**（§8.2 第二梯队第 8 项）。`annotation/src/codegen.rs` 原先每个骨架各写 8 行逐字相同的字段列表（`group`/`shape`/`dtype`/`axes`/`shape_ctx`/`cost`/`sample`/`reference`），合起来是 152 行 `else if`，而真实差异只有四类：模块名、`group` 的写法、入口是否泛型、是否附带粒度断言。现在是一张 12 行的 `SKELETON_TABLE`（`codegen.rs:262-286`）加 6 个小函数，加一个骨架 = 加一行数据。`render_model` 实测 **ccn 15 → 2、nloc 233 → 19**，全仓最长函数因此从它变成 `prof/src/prof.rs:validate`（202 行）。三个例外留在数据里而不是回到分支：`gemv` 的粒度取 `Format::GRAIN` 并生成两条编译期断言、`gemm` 的模块名就是注册名（`module = None`）、`flash_attention_f32` 的 `reference` 由一张按注册名的改写表给出——后者就是 P1-5「宏知道算子名」的全部剩余。**行数没有下降**：`codegen.rs` 761 → 794（+33），因为表从两列变成五列、每一件新东西都带一段"为什么"；本轮买的是单函数复杂度与"加骨架只加一行"，已按实测记录，不按预估的 −80 记。`crates/annotation` 用例 53 → 54（新增锁表测试，并在既有测试里加了两条反向断言：表里每行必须在闭集内、表长必须等于闭集长度）；`[[gate.debt]]` 251 → 250（`render_model#0` 实测 2/19，删除成立）。顺带修 `syntax.rs:452` 的"缺省为 `gemm`"（实际是 `gemv`）与 `syntax.rs:384-387` 的"自己的分支"（现在是"自己的一行"）。

- **`data` 的可执行代码覆盖率收敛到 100%**。逐行覆盖报告暴露 `Dataset` 的三个 trait 方法（`cols`/`batches`/`remaining_batches`）在具体类型上的同名调用全部解析到 inherent 方法，trait 转发路径从未被执行（区域覆盖 96.3%）。新增一个通过 `&mut dyn Dataset` 取数的测试，走完整对象安全路径，验证三个转发与 `next_batch` 流；`batch`/`dataset`/`split` 三个含逻辑文件的行与区域覆盖均为 100%（`lib.rs` 仅含 `mod`/`pub use` 与文档，无可执行体，其 0% 是工具对纯声明模块的已知现象）。Miri 复测零 UB，全 workspace clippy（`-D warnings`）、测试零失败。

- **elementwise 的计时不再手写在测试里，改为 criterion 基准**。`crates/benchmarks` 原先用 `WARMUP = 8` / `TIMED = 64` 加 `Instant` 自己实现了一遍预热与计时——而预热多久、采样多少次、测多久正是 criterion 的配置项（`--warm-up-time` / `--measurement-time` / `--sample-size`），手写这套既不可配置也没有统计口径。现在该文件删除，`crates/benchmarks` 只声明"一次迭代是什么"（`bench_function`）与"一次迭代处理多少元素"（`Throughput::Elements`，报告直接给 elems/s），测量细节交给 criterion；`llm-benches` 因此不再需要 `tool` 这个 dev 依赖（回退判定仍由 `tool::regress` 的单元测试锁住，固定硬件上的 bench job 再接线）。同步：`docs/ARCHITECTURE.md` 的边表 75 → 74、`+dev` 11 → 10，`span_parse` 文档里已不存在的 `llm-regress` 改为 `tool::regress`。

- **算子层的四份表达式缓存收敛为一份，`gemv` 的三份输入校验上提**（§8.2 第二梯队第 5、6 项）。四条骨架原先各写一份 `OnceLock<Result<Formula, Error>>`——`gemv::Cache`、`elementwise::BinCache`、`unary::UnaryCache`、`reduce::ReduceCache` 逐字同构（139 行），差别只有表达式文本、操作数表与错误帧；现在只剩 `einsum::FormulaCache`（`crates/op/src/einsum/cache.rs`，61 行），三件事都成了 `get` 的参数，`site` 由**骨架**给出（同一条骨架的所有精度共享一个位置，重复写在每个实现里只会多出几份可以写歪的事实）。**没有做"泛型 `static`"**：`static` 仍留在各实现的 `cache()` 里——写在泛型作用域里的 `static` 在 Rust 里只有一份，`Dense<B>` 与 `Quantized<G>` 会共用同一个单元格，先跑的那一份会让另一份拿到错误的轴表（`gemv.rs` 原注释记录过这次实测）。资产是"一份逻辑"，不是"一份单元格"。同一轮里 `OperandSpec` 从 `gemv` 搬回 `einsum::formula`：它描述的是表达式，而 `elementwise`/`unary`/`reduce` 三条兄弟骨架原先都在反向依赖 `gemv`。校验侧：骨架的 `reference()` 一直声称"输出非空、输入够长、权重够长由骨架负责"，但三条 `forward` 各抄了一遍，`gemv_i8` 还为此专门写了一个 `layout()`（32 行、ccn 6）；现在收成 `non_empty` / `scaled_groups` / `required_weights` 三个 helper（错误文案逐字保留，上界改成饱和乘，溢出即判"长度不足"而不是回绕成轻松满足的小数），`gemv_i8::layout` 整段删除，格式只保留真正的差异（int4 的偶分组、稠密格式不该收到标度）。**实测**：`crates/*/src` 55 221 → **55 137** 行（−84），文件 172 → **173**（+1 为 `cache.rs`），`[[gate.debt]]` 252 → **251**（`layout` 消失；`gemv_i8::forward` 一度因 "3 个 `for` + 2 个 `?`" 被门禁判 ccn 6，把内层点积改成一次 `map().sum()`（与原循环同为从左到右折叠，逐位相同）后回到 5），**没有新增或抬高任何基线**；用例 **60 目标 / 1 066 通过** 与改动前逐项相同；`docs/ARCHITECTURE.md` 的所有权条目 `Cache` → `FormulaCache`，`[[ownership.unlisted]]` 里三个已不存在的类型名一并删除。

- **`diagnostic` 正交拆分并收敛构建期内核**。错误类型按职责拆为 `code.rs`（`Code` 闭集与退出码契约）与 `error.rs`（`Error`/`Result`/`from_io`），`lib.rs` 只声明模块并 `pub use` 原路径，20 个下游 `use diagnostic::...` 零改动。三个构建脚本重复的扫描/读取/表达式生成收敛到 `diagnostic::buildgen`（`collect`/`read`/`emit`/`relative`），`toml` 作为可选依赖关在非默认 `build` feature 后，默认构建仍是纯 std、零依赖（`cargo tree -e normal` 无 toml），运行期无解析器（ADR 0022）。buildgen 位于 `src` 内，为满足「禁止 panic/裸 unwrap」调用门，失败一律以 `diagnostic::Result` 返回、panic 边界留在各 build.rs；并把 `collect`/`value`/`scalar` 拆成小函数，将圈复杂度压进阈值，删除已达标的 `lib.rs:fmt#1` 基线。另清理 `cutile` 无构建脚本却残留的 `[build-dependencies] toml`。三个生成产物字节级对拍一致（除被删除的基线），`docs/ARCHITECTURE.md` 标注 `diagnostic::toml` 已在 ADR 0022 移除。

### Fixed

- **同族长度缺陷收口：文件里的偏移不再参与未检查的加减**。上一轮修完四条 P0 时留下四条"同族未复核"，本轮按同一判据（长度只要来自文件，就不许参与未检查的加/减）处理完，并在过程中发现第五条同类违例。① `zip::slice` 的 `bytes.get(at..at + len)` 原先在 `at` 是 ZIP64 的 `u64` 原样转成的 `usize` 时会算术溢出 panic（`at = usize::MAX` 时 `attempt to add with overflow`），release 下回绕成 `get(start..start-4)` 再落成 `None`；改为 `checked_add` 并报「ZIP 字段偏移溢出」，保留"偏移本身溢出"这个事实而不是退化成"越界"。② `zip::zip64_extra` 的 `stop = extra_start + extra_len`、`at = body + size` 未检查，且按 `field * 8` 取 u64 时**不看子字段声明的 `size`**——声明 8 字节的字段能读到下一段的 8 字节并返回 `Ok(错值)`；现在推进用 `saturating_add`，取值交给新函数 `zip64_field`，后者要求 `(field+1)*8 <= size`，否则报「ZIP64 extra 声明长度不足，字段被截断」。③ `gguf::raw` 的 `data + offset` 用了 `checked_add` 但随后的 `start + len` 没有（`offset` 是文件里的 `u64`，取到 `usize::MAX - data` 即 debug panic），改 `saturating_add`。④ `gguf::Reader::array_count` 的 `len - pos` 改 `saturating_sub`：不变式确实成立，但改法把"依赖别处的正确性"变成"本处自证"。⑤ `axes::search` 末尾的 `assert!(verified(..), "raw 基线必须逐位还原")` 改为 `verified(..).then_some(best).ok_or_else(baseline_broken)`——公开函数本来就返回 `Result`，库内 panic 没有理由。**四条回归测试是先红后绿的**：`gguf.rs:386`、`zip.rs:167`、`zip.rs:130` 三处报的是 `attempt to add with overflow` 原文（证明踩在真实溢出路径上），第四条报的是"字段 1 超出声明的 8 字节"（证明旧代码把邻居数据当成了字段值）。写法上有一条被门禁逼出来的结论：`ok_or_else(|| ..)` 里的 `||` 会被复杂度门禁记成一个布尔分支，第一次改动因此把 `axes::search` 从 ccn 8 推到 9 并被拦回，最终改为传函数项；与此同时 `zip64_extra` 抽出字段读取后 ccn 降到限额内，棘轮主动要求删除其基线条目（`[[gate.debt]]` 253 → **252**，本轮没有任何限额被抬高）。实测：`cargo test -p llm-weights --lib` **174 通过**（原 170 + 4）、`cargo test --workspace --all-features` **60 目标 / 1 066 通过 / 0 失败 / 1 忽略**、`crates/*/src` 172 文件 / **55 221** 行、clippy 与 rustdoc 各 0 告警、`cutile doctor --device cpu` 退出码 0。**有意不改**：`zip::data`、`reader::read` 等处的直接索引（不变量由构造函数建立）、循环内自增偏移（上限由成功的切片给出）、`dsa::Xof::take`（内部类型、偏移不可由外部给定，记为已知 panic 面）、`rans` 的全局分配上限（属策略而非长度校验）、ZIP64 的字段顺序假设（现为"要么正确、要么报错"，按 sentinel 重排字段留待真实样本）。

- **四条 P0 崩溃路径修复，外加 `dsa` 提示位解包的第五处越界**。原则只有一条：长度由调用方契约给出，不由被解析的字节自报。① `rans::decode` 不再拿帧头自报的 4 字节长度当分配依据：`declared_len`（`rans.rs:380`）先与调用方上限比较，上限来自段描述符的类型与形状（`reader.rs:84-86`）或元素数推出的 `payload_cap`（`axes.rs:173`）；② `bin::pop_pairs`（`:448`）在 `chunks(2)` 前判奇偶（`:450-452`），奇数 `SETITEMS` 返回 `invalid`；③ `Snapshot` 的标签表改为一次判完（`tags()`，`train/src/snapshot/mod.rs:346`），`encoded_len`/`encode`/`encode_into` 都走 `tags()?`，不支持的 `DType` 现在是 `unsupported` 而不是 `expect`；④ `dsa::pk_decode`（`:378`）要求长度等于 `pk_len()`、`sig_decode`（`:415`）不小于 `sig_len_min()`（`:359`），截断的侧车不再进切片；⑤ 新发现：`hint_unpack_dyn`（`:715`）的 stop 计数可与段长自相矛盾，原写法 `y[index..total]` 会以 `start > end` panic，改为 `get(index..total).is_none_or(..)`。两条修法各自把该函数 ccn 推到 9，都被静态门拦回，最终落到登记值 8——`[[gate.debt]]` 未抬高。五条回归测试：`rans.rs:470`、`bin.rs:582`、`snapshot/tests.rs:22`、`envelope.rs:161`、`dsa.rs:766`，其中 P0-4b 做了「删掉修复即 panic `slice index starts at 4 but ends at 0`」的反向验证。实测：`crates/*/src` 54 519 → **54 717** 行、`[[gate.debt]]` 255 → **254**、用例 **1 057 通过 / 0 失败 / 1 忽略**、`cutile doctor --device cpu` 退出码 0。残余风险（未做）：`rans` 的上限仍由文件里的形状推出、没有全局封顶；`gguf.rs:199,358`、`zip.rs:128-142`、`axes.rs:89` 四条同族缺陷未复核。

- **平台门禁补齐：`storage` 的 `mmap` feature 现在只在 Unix 编译**。全仓唯一的 `unsafe` 岛是 `mmap(2)` 的薄封装，模块门禁却只看 feature 不看目标，于是 Windows 上 `cargo test --workspace --all-features` 必红——本轮用交叉目标把它从「推断」变成「已确认」：`cargo check -p llm-storage --features mmap --target x86_64-pc-windows-msvc` 报 `unresolved import std::os::unix`（`mmap.rs:14`）与 `no method named as_raw_fd`（`:71`）。修法是 `cfg(all(feature = "mmap", unix))`（`storage/src/lib.rs:29-32`，模块与再导出同改），唯一消费者 `weights/tests/round_trip.rs:162` 加 `#[cfg(unix)]` 而**不是**在非 Unix 上 `return`（那会变成可空转的绿）。验证：`cargo check --workspace --exclude llm-benches --all-targets --all-features --target x86_64-pc-windows-msvc` 退出码 0——这是全仓第一次在 Windows 目标上整体通过类型检查，因此「`mmap` 是唯一缺口」也有证据；`llm-benches` 被排除是因为 `criterion → alloca` 的构建脚本需要目标平台的 C 编译器，且只做了类型检查、没在 Windows 上运行测试。顺带修三处文档漂移：`docs/ARCHITECTURE.md:99` 的「现有两条算子」按 `op/src/lib.rs:111` 的 `[Registration; 22]` 改为 22 条；`crates/op/README.md:108` 整节（标题就是「现有的两条算子」、表只有 2 行）改为按族 7 行表并写明「清单不在这里维护，`REGISTRY` 是唯一事实源」，避免 README 成为第二份算子清单；`crates/op/README.md:147-151` 的依赖段（`llm → cutile → op → einsum`、`diagnostic` 的 `toml` 子模块）与 `crates/weights/src/lib.rs:4-10` 的拥有/依赖段按 `Cargo.toml` 与 `pub mod` 实况重写。另删 `weights/src/seg.rs` 里 `validate` 的空循环（`for axis in &self.shape { let _ = axis; }`），把「轴长允许 0」的决策移进文档注释；复核发现 `tool/src/ccn.rs:138-142` 的死递归与 13 处 `pack::*` 诊断锚点在更早的提交里已消失。实测（**本轮自身改动**）：`crates/*/src` 54 717 → **54 727** 行、用例 **1 057 通过 / 0 失败 / 1 忽略**（不变）、`[[gate.debt]]` 254（不变）、成员 25 与边 75（不变）；复核时全仓为 **1 062 通过 / 0 失败 / 1 忽略**、172 个源文件 / 54 980 行——差额来自同期另一个进行中的 `diagnostic` 拆分（新增 `code.rs`/`error.rs`/`buildgen.rs`），不是本轮改动；那次并发改动中途的两次红灯已随一次性审计文档一并归档删除，此处只保留结论。`cutile doctor --device cpu` 退出码 0。

### Added

- **运行期删除 TOML 解析器：语料改在构建期解析（ADR 0022）**。ADR 0021 只解决了「解析器住谁家」，没回答「运行期为什么要解析」：`ops/*.toml`、`devices/*.toml`、`lints/vocabulary.toml` 都是随二进制发布的编译期资产，而唯一属于用户的输入是配置文件，它只有三个路径键——这两条都意味着常驻解析器是纯成本。于是新增三个 `build.rs`（`crates/{op,prof,tool}/build.rs`）用 `toml` **构建期依赖**把语料展开成 `diagnostic::data::{Table, Value}` 字面量（`op::declared` / `prof::embedded` / `tool::vocabulary`），入口随之改为 `OpDecl::parse(&Table, tmpl)` 与 `Profile::parse(&Table)`，`op::load`/`dir`/`BUILTIN`、`prof::load`/`dir`/`Archive`/`builtin`、`cutile::builtin::{from_dir,from_repo,from_layers,disk_profiles}` 与 `ProfileError::Syntax` 全部删除——运行期没有「按目录装载语料」这件事。数据树补上构造与路径改写（`Table::from_pairs`、`Value::text/int/real/flag`、`Table::{set,push,remove,at,at_mut}`），原先靠拼接 TOML 文本制造非法声明的测试改为表改写，类型仍由编译器检查。运行期配置层再砍一层：`cutile::settings` 只剩「内置默认 < `CUTILE_*` < `--set`」，**没有配置文件**、没有 `schema` 字段、没有文件相对路径基准；`missing`/`io` 两个分支随读取一起消失（退出码契约不变）。`fuzz` 的 `Toml`/`Profile` 两个目标删除（没有运行期入口可灌），`fuzz → +dev: prof` 成为死边并删除。实测：`crates/*/src` 54 959 → **54 519** 行（−440，删掉 585 行解析器）、`[[gate.debt]]` **271 → 255**、成员 25 不变、边 **76 → 75**（`+dev` 12 → 11）、用例 **1 052 通过 / 0 失败**、`cargo tree -e normal` 里仍无 `toml`/`serde`（运行时第三方依赖 0）。

- **`llm-config` 解耦删除：TOML 语法层并进 `diagnostic`（ADR 0021）**。设置层在 ADR 0020 里已经搬进 `crates/cutile/src/settings/`，只剩 583 行语法层；改名 `llm-toml` 的写法被否决——成员数、边数、指向它的六个 `use` 一个都不会变，那只是把名字换得更准确一点。真正的事实是那条边本来就冗余：`op`（`ops/*.toml`）、`prof`（`devices/*.toml`）、`tool`（`lints/vocabulary.toml` 阈值）、`cutile`（经设置层）、根包与 `fuzz` 全部已经依赖 `diagnostic`（错误类型全仓共用），于是语法层直接进同层零依赖的那个成员：`crates/config/src/toml.rs` → `crates/diagnostic/src/toml.rs`（`pub mod toml;`），`crates/config` 目录删除。**成员 26 → 25，依赖边 82 → 76，新增零条边**（消失的六条是上述成员各自指向语法层的那一条）；`prof` 的 `pub use llm_config::toml` 再导出随之删除——它自己有 17 处 `crate::toml`，说明「借道 `prof`」一直在发生，现在改为 `use diagnostic::toml`。语法错误仍只带行号与说明（`ParseError`），类别由调用方按语义给：语法层不该知道一次失败该算 `invalid` 还是 `missing`。门禁同步：`[gate.debt]` 8 条键改路径（ccn / nloc 逐条不变，搬家不降复杂度）、`[[ownership.unlisted]]` 的 `config` 条目改为 `diagnostic`，`tests/architecture.rs` 删掉 `member == "config"` 例外（成员不存在了），改成**文件级**豁免 `crates/diagnostic/src/toml.rs`——那里的 `String` 是消息片段，对外一直是 `ParseError`，例外因此是被收窄而不是取消。用例 62 目标 / 1 063 通过 → 60 目标 / 1 062 通过：少的 1 条是 `llm-config` 自己的 doctest 目标，`toml.rs` 内 13 个测试随文件继续执行。文档同步：`README.md` 成员表与依赖图、`docs/ARCHITECTURE.md` 层表/边表/所有权表/吸收链、`crates/cutile/README.md` 的配置边界、ADR 0005 的传递依赖说明。

- **`cutile` 收掉最后几处「同一事实两份」：设备探测、Metal 驱动、MSVC 查找、模板渲染**。① `probe::detect_cpu` 不再用 `cfg!(target_feature)`——那回答的是「这个二进制按什么指令集编的」：默认 baseline 构建下永远报不出 `avx2`，`-C target-cpu=native` 的产物拷到别的机器又会虚报能力；现在改 `is_x86_feature_detected!` / `is_aarch64_feature_detected!` 运行期问 CPU，测试与 cpuid 对拍。② `probe::fake` 不再手写 nvidia/intel/ascend/metal 四张指纹，改为从档案自己的 `[fingerprint]` 反查（`Exact` 取原文、`Range` 取下界、`Contains` 取第一个子串、`Set` 取前缀加成员、`OneOf` 取第一个分支），加一台设备只需要一份 TOML，不再是「漏改代码只有在运行期报『没有档案匹配』」。③ Metal host 驱动里写死的三处 `8` 改为 `WITNESS_MAX_OUT`，由 `witness::MAX_OUT` 注入 `#define`——MSL kernel 侧一直是 const 泛型 `{max}`，驱动侧是裸字面量，这是跨 FFI 边界唯一没有编译期保护的重复（`MAX_OUT` 一变就错位）。④ `find_cl` 不再写死五个开发机盘符、也不再 `sort()` 后取「最新」（`9.29` 按字符串序排在 `17.10` 之后，多版本共存时会选中更旧的一套）：改 `vswhere -find` 加 `VCINSTALLDIR`/`VSINSTALLDIR`，按版本号数值比较，标准安装路径不再漏掉。⑤ 见证语法门 `parse_all` 改走 `witness::source` 的「内嵌优先」：把文本落到一次性目录再交给语言前端，不再直接把 `witness_root(settings).join(file)` 交给 clang——在没有源码树的发布机上，原来会把「本机没有这份源码」报成「见证被自己的语言拒绝」。⑥ `substitute::render` 变成 `ast::render(&ast::build(..))` 的薄包装，槽位解析不再有两份逐行相同的循环（`ast_renders_byte_identical_text` 保留为回归门）。⑦ 判定型返回值补 `#[must_use]`（`cap::Path::is_head`、`NeedReport::is_satisfied`、`FragmentGap::is_aligned`、`Timing::min_ns`/`p50_ns`、`emit::Source::lines`/`ext`），删掉只有测试在用的 `Proof::with`，`cap` 里两处指向旧 `audit::witness` 的文档改到 `crate::witness`；`DESIGN.md` §5.4 的 CPU 指纹键从虚构的 `isa = …` 改为代码实际读的 `arch`/`features`。复杂度基线净减 3（`find_cl`、`gate::parse_all`、`substitute::render` 已达标，条目删除，无新增）。

- **`cutile` 删掉 `.config/report` 与 `witness/` 两层目录，四条命令共用一条流水线**。`.config/report` 这个名字从一开始就不成立：它不产出“报告”（排版在 L7 的驱动方），却让 `doctor` / `verify` / `bench` / `execution` 各自重写一遍「找算子 → 取模型 → 建立证据 → 协商 → 选 tile → 校验签名 → 拼源码」，`.config/report` 700 行里还有 620 行是测试。现在四个命令是 crate 根的 `cutile::{bench, doctor, verify, execution}`，共用 `pipeline`（`Run` = 算子 + 模型 + 档案 + 证据，`Prepared` = 计划 + 源文本 + 缓存键，`Prepared::harness` 负责把样本变成实参），CUDA 路径也走同一个 `Prepared`；`CacheWhy` / `CacheState` / `why`（`doctor --why` 的口径）留在 `doctor`，`CACHE_LIMIT` / `cache_gc` 单独成 `gc`（`cache.rs` 有 1,245 行的行数棘轮，命令策略不该塞进去），`Fragment` 与 §16 的两个门槛归 `emit`。`witness/` 同样删除：`#[annotation::frontend]` 生成的是 `crate::witness::Language`，这层目录只靠 `pub use` 转发，现在 `witness` 是真正的顶层模块（契约 + 证据 + 编排），执行前端在 `device`、语法前端在 `gate`。顺带三处去重：片段「去重复终结符 + 按调用点缩进对齐」抽成 `substitute::place` 由字符串路径与 AST 路径共用；`reference::tolerance` 不再用 `_ => 0.0` 兜底，24 个 `DType` 变体逐条表态（新格式必须在编译期决定阈值，而不是静默落到“位精确”）；`clang` / `metal_toolchain` / `HOST_FLAGS` / `c_compile_inputs` / `harness` 从 `witness` 移到 `device`——它们本来就是程序生成与工具链探测。`lints/vocabulary.toml` 里 20 条 `.config/report` 的键改道到新路径，`docs/ARCHITECTURE.md` 的 L3/L7 行与 `cutile/DESIGN.md` 的预算段同步。

- **`data` 的契约改成「可验证」，并把门禁做成双向**。`Batch`/`Chunks` 现在借用调用方的 token 缓冲（每批零分配、零拷贝），批几何搬进 `Chunks::new` 构造期校验（列宽 0 / 批行数 0 / `rows * cols` 溢出都报 `Code::Invalid`），于是 `Dataset::next_batch` 不再收 `rows` 参数、`None` 只剩「剩余不足一整批」一个含义。三个含混的「行」拆成互不重叠的三个计数：`cols`（列宽）、`batches`（完整批总数）、`remaining_batches`（还没取走的批数），`Chunks::rows` 只表示每批行数——此前它返回的是批数且无视游标。`Split::new` 不再把 100% 验证集静默夹成 0%（直接报错），百分比算术换成不会溢出的 `u128` 中间量（`saturating_mul` 在 `total` 接近 `usize::MAX` 时会把比例放大近一倍）。删掉 `Batch::len` / `is_empty` 这对同义/恒假的死 API；crate 按类型拆成 `batch` / `split` / `dataset` 三个模块，`lib.rs` 只留文档与再导出。

- **所有权门禁现在是双向的，L7 组装点不再定义类型**。`tests/architecture.rs` 新增两条断言：每个成员的 `src` 里的 `pub struct|enum|trait` 必须出现在 `docs/ARCHITECTURE.md` 的所有权表，或登记在 `lints/vocabulary.toml` 新增的 `[[ownership.unlisted]]` 存量基线（基线不允许过期：某个名字进了表或不再是 `pub`，条目必须删除）；`src/**/*.rs` 不得出现任何顶层类型定义。后一条立刻兑现了它守的东西：`src/main.rs` 里的 `LmHead` 迁进 `crates/nn/src/lm_head.rs`，成为 `nn::LmHead`（`nn` 里本来就有同名的 `lm_head` 声明，因此是整体搬走并合并，而不是并存两份）。

- **`dtype` 的闭集契约上移到类型层，访问器与行表的对应也有断言了**。`Format` 的 supertrait 补上 `Display`：每个闭集的对外名字过去只是各 impl 各自遵守的约定，泛型代码要额外写一条 `where F: Display` 才打印得出，现在它属于闭集定义本身；同处补一节「哪些枚举有规范名」，写明 `dtype::Class` 与两个 `Scale`、两个 `Family` 是分类枚举、刻意没有 `Display`/`FromStr`/serde——给内部建模细节编一套对外名字，等于把建模方式固化成一份随时会过期的承诺。测试同步收口：`QType`/`MxType`/`Element` 各加一条「每个公开访问器 == 自己那一列」的断言（`bits()` 与 `block_bytes()` 同为 `u32`，把列取错此前不会让任何测试变红，而 `scale()`/`family()` 在 crate 外根本没有读取点）；删掉 `tests/geometry.rs` 里与 `dtype.rs` 内联测试逐字相同的 `bytes_for` 字面断言（「字面事实归 `src`、性质归 `tests`」这条分界此前被自己违反），以及 `element_table_is_locked` 里一条类型已经锚定、永不失败的 `assert_eq!`；`mxtype.rs` 两个同名 `discriminant_fits_one_byte` 改名为 `mx_type_*` / `element_*`，`Element` 的 `Format` 实现改为转调 inherent 访问器（与另外三个闭集同构，与「几何公式只有一份」的分工一致）。文档修正：`tests/mx_spec.rs` 标题错字、README 的「Identity is locked」漏掉元素行表与访问器断言、`Cargo.toml` 描述漏掉元素集合、ADR 0018 里「三个格式闭集各实现一次」改回四个。

- **跨模块把「一个事实只写一遍」推过 `dtype` 边界**。`tensor` 里 `u8` 曾被标成有符号 `I8`（与早先 `weights::bin` 的同一类错误），`Shape::element_bytes` 也另写了一份字节公式——现在 `u8 → U8`，字节数委托 `DType::bytes_for`。`annotation` 的 `skeleton` 闭集与生成分支一一对应：新增 `gemv` 成员并设为缺省，`gemm` 只认 `gemm_f32`/`gemm_i8`/`gemm_i4`（此前写错名字会静默落到 gemv），重复的 `skeleton` 键会被拒，三种 gemm 分支收成一次表驱动调用。`serve` 的 `Kind::from_code` 由 `ALL` 推出（不再有第三张手写表），校验和测试开始覆盖帧头（把 `chunk` 改成同样合法的 `error`，只有 CRC 能发现），校验尾长度收敛为一个 `trailer_len`。`engine` 的「下一页」公式抽成 `next_pages`，`Sequence::new` 拒绝上下文长度溢出，`Cache::release` 不再把「从未分配」报成「重复归还」。

- **散落在各模块的诊断与门禁修正**。`testkit::approx_eq` 现在对相同的无穷判等（`inf - inf` 是 NaN，原来会被判不等）；`tool` 的代码度量删掉一段算完即丢的递归、`Judgement` 补上 `#[must_use]`；`storage::Mmap` 修掉非 64 位目标上的编译失败（`cfg` 把函数体整块裁空）与 `u64 as usize` 截断，`pack::` 这一已删除 crate 的残留引用改为 `weights::`（`weights` 的全部错误站点名同步）；`prof` 的区间诊断不再打印假不等式（`16 < 8`），测试不再保护缺陷；`shard` 的集合通信用例改成可观察断言；`config` 的键表由测试锁定内容，并写明 `ALL` 是唯一没有穷尽 `match` 保护的一处。同时把棘轮基线里因早先重构而失配的四条（`weights::bin` 函数改名、`prof::atom` 函数编号、`gguf` 调用数、`annotation::syntax` 行数）对齐。

- **`dtype` 补齐互操作，`Element` 收进闭集纪律**。四个格式闭集（`DType`/`QType`/`MxType`/`Element`）现在共享同一份身份读法：`Element` 的位宽与标量对应从两条 `match` 变成 `ELEMENT_SPECS` 行表（name / variant / bits / dtype），`ALL` 锚定表长，`Format` / `Display` / `FromStr` 都读这一行；全部闭集实现 `FromStr`（零分配错误 `UnknownName`），`DType` 另有 `From<DType> for u16` / `TryFrom<u16>`（`UnknownCode`），并穷举 65536 个 `u16` 验证 code 是双射。刻意不给 `Default`：默认精度会静默宣布一个没有被声明的类型。`serde` 与 `bytemuck` 作为**默认关闭**的可选 feature 接上：serde 的线上名字直接取 `Format::name`（不再用 `rename_all`——`MxType` 的 `mxfp8_e4m3` 并不是变体名的小写），bytemuck 给 `NoUninit` + `CheckedBitPattern` 而不是对枚举不成立的 `Pod`；默认特性集仍是零依赖。`MxType` 另用 OCP v1.1 的数值表与手写字节块对拍了块长、每块字节与 E2M1/E4M3 解码（`tests/mx_spec.rs`），真实模型字节流仍待 `weights` 接入 MXFP4 反量化后再补。

- **线协议第 2 版：帧第一次能用在真实连接上（ADR 0016）**。`proto` 之前只有"10 字节帧头 + 载荷"的整帧编解码，按那套接口写不出一条真实连接：连接上读到的是字节流（半帧、多帧都要调用方自己切，而"这一帧多长"的规则每多一个调用方就多一份可能与 `parse` 不一致的实现），载荷长度是 `u32`（对端写 `0xFFFF_FFFF` 就要接收方替它分配 4 GiB），载荷本身没有任何覆盖（一个比特翻转会被当成合法数据送进模型，只在若干层之后表现成"这轮输出很奇怪"）。现在三处一起补：一，`Decoder` 是"字节 → 帧"的全部状态——`push` 追加、`next_frame` 取帧，返回 `Result<Option<Frame>>`（`None` 是"还没到齐"，不是错误），交付的 `Frame` 借用缓冲（零拷贝，要留住才用 `envelope()` 换一份），`finish` 区分"读完"与"读到一半断了"；失败是一次性的（帧头坏了没有可靠的重同步点，第一次失败之后每次调用都返回同一个错误，复用对象须显式 `reset`），内存有上界（超限声明在 10 字节帧头到达时就被拒，已交付前缀在 `push` 里按"挪掉的不少于留下的"压缩，摊还仍是每字节 O(1)）。二，`MAX_PAYLOAD = 1 MiB` 成为协议常量而不是配置项：接收方上限必须与发送方一致，否则"本端写得出、对端收不下"会变成一个只在特定连接上出现的失败。三，第 2 版在帧尾加 4 字节 CRC-32（覆盖帧头与载荷），"长度对得上、内容坏了"因此在解码期失败；帧头一个字未动，第 1 版帧照旧可读且版本在往返里保留，`serve` 的调用点一行未改。版本 0 与"比本端新"也分开报（`invalid` 对 `unsupported`），编码失败不留半帧（`encode_into` 先自检再写，失败时调用方缓冲原样）。新增 25 条用例：字节布局对照（v1 / v2 逐字节差异）、逐字节喂入、64 种随机切分、坏魔数 / 超限声明 / 载荷翻转之后的"不再重同步"、断连与 `finish`、版本 0 与过新、失败编码不动缓冲、100 轮收发后缓冲不增长，以及 CRC-32 与逐位定义、标准检验值 `0xCBF43926` 的对拍。文档同步：ADR 0016、架构所有权表补 `Frame` / `Decoder`。

- **见证补上第三种结论：语言门**。host 自检只能证明"这段代码在 C 里成立"，而 `metal/witness.metal` 曾长期写成 `float *out`——MSL 没有默认地址空间，host `cc` 绿灯、`xcrun metal` 编不过，等于那份证据从未在自己声称的语言里存在过。现在设备不在本机时，见证另交给**它自己语言的解析器**只解析一遍（不链接、不执行）：`cuda` → `clang -x cuda --cuda-host-only -nocudainc -nocudalib -fsyntax-only`（`--cuda-host-only` 让 `__syncthreads()` 这类设备内建在 host 函数里被拒），`opencl` → `clang -x cl -cl-std=CL3.0`（没开 `cl_khr_fp64` 时 `double` 是错误——`cc` 照收），`ascendc` 没有开源前端就 `Absent` 并说清是哪门语言。语法前端与执行前端同构：`Front` 只加 `TOOL` / `CHECK` 两条编译期常量，语言身份仍由 `#[annotation::frontend(..)]` 生成，分发是 `const GATES` 表，新增一门语言 = 加一个类型 + 一行。门不执行任何一行，所以**不改 `grade`**；但它被拒绝会记进 `Evidence::failure`——那是证据缺陷，与"本机没有这台设备"分开报。`Evidence` 新增 `language`，`doctor` 的 `[证据]` 行列出现在有三半，`verify` 的失败文案也会带上"语言门 … 通过（N 条）"。新增 3 条用例（两台加速器的见证必须在自己语言里编过、设备执行覆盖时免开、OpenCL 拒收裸 `double` 而 host C 放行）与 1 条契约（`impl Front` 必须标注解且在 `GATES` 里）。实机结果：`nvidia` 17 条经 `-x cuda` 通过、`intel` 17 条经 `-x cl` 通过、`ascend` 如实缺席、`cpu`/`metal` 免开。

- **注解、格式与注册表三处互为判据（P0-11）**。类型系统守住了"注解说的是哪份格式"，却守不住三件纯文本事故，现在各有判据：① **格式自证**：`Format::CHECK`（编译期）要求 `GRAIN > 0` 且派生轴的除数**就是**形状粒度（`gemv::derived_matches_grain`）——`cost` 用派生轴除数造轴、`sample` 用粒度算每行标度个数，写歪一个数不会让任何一步失败，只会让代价与样本说的是两种语义；生成物里的 `const _: () = <别名 as Format>::CHECK;` 是求值点（不被引用的关联常量不报错）。② **注解 ↔ 输出类型**：`Signature::output_type()` 与 `CType::dtype()` 把 `.k` 的可写输出读成 `DType`，`#[model]` 生成的校验与 `Model::dtype` 逐项比较（注解写 `f16` 而内核写 `float *y` 是测试红灯）；`backend::harness` 另加一道同源闸，非 `float` 输出在拼任何 C 之前就被拒，诊断带算子名与缓冲名。③ **注解 ↔ 注册表**：`tests/contracts.rs` 新增双向门禁——标了 `#[model]` 必须在 `REGISTRY` 里有同名条目，注册表的 `model:` 必须是 `..::MODEL`（手写 `Model { .. }` 字面量即红灯），语义表因此只有一个生产者，且没有豁免节。文档同步：`docs/ARCHITECTURE.md` 的注解段与守门表、`cutile/DESIGN.md` §11.1 的阈值段、`op/README.md` 的"三处自证"、`lints/vocabulary.toml`（并修掉"dylint 规则也在读这份白名单"的不实描述——那条规则尚未落地）。
- **GEMV 的两种精度收成「泛型骨架 + 格式契约 + `#[model]` 注解」**。`gemv_i4` 与 `gemv_f32` 原先各写一份代价、样本、校验与遍历——两边的 `formula` / `cost` / `sample` / `reference` 逐字相同，只有权重格式不同。现在骨架只有一份（`op::gemv`）：`Format` trait 承载"随精度变化"的部分（`Elem`、`EXPR`、`OPERANDS`、`GRAIN`、派生轴、标度、`row`、`weights`、`forward`），`cost` / `sample` / `reference` 按格式**单态化**，`Model` 里仍是三个 `fn` 指针——热路径上没有 `match op`、没有枚举分发。进位到类型里的只有真正静态的东西：`Dense<32>` 的块宽直接出现在 `chunks` 上（热循环），`Quantized<32>` 的声明粒度决定派生轴除数与标度个数（冷路径）；实际分组大小留在数据里，因为模板本身就把它当 `int gsz` 实参收，训练侧按 8 元素分组调用同一份 `forward` 因此不需要第二份单态化副本。`MODEL` 改由 `#[annotation::model(name = .., shape = [..], dtype = ..)]` 生成（标在 `type 名字 = <具体格式>;` 上）：`cost` / `sample` / `reference` 写成 `<名字 as Format>` 的实例、`group` 取 `Format::GRAIN`，另加"生产形状是粒度整数倍"的 `const` 断言与一条"注册表里的语义就是这份 `MODEL`、样本与 `.k` 签名一致"的测试——注解与泛型 bound 强绑定，别名不实现 `Format` 就编译不过。实测踩到一处 Rust 语义陷阱：`static` 在泛型作用域里只有一份，写在泛型函数里的表达式缓存会让两种格式共用同一格（i4 一度拿到稠密的表达式），现在缓存由具体格式持有（`Cache`）；同时修掉 `syntax` 里四处仍按旧签名调用 `target_name` 的用例。文档同步：`docs/ARCHITECTURE.md` 的边（`op → annotation`）与所有权条目、`op/README.md` 的"一份骨架，多条精度"一节。
- **见证执行器收成「注解 + trait + const 泛型 + 静态表」，并给 Metal 接上真正的 GPU 执行**。此前的执行是按 `emit` 写的 `match`，而 Metal 的见证文件只跑过 host `cc` 自检——`float *out` 在 MSL 里是语法错误（没有默认地址空间），于是那份"证据"从未在自己声称的语言里编译过。现在分三层：`#[frontend(emit = .., ext = .., tag = ..)]`（`crates/annotation`，与 `#[layer]` 共用同一趟 `expand_with` 展开与同一套类型名判据）生成 `Language::EMIT` / `EXT` / `TAG`，并展开一条 `const _` 断言校验 `EXT` 等于 `EMIT.ext()`、`TAG` 非空（`same_text` 是 `const` 版逐字节比较，`str ==` 不是 `const`）；手写的 `Device`（`available` / `program::<OUT>` / `run`）给出"怎么跑"；`Frontend: Language + Device` 把两半绑死，只标注解不写执行、或写了执行没标注解都拿不到 `Frontend`。分发是一张 `const FRONTENDS` 表，每一项都是单态化后的具体函数——没有 `match`、没有 `dyn`，新增设备 = 标一个类型 + 加一行。每条见证的写出上限以 const 泛型 `OUT` 传进两个前端，生成的 C / MSL 里缓冲区尺寸是字面量。`metal` 前端走 `xcrun metal` → `metallib` → 一段 host 驱动把 `cutile_witness` 发射到 GPU 再回读（计数走 stderr、值走 stdout，与 C 同一个 `compare`），`M5` 的 17 条见证因此拿到 `Scope::Device` 的结论；等级仍是 `L1`，因为 `L2` 要的是**内核**跑通并对拍。顺带纠正一处自证循环：`run_all` 不再按 `grade >= L2` 筛，等级是执行的结论而不是前提。新增 2 条用例（Metal 的全部见证在 GPU 上跑通；`float *out` 这种纯 C 签名必须被 `xcrun metal` 拒绝且报错点名 address space）。

- **训练侧的缓冲抽象收成“泛型 + 关联常量”，参考内核不再写 `match`**。`train::Data` / `View` / `ViewMut` 原先各自按两个变体手写一份 `match`，`Pool` 的两条分支各写一遍“取块—截断—零值分配”，`fits` 与 `empty` 又各写一份 `DType` 判断；现在密封 trait `Lane`（`f32` 与按位打包 `u8` 两个实现，带 `KIND` 关联常量与按 lane 取切片的两个方法）把这些按 lane 的单态化实现收成一份，`Kind::of` 是全模块唯一一次“元素类型 → 存储形态”的判断（`F16` / `BF16` 在那里就是 `None`）。内核侧新增 `Inputs`：`(f32, u8, f32)` 这种输入组合本身就是类型，`unpack` 一次校验元数与逐格 lane、并报出是第几格，`gemv_i4` 的适配器因此只有三行。`Reference` 也不再写 `match op { … }`：它查一张定长表，**长度由 `op::REGISTRY.len()` 给出、名字与顺序由 `const` 断言逐条对上**，所以“注册了算子却忘了给参考实现”从运行时的 `unsupported` 变成编译错误；顺带补上 `gemv_f32`——它一直在注册表里，但训练侧没有可重放的标量内核（只有 `gemv_i4` 一个分支）。新增 2 条测试：`gemv_f32` 的重放与自己标量参考逐位相同；元数/lane 不符时点名第几格且失败的那次不留占用。

- **检查点第 3 版：名字与布局都只写“相对上一条”的差，文件小 71%，校验快 15 倍**。第 2 版把每个条目的名字整串写进字节，可 2 万条目的头部里名字占了 400 KB / 720 KB——而真实权重表的名字是“同层同前缀、同部件同后缀”的（`blk.3.attn_q.weight` 紧跟着 `blk.3.attn_k.weight`）。现在每条只写“共享前缀长度 + 共享后缀长度 + 中间那几个字节”，后缀只从两条名字都还没被前缀吃掉的那一段里取，所以前后两段永远不会重叠；元素类型与形状也相对上一条，同型同形（同一层的 `gate` / `up` / `down`）只花一个字节的标签 `0`。**紧凑不能以“同一份快照有多种字节形态”为代价**：共享必须是最长写法、同布局必须写 `0`，解码器对每条重算一遍并拒掉任何“还能省却没这么写”的编码，于是 `encode(decode(b)) == b` 依旧成立，校验和也依旧等于“这份字节就是这个检查点”。CRC-32 从逐字节查表换成八路查表（第 k 张表 = 第 k-1 张再过一个字节，编译期展开），定义一个比特没变、仍能用逐位定义独立复算（`0xCBF43926` 与随机字节两条测试都还在）。实测 2 万条目 / 4096 维量化头部：720 030 → 208 021 字节（−71.1%），只验字节 1.19 ms → 82 µs（15×），`encode` 1.52 → 0.62 ms，完整解码 3.41 → 2.56 ms（其余几乎全是 `weights::Header::new` 在 2 万个长名字上做重名检查的 1.5 ms，那是所有权在 `weights` 的既有成本，本轮不动）。格式变更因此把版本推到 3：第 2 版从未发布，证书式地拒绝比“看起来能读”安全。新增 2 条测试（最长共享写法：前缀与后缀各少写一条；布局重复必须写 `0`，含“第一个条目不能共享布局”），并把跨写缓冲那条测试改成 4096 条目、额外断言体积必须远小于名字整串写下的总量——否则共享在某一轮里被悄悄关掉，只有文件变大而测试仍是绿的。
- **框架第一次用真实权重算出了数（ADR 0014）**。此前两条证据各管一头：`cutile verify` 证明"这台机器能算这条算子"（用的是算子自己的合成样本），`llm status --weights` 证明"这些层至少要多快"（纯算术）。两者之间有个洞——真实权重从没进过内核。现在补上三段：`op::model::bind` 是真实数据进入样本的唯一入口（只换 `x` / `w`，标量与输出缓冲仍来自算子样本，因此 ABI 不被真实数据顺手改掉；量化算子的打包缓冲以 `unsupported` 拒绝，不让内核去解读半份字节）；`audit::execute` 用调用方给的样本走同一条流水线（协商 → tile → 拼接 → emit → 编译 → 执行），执行前仍要求见证证据、对拍不过即错误，结论是 `Execution`（内核输出 + 标量参考 + 误差 + 变体/tile/键）；`nn::Layer::linear` 把"这一层能不能被执行"写成类型问题——`#[layer]` 注解层没有参数表，返回 `None`。组装点新增 `llm run --weights <文件> [--input a,b,c] [--device d]`：只跑一条连续链，遇到宽度不等、没有算子的编码、条目元素数不符就停下并说明，每层只按需读自己那条条目。实测两层的稠密链（`8→4`、`4→4`）输出 `-0.195312、-0.765625、-1.468750、-2.968750`，与独立手算逐位相同，每层 `max_rel_err = 0`。**边界如实记录在 ADR**：f16 / bf16 仍无算子、U4 是打包字节，因此今天覆盖 f32 稠密模型；样本仍以 C 字面量内嵌进 harness，大层需要二进制数据通道（下一步）；批大小为 1，尚无注意力与 KV。
- **第二条算子落地，而且"加算子只加数据"第一次被证伪过（ADR 0013）**。此前仓库只有 `gemv_i4` 一条算子，`nn::WeightFormat::operator` 也只把 int4 映射到它，于是真实 safetensors（绝大多数条目是 `F32`/`F16`/`BF16`）一律算不出下界——`llm status --weights` 只能报"这个格式还没有投影算子"。现在新增 `gemv_f32`（`y[n] = dot(w[n, :], x[:])`）：`ops/gemv_f32.toml` + `.k` + `src/gemv_f32.rs` 三件数据，加 `REGISTRY` 里一条注册，`cutile` 的协商/拼接/emit/缓存/审计与 `prim` 的片段**一行未改**，用的是 CPU 档案里同一批 9 个 L2 原语。真机证据：`cutile verify --op gemv_f32 --devices cpu` → `✓ L2 | variant pipe2 | max_rel_err 0.000e0 | 片段 51 行`，形状矩阵含 `1x1,2x3` 这种非块整数倍，模板用 `len` 截断尾块、参考实现按同一块宽累加，所以对拍比的是同一份舍入顺序。三条刻意取舍：格式 → 算子名是映射而非默认值（同一份 `[4096, 4096]` 权重，f32 读下界 64 MiB、int4 10.5 MB，互相顶替就是报一个错得很像对的数）；稠密的形状粒度是 1（nibble 对齐的约束不该溢出到 f32，宽度 100 也能算）；代价仍由一条表达式 `nm,m,n->n` 推出，本 crate 不写流量常数，`nn::Bounds` 只是把 `read`/`write`/`flops` 取过来。顺带修掉两处会让"按位置取算子"静默换人的地方：新增 `op::builtin_named(name)`（与 `model(name)` 对称），`cutile` 的测试与语义测试改为按名字取；`nn::Bounds::of_op` 与一条测试里遗留的 `(model.cost)(m, n)` 调用改为 `ShapeContext`（具名轴），那是并行重构留下的中间态。**未做且已经写进 ADR**：f16 / bf16 仍然推不出下界，补齐它们需要 16 位的 `cvt` 片段与模板，属于片段层和档案层的改动。
- **模糊测试契约终于有驱动方，而且"不 panic"不再靠调用方自觉**：`crates/fuzz` 一直写着三条契约，却没有任何真实入口，第一条（不 panic）被推给"驱动方必须把它当作失败"——于是每个驱动都要自己 `catch_unwind`、自己拼证据，panic 一发生就连同已跑过的用例一起消失，也拿不到"哪条输入打死的"。现在 `sweep` 自己捕获：新增 `Behaviour::Panicked { message }`，从 payload 取回说明（`&str` / `String` 两种形态都认），第一条违约时返回带着**用例号与输入原文**的 `Finding`，消息形如"不 panic：任何输入都必须被解析成结论，而不是打死宿主"。配套新增 `tests/parsers.rs`：把 `config` 的 TOML、`prof` 的档案、`cli` 的参数文法、`weights` 的 JSON 与 safetensors 头部五个入口按"合法种子 + 逐字节扰动"灌进去（每个 512 条 × 每条跑两遍），并先断言"种子本身必须被接受"——否则整个遍历只是在错误路径上打转。失败时打印可复现的输入（UTF-8 就原样，否则十六进制）。
- **量化权重终于读得出来了：分组与宽度出自同一次推断（ADR 0012）**。此前真实 4 位量化 checkout 一个都读不出来：`blk.0.attn_q.scales`（秩 2 的标度条目）被当成一层额外投影，`blk.0.attn_q.weight`（`U8`）被说成"i8 没有这个格式的投影算子"而无法计价，而 `gemv_i4` 声明的 32 个元素一组可能根本不是文件里的分组。新增 `weights::quant`：`Header::quantization` 认出四种标度命名族（`{name}_scale` / `{name}_scales` / `{base}.scales` / `{base}.scale`），由标度形状推出 `group = in / groups`，并用**字节数事实**（`out × ceil(in / 2)` 必须等于条目长度）验证"它到底是不是 4 位打包"。关键是 `Quantized` 一次给出 `group()`、`in_features()`、`out_features()`——8 位元素类型的第二维是**字节数**，若把 `shape[1]` 当输入宽度，下界会正好少一半且不报错；现在宽度与分组同源，同一层权重的打包字节与解包视图（`U4 [64, 128]`）推出逐位相同的下界（`a_quantized_entry_and_its_unpacked_view_declare_the_same_floor`）。三种结局按"能不能解释成另一种合法方案"分界：没有标度伙伴 → 不是量化；形状自相矛盾（秩不对、输出维不等、不整除、`groups == 0`）→ `invalid`；打包口径不符或标度不是浮点（int8 逐元素、整数标度）→ 不猜、由元素类型给出诚实理由。`nn` 侧只有两处改动：`Stack::from_header` 用 `Header::scales_names()` 排除标度条目（与推断分组同一份规则），`projection` 对量化条目先问标度再取宽度。实测真实形态文件（`U8 [64, 64]` + `F16 [64, 4]`）→ 1 层、可计价、未覆盖为空；`scales [64, 1]`（group 128）如实点名"内核声明的是 32 个元素一组"。
- **检查点文件从"能编解码"变成"能自证"：第 2 版格式带 CRC-32 尾部，写入是原子替换**。`crates/checkpoint` 之前是固定宽度编码（事实各 8 字节、名字长度 2 字节、偏移 8 字节），既没有校验和也没有落盘路径，三处都是"错了会很贵"的位置。一，**字节自证**：`LLMC` 魔数 + 版本之后，整个正文由尾部 4 字节 CRC-32（IEEE 802.3，256 项查表在编译期生成）覆盖，任何一位翻转或截断都在 `decode` 里失败，不会再出现"读出来是另一个检查点、第 N 步才发现参数与步号对不上、而旧状态已被覆盖"；校验范围不是一个手写区间，而是正文写进一个"密封"的接收器（`Sealed`），因此新增字段不可能忘了进校验。二，**编码紧凑且是单射**：长度、计数、形状与偏移全部 LEB128，偏移按条目顺序存增量（`Header` 已保证有序），解码器再拒掉非规范写法（如 `0x87 0x00` 也是 7）与超过 `u64` 的整数，于是同一份快照只有一种字节形态，`decode(encode(x)) == x` 与 `encode(decode(b)) == b` 同时成立；一份 224 条目、32 层、名字与维度都真实的头部从 11 084 字节降到 7 021 字节（−36.7%）。三，**长度先算准**：`encoded_len` 与 `encode` 共用同一个编码器——统计长度只是换了一个只做加法的接收器，没有第二份算术；`write` 按它一次分配、先写同目录临时文件再 `rename`，于是"承诺的字节数 == `metadata().len()`"，中途崩溃既不留半截检查点也不毁掉上一份。质量上：`Facts` 把步号 / 并行度 / 层数 / 批行数收成一个**可整体比较**的类型（此前 `Snapshot::new` 是五个位置化参数、其中三个同为 `usize`，传反了得到的仍是一个"合法"检查点），零值校验跟着类型走，`Snapshot::new` 因此不再返回 `Result`，`encode` 也不再返回 `Result`（编码没有可失败的操作）；解码在读之前先按"每条目至少 4 字节"给条目数封顶、按剩余字节给维度容量封顶，伪造的计数换不来一次大分配；条目出错时补上"第 N 个权重条目"的上下文。新增 `Snapshot::write` / `Snapshot::read`（`missing` / `io` 的分类仍由 `Error::from_io` 给，不重写一遍）。新增 15 条测试：**整个文件逐位翻转 8×N 次都必须被拒**、重算校验和之后条目表仍被自证（重名 / 区间重叠）、CRC 与逐位定义及标准检验值 `0xCBF43926` 对上、变长整数每个宽度边界、UTF-8 名字按字节计、非规范与超宽整数、声明计数超限、CRC 正确但正文截断时的偏移报告、尾部多余字节、缓冲区不足、200 组随机条目表、写读往返与临时文件零残留、路径不存在 / 目录不存在 / 文件被改动带路径报错。四，**写盘与校验各走一遍**：`write` 把编码器直接接到文件缓冲（`FileSink` 包一层 `BufWriter`），整份检查点从不进内存——条目表大到几十万条时，这是"写得下"与"写不下"的区别，而不是快慢的区别；`encoded_len` 不再顺带算一遍 CRC——长度与校验是两份工作，正文长度加固定 4 字节校验和就是整份文件（2 万条目 / 720 030 字节上，长度从 1.25 ms 降到 51 µs，`encode` 随之从 2.6 ms 降到 1.5 ms；只验字节 1.19 ms，完整解码 3.4 ms，其中 1.2 ms 是同一遍 CRC）；解码对"声明的条目数"只预分 1024 个槽位、其余按需增长，伪造的计数换不来一次大分配；变长整数改为在切片上游标，报错时的偏移仍然精确。五，**恢复前的第二道关**：`Snapshot::verify` 只验字节（魔数 / 版本 / CRC-32，不建表，供巡检与下载后自检），`Snapshot::check_weights` 要求检查点引用的条目表与手头的权重文件逐条相同（数据段起点、条目数、每个条目的名字 / 类型 / 形状 / 偏移），差异按第一条报出——"文件都对、但配错权重"在字节层面完全看不出来，只会在训练若干步之后表现成 loss 不降，那时旧检查点往往已被覆盖。新增 3 条测试：跨过 8 KiB 写缓冲的 512 条目表逐字节对上且不留临时文件、只验字节的三条拒绝路径（校验和 / 截断 / 魔数）、权重表六种差异各报出第一条。

- **签名对齐的第六段：两层共用一套签名词汇，属性键属于原语**。`[needs]` 与 `[[provides]]` 是同一个能力签名的两个方向，却各有一个解析器：档案侧只认 `m`、只认数组形式的 `dt`/`space`，声明侧认 `m`/`bytes`/`extent` 与标量写法——同一份签名换个合法写法，档案层就会**静默少一个约束**（`space = "FAST"` 落成空集，直到协商才表现成"缺空间"，离原因很远）。现在 `parse_provide` 接受与 `[needs]` 相同的别名与标量/数组写法，形状不对（`space = 3`、`dt` 里混进数字）直接报错，不再当作"没写"。第二处裂缝是"一个签名由哪些参数构成"没有闭集：`load.sat`、`mma.lanes` 能写进两层，而协商只比对需求里出现的键，这些键永远不会被检查、也不会被片段消费。新增 `prof::atom::Prim::attr_keys`（16 个原语各自的属性闭集，`exec` 对所有原语成立），`Profile::validate` 与 `op::decl::parse_need` 用同一张表拒收，诊断点名原语与该原语真正允许的参数。现有 85 条声明的属性键全部落在这张表内，因此这次是把一直隐含在数据里的规则写明，不改任何设备数据。新增 5 条用例：档案接受 `bytes` 与标量写法、字段形状错即报错、档案里 `load.sat` 被拒、声明里 `store.sat` 被拒，以及一条跨层等价——同一份签名用两种写法分别写在两层，协商用的覆盖关系两个方向都成立。

- **层的下界按方向分账，并让"无法计价"成为一等结果（ADR 0011）**。`einsum::Cost` 与 `nn::Bounds` 现在把流量拆成 `read` / `write` 两笔（`bytes()` 保留为派生量），`Formula::cost` 按操作数自己的 `Access` 分别累加——读的上限与写的上限不是同一个数，合成一个总量之后"这层卡在哪一侧"就再也判断不了。`Bounds::moves_within` 把"融合到底省了多少"变成可检查的不变量（每个方向都不更高），并明确"总量不变、只是把读挪成写"不算省。权重编码也不再是一个全局常量：新增 `nn::WeightFormat`，`Stack::from_header` 从 `Entry::dtype()` 取格式，格式 → 算子名由 `WeightFormat::operator()` 给出（int4 的分组从 `op` 的语义表读，不另写一个 32）；没有算子的格式（f32 / f16 / bf16 / i8 / 未声明分组的 4 位打包）仍然是合法的层声明，但 `Linear::bounds` 返回 `unsupported` 而不是拿 int4 的数字顶上——这修掉了一个真实的错报：`[4096, 4096]` 的 f32 权重有 64 MiB，旧代码按 int4 报成 10.5 MB，差 6.4 倍。`Layer` 契约新增 `bounds() -> Result<Option<Bounds>>`：`Ok(None)` = 层合法但没有可计价的形状（注解层没给 `in`），`Err` = 有算子却算不出来；`Stack` 因此收 `Box<dyn Layer>`，`floor()` 返回 `Floor { total, unpriced }`——一张 LoRA 表（`[4096, 16]`，16 不是分组整数倍）或一张 f32 权重只让自己进"无法计价"清单，其余层照常相加，不再让整张模型算不出下界。`#[layer]` 注解新增可选 `in = <宽度>`，生成的 `bounds` 与 `Linear` 走同一条 `Bounds::of_ops`（空算子列表是 `Bounds::ZERO`，组合层就是各算子之和）。同时删掉 `nn::Bounds::calls_per_second_ceiling`：它是 `cutile::plan::predict` 的半个副本、且零调用方，roofline 编排只留在 `cutile`。`llm status` 的层/模型下界改为"读 … / 写 …"两段，并把无法计价的层连同原因单独列一行。

- **声明层的装载路径整体收紧：模板只切一遍、注册只有一张表、解包砍掉两次转换**。`crates/op` 之前的形状是"能跑"，但装载期把同一份模板 `lex` 了三遍（槽位、原语、调用参数各一遍），每遍都把整篇文章重新拷成 `String`；`OpDecl::parse` 还为判一个上限把笛卡尔积全部物化再丢掉；`gemv_i4` 的最内层循环每元素做一次 `i32 → f32` 转换与一次 `scale` 乘法。现在：一，`tpl::Template` 是"切一次、查三遍"的视图，`Segment` / `Arg` 借用模板正文，文本段与槽位名一个字符都不拷，只有 `@原语` 的参数表每处一次小分配；`OpDecl::parse` 拿到的就是这一份切分结果。二，新增 `decl::tiles(op)` 只数不物化（与 `expand` 同一个判据与消息），装载期不再为丢弃的 `Vec<TileCombo>` 克隆轴名；`expand` 保留给真要组合的调用方，容量一次算准。三，`REGISTRY` 成为唯一的算子注册表：名字、声明、模板与语义表同属一条 `Registration`，`BUILTIN` 由 `const fn` 投影、`model()` 从同一张表查——加算子只改一处，不可能出现"内嵌里有、语义表里没有"；`Op` 的三段文本改用 `Cow<'static, str>`，`builtin()` 因此只剩一次 `Vec` 分配（`nn` 每次建栈都会调它）。四，`gemv_i4::forward` 换成按组切片迭代：`chunks_exact` 消掉内层的下标乘加与边界检查，解包走一张 16 项的 `f32` 查表（`nibble − 8` 全是精确小整数），`scale` 从每元素一次降到每组一次——4096×4096 int4 GEMV 的同一份内核（release、单线程、`black_box` 输入）从 192–194 ms 降到 69–82 ms，2.3–2.8×，且**结果逐位相同**（`max_rel_diff = 0`）；`formula()` 用 `OnceLock` 只解析一次（`cost` 每层都会被调），`parse_need` 不再克隆整张内联表。**顺带修掉一个诊断缺陷**：`lex` 跳过 `{{...}}` / `@原语(...)` 时没有统计其中的换行，跨行调用之后的每条诊断都会指回上一次换行；现在跳过时按被消费的字节数行进行号，`the_line_number_survives_a_multi_line_call` 把它锁住。质量上：`check_monotonic` 泛化到 `AsRef<str>`（装载期直接传借用的变体名），`LayoutSpec` / `Schedule` / `TileCombo` 的 `Display` 直接写进调用方的缓冲而不是先建一个中间 `String`。新增 10 条测试：多行调用后的行号、切分不吃字符、命名/位置参数保持原文、未知原语退回文本、未闭合构造不被猜成半个、计数与展开同口径、超限诊断点名该剪的轴、借用名的单调链、每条内嵌算子都有语义表，以及一条手写求和（不经过本文件的任何循环）钉住替换后的内核算术。

- **见证补到最后一台设备：`cpu` 的 `L0` 见证不再写而不跑**。见证矩阵此前只覆盖四台加速器，`cpu`——唯一能产出 `L2` 证据的那台——反而有空缺：`mma`/`scan`/`exp2`/`rcp` 四个它在标量意义上真能算的原语根本没有声明，`atomic` 有声明却没有见证。现在 `cpu/witness.c` 补齐 5 个入口（2×2 分块乘加、含前缀和、逐次加倍的 `exp2`、牛顿迭代的 `rcp`、单线程读改写），档案相应补四条 `L0` 签名并把 `atomic` 从 `L1` **降为 `L0`**（单线程只有读改写，`impl` 改为 `rmw_scalar`）——矩阵补的是覆盖，不是等级。同时修掉一个真实的执行链缝隙：`self_check` 此前只被非 C 设备调用，而 `run_all` 只挑 `grade ≥ L2`，于是 `cpu` 的低等级见证既没有设备结论也没有自检结论。现在自检的范围改为"设备执行跑不到的那些"（非 C 设备全部，加 C 设备的 `L0`/`L1`），`verify` 对每台设备都跑，`Verified` 新增 `self_checked` 字段，`verify --devices cpu` 因此显示 `见证 9 条，自检 5 条`（`max_rel_err 0`、key `fc79113cc543d8b6` 不变）。空白格也不能再靠沉默：`shuffle`/`broadcast` 要求 `[simt].group > 1`，`Profile::validate` 把这条做成装载期判据，且只对**写了** `[simt]` 的档案生效（`ascend` 的 lane 交换由向量单元承担，默认值 1 不代表没有并行）——为此 `Simt` 增加 `declared`。新增/改写 4 条用例：单 lane 设备声明 lane 交换被拒、每台设备的每条声明恰好被执行一次且两个范围相加等于全部声明、`cpu` 的 `L0` 见证必须真被自检、`verify` 的两个计数覆盖全部声明。

- **权重文件的第二条读取契约：`Loader` + 只读头部的 `safetensors::File`（ADR 0010）**。ADR 0009 的"`Reader` 已经把接口留好了"经复核不成立：`Reader::read` 返回 `&[u8]`，生命周期绑在 `&self` 上，文件后端在"不装 `unsafe`、不引 mmap 依赖"的前提下交不出这个借用。现在契约按**所有权**拆成两条：`Reader` 把字节借出来（内存切片、映射、测试夹具），`Loader` 把字节拷进调用方缓冲区（文件、压缩包、远端分块）——后者只有一个必须实现的方法 `copy_into`，`load` 是它的分配封装，因此长度检查（`check_capacity`）只有一处。新增 `safetensors::File`：`open` 只读 8 字节前缀与头部 JSON，数据段留在磁盘上，条目按 `seek + read_exact` 读；自证合法的口径一个字没放宽——用 `metadata().len()` 核对"头部 + 条目声明的数据段正好等于整个文件"，被截断与尾部多余字节都在**打开期**以 `invalid` 失败，而不是等读某个条目时崩在别处。`Archive::parse` 与 `File::open` 共用 `split_prefix` / `parse_header` 一份实现，两个后端解出的头部必须逐字段相等（`both_backends_produce_the_same_header_and_the_same_bytes`）。`llm status --weights` 因此不再 `std::fs::read` 整份文件：输出 `只读头部 X / Y 字节`，并按需读**最小的那一条**条目作为"字节能真取出来"的证据（超过 1 MiB 的样例如实报"未读"，不为了让输出好看而抬高上限）。实测 1 GiB 稀疏文件上 `header_len` 仍是头部那几百字节、`file_len` 是 1 GiB；截断一个字节即退出码 2。

- **训练侧的真实激活缓冲 `train::Buffers`：`autograd::Store` 终于落地**。`Replay` 只决定"重算哪一步、什么时候还"，值从哪来它不管；`Buffers` 把这个口子接到真内存上——`compute` 按 `autograd::Sizes` 登记的（形状，元素类型）取一块缓冲、把只读视图交给 `Kernel`、结果写进去，`free` 当场把 `live` 降下来并把块还给内部空闲表（显存收益发生在这一刻，不在作用域结束）。三个刻意的取舍：一，缓冲只有 `f32` 与按位打包的 `u8` 两种（`unsafe_code = deny` 之下 `Vec<u8>` 不能重解释成 `&[f32]`），`F16` / `BF16` 直接报 `unsupported`，把"哪种元素能算"提前到编译期；二，`gemv_i4` 的算术不在本层重写——`op::gemv_i4::forward` 从标量参考里提出来的正是那段循环，于是重放值与参考值比较的是同一份语义，还省掉了每次重算的一次拷贝与一次分配；三，空闲表按**正好相等**的块长分档，稳定态"算一块、还一块"不再向系统分配器要内存，记账也不必打折。质量上：`load` 校验数据与登记元数据一致（装错类型当场失败，不会在训练中途变成一串无意义的数）、`compute` 在读输入前先确认输入在场（读已释放的值不会算出一块假值）、实测 `peak` 可与 `Plan::peak` 对账（减掉常驻量），`reset_peak` 让前向与重放各测各的。新增 13 条测试（11 条缓冲、2 条内核）：与参考逐位一致、交还当场降 `live`、复用块不残留上一轮的值、已释放输入不被读、未登记形状 / 未知算子 / f16 各自报对类别、装入类型不符被拒、50 轮不增长、清零后复位，以及一条端到端——用 `Plan` + `Replay` 驱动紧预算重放，逐步核对每个输入与前向逐位相同、实测重算步数等于计划承诺、执行完只剩常驻值。
- **真实权重文件能读出来了：safetensors → 层栈 → 模型下界（ADR 0009）**。此前 `weights` 拥有 `Header` / `Entry` / `Reader` 却没有任何解析器（唯一调用方式是 `Header::new(Vec::new(), 0)`），而 `nn::Bounds` 的层清单是组装点手写两个 `Linear`——"模型每 token 至少多快"因此无法对任何真实文件成立。现在两段都补上了：`weights::safetensors::Archive::parse` 读 8 字节小端头部长度 + JSON，校验"`shape × dtype` 推出的字节数 = 声明的区间长度"、"数据段正好被条目覆盖"、字段只允许 `dtype`/`shape`/`data_offsets`、`dtype` 只在 `{F32,F16,BF16,I8}` 闭集内映射（`U8` 明确拒绝而不是猜成 `I8`），并 `impl Reader` 直接返回文件里的字节切片；JSON 读取器是手写的受限实现（`crates/weights/src/json.rs`），小数/指数、重复键、嵌套过深、值后多余内容全部按 `invalid` 带字节位置报出。`nn::Stack`（层栈）从 `Header` 推层：秩 2 的条目按 `[out, in]` 成为投影、名字去掉 `.weight`，**其余条目进 `uncovered` 而不是被静默丢弃**（范数、偏置、位置编码必须被看见），去掉后缀后重名与零宽条目都在构造期拒绝；`floor()` 就是 `Bounds::plus` 的累加，错误点名具体层。组装点因此有了可执行路径：`llm status --weights model.safetensors` 打印"读到什么"（条目数 / 数据段 / 载荷）与"算出什么"（层数 / 总下界 / 未覆盖条目），不给文件时退回声明层并明确写出来。实测 3 条条目（2 个矩阵 + 1 个范数）→ 2 层 / 1 条未覆盖；`cargo test` 锁住了 happy path、未覆盖清单与"文件不存在是 `io` 而不是 panic"。
- **`prim` 的内嵌表改成编译期生成，并加上"源码调试 / 设备镜像发布"两种模式**：`BUILTIN`（片段）与 `WITNESS`（见证源码）此前要手写登记——每加一台设备补一行 `include_str!`，漏一行的症状是"发布出去的二进制里没有它"，只有 `builtin_matches_disk` 能发现。现在 `crates/prim/build.rs` 在编译期扫 `<vendor>/`，一次生成三张表：片段、见证，以及新的 `DEVICES`（每台设备的**原语枚举 + 映射表**：有哪些原语、每个原语在镜像里的入口叫什么）。入口名取自档案里 `(原语, impl)` 那条供给的见证函数，所以映射表指向的一定是那份见证源码里真实存在的 `int func(`——`every_entry_is_a_function_in_the_witness_source` 把它变成判据。生成与校验走同一趟，四类数据错在构建期就失败：文件名不在 `Prim` 闭集内、片段第一行的 `cutile-sign` 在档案里没有对应 `provides`、那条供给没有见证、目录名与 `meta.vendor` / `meta.emit.ext()` 不一致。`Device` / `Image` / `Status` 是新的数据层（`src/device.rs`），生成代码只有常量，所有权表照旧"一个类型一个主人"。feature `device-image` 打开时，按**输出语言**（不是厂商）把 `<vendor>/witness.<ext>` 编成设备镜像并 `include_bytes!` 嵌入：`metal` 走 `xcrun metal -c` + `xcrun metallib`，实测产出带 `MTLB` 魔数、17 个已定义符号的 `metallib`；关着（默认，源码调试）只有源码，运行期照旧 emit 后编译。没有镜像不是错误——`Status` 是闭集，`NoRecipe`（这门输出语言还没有配方）与 `Skipped { reason }`（工具链不在本机 / 编译失败，附第一行 stderr）各说原因；发布流水线要"缺镜像即失败"时设 `CUTILE_REQUIRE_IMAGE=metal`，它把"缺镜像"升级成构建失败。**没有做**的是把片段编进镜像：片段是语句块，`$0..$3` 的绑定由算子模板的 `@prim(...)` 给出，脱离模板编译只能得到与真实调用无关的代码，所以镜像里目前是见证编译出的设备侧库（一份完整翻译单元、每条签名一个入口）；CUDA / OpenCL / AscendC 也还没有配方，因为它们的镜像只收 `__global__` / `__kernel__` 入口，把 host 见证编进去只会得到空产物。编译不传档案的 `[compile].flags` / `preamble`：见证是 host C ABI（`int func(float *out)`），与片段语言无关，`cutile verify` 的本机自检因同一原因跳过它们——`-std=metal3.1` 会要求指针写明地址空间，而见证里的 `float *` 故意没有。新增 8 条用例：目录与表一一对应、每台设备的档案自洽、入口在源码里、枚举有序唯一、枚举与片段表同源、`status` 与 `image` 一致、调试模式无镜像、发布模式里镜像的魔数与符号数。组装点 `llm status` 多一行 `片段层 prim`，逐台设备列出 `=源码` / `=metallib N B/M 个符号`——它同时是"镜像真的进了二进制"的证据（`--features device-image` 构建出的 `target/debug/llm` 里能搜到 `MTLB`，默认构建搜不到）；feature 从根包逐层透传（`llm → cutile → prim`），所以发布只用 `cargo build -p llm --features device-image`。
- **签名对齐的第五段：模板调用点的能力参数必须被 `[needs]` 允许**：`@cvt(..., sat: off)` / `@reduce(..., op: sum, axis: GMAX)` 的命名参数会原样拼进片段文本，决定"这次生成用的是哪条签名"，但此前只被 `substitute` 当成占位符取值——`needs.cvt.sat = ["off"]` 与模板里的 `sat: on` 可以同时存在，于是设备按 `off` 被协商验证、代码按 `on` 生成，同一条能力签名有了两份事实源。现在 `op::decl::OpDecl::check_call_arguments` 在装载期逐变体比对：闭集内的能力属性键（`sat` / `op` / `axis` / `kind` / `accuracy` / `group` / `lanes` / `src` / `dst` / `stages` / `exec`）传了就必须在 `[needs]` / `extra_needs` 里声明、且取值被声明允许，闭集之外的命名参数（`along:`）是片段级提示，留给片段。错误带算子名、变体名、模板行号与原语名。新增 3 条用例：`sat: on` 与声明矛盾、`accuracy` 未被声明、`along: 32` 不被误伤。
- **重放执行 `autograd::Replay`：把显存计划变成可执行的指令序列**。`Plan` 只说"谁保留、谁丢弃"，真正要有人去重算。`Replay::of(&tape, &schedule, &plan)` 编出两张扁平区间表——反向第 `k` 步之前重算哪几步（前向拓扑序）、之后交还哪些激活——`run(&tape, &schedule, &mut store, backward)` 按表驱动整次反向：先重算，再让调用方算这一步的梯度，最后按释放点还缓冲。数值不在这一层：`Store` 只有 `compute` / `free` 两个方法，真实缓冲归 `train`，测试里是一张标量表，于是"重算出来的值"与"前向真值"可以逐位对拍。执行期零分配、不递归、不查哈希；同一步里共享的祖先只展开一次（`stamp` 换代），所以实际重算步数只会小于等于 `Plan::replayed`，一旦更大就直接报错——承诺与实测可对账。这次还修了计划里的一个真错：释放点原先只取"最后一次直接消费"，而一个检查点可能一边被深层节点直接用掉、一边是某个更浅的被丢弃值的重算起点，后者排得更晚；现在重算需求沿被丢弃的链按编号降序传播一遍，"最后一次需要"才真正是最后一次。零字节的激活（标量、或调用方还没登记形状）也不再被丢弃：留着不占预算，丢了只烧算力。新增 6 条测试：紧预算逐位对上承诺值、装得下时零重放但仍按时释放、检查点活到最后一次需要、锚定更晚重放的检查点不被提前释放、两万层深链不递归、随机图 200 轮"不读已释放的值且峰值不超计划"。
- **Apple GPU（Metal）成为第四台有档案的设备**：`prof/devices/metal.toml` 按 DESIGN §5.3 的 Apple 行落地——`emit = metal`，三级存储的物理名是 MSL 地址空间（`GM`=device / `FAST`=threadgroup / `REG`=thread），threadgroup 32 KiB 与 32 bank、simdgroup 32（`groups_max = 32` 即 1024 线程上限的折算），峰值写成可求值的推导式（`20*128*2*1.6e9`、`256/8*9.6e9`），指纹用机器名 + `gpu_cores 10..80`——DESIGN 原写的 `os >= 26` 没有进指纹，因为 `probe` 目前只采集 CPU 的 `arch`/`features`，写一条当前永不满足的条件只会让档案更严格地不命中。17 条 `provides` 覆盖整个原语闭集（P0 的 9 个为 `L1`，`mma`/`shuffle`/`broadcast`/`scan`/`exp2`/`rcp` 为 `L0`），`dot` 拆成 `simd_dp4a`（i8×4→i32）与 `simd_fma`（f32）两条签名。片段与见证同时落地：`prim/metal/` 的 9 段 `.metal` 片段（第一行 `cutile-sign` 与协商会选中的 `impl` 逐条对齐）与 `witness.metal` 的 17 个入口（一个签名一个函数），分别登记进 `prim::BUILTIN` / `prim::WITNESS`，`prof::BUILTIN` 与其同步，`probe::fake("metal")` 给出 M5 Pro 的指纹。判据都已实测：`doctor --op gemv_i4 --device metal` 选中 `metal(m5)` 并逐个需求报 `L1 < L2（未验证视为不具备）`；`verify --devices metal` 按“缺设备即失败”退出，同时报“见证自检 17 条通过（host C ABI，不含设备执行）”；`audit` 的 host 自检与 `cutile::builtin` 的两条矩阵判据现在都覆盖四台加速器（71 条见证）。**没有做**的是 Metal 的执行后端：P0 只有 C 能编译执行，因此 `emit = metal` 的设备拿不到设备级证据，等级不变；`simdgroup_matrix`、`simd_dp4a`、双缓冲 `async_copy` 这类 tile 落点只写在档案与见证里，属 P3。
- **显存计划 `autograd::Plan`：预算决定留哪些激活、何时释放、重放多少步**。有 `Tape` 与 `Schedule` 之后，"中间结果放不下时留哪几个"仍然没有主人。`Plan::of(&tape, &schedule, footprint, budget)` 只做算术：`Footprint` 是泛型 trait（无 `dyn`、无 `Box<dyn Any>`、无 `unsafe`，与 `unsafe_code = "deny"` 相容），`Sizes` 是它的现成实现（`Shape` × `DType`，4 位元素按位打包）——形状不必进反向图，`Tape` 只用来判"叶子还是算子输出"，于是参数与输入 token 不会被误算成激活，detach 出去的值则被判定"本次调度里无处重放"，只能留不能丢。产物是三个可核对的数：`saved`（反向一开始就必须存在的检查点）、`release`（**最后一次**被消费之后即释放，不按作用域）、`peak` / `replayed`（保守峰值与重放步数，`peak <= budget` 由构造保证）。段长由预算解出来而不是写死常数：等字节时峰值约为 `total/√n` 的两倍，`BLK = 256` 只是这条公式的一个特例；`floor` 是任何计划都绕不过的下界（一次算子的输入必须同时在场），预算低于它直接以 `invalid` 报"放不下"，而不是给出一个实际会超的计划。图上的最优 pebbling 是 PSPACE 难的，所以这里不声称最优，只给可核对的峰值、下界与重放代价；与 burn 的 `checkpoint` 相比没有 `ComputingProperty`、`n_required` 引用计数与策略 trait——释放点由"最后一次被消费"直接给出，重放执行留在 `train`。所有权表补 `Plan` / `Budget` / `Choice` / `Footprint` / `Sizes` 一行（与 `cutile::Plan` 的 tile 选择不是同一件事）。新增 12 条测试：全留、低于下界、紧预算买重放、释放点跟随最后一次消费、零预算、确定性、形状表换算、死分支不进计划、detach 值只留不丢、两万层不递归、随机图 200 轮不超预算。
- **autograd 拆成五个职责模块，调度与账本变成可复用对象**：`lib.rs` 只留文档与再导出，图（`graph.rs`：`Requirement` / `Var` / `Node`）、记录（`tape.rs`：`Tape`）、调度（`schedule.rs`：`Schedule`）、账本（`account.rs`：`Account`）与错误构造（`error.rs`）各居一处，`lib.rs` 只剩文档与再导出，公开面按职责收拢，与 `burn-autodiff` 的 `graph/` + `grads` 划分同构。训练循环不必每步新建对象：`Tape::clear` 清图，`Tape::plan_into(&mut Schedule, loss)` 把计划重写进同一个 `Schedule`（步骤表、终点表、可达位图 `clear` 后重填，容量留在原处；失败的规划不留下半份计划），`Account::rebuild(&Schedule)` 重置扇出，`Account::settle_into(step, &mut ready)` 不再为每个节点分配一个 `Vec`。可达性改为**按编号降序单遍扫描**：边只能从小编号指向大编号，消费者总在生产者之前被处理，拓扑序的逆序天然成立，不需要显式栈也不需要递归（两万层深链仍是 O(节点数)）。新增 `Schedule::levels()`：同一深度切成一层，层内互不依赖，可并行或合批；`Account::ready` 改为对不进图的变量恒为 `false`，常量不会再被误报成"梯度已齐"。数值仍不在本层：梯度、前向激活归运行时缓冲，反向公式归 `op` 的反向表。`Tape::var(id)` 把运行时缓冲的稠密编号换回变量（不校验的构造会让打错的编号变成一张看起来正常的图）。测试从 22 条增加到 33 条，新增缓冲复用（步骤表 / 终点表 / 位图地址不变）、换 loss 重新规划、分层就绪、空账本拒绝结算与账本重建；另有 `tests/gradcheck.rs` 用五个光滑算子（`add` / `mul` / `scale` / `square` / `neg`）跑 200 个随机图，梯度与中心差分逐参数对拍（口径 `|Δ| ≤ 2e-3 + 0.05·max(|fd|, 1)`），并验证重复反向逐位一致、分层执行与顺序执行等价、断开分支与图外变量恰好为 0。
- **`checkpoint` 成员（目录 `crates/checkpoint`）：检查点文件**。`train::Checkpoint` 只记"恢复所需的事实"，但"这些事实怎么落盘、版本怎么认、条目偏移怎么校验"此前没有主人。现在 `Format`（魔数 `LLMC` + 版本，`check` 只认当前版本）与 `Snapshot`（步数 / 并行度 / 层数 / 批行数 + `weights::Header`）负责这一段：`encode` / `decode` 是纯字节的往返，条目表在解码时仍交给 `weights::Header::new` 重新校验（重名、区间重叠、起点早于数据段），截断、版本不符与尾部多余字节都以 `invalid` 带上偏移报出。`train::Checkpoint::snapshot(&weights::Header)` 把内存事实组合成快照，删掉 `checkpoint` 会让 `train` 编译失败；层表把它放在 L6 最左（`checkpoint` → `train`），依赖 `weights` / `tensor` / `diagnostic`，依赖边 79 → 83。新增 6 条单元测试（往返、魔数、版本、截断、尾部多余、零层数）与 `train` 侧一条快照往返用例。
- **autograd 从"只给一个逆序"升级为反向图调度**：`Tape` 现在按 `Requirement`（`None` / `Backward` / `Grad`）维护需求闭包——`record` 从输入推出需求，`record_with(.., None)` 是显式 detach，`retain` 只允许把**还没被引用**的叶子升级为梯度终点（已被引用即 `invalid`，因为父节点的需求在记录时就定了，事后改会让剪掉的输入悄悄漏梯度）；每个节点同时记深度（输入最大深度加一）。`Tape::backward(loss)` 返回 `Schedule`：只排 loss 可达的节点，顺序为"深度降序、同深度编号降序"，既是合法的反向拓扑序（输出总比输入深），也是并行提示（同深度互不依赖）；`endpoints` 只列可达的 `Grad` 变量，于是没参与 loss 的参数不会进调度、也不会被分配梯度缓冲。`Schedule::account()` 给出 `Account`：按可达步骤的输入位置数出扇出（同一变量在一次调用里出现两次就是两份贡献），`settle` 在梯度未收满、步骤已结算或步骤不属于本调度时返回 `invalid`，收满时返回因此可执行的变量——"谁的梯度已齐"不再靠调用方自觉。实现是稠密 `u32` 索引的平行数组（需求 / 深度 / 生产者）加显式栈遍历：无哈希、无递归，两万层前向照常排。数值仍不在本层：梯度与前向激活归运行时缓冲，反向公式归 `op` 的反向表。所有权表补 `Node` / `Requirement` 与 `Schedule` / `Account` 两行。`Tape::clear` 保留容量地清空，训练循环每步复用一张图时稳定态不再分配。新增 22 条单元测试（含随机图逐条不变量对拍、两万层深链不栈溢出、乱序与重复结算被拒）与一条标量引擎跑完整反向再对解析梯度的用例（`d/dx (x+y+xy) = y+1`、`d/dy = x+1`）。
- **见证矩阵补齐：闭集的 16 个原语 × 4 台加速器**：此前只有 P0 用到的 9 个原语有声明与见证，`mma` / `shuffle` / `broadcast` / `scan` / `exp2` / `rcp` 在全部档案里一条都没有——`nvidia.toml` 写着 `flops_f32 = 94.6e12`，却没有一条张量核能力，而 `intel` 连 `atomic` 都漏了。现在加速器各补上缺失的签名（`wgmma`、`dpas`、`mmad_tile`、`shfl.idx`、`sub_group_shuffle`、`vcumsum`、`ex2.approx`、`native_recip`、`vrec` …），每条签名配一份自己的见证函数（`mma` 是 2×2 tile 乘加、`scan` 是含前缀和、`exp2` / `rcp` 是 2 的幂与倒数的标量等价），`nvidia` / `intel` / `ascend` 分别有 17 / 17 / 20 条（`metal` 按同一判据落到 17 条），矩阵共 71 条；`ascend` 原先让 `quant` / `load` / `load_async` 的两条签名共用一个入口，现在各有一份。等级仍是 `L0`：有见证源码，没有片段文本（P0 模板不调用它们），也没有真机执行。新增测试 `every_accelerator_declares_the_whole_primitive_set`——它第一次运行就抓出了漏掉的 `intel/atomic`。
- **一条见证只能证明一条签名**：`Profile::validate` 新增判据——同一个 `{ file, func }` 不能被两条签名引用。否则"每条 provides 都有见证"会退化成"每条都指向同一份证据"，执行通过的结论无法归属到具体实现（`mmad` 与 `scalar_fma` 是两种算法）。新增用例 `a_witness_may_serve_only_one_signature`。
- **签名对齐的第四段：片段声明自己实现哪条签名**：同一原语在档案里可以有多条签名（`dot` 的 `dp4a` 与标量 `fma`），片段层每个原语却只有一段文本；此前"这段文本写的是哪一条"只写在注释里，`needs`↔`provides` 与 `provides`↔`witness` 都已可判定，唯独 `provides`↔片段靠人读。现在每段片段的**第一行**是机器可读的 `/* cutile-sign: <prim>/<impl> */`（四台设备共 36 段片段补齐），`prim::Fragment::parse` 在装载期解析并拒绝"缺声明 / 原语名与文件名不一致 / `impl` 为空"；`cap::fragment_gaps` 与 `negotiate` 用同一张需求表、同一个 `select`，逐变体逐需求比对"协商会挑中的实现"与"片段声明实现的实现"，`Catalogue::validate` 把差异变成装载期错误（写错签名是 `invalid`，缺文本是 `missing`）。于是"能协商"与"能生成"不再可以同时为真却指向不同的算法。新增测试 `every_fragment_declares_the_implementation_negotiation_picks`（四台设备逐条对齐）与 `a_fragment_declaring_another_implementation_fails_the_load`（把 `nvidia/dot.cu` 改成声明 `dp4a` 即装载失败），`prim` 侧另有四条解析用例。等级不变：这条判据只说"文本与签名同构"，不构成设备证据。
- **P0-10 编译参数进入产物身份**：能力一直是「片段 + 工具链 + 参数」三元组，前两项已经由见证执行（P0-1）与实测工具链身份（P0-8）落下，参数这一项没有——`ArtifactKey` 只有源文本 + 工具链 + 主函数。算子的缓存键恰好把 `[compile].flags` 哈希了进去，见证程序却按 `标记;程序文本` 定名：于是改一次档案的 `flags`，内核会重编，见证却复用旧参数编出的可执行文件，「这条见证跑通过」对着的是一份不再会被编出来的程序。现在 `ArtifactKey` 多了 `params`（算子取 `CacheKey::flags_hash`，见证取实际那一行 `flags` 的指纹），`build` 的编译与无编译器复用两条路径都按它匹配；缺 `params=` 的旧产物一律算"没有身份"（缓存可重建，重编一次远比把证据挂到未知参数上便宜）。三条命令把参数写进报告：`doctor` 的 `device` 行给出 `参数 -O3 -std=c99`、`[证据]` 段分别写设备执行与自检各自的参数（自检固定 `-O2 -std=c99`，host C 不认设备的 `-arch=…`），`verify` 的结论行与 `bench` 的产物行也带上它。实测：`-O3` 下 `key a073f1d3c75cd006`、产物 `复用 3`；把 `cpu.toml` 改成 `-O2` 后 `key dee9be4a1ee17a5a`、`复用 0`（见证也重编重跑）；改回 `-O3` 又回到旧键与 `复用 3`。
- **P0-9 证据前置到三条命令**：`verify` 早就会把 `grade ≥ L2` 的见证编成可执行文件跑一遍，但 `doctor` 与 `bench` 仍在直接读档案里的 `grade`——同一台设备因此可以同时"协商可行"（`doctor` 打印 `path: pipe2`）与"见证不成立"（`verify` 失败）。现在 `audit::witness::establish` 把设备执行与自检合成一份 `Evidence`（依据 `Basis`、设备执行条数、自检条数、失败原因、`cap::Proof`），三条命令共用它协商。`cutile::cap` 因此多了一个输入：一条供给要被认可，除了档案写 `L2`，还得出现在**这次执行**的证据里（`negotiate_with` / `inspect_with`）；失败原因分成三种——参数不匹配、`L1 < L2`、"L2 但没有这次执行的见证"——不再被压成一句话。两条供给都能覆盖同一需求时，先取真的跑过的那条再比等级：否则一份自述过来的 `L3` 会把本次唯一有见证的实现挤掉，协商结果取决于哪个数字写得更漂亮。`doctor` 先跑证据再协商，见证不成立时仍交报告（`[证据]` 段写出具体失败，`no_path` 的每一行都说明"欠一次执行"）；`bench` 直接拒绝计时（`invalid`，退出码 2），`Bench::proven` 报告这次用了几条见证。本机执行不了的设备（非 C 档案）只能按档案自述协商，`Basis::Declared` 把这件事写进报告：`doctor --device nvidia` 打印 `依据 档案自述（本机执行不了这台设备） | 设备执行 0 条 | 本机自检 17 条`。`negotiate` / `inspect` 保留为"按档案自述"的旧口径，只服务于 `verify` 的 §16 协商判据与这类解释性报告。
- **P0-8 工具链身份来自实测**：档案里的 `[meta].tool_ver` 一直是手写常量（`cpu.toml` 写 `clang-17`，本机其实是 Apple clang 21），而它**同时**被当成缓存键里的"工具版本"与产物身份里的"编译器"——两处都在替一个从未被读过的版本作证。现在 `cutile::backend` 新增 `Toolchain`：`probe` 跑一次 `--version`，`parse_version` 认 clang / gcc / nvcc 三种写法（`Apple clang version 21.0.0 (clang-2100.4.1.1)`、`gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0`、`release 13.3, V13.3.44`），`id()` 是 `命令@版本`，解析不出就不猜（`cc@-`）。产物身份 `ArtifactKey.tool` 从命令名换成这条身份：升一次编译器就是另一份代码生成，旧产物不再冒充新编译器的产物；`CacheKey` 相应**删掉** `tool` / `tool_ver`——源文本与编译器无关，换编译器只换产物，不换文本与它的指纹。`verify` 的结论行、`doctor` 的 `device` 行与 `--emit-ledger` 台账都同时打印"档案登记 / 实测"，`Verified` 新增 `tool_ver` / `toolchain` / `toolchain_agrees`；不一致**不改等级**（证据可以在别的机器上取得，登记值与 `checked` 同一性质），但差异必须可观测，否则换机器换编译器会静悄悄地发生。`prof` / `prim` 的 `tool_ver` 文档同步写清楚它是登记值。
- **P0-7 数据层自己的错误类别**：`prof` / `prim` 的失败出口不再是 `Result<_, String>`——档案解析（`ProfileError`）、片段签名声明、`Span` 区间与常量推导式求值都返回 `diagnostic::Result`，类别由**产生错误的那一层**给：写错是 `invalid`，文件不在是 `missing`，读不出是 `io`。`diagnostic::Error::from_io` 收下"路径不存在 vs 存在但读不出"这条唯一判定（它同时是退出码 `4` 与 `5` 的分界），原来各写一遍的 `format!("读取 {} 失败：{error}")` 一并删掉。`cutile` 的两处边界（`builtin::from_layers`、`build::fragments`）从"猜一个 `Missing`"改为挂上下文帧后原样上抛，于是 `data.root` 指错仍是 `4`，而片段第一行不是签名声明从 `4` 变成 `2`——这正好是 DESIGN §4.2 想要的可区分性。装载期的“一次报全”也保留了下来：`Result` 只能携一个错误，所以 `Error::also` 把其余失败作为同级诊断逐行追加（首条决定类别，其余各自带自己的类别），不再把它们 join 成一行 `invalid`。同时把“见证文件不在或读不可”单列为 `ProfileError::WitnessFile`（`missing`），同一份文件的重复缺失只报一次。`tests/architecture.rs` 的扫描范围扩到 `crates/prof/src` 与 `crates/prim/src`，边集随之从 83 变成 85（`prof → diagnostic`、`prim → diagnostic`）。
- **P0-6 编译产物入缓存**：缓存命中原先只省掉 emit，真正慢的 `cc` 每轮都在跑。现在产物有自己的身份 `ArtifactKey` = 源文本键 + **真正调用的编译器** + **拼进去的 `main()`**（形状、样本、重复次数都在其中），落在同一条目的 `bin/<hash>`；`cutile::build::execute` 先查产物再决定是否编译，命中时连编译器都不需要——`verify`/`bench` 的第二轮不再调用 `cc`，实测 4.6 s → 1.55 s，且命中路径不看 `PATH` 上有没有编译器。`backend` 相应拆成 `compile`（产出可执行文件）与 `run`（跑可执行文件），编译在 `work.dir` 的一次性沙箱里进行、产物入缓存后沙箱即删，`work.dir` 不再残留第二份可执行文件。见证程序走同一条路（`Cache::put_program` 按程序文本登记成普通条目，`meta.txt` 记 `program=` 而不是算子指纹），两轮 `verify` 因此都不再重编见证，见证产物也进入 GC 的账。存量管理同步修掉两处真错：占用改为**递归**统计（`bin/` 里的产物原先不计入，GC 看着 2 GiB 上限却删不掉真正占空间的东西），原子落盘的临时名带上进程号与序号（同一个 key 的两个并发构建原先共用一个 `source.tmp`，先 `rename` 的一方会让另一方拿到 `ENOENT`）。报告面增加证据：`verify` 打印产物份数与复用数，`bench` 明确写出这一轮是"复用，未调用编译器"还是"新编译并落盘"。
- **P0-5 错误类别贯穿三层边界**：`diagnostic::Code` 新增 `exit_code()`（`Invalid` 2 / `Unsupported` 3 / `Missing` 4 / `Io` 5 / `Internal` 70），`src/bin/cutile.rs` 的 `main` 从"自己决定 1 还是 2"变成"打印诊断 + 按类别退出"，脚本因此能区分"这台设备没有这条能力"与"我把参数写错了"。`op`、`cutile`、`audit` 的全部 `Result<_, String>` 换成 `diagnostic::{Error, Result}`：声明/profile/调度/布局解析与样本对齐是 `invalid`，缺缓冲/标量/模板/档案/见证是 `missing`，容量、协商与缺编译器是 `unsupported`，文件与编译器子进程是 `io`，本层实现漂移（`check` 已过却取不到标量、见证计数与打印不一致）是 `internal`。`OpError`、`SigError`、`SubstituteError` 三个模块私有错误类型随之删除，行号降为 `Error::with("第 N 行")` 的一层上下文；`prof` / `prim` 仍是存量（在边界处显式给类别，不做无类别的 `From<String>` 猜测）。`cutile` / `audit` 因此各多一条指向 `diagnostic` 的边（77 → 79）。
- **见证的第二种执行范围：本机自检**：`audit::witness` 拆出 `Scope::{Device, Host}` 与共享的执行器，新增 `self_check`——把每台设备**已声明**的见证（不再要求 `grade ≥ L2`）按 host C ABI 编译执行并与 `expect` 对拍。见证的 ABI 与片段语言无关，所以自检跳过档案的 `[compile].preamble` 与 `flags`（`#include <cuda_fp16.h>`、`-arch=sm_100a` 在本机要么不存在、要么只对真机编译器有意义），因此 `nvidia` / `intel` / `ascend` 的 35 条 L0/L1 见证第一次真正被编译执行，期望值写错会在今天失败而不是等真机接入。`verify` 对非 C 设备先跑自检、再如实报"本机没有该设备"，自检失败是硬错误；通过后把条数作为诊断的一帧追加——`本机没有该设备… → 见证自检 11 条通过（host C ABI，不含设备执行）`。结论的效力写在 `Scope::Host` 里：它只证明见证源码与期望值自洽，**不**证明设备具备该能力，因此不改写任何等级，协商结论仍由 `L2` 决定。新增测试：三台加速器每条声明见证都要跑（且 `scope == Host`）、自检不受设备 preamble/flags 影响、期望值错一位即使自检也失败、`verify` 在缺设备时先报自检条数。
- **P0-4 内核 ABI 单源**：新增 `op::sig`，装载期从 `.k` 的入口函数解析形参名、类型与可写性；`OpDecl` 持有签名，`Model` 删掉 `call_args` 与 `output` 两个字段，`Sample` 改为具名标量（`M`/`N`/`gsz`）。执行侧（`cutile::backend` 与 `audit`）只按签名顺序派生实参与输出名，并在 emit 之前把签名与样本逐项对齐：把 `const float *scales` 写成 `double *` 会在装载期报出算子名与形参名，不再等 `cc` 的指针类型错误。
- **签名对齐的第三段：加速器片段层可拼接**：`nvidia` / `intel` / `ascend` 三个片段目录补齐 P0 模板用到的 9 个原语，并登记进 `prim::BUILTIN`（此前只登记 `cpu`，发布出去的二进制里没有这三台的片段文本，只能靠源码树或 `data.root`）。片段按 `meta.emit.ext()` 的扩展名（`cu` / `cl` / `cpp`）写，语义是 **P0 的标量路径**——`dot` 对应档案里 `fma` / `scalar_fma` 那条签名，`dp4a` / `mmad` / `cp.async` 这类 tile 落点仍留给 P3，因此三台设备的等级仍是 `L0`/`L1`，"未验证视为不具备"的协商结论不变。新增测试 `every_device_fragment_set_splices_the_p0_template`：四台设备的片段都必须能把 `gemv_i4` 模板在全部 tile 组合下拼出来，参数个数或命名参数写错不必拖到构建期才暴露。
- **层的性能下界 `nn::Bounds`（ADR 0008）**：`bytes` / `flops` 不再手写，而是由层声明的算子在 `op` 里的语义推出（`Bounds::of_op` → `op::model(op).cost(m, n)`，与 `cutile::plan` 同一份 `einsum` 定义），因此新增算子不会多出第二份流量公式。`Bounds` 不含设备知识：`intensity()`（flop/byte）是 roofline 横轴，`calls_per_second_ceiling(peak)` 的峰值由调用方传入。字段私有且只此一个构造入口，调用方拼不出"我以为是这么多"的下界；`plus` 用于一串层的累计。组装点 `llm status` 多一行 `blk.0.attn_q 4096 → ≥ 10.52 MB / 33.6 MFLOP，强度 3.19 flop/byte`——这一行是声明，`cutile bench` 的达成率是事实，两者相减才是融合/量化/布局真正赚到的部分。测试按 burn 的 `a_projection_declares_its_floor` 写法逐项钉住数字，并覆盖"层能构造、但形状对算子不合法因此没有下界"这一真实中间态。
- **P0-3 架构契约可执行**：新增 `tests/architecture.rs`，读 `cargo metadata` 与源码，把 `docs/ARCHITECTURE.md` 的三张表变成判据——成员集合与 `members` 一一对应；边集（含总数与 `+dev` 数）与文档完全一致；每条边只能指向更低的层或同层更靠左的成员；所有权表里的每个类型在自己声称的 crate 里确有 `pub` 定义。顺带把 `einsum` 归位到 L2 最左（`op → prof` 原先逆向），同层规则改为“只能依赖更靠左的成员”，CI 的 `test` job 从此就是这份契约的守门人。
- **P0-2 计量契约**：`bench` 改为 `CLOCK_MONOTONIC` 墙钟计时，warmup 不计入样本、逐次采样并报告 `min`/`p50`；亚微秒内核自动按批量折算（校准 + `CUTILE_BARRIER()` + `volatile` 出口读，挡住编译器把重复调用提走或删掉），并打印输入工作集与档案 `FAST` 容量。达成率分成 `Reached` / `Below` / `OverPeak`（>100%）/ `CacheResident`（工作集装得进缓存）/ `Unknown`，后两类明确打印「口径无效：数据在缓存中」且不参与 L3 判定。峰值常数进入可执行校验：`src` 只接受可求值的数值表达式，装载期与 `value` 比对。
- **P0-1 证据链闭合：见证可执行、形状矩阵、模板原语 ⊆ `needs`**。档案里的 `witness` 从"一个路径"变成可执行的 `{ file, func, expect }`（统一 ABI `int func(float *out)`）；`cutile verify` 先把**每一条** `grade ≥ L2` 的 `provides` 的见证编成可执行文件并执行，值与期望不符即整台设备失败，`grade` 不再是自述。随后 `verify` 对 `(group,1)`、`(group*2,3)`、`(4096,4096)` 三个形状各自编译对拍，并校验 `M` 非组大小整数倍的负例必须被 `parse_shape` 拒绝；声明层在装载期新增 `模板原语 ⊆ [needs] ∪ [extra_needs] ∪ 调度` 校验，漏写需求不再拖到 `substitute` 才报错。顺带修正 `quant` 见证里沿用的错误期望值（`0xF0` 解包应为 `-4.0`，不是 `-0.5`）。
- **注解层拆成三层，错误带上 span**：`annotation` 现在分 `syntax`（纯 token 模型与文法，不接触 `proc_macro`）、`codegen`（声明 → 代码，纯字符串）、`lib`（`TokenStream` 适配与 span 定位）。文法与渲染因此可以被单元测试（31 条），适配层用 `Streams` trait 做依赖倒置、测试用字符串实现它；错误不再笼统指向整个调用点，而是指向写错的那个 token。生成的名字统一带 `__layer_` 前缀（不占用用户方法名），运行期失败的消息里带层名。
- **配置键集进入身份**：`App::KEYS` 让文件层、环境变量层与 `--set` 只认本应用登记的键；`Error::UnknownKey` 直接列出该应用可用的键，`Settings::entry` 对未登记的键返回 `None`，`describe()` 只打印本应用的键。
- **非连续视图**：`tensor::View::strided` 支持转置/切片这类非连续布局，构造时校验步长轴数、零元素口径与最高偏移；`View::row` 改为按步长判断（列不连续时返回 `None`，而不是把跳着取的元素当成一片内存），并新增 `View::is_contiguous`。
- **`fuzz` 成员（目录 `crates/fuzz`）**：模糊测试的契约层——`Target`、确定性 `Seed`、`Behaviour`、`Finding` 与 `sweep`，把"同一输入结论稳定""拒绝必须给出非空消息"写成可执行判据（按 ADR 0004 替换了原来的 `add(2,2)` 占位）。
- 工作区 lint 加严：`unsafe_code = "deny"` 与 `clippy::{dbg_macro, todo, unimplemented} = "deny"`，让"临时手法留在代码里"变成红灯。
- **`annotation` 成员（目录 `crates/annotation`）：`#[layer]` 属性宏**。把“名字 + 输出形状 + 算子列表”写成属性，展开成 `impl nn::Layer`（`OnceLock` 构造一次规格）加一条 `#[cfg(test)]` 校验（算子被删而注解没改时 `cargo test` 失败）。手写解析（不引入 `syn` / `quote` / `proc-macro2`），错误一律走 `compile_error!` 且只报注解文法；不建运行期注册表（ADR 0005）。
- **`tensor::View<'a, E>` 与 `Element`**：安全的泛型零拷贝元素视图（形状 + 步长 + 调用方的 `&[E]`），按元素类型单态化，秩/越界检查返回 `None` 而不 panic；与字节视图 `tensor::Tensor` 分工明确（元素计算 vs 权重/打包类型）。
- **`cli` 成员（目录 `crates/cli`）：命令行前端**。`Command`（`Status` / `Verify { op }` / `Bench { op }` / `Help`）、`Invocation::parse` 与唯一的 `USAGE` 文本；只解析与分派，业务留在各层，退出码由 `llm` 决定。`llm` 的 `main` 因此收敛成“解析 → 分派 → 退出码”三步。
- **配置身份参数化**：`llm_config::App`（`DIR` / `FILE` / `ENV_PREFIX`）与 `Settings<A: App = Cutile>`；`Key` 只登记环境变量**后缀**，完整变量名由前缀拼出；`Roots` 只拼目录，不认识产品名（ADR 0006）。新增测试证明第二个应用读自己的 `demo.toml` / `DEMO_*`、读不到 `CUTILE_*`、默认目录互不干扰。
- `llm` 组装点新增两行摘要：注解声明的层（`nn::LmHead` → `[4096]`，算子 `gemv_i4`）与零拷贝视图（借用调用方 4 个元素求和）。
- 架构文档同步：`annotation` 作为工具成员列在 L1，`tensor` 的所有权扩展到 `View` / `Element`，`llm-config` 扩展到 `App`；当前 31 个成员 / 77 条依赖边（`einsum` 归位到 L2 最左，`audit` 作为独立成员列在 L3）。
- **架构契约**：`docs/ARCHITECTURE.md`（七层表 + 依赖规则 + 所有权表 + 新增成员三步）；每成员至少一条边、全部 `publish = false`、仓库只有一个 `target/`；P0-3 起由 `tests/architecture.rs` 在 `cargo test` 里执行。
- **16 个占位成员改成接口形状**（占位口径现见 `docs/ARCHITECTURE.md` 第五节）：`diagnostic`（`Error`/`Code`/`Result`）、`testkit`（`approx_eq`/`device_enabled`/`scratch_dir`）、`mem`（`Arena`/`Region`）、`tensor`（`Tensor`/`DType`/`Shape`/`Strides`）、`storage`（`Layout`/`Order`/`remap`）、`exec`（`Stream`/`Graph`/`Launch`/`Executor`）、`tokenizer`（`Tokenizer`/`Token`/`Bytes`）、`weights`（`Header`/`Entry`/`Reader`）、`kv`（`Page`/`Table`/`Cache`）、`autograd`（`Tape`/`Var`）、`comm`（`Group`/`Reduce`/`Collective`）、`proto`（`Envelope`/`Kind`/`Version`）、`data`（`Batch`/`Split`/`Dataset`）、`nn`（`Layer`/`LayerSpec`/`Linear`，层定义必须引用 `op` 已声明的算子）、`shard`（`Topology`/`Partition`，rank 布局与并行组）、`train`（`Schedule`/`Step`/`Checkpoint` + 恢复拓扑校验）、`engine`（`Sequence`/`Scheduler`，准入即占页）、`serve`（`Route`/`Endpoint`/`Response`）。
- `tool` 成员（目录 `crates/tools`）：`Report` / `join_names`，把人可读输出从组装点里抽出来，避免三处各写一套排版。
- `benches` 成员（目录 `crates/benchmarks`）：`Case` / `Baseline` / `Judgement` / `Direction`，把“方向 / 相对预算 / 回退与改进”写进类型，供后续 bench job 直接判定。
- 根包 `llm` 变成真正的组装点：逐层构造最小实例并打印一份摘要（`op` / `prof` / `tensor` / `weights` / `train` / `serve` / `tokenizer` / `kv` / `engine` / `exec` / `tool` / `diagnostic`），这也是“依赖图能编译且公共 API 可用”的端到端证据。

- `op` 成员（目录 `crates/op`）：声明层独立成 crate——`gemv_i4.toml` + `gemv_i4.k` 从 `crates/cutile/ops` 迁入，`src/lib.rs` 提供 `BUILTIN`（内嵌）、`builtin()`、`dir()`、`load()`；`cutile` 单向依赖它，新增算子不必动机制代码。
- `prof` 成员（目录 `crates/prof`）：档案层独立成 crate——`devices/*.toml` 从 `crates/cutile/devices` 迁入，`src/lib.rs` 提供 `BUILTIN`（内嵌）、`builtin()`、`embedded()`、`dir()`、`load()`；`cutile` 单向依赖它，新增设备不必动机制代码。`proto` 占位成员补齐元数据与 crate 文档，使 workspace 门禁可整体通过。
- `cutile` P0 纵切面实现（约 4,900 行 Rust + 数据）：`atom`（16 原语闭集 / 参数化能力签名 / 逐字段差异）、`prof`（档案 schema + 指纹规则 + L2 见证校验）、`cap`（协商 + 降级链 + 可解释诊断）、`layout`（4 构造子 / 正规形 / repack 计费）、`space`、`sch`、`lower`（词法 / 轴展开 / 片段拼接 / emit）、`cache`（FNV 指纹 + 锁 + 原子落盘）、`plan`（roofline）、`reference`（标量参考 + 对拍口径）、`backend`（`cc` 编译执行）、`probe`、`builtin`、`toml`（手写受限解析）。
- `prim` 成员（目录 `crates/prim`）：片段层独立成 crate——`crates/cutile/prim/cpu/*.c` 迁入为 `crates/prim/cpu/*.c`，`src/lib.rs` 提供 `BUILTIN`（按 `(vendor, ext)` 登记的内嵌片段）、`builtin()`、`load()`、`dir()`、`layer_root()` 与 `FragmentSet`；文件名不在原语闭集内即装载报错，`witness.<ext>` 与片段同目录同扩展名、装载时按名字跳过（旧目录扫描会把它当成非法片段名而失败）。`cutile` 的 `lower/substitute` 只剩拼接（`pub use prim::FragmentSet`）。
- 见证路径语义收敛：档案里的 `witness` 改为**相对片段层根**（`cpu/witness.c`），仓库内根是 `crates/prim`、`data.root` 下是 `<data.root>/prim`；`Catalogue::root()` 随之删除，`cutile` 不再需要知道自己的清单目录。
- 三层数据：`crates/op/ops/gemv_i4.toml` + `gemv_i4.k`（声明层）、`crates/prof/devices/{ascend,cpu,intel,nvidia}.toml`（档案层）、`crates/prim/cpu/`（片段层，9 段 + 见证）。
- **执行单元进入闭集**：`AttrKey` 新增 `exec`，档案新增 `[units.<name>]`（单元名 → 可访问的存储层），`Profile::validate` 校验 "`exec` 必须是已声明单元" 与 "该单元访问得到这条供给声明的存储层"，于是"昇腾 Cube 只能访问 L1/L0"是可校验的数据而不是注释；`Need` 写 `exec = ["vector"]` 即"我要 vector 实现的点积"，协商仍是同一次集合/区间判定。
- 昇腾档案 `crates/prof/devices/ascend.toml`：`GM` / `L1` / `L0C` 三级存储 + `cube` / `vector` / `mte` / `fixpipe` / `scalar`五个执行单元，10 条 `provides` 全部 `L0`（P0 无真机证据，未验证视为不具备）；`probe::fake("ascend")` 让指纹选档可离线回归。
- `prof` crate 承接档案 schema 与能力词汇表：`crates/prof/src/{atom,prof}.rs`（从 `cutile` 迁入），`cutile` 只 re-export（`pub use ::prof::{atom, prof}`），`SCHEMA_VER` 升到 3。
- CLI：`cutile doctor` / `verify` / `bench`（`src/main.rs`）；`verify` 在缺设备时返回非零码；`doctor` 新增 `units:` 一行，直接显示档案声明的执行单元与可访问存储层。
- `llm-config` 从占位成员变成真正的配置基座（目录 `crates/config`，包名 `llm-config`）：`Key` 键闭集（`cache.dir` / `work.dir` / `data.root`，只登记有消费方的键）、`Kind`/`Value` 类型校验、`Sources` 三层来源（默认 < 用户级文件 < 环境变量 < 命令行）与 `Origin` 可追溯、`Roots` 绝对路径解析、`Settings` 一次性加载并向后传递；配置文件出现未登记的键直接报错，`schema` 缺省按当前版本处理。
- `crates/config/README.md`：说明三条规则（键闭集 / 唯一来源 / 绝对路径）、键表与"什么不该放进来"的边界。
- 初始工程：`src/main.rs` 入口，`Cargo.toml` 元数据（双许可、`publish = false`）。
- 工程门禁配置：`rust-toolchain.toml`、`rustfmt.toml`、`clippy.toml`、`[lints]` 段、`.editorconfig`、`.gitattributes`。
- GitHub Actions CI：fmt、clippy、test（三平台 + MSRV）、doc 五个 job。

### Fixed

- **端点把"请求不合法"当成了服务端故障**：`serve::admit` 用 `?` 把 `engine::Scheduler::admit` 的 `invalid` 原样上抛，而端点契约里 `Err` 意味着"服务端自己坏了"——于是 id 已在批里、序列已结束、页宽与调度器不一致这三种**客户端能改**的错，最终表现成 5xx 或直接断连接，客户端连"我哪里写错了"都读不到。现在只有真正的服务端故障（`internal` / `io`）继续上抛，`invalid` 落成 `400` 且载荷带原话；准入因此有三个不同结局：进去（200）、现在满（可重试 503）、永远装不下或请求自相矛盾（400）。新增两条用例：重复 id 与页宽不一致都是 400 且不多占名额、不多耗页；已结束的序列（先跑完再 `finish`）同样是 400 并报"已结束"。
- **训练缓冲的两条承诺只写在文档里**：`train::Buffers::load` 的文档说"登记的元数据与实际数据必须一致"，可实现是`if let Some(...)`——变量没登记形状时整段校验被静默跳过，任意长度、任意元素类型的数据都能占住一个登记为 4 字节的槽位，之后每次 `compute` 都算在一个与数据无关的形状上（错得毫无痕迹）。现在缺登记是 `internal` 并指名"调用方漏了 `Sizes::set`"，与 `compute` 同口径。同一层的分配也从 `vec![0.0; n]` 改成 `try_reserve_exact`：`n` 来自登记的（形状，元素类型），而形状可以是从权重文件头读出的不受信任输入，超大形状原先会直接 `abort` 整个进程——现在是一条带类别的错误，且失败时不留半块缓冲区。新增两条用例：未登记的变量被拒且 `live()` 不变；`[usize::MAX]` 的输出形状报 `invalid` 而不是打死宿主。
- **量化推断的"第三条硬约束"是假的，公式也写错了**：`weights::quant` 的模块文档声称"打包权重字节数必须等于 `out × ceil(in / 2)`，对不上就返回 `None`"，但本层从未检查过——`Entry::len()` 由 `shape × dtype` 推出、safetensors 又要求声明区间等于这个乘积，所以从 `Header` 造出来的条目**结构上**必然满足它，再比一次等于拿一个值与自己比。而且真实口径是 `ceil(out × in / 2)`（4 位当作一整块紧贴位数组、**按行不补齐**），原文那个公式在输入宽为奇数时就会差一个 nibble。文档改成两条硬约束 + 一段"为什么第三条不在这里"，并说明它挡不住真正含糊的那件事（8 位条目到底是逐元素 int8 还是两个 nibble），那件由上层拿分组与内核声明一比来拒；同时把测试夹具的错公式一并改正，新增一条用例把 `ceil(out × in / 2)` 钉住。
- **切分拓扑的 `world()` 会饱和、`group()` 会回绕**：`Topology::world` 用 `saturating_mul`（`tp · pp · dp` 溢出时谎报`usize::MAX`），而 `group` 用不检查的 `*`——两个口径不一致的后果是"检查通过、rank 错位"：`check(usize::MAX)` 会因为饱和值恰好相等而通过，随后 `group` 用回绕的乘积发出一串看起来完全合法的 rank，训练只在若干步后表现成各进程算的不是同一个模型。现在 `Topology` 在构造期就把乘积算好并校验它**必须能用 `u32` 表示**（rank 本来就是 `u32`），`world()` 因此是精确值，`group` 里所有乘积的上界都是它，不可能回绕；超出 `u32` 的拓扑直接 `invalid` 并说明"rank 布局无解"。新增两条用例：一条锁住溢出/恰好超出 `u32` 都被拒、`65_535 · 65_535` 这种"贴着上限"的乘积仍然合法且 `world` 精确；另一条按五组拓扑验证三个维度的组各自把 `0..world` **恰好切一遍**（每个 rank 出现且只出现一次、越界的第 N 组被拒）。
- **解锁建议会把两个变体拼成一句话**：`cap::negotiate` 记下了“缺得最少的变体”的缺项清单，却用 `chain.first()` 的变体名去配它——首选缺 `dot` + `load_async`、兜底只缺 `dot` 时，诊断会说“补齐 `dot` 后可用 pipe2”，而 pipe2 还缺 `load_async`，补完仍不可用。现在 `closest` 连同变体下标一起记，`Unlock.variant` 与 `missing` 必然同源，建议行同时打印档位与代价（`可用 direct（第 2/2 档，预计 mma_share 0.35）`）。
- **区间的结束偏移会溢出**：`mem::Region::end()` 是 `offset + len`，而 `Region` 可以由 `Region::new` 从反序列化的偏移造出来——`Region::new(usize::MAX, 8)` 一调 `end()` 就在 debug 构建里 panic，`Arena::slice` / `slice_mut` 文档承诺的"越界返回 `None`（不 panic）"因此是假的。现在按 `saturating_add` 饱和到 `usize::MAX`：越界区间只会被 `slice` 判成 `None`；用例同时锁住"合法区间不受影响"与"贴着末尾的零宽区间也越界"。
- **批级"谁在等页"没有回答**：`engine` 只给了单步的 `Progress::Blocked`，服务端想知道"这一批卡在谁身上"只能自己重算页数，而重算就要复制 `advance` 的页数公式。现在 `Running` 记一个派生值 `needs_page`（在准入与每次推进后刷新），`Scheduler::blocked()` 直接给出阻塞集；判据是精确的——页表一直按长度刚好够用，下一步最多多要一页，所以"要新页且空闲为零"就是全部条件。随机对拍用例在每一步断言"`blocked()` 命中某序列" 与 "`advance` 对它返回 `Blocked`" 一致（400 步）。
- **帧头里的长度会算出算术溢出**：`Envelope::parse` 用 `bytes.get(HEADER_LEN..HEADER_LEN + len)` 切载荷，而 `len` 是对端发来的 `u32`（可以是 `0xFFFF_FFFF`）：32 位平台上 `10 + 4294967295` 溢出，debug 构建直接 panic——十来个字节就能打出一个拒绝服务。现在先切帧头再比长度，错误消息不变；顺带补了两条用例：极限声明长度必须是 `invalid`，以及一个缓冲里有多帧时只解析第一帧、剩下的按 `HEADER_LEN + 载荷长度` 切给下一帧（流式解析的口径写进文档）。
- **"永远装不下"被当成"稍后重试"**：`serve::admit` 把任何拒绝都翻成 503，于是 `max_new` 大到最坏情况超过整个页池的请求会得到"请稍后重试"，客户端重试到天荒地老也没有用。现在 `engine::Scheduler::fits` 是准入里那条判据本身并对外暴露，服务层据此分两种拒绝：现在满 → 503（可重试），最坏情况超过页池 → 400（只有调小 `max_new` 或加大页池才有用），而且后者连一页都不占，重试不会白耗容量。
- **形状的乘积会溢出**：`Shape::numel` 用的是 `iter().product()`——debug 构建里一个来自权重文件头的巨大形状直接 panic，release 构建里回绕成一个小数（`[2^32, 2^32]` 得到 0），于是 `Tensor::new` / `View::new` 的"缓冲够长"判定被绕过，得到一个形状与缓冲不符的张量。现在按 `saturating_mul` 折叠，溢出饱和到 `usize::MAX`：下游只会按"缓冲不够"拒绝，`View::strided` 里现有的 `checked_mul` 照旧报"形状与步长算出溢出"；新增用例同时锁住"真有一轴为 0 才是 0 个元素"。
- **"页不够就拒绝"只在准入那一刻成立**：`Scheduler::admit` 按提示长度占页，之后 `Sequence::step` 只加计数——生成到第 `max_new` 个 token 时页表可能只覆盖提示那几页，与 `engine` 文档承诺的"绝不让 KV 缓冲跑到一半溢出"正好相反。现在批里唯一的生成入口是 `Scheduler::advance`：先按"这一步要写的位置"预留页（`kv::Table::grow` 抬容量，页仍由 `Cache` 发），池不够时返回 `Progress::Blocked` 且序列、页表、页池都不动，调用方可以等别的序列 `finish` 之后原样重试；`Sequence::step` 因此降为同 crate 可见，"绕过页账的生成"写不出来。准入也顺手挡掉三种会白占页的输入：页宽与页池不一致、序列已经结束，以及最坏情况（`Sequence::pages_required`）连独占整个页池都跑不完的请求——最后一种是活性问题：它迟早 `Blocked`，而批里没人能结束，页也就永远回不来。顺带把 `Cache::release` 的重复归还判定从 `free.contains`（O(总页数)）换成每页一字节的存活位图（O(1)）——一次 `finish` 要逐页归还，线性扫描会把回收变成 O(总页数 × 页数)。用例覆盖跨页增长（长度 4 仍一页、5 转两页、9 转三页）、阻塞无副作用、页宽不一致与"已结束的序列不再准入"，并用 400 次随机准入/推进/结束对拍"占用 = 页表之和""每张页表最多短一页""空闲 + 占用 = 页总数"。

- **见证矩阵把“没有 `[simt]` 段”当成单 lane**：`every_device_covers_the_primitive_set_or_explains_the_gap` 只判 `simt.group <= 1`，于是昇腾（不写 `[simt]`，lane 交换由向量单元 `vgather`/`vbroadcast` 承担——`prof` 的文档与 `Profile::validate` 都是这么认的）被读成“声明了 `group = 1` 却又有 `shuffle`”，档案装载失败、`cutile` 三条命令全部不可用。现在按 `simt.declared` 区分，与 `prof` 的判据同一口径。
- **“流水掩盖 ≤ 20%”只是文档承诺**：DESIGN §12.2 判据 1 写着首选变体必须把未掩盖的非 MMA 时间压到 20% 以内，但 `(1 - mma_share) / stages` 只被算进 roofline 估值，没有任何判定——声明里给 `pipe<2>` 配 `mma_share = 0.10`（未掩盖 45%）照样能装载。现在公式与阈值在 `op::sch`（`Masking` / `UNMASKED_LIMIT` / `check_masking`），装载期只判降级链首选，不达标即拒绝解析并指出该换 S3/S4；兜底变体不判（它们存在的理由正是“设备缺原语时仍能跑”），账照样进 `doctor`/`bench` 的计划行。
- **“产物命中时连编译器都不需要”只是文档承诺**：`ArtifactKey` 里带着真正调用的编译器，于是没有编译器就拼不出产物文件名，`audit::verify`/`bench`/见证执行都在流水线最前面把 `detect_toolchain()` 当必需项报错——一份已经验证过的产物在这台机器上反而跑不了。现在每份产物旁边写一行身份（`bin/<hex>.key`，`ArtifactKey::describe` 原文），`build::execute`/`run_program` 的 `tool` 变成 `Option<&str>`：没有编译器时按“源文本 + 主函数”在条目里找候选，唯一一份就直接复用（`verify` 实测 `产物 3 份（复用 3，cc）`、`bench` 照常出数），多份候选取哪一份都是替用户选代码生成、也就替 `bench` 决定耗时的来历，因此不猜——列出编译器名让用户装一个或清掉多余的产物；`--repeat` 变了这类“主函数不同”的情况则明确报“没有这份产物”。见证程序的条目名也随之只由程序文本决定（编译器是产物身份的一部分），同一份程序的 `cc`/`clang` 产物共用一个条目。
- **同一个缓存键的并发编译共用一个沙箱目录**：`build::cached_run` 的沙箱名只是 `<key>-<pid>`，于是同一进程里两条路径同时编译同一个程序（`verify` 的自检与形状矩阵、两条自检用例）会写进同一个目录：先完成的那条删掉沙箱，另一条的 `cc -o` 就失去输出路径（linker 报 `cannot open output file`）。现在目录名再带一个进程内序号，`cargo test -p llm-audit` 并行跑不再偶发失败；见证变多、编译变慢之前，这个窗口只是恰好没被撞上。
- **崩溃残留的缓存目录既不计入占用也不被回收**：扫描只认带 `source` 的目录，`source.tmp` / 只有 `meta.txt` 或锁的半截目录被当成"别的工具建的目录"跳过，于是它占着磁盘、永远等不到任何一次"超限"，`doctor --why` 还会因为它的差异为空把"只写了一半"说成"命中"。现在每条目录都有状态（`cache::State::{Complete, Partial}`），`cache gc` 把与上限无关的残留单独清掉并在报告里分行计数，`CacheWhy::state` 先分"从未落盘 / 半截落盘 / 一致 / 不同"。
- **一次崩溃会让某个缓存 key 永久卡死**：`build.lock` 是 `O_EXCL` 文件而不是 flock，持有者被 `kill` 后没人能释放它，之后每次构建都要等满 5 s 再报"确认没有并发编译后删除它"。现在陈旧锁（`cache::STALE_LOCK` = 60 s，而一次 emit 是毫秒级、等锁上限 5 s）由取锁方自己删掉重试，活着的锁照旧不抢；`cache gc` 也把陈旧锁当成残留一并回收。
- **缓存的容量上限与“为什么没命中”只是文档承诺**：DESIGN §10.3 写着 LRU + 2 GiB 回收与 `cutile doctor --why <key>`，但 `cache.rs` 只做指纹/锁/落盘，缓存目录只会一直变大，旧键也只能靠人肉 diff 两行拼接文本。现在 `cutile cache gc [--limit <字节>]` 按 LRU 回收到上限以内（读命中更新 `meta.txt` 时间戳，`build.lock` 还在的条目跳过并计数），`doctor --why` 用 `CacheKey::fields` 与 `meta.txt` 逐项比较、只列出真正不同的字段。
- **峰值常数与自己的推导式对不上**：`cpu.toml` 的 `flops_f32 = 108.8e9` 与 `8 cores × 2.6 GHz × 8 lanes × 2 flop = 332.8 GFLOP/s` 差三倍，`bw = 68.0e9` 与 `2 × 8 B × 2666 MHz = 42.7 GB/s` 也不一致；`intel.toml` / `nvidia.toml` 的推导式同样对不上自家数字。现在 `src` 必须是可求值表达式且与 `value` 一致，三份档案的峰值已按各自公式重算（CPU 带宽 42.656 GB/s，NVIDIA 稠密 f32 94.6 TFLOP/s、HBM3e 8.192 TB/s）。
- **`bench` 的带宽达成率差了 1e9 倍**：旧实现拿 GB/s 直接除档案的 B/s 峰值，实测 0.48 GB/s 对 68 GB/s 会打印 `0.0%`。现在按 `bytes / ns × 1e9 / peak` 换算，同一台机器上 `bench --shape 1024x1024 --repeat 20` 打印 `0.7%`；换算抽成 `cli::throughput` 并有单元测试锁住量纲。
- **编译参数从来没生效**：`backend::compile_and_run` 写死 `-O2 -std=c99`，而档案 `[compile].flags` 只进了缓存指纹。现在 `flags` 由调用方传入并原样交给 `cc`，`cpu.toml` 的 `-O3` 因此真的生效，编译失败信息也会打印实际使用的参数。
- **缓存锁失败被静默吞掉**：`build_source` 里的 `cache.lock(&key).ok()` 改成 `?`，拿不到锁时返回带锁文件路径的错误，而不是与另一个进程同时写同一份 `source`。
- **`swz<B>` 的移位溢出**：`space::check_alignment` 原先用 `1u32 << B`，而 `LayoutSpec::parse` 接受 `u8` 到 255，声明 `swz<40,3,3>` 会让 `doctor` panic。现在用 `checked_shl` 先挡住溢出，再与档案 bank 数比较并报错。
- **分组轴被算进了输出形状**：`Axes::output` 原先只排除规约轴，于是 `scales[n, g]` 的派生轴 `g` 也进了 `output_shape`（`y[n]` 被说成 `y[n, g]`，`[4]` 变成 `[4, 2]`）。现在只取批轴与自由轴——分组轴是"同一批元素的另一层看法"，本来就该只计流量。
- **`doctor` 的 `[存储]` 段把中转缓冲冒充成容量**：旧行 `FAST=L2 (128 B)` 里的 128 B 是 `GMAX × f32`，而档案声明的 FAST 容量是 512 KiB——既不是容量也不是用量。现在逐层（`GM`/`FAST`/`REG`）打印档案容量与构建期契约真正比较过的声明用量（`audit::Storage` + `audit::Declared`），容量只有 `space::capacity` 一个出处。
- **构建期存储契约曾排在缓存命中之后**：声明的 `bytes` 上界不改变源文本、因此不在缓存键里，于是同一份声明先按合法值落过盘、再改大就会直接复用旧文本，越界声明免检。契约现在先于缓存查询执行，`build::contract` 返回 `Vec<Usage>`，并新增用例证明缓存命中不能跳过判定。
- **`bench` 的性能判据只在失败时出声，而 §16 的「原生 85%」没有可测的分母**：达成率的口径（`Unknown` / `OverPeak` / `CacheResident`）与判定（`Reached` / `Below`）混在一个枚举里，前端又把它包在 `if let Some(peak) && let Some(ratio)` 中——档案没声明峰值时 `bench` 一行结论都不打印，"没说话"于是被读成"通过"；同时 §16 的判据表写着"性能 ≥ 原生 85%"，而仓库里没有原生库的对照实现，那个分母要等 P1 才有。现在 `audit::Performance` 把达成率与它的尺子绑在一起（`is_measured` / `needs_specialisation`），`Bench::performance` 取代 `Bench::verdict`；`bench` 每轮都打印一行结论——`Reached` 明说"尺子是 roofline：实测 ÷ 档案峰值，不是相对原生库的 85%"，`Below` 给出 §16 的"这台设备承认需要特化代码"，口径无效或没有峰值时打印"未测量，不参与 Go/No-Go"。§16 的判据行随之改为 `roofline 达成率 ≥ 70%`（与 §12.1 的 `Reached`、§12.3 的 L3 门槛同一个数），85–95% 那组"相对原生库"的数字只留在 §12.3 当目标。

### Changed

- **`data` 的两个构造函数把「行」与「列」写反了**：`Chunks::new` 收 `(cols, rows)`、`Batch::new` 收 `(rows, cols)`，而两种顺序都通得过校验——同一对数字换个构造函数就是一次静默转置，正好是这个 crate 声称要挡住的那类错值。现在两者都按 `(rows, cols)` 收几何、与 `Shape::new([rows, cols])` 同序，并有用例把「2 行 3 列」钉死。同批：`Chunks` 的游标从「已取走的 token 数」改成「已取走的批数」（`batches` 在构造期算好），`batches()`/`remaining_batches()` 因此退化成减法、`next_batch` 退化成一次比较加一次切片，不再有两条只能表示「取完了」的 `?`；`cols()`/`batches()`/`remaining_batches()` 补上 inherent 实现（此前只存在于 `Dataset` 实现里，不 `use` trait 就调不到）。`Chunks` 现在是 `Iterator + ExactSizeIterator + FusedIterator`：`len()` 与 `size_hint` 都取自 `remaining_batches`，`for` 循环与 `collect()` 不必自己数还剩几批，也不会与 `next_batch` 走出两个游标。`Split` 的百分比算术从 `u128` 中间量改成 `(total / 100) * p + (total % 100) * p / 100`：两项都在 `usize` 内，和恰为 `floor(total × p / 100)`，于是两条在现有目标上不可能触发的 `Code::Internal` 一并删掉。

- **`dtype` 的测试合并重复部分：同一个契约只写一次**。四处重复被收掉。一，「trait 视角 == `const fn` 视角」原来拆成三条测试、十段按类型手抄的循环（`tests/identity.rs` 两份、`tests/geometry.rs` 一份），现在是一处按类型展开的断言——新增一个闭集加一行，多一个块格式加一个 `block`，身份与块几何由此共用一个断言点；`tests/geometry.rs` 只留性质（整块取整、单调、饱和、`Layout` 两条手臂）。二，互操作的两条出口（`serde` 的线上名字、`bytemuck` 的受检字节）原来按四个闭集各抄四遍，现在各是一个泛型 helper（`Format + Serialize + DeserializeOwned` / `Format + NoUninit + CheckedBitPattern`），新增闭集只是多一行调用，漏一个闭集也不再可能无声无息；顺带把「写出的字节 = 行表下标」这一条显式化。三，`tests/mx_spec.rs` 的 OCP MXFP4 与 NVFP4 两条金标用例结构逐字相同，现在共用一张字面用例表（格式、块长、块字节、元素、字面块、逐项期望值）与一套断言，scale 由 `MxType::scale()` 选解码器——`scale` 这一列因此也被数值校了一遍。四，`tests/geometry.rs` 里标量轴的单调扫描与块格式那一份是同一段循环，现在共用一处（按轴传访问器与标签），只有饱和判据各留各的口径：块格式给下界，标量给精确上界。文档同步：`format.rs`/`lib.rs` 与 ADR 0018 里「身份归 identity.rs、几何归 geometry.rs」的说法改为「两条读法的对拍在 identity.rs，geometry.rs 只管性质」。

- **`dtype` 的测试按「被测契约」分文件，字面事实锁全量**：`tests/formats.rs` 一个文件里混着身份契约、块几何与 `Layout`，`tests/mxfp4.rs` 实际覆盖 NVFP4 与全部 MX 几何——两个文件名都对不上内容，而且同一批事实在内联测试与集成测试里各写了一遍。现在按对象拆成 `tests/identity.rs`（四个闭集的身份读法）、`tests/geometry.rs`（块几何、单调与饱和、`Layout` 两条手臂）、`tests/mx_spec.rs`（OCP v1.1 与 NVFP4 的字面块）、`tests/interop.rs`（可选 feature）；每个闭集的字面事实收进一张与 `ALL.len()` 锚定的行表：`DType` 的 code/名字/变体/位宽/类别、`QType` 的块长/块字节/位宽/scale/族、`MxType` 的块长/块字节/元素/scale/族。此前只锁了子集——`QType::bits` 25 取 8、线上 code 24 取 22、NVFP4 的 scale 与块长没有逐项锁。删掉的是重复与自证的断言：`FromStr`/名字往返在四个闭集上由泛型契约统一测一次，`bytes_for` 与块几何不再用实现里的同一公式重算。
- **`proto` 的帧格式升到第 2 版**：`Version::CURRENT = 2`，每帧末尾 4 字节 CRC-32（`TRAILER_LEN`），`Version::FIRST = 1` 仍可读，`HEADER_LEN` 不变；代价是每帧多 4 字节与一次校验，换来“长度对得上但内容坏了”在解码期就失败。
- **注解里的闭集不再是自己的一份手抄表**：`#[frontend]` 的 `LANGUAGES`（9 个）与 `#[model]` 的 `DTYPES`（3 个）此前是两串字面量，与 `prof::Emit` / `prof::DType` 之间没有任何检查——词汇表增删一个变体，注解只是安静地留着一个匹配不上、或者永远用不上的名字。现在两者都是**等式**：`LANGUAGES` 与 `Emit::ALL` 逐项相等，`DTYPES` 与「`cutile::reference::tolerance` 给出非零阈值」的 `DType` 集合逐项相等（新增门禁 `tests/contracts.rs::annotation_names_come_from_the_vocabulary`，并用 `Debug` 变体名核对 `codegen` 的「首字母大写」映射）；此外每个注解调用点新生成一个 `#[cfg(test)]` 模块，把字面量与变体对齐（`Emit::parse(emit) == Language::EMIT`、`DType::parse(dtype) == Some(MODEL.dtype)`），于是失败指向被标注的那一行。`annotation` 仍是零依赖：核对住在门禁与生成的代码里，而不是给过程宏拉一条到 `prof` 的依赖边。
- **闭集与位集是两份手抄的事实源**：原语 / 数据类型 / 存储层 / 属性键 / 证据等级各自维护`ALL` + `as_str` + `parse` + `index`，`DTypeSet` / `SpaceSet` 又各自手写一遍位运算（`u32` + `1 << index`），加一个元素要改四五处、且有 `index` 与 `ALL` 顺序漂移的风险。现在 `prof::atom` 只留一个契约：`Closed`（`ALL` / `as_str` / `index`，`parse` 派生）+ `Set<E, WORDS>`（const 参数控制位宽，`const FITS` 在单态化时断言容量）+ `inherent_names!`（同名固有转发，调用点写法不变）；`DTypeSet` / `SpaceSet` 成为别名，`Emit`、`op::Skeleton` 也改成同一个契约。新增门禁 `tests/contracts.rs` + `lints/vocabulary.toml`：禁止再手写闭集表或位集，白名单必须带理由且过期即失败。

- **§16 的「片段 ≤ 400 行（内部目标 240）」原先只打印数字**：`verify` 的结论行与 `doctor` 的 `[emit]` 都只写`source 53 行`，读不出这条判据是达标还是超标。现在 `llm_audit::Fragment` 携带行数与口径（`FRAGMENT_LIMIT = 400`、`FRAGMENT_GOAL = 240`），`Built` / `Verified` 都换成它，`within_limit()` 就是“行数 ≤ 400”这条判据本身；两处输出共用同一句 `片段 53 行（上限 400：达标；内部目标 240：达标）`，上限管成败、内部目标只是余量。
- **DESIGN §16 的「协商 3/3 命中首档」原先没有出分的地方**：判据表里写着“首选没被选中说明 `needs` 粒度错”，但 `verify` 只报逐设备执行结果——把 cpu 换成 `cpu,nvidia,intel` 也读不出这个分数，缺设备时更是只剩两种断言。现在 `cap::Path::is_head()` 给出档位，`llm_audit::Negotiation`（首档 / 降级到第 k 档 / 链上无可行变体（列出首选缺的原语）/ 无档案）与 `VerifyReport::head_hits()` 一起进报告，`verify` 结尾打印`协商首档 1/3` 与逐设备一行；`doctor` 的 `path:` 行标注 `第 1/2 档（首档）`。协商只读声明与档案，所以本机没有该设备时照样有答案，与“缺设备即失败”各自成句。
- **`Plan` 的掩盖项换成一份账**：`Plan.unmasked: f64` 变成 `Plan.masking: Masking`（`mma_share` / `stages` / `unmasked` 一起走），roofline 的未掩盖项与计划行打印的百分比因此是同一个数；计划行自带阈值结论（`未掩盖 19%（阈值 20%，达标）` / `未掩盖 65%（阈值 20%，未达标：降级兜底）`）。
- **组装点的失败退出码改由错误类别决定**：`llm` 的 `status` 分支此前对任何错误都返回 `ExitCode::FAILURE`（1），于是"权重文件不存在"（`io`）与"`dtype` 不在闭集内"（`unsupported`）在脚本里无法区分。现在走 `Code::exit_code()`（`invalid` 2 / `unsupported` 3 / `missing` 4 / `io` 5 / `internal` 70），与 `cutile` 二进制同一份契约。实测：文件不存在 → 5，`U8` 权重 → 3。
- **`lower/` 平铺成两个根级模块**：`cutile` 其余模块都是单文件，只有 `lower/` 占着一层目录，而它真正实现的只有 ④ 拼接（`substitute`）与 ⑤ 成文（`emit`）——`OpDecl` / `Variant` / `expand` 本来就是从 `op::decl` 转发的路径，`mod.rs` 只是把它们与两个子模块重新导出。现在实现移到 `src/substitute.rs` 与 `src/emit.rs`，转发上移为 `lib.rs` 的 `pub use ::op::decl`（与 `layout` / `sch` 同一模式），调用点从 `cutile::lower::{Source, FragmentSet, fnv1a, expand}` 变为 `cutile::emit::{Source, fnv1a}`、`cutile::substitute::FragmentSet` 与 `cutile::decl::expand`；两步的名字仍留在模块路径里，DESIGN §10 的 ④/⑤ 标注因此还能对上代码。行为不变：`emit`/`substitute` 的单测、`builtin` 的四设备拼接用例与 `audit` 的缓存键逐字不变。
- **构建期存储契约的输入变成可报告的数据**：`space::capacity` 统一了"档案容量"的取法（`check_capacity` 与报告共用），`build::contract` 从只判不报改为返回 `Vec<Usage>`（原语 / 层 / 字节），`doctor` 的 `Storage` 由"一个中转字节数"改为"每层：容量 + 声明用量"。
- **层声明算子时多一条校验：必须有语义表**：`nn::LayerSpec::new` 原先只要求算子出现在 `op::builtin()`，现在同时要求 `op::model()` 能拿到它的语义表——只声明不实现的算子推不出 `Bounds`，也无法参与 `verify` 对拍，因此在层构造期就以 `Code::Missing` 拒绝，而不是拖到 emit 阶段才报"未知算子"。`Linear` 的算子名收敛为一处常量 `Linear::OP`（构造与下界共用）。
- **生产形状与输出类型回到算子声明**：`op::model::Model` 新增 `shape`（`gemv_i4` 为 `(4096, 4096)`）与 `dtype`，`audit` 的 `doctor`/`verify` 形状矩阵、`bench` 缺省形状与 `verify` 的对拍阈值都从它推出，不再各写一份 `4096x4096` 与 `* 4`。`--shape` 因此没有缺省值（不给就用算子的生产形状），`Bench.shape` 变成 `Option<String>`，`doctor` 头部打印 `生产形状 MxN`；形状矩阵的生产形状也不再“4096 不是组大小整数倍时退回”，算子的生产形状本来就是它的合法输入，并由一条测试锁住。
- **缓存键不再含凭空的数据类型与形状**：`CacheKey.dtype` 从字符串 `"f32"` 变成 `DTypeSet`（`[needs].dt` 的并集），`layouts` 从“用编出来的 tile×64 实例化出的正规形”变成声明式（`row` / `col` / `swz<…>` / `pad(p)`）；`doctor` 的 `[布局契约]` 段随之改为 `[布局]`，只打印 `槽位 = 声明式`。代价是缓存键变化（`fc79113cc543d8b6`），下一次运行会重建一次源文件。
- **并发取锁改成有上限的等待**：另一个进程正在为同一个 key 生成源文件时，等待它落盘并复用其产物（上限 5 s），只有残留锁文件才会带着路径报错；并行的 `doctor`/`verify` 不再因为共享缓存键而互相失败。
- **加速器补上见证源码，见证校验与等级解耦**：`nvidia` / `intel` / `ascend` 的每条 `provides` 现在都带 `witness = { file, func, expect }`，对应的 `nvidia/witness.cu`、`intel/witness.cl`、`ascend/witness.cpp` 按设备语言登记进 `prim::WITNESS`（与 `BUILTIN` 分开：见证不是片段）。`dot` 的 `dp4a`（i8×4）与 `fma`/`scalar_fma`（f32）各用自己的入口函数，`mmad`（f16 分块）用 `witness_dot_f16`；`quant` / `load` / `load_async` 的标量与特化实现语义相同，共用函数。`Profile::validate` 改为**声明了见证就校验**（文件可读、入口函数在源码里、期望值非空且不超上限），不再只看 `grade ≥ L2`——这样 L0/L1 的证据可以先进数据，路径或函数名写错在装载期就暴露，而不是等升 L2。等级本身不变：本机没有 CUDA/OpenCL/Ascend 工具链，P0 的见证执行器也只执行 `emit = c`，三台加速器仍是 `L0`/`L1`，协商继续把它们视为不具备。
- **`benches::Verdict` 改名 `Judgement`**：它与 `audit::Verdict` 同名但语义不同——后者是**达成率口径**（`reached` / `below` / `over-peak` / `cache-resident` / `unknown`，回答“这次测量能不能与档案峰值比较”），前者是**基准回退判定**（`pass` / `regressed` / `improved`，回答“与基线相比是否回退”）。两者会同时进同一份 CI 报告，同名等于让人读错结论；改名后 `as_str()` 的稳定短名（`pass` / `regressed` / `improved`）不变，`audit` 侧一行未动。
- **同一原语可以按 `impl` 拆成多条签名，加速器档案据此与 P0 需求对齐**：`gemv_i4.needs` 要的是标量 f32 点积（`dt=[f32]`、`lanes=1`），而 nvidia / intel 只声明了 `dp4a`（i8×4），ascend 把 `dot` 挂在 Cube 的 `mmad` 上；`load_async` / `load` 的 `dst`、ascend 的 `quant.space` 也各差一层。这些签名与 P0 不同构，补见证救不回来，而旧 schema 只允许"每个原语一条供给"，唯一出路就是把 `dt`/`lanes`/`dst` 少写几个值——`covers` 是集合包含，少写字段等于声称"也支持"。现在 `Profile::providers` 返回同一原语的全部供给，`cap` 先挑能**覆盖**需求的那条、再看证据等级、都不覆盖时取差异最少的那条做诊断；`Profile::validate` 改按 `(prim, impl)` 查重，`Catalogue::validate` 里"声明了两条，协商无法取舍"的旧互斥随之删除。三份加速器档案同步补齐：nvidia / intel 新增 `dot = fma`（f32、lanes 1）并把 `load_async` 的目标扩到 `GM`（`cp.async` / `async_work_group_copy` 落不到全局，目标是 GM 时退化为预取 + 普通加载），ascend 新增 scalar 单元的 `quant` / `dot` / `load` / `load_async` 四个签名（MTE 只到 L1、Cube 摸不到 GM，这些需求本来就该由标量单元满足）。`doctor --device {nvidia,intel,ascend}` 现在每条 need 的 `have` 行都结构同构，唯一剩下的阻挡是 `L0`/`L1` 的证据等级——`verify --devices cpu` 输出逐字不变（key `83a3ecdb31cd1225`）。
- **审计层独立成成员 `llm-audit`（目录 `crates/audit`）**：`doctor` / `verify` / `bench` 的计算与见证执行器（`witness`）不再住在内核里，`cutile` 只保留机制（协商 → 拼接 → emit → 编译执行），命令策略（形状矩阵、重复次数、达成率口径、`L3_THRESHOLD`）与它们的报告类型归 `audit`。数据装载仍在内核（`Catalogue::load` / `resolve`），排版与退出码仍在前端 `src/bin/cutile.rs`；层表把 L3 写成 `cutile` → `audit`（同层先后），依赖边 77 条。顺带修正所有权表：`Catalogue` 的归属从 `op` 改为 `cutile`（它一直是 `cutile::builtin` 的实现，而非声明层的类型）。内核里原先埋在 `audit.rs` 的 ⑤ 步骤（指纹 / 查缓存 / 取锁 / 拼接 / 落盘）提为独立模块 `cutile::build`，审计从此只经 `build::source` 与 `build::dir` 取文本与目录；`doctor` / `verify` 输出逐字不变（`83a3ecdb31cd1225`、53 行、形状矩阵 `32x1,64x3,4096x4096`）。
- **诊断排版收回驱动方，审计报告只剩数据**：`cap::NoPath::render` 删除（§8.2 的失败文本仍由 `Display` 给出，但不再提供"排好版的一段"），`cap::explain` 换成返回 `VariantCheck` 的 `cap::inspect`——只给逐变体的判定与失败明细，`✓ 选中`/`·` 这类标记与缩进归前端。`audit::Doctor` 的 `negotiate` / `settings` / `storage` 与 `Built` 的 `key` / `detail` 四个预渲染字段改为数据（`variants` / `no_path` / `Storage` / `CacheKey`，配置表由驱动方调 `Settings::describe`），`space::describe` 随之删除。顺带消掉 `doctor` 的重复协商（`explain` 原先在内部又跑一遍 `negotiate`）。`doctor` / `verify` / `bench` 输出逐字不变（`FAST=L2 (128 B)`、`path: pipe2 (schedule pipe<2>) mma_share 0.62`、缓存键 `83a3ecdb31cd1225`）；其中 `[存储]` 段随后改为逐层打印档案容量与声明用量（见本版 `Fixed`）。
- **声明解析搬进 `crates/op`，`cutile` 只剩机制**：`OpDecl` / `Variant` / `Axis` / `TileCombo` 与轴展开（原 `cutile/src/lower/expand.rs`）、布局构造子（`layout.rs`）、调度模板（`sch.rs`）、模板词法（`lex` / `Segment` / `Arg` / `SubstituteError`）现在分别是 `op::decl` / `op::layout` / `op::sch` / `op::tpl`。`cutile::layout` 与 `cutile::sch` 改为 `pub use ::op::…` 转发、`cutile::lower::expand` 收敛为 `lower::decl`，模块路径不变而闭集只剩一份；`op` 因此依赖同层的 `prof`（原语闭集）与 `llm-config`（受限 TOML 解析），`cutile` 侧少约 900 行，"改声明语法不动机制层"从口号变成结构。
- **CLI 拆成「语法 / 计算 / 排版」三处**：`cutile::cli` 删除，`CutileCommand` 与用法文本移到 `llm-cli::cutile`（选项按子命令分组，`verify --shape` 这类错配不再被静默忽略）；`doctor` / `verify` / `bench` 的计算变成 `cutile::audit` 的结构化报告（`Doctor` / `VerifyReport` / `Bench`，模块内没有 `println!`），排版与退出码在根包新增的 `src/bin/cutile.rs`（L7 入口，不定义新类型）。命令随之从 `cargo run -p cutile -- …` 变为 `cargo run --bin cutile -- …`，根包多两条依赖边（`cutile`、`llm-config`）；`verify` 输出逐字不变（key `83a3ecdb31cd1225`）。
- **代价不再手写：新增语义层 `einsum`（目录 `crates/einsum`）**。`cutile` 的 roofline 只消费 `(flops, bytes)`，而这两个数此前由每个算子在 `op` 里手写一遍常量（`gemv_i4` 是 `m*n/2 + m*4 + n*(m/32)*4 + n*4`）。现在一条求和表达式加每个操作数的位宽/方向就够了：`Expr`（显式模式解析，拒绝省略号、对角与隐式模式）、`Extents`（轴长度，含 `group` 声明的**派生轴**）、`Operand` / `Access` / `Bits`（4/8/16/32/64 位闭集）、`Axes`（批 / 自由 / 规约 / 分组四类角色）、`Cost`。`gemv_i4` 只剩 `EXPR = "nm,m,ng,n->n"` 与四个操作数；`Model::cost` 变为 `fn(usize, usize) -> Result<Cost, Error>`，形状与表达式不自洽（如列数不是量化分组的整数倍）直接报错，不再在公式里静默截断因子。`Cost` 的所有权移到 `einsum`，`op::model` 只做再导出，`cutile` 的依赖不变。实测 4096×4096 的 `flops`/`bytes` 与迁移前逐位相同，`verify` 的缓存键仍是 `83a3ecdb31cd1225`。
- **算子语义搬回 `crates/op`**：`gemv_i4` 的标量参考、确定性样本、`flops`/`bytes` 代价与内核实参原本散在 `cutile` 的 `reference.rs`、`backend.rs`（`GemvData` + 写死签名的 `harness`）、`plan.rs`（`gemv_i4_traffic`）与 `cli.rs`（`sample_data`，组大小与形状写死），一处算子四个位置。现在统一为 `op::model::{Cost, Sample, Buffer, Model}` + `op::gemv_i4`（`GROUP`/`MODEL`），由 `op::model(name)` 注册；`plan::select` 改为接收 `Cost`，`backend::harness` 按样本缓冲名与类型渲染 C 数组并返回 `Result`，`cli` 的形状校验改用模型的组大小。新增 `the_model_group_matches_the_declaration` 盯住声明与模型的组大小漂移；`verify` 输出逐字不变（key `83a3ecdb31cd1225`）。
- **`cutile plan` 不再声称在 tile 候选之间做选择**：`predict` 只吃 `(flops, bytes, mma_share, stages)`，每个候选拿到的预测完全相同，所以"取最快的"恒等于"取第一个"。现在 `select` 按声明顺序取 `[axes]` 笛卡尔积的第一个组合（轴取值顺序即优先级，与降级链同一口径），`gemv_i4` 的 `GMAX` 相应只声明 `[32]`（组大小由 `needs.quant.group` 固定成 32，更大的取值只多占栈）。`DESIGN §9` 拆成 P0（取声明值 + roofline 估值）与 P3（plan 表 + ±1 档 3 候选微基准）。
- **构建期存储契约接上流水线**：`build_source` 现在对协商出的变体做两项校验——声明需求的 `bytes` 上界不得超过它声明的每个存储层容量（`space::check_capacity`），`[layouts]` 里每个 `swz<B,…>` 的 `2^B` 不得超过档案 `mem.fast_bank`（`space::check_alignment`）；报错带槽位名与 need/have。
- **`substitute::render` 只返回文本**：`Rendered` 结构体与 `emit::fragment_hashes` 删除（缓存指纹本来就用 `Prim::ALL` 重算），签名里的 `variant` 参数一并去掉——渲染阶段确实不需要它。
- **配置调用点显式写身份**：关联函数路径不套用类型参数默认值，所以 `Settings::load()` / `Settings::defaults()` 改为 `Settings::<Cutile>::…`（`cutile` CLI、配置测试与文档示例同步）；新增应用写 `Settings::<自己的 App>::…`，不复制分层逻辑。
- `nn` 公共 API 再导出 `tensor::Shape`：`Layer::output` 本来就暴露它，注解生成的代码与消费者因此不必额外依赖 `tensor`。
- **`crates/*` 不再等于“架构成员”**：新目录必须先写 `description` / `readme` / `publish = false`、写清 `//!` 文档、在 `docs/ARCHITECTURE.md` 的层表里有一行，并至少有一条依赖边，才算是架构成员。
- `nn` 与 `proto` 不再是 `add(2, 2)` 占位：`nn` 的 `LayerSpec` 会把算子名对照 `op::builtin()` 校验，`proto` 落地 10 字节帧头的编解码（魔数 / 版本 / 类型 / 载荷长度）。
- `train` / `serve` 的测试依赖（`tensor` / `exec` / `kv`）以 `[dev-dependencies]` 声明，不影响发布依赖面。

- **片段集合不再按厂商分支**：`FragmentSet::builtin_cpu()` 改为 `FragmentSet::builtin(vendor)` + 内嵌表 `FragmentSet::BUILTIN`，CLI 的 `vendor == "cpu"` 判断消失——未登记的 vendor 一律走 `data.root` 的 `prim/<vendor>/`，接入新设备仍然只改数据。
- **档案层迁出 `cutile`**：`crates/cutile/devices/` → `crates/prof/devices/`；`cutile::builtin` 改为从 `prof` 取档案（内嵌副本与磁盘副本两条路径都只经 `prof::builtin()` / `prof::load()`），测试里的 `include_str!("../devices/*.toml")` 改为 `prof::embedded()`。`data.root` 语义不变：仍指向一个含 `ops/`、`devices/`、`prim/` 的数据根；档案里的 `witness` 仍以 `cutile` 根为基准解析。同时修正 `cutile` 的 `description` 与项目文档中的分层说明（三层数据分属 `op` / `prof` / `cutile`）。
- **声明层迁出 `cutile`**：`crates/cutile/ops/` → `crates/op/ops/`；`cutile::builtin` 改为从 `op` 取声明（新增 `from_repo()` 对拍磁盘与内嵌副本），并校验“文件名即算子名”（与 `[op] name` 不一致直接报错）。`data.root` 语义不变：仍指向一个含 `ops/`、`devices/`、`prim/` 的数据根。
- **TOML 解析器**从 `cutile` 移到 `llm-config`（`cutile::toml` 保留为转发路径）：实现只有一份，`llm` 将来读自己的配置不必反向依赖 `cutile`。
- `Cache::from_env()` 改为 `Cache::from_settings(&Settings)`：缓存目录不再按当前工作目录拼 `target/cutile-cache`，而是取平台用户缓存目录（`~/Library/Caches/cutile/cache`、`$XDG_CACHE_HOME/cutile/cache`、`%LOCALAPPDATA%\cutile\cache`）。
- **默认目录改为平台用户级目录**：`Roots::discover()` 不再“向上找 workspace、找不到就用 CWD”，而是取 `~/Library/Caches`（macOS）、`$XDG_CACHE_HOME`（Linux/BSD）、`%LOCALAPPDATA%`（Windows）；配置目录同理。发布后的二进制在任意目录运行都命中同一份缓存，不会在 CWD 里造 `target/`。
- **发布路径修正**：内置数据是编译期内嵌的，CLI 不再用 `CARGO_MANIFEST_DIR` 校验见证文件（发布机上没有源码树）；`verify --emit-ledger` 写到 `<work.dir>/ledger/` 而不是源码树。
- 环境变量与 `--set` 的相对路径改按**启动目录**解析（符合 shell 习惯）；配置文件里的相对路径仍按该文件所在目录解析。
- `cutile` 的 `verify` / `bench` 工作目录与可选数据根（`data.root`）统一由配置层提供；`doctor` 输出新增 `[配置]` 段逐键标注来源；CLI 支持 `--set key=value`，配置层错误以退出码 2 结束。

- 设计文档同步实现细节：模块表补 `reference.rs` / `toml.rs` / `main.rs`，并明确 `axes`（tile 候选）与 `[[variants]]`（降级链）是两个维度。
- 设计文档重写为 v1.0《cutile DSL —— 设备无关的替换式 kernel DSL》：改为以设备无关为核心的三层声明（算子声明 / 设备档案 / 原语片段）+ 三组闭集（16 原语、4 布局、4 调度）+ 无 IR 的 lowering，并把预算拆成 P0 / P3 两档（当前值见 DESIGN §13.1：2,640 / 9,520）。
- `README.md` 的成员表与目录树同步：`cutile` 职责改为“设备无关 kernel DSL”，并链接 `crates/cutile/README.md`。

- MSRV 提升到 1.100：`Cargo.toml` 的 `[workspace.package] rust-version`、`clippy.toml` 的 `msrv`、CI 的 `msrv` job 与 `README` 统一为 1.100。
- `rust-toolchain.toml` 的工具链改为 `nightly-2026-09-01`（rustc 1.100.0-nightly）：1.100 正式版发布前只能用该 nightly 构建，发布后把 `channel` 改为 `"1.100"` 即可。
- CI 的 fmt / clippy / test / doc 四个 job 改用 `nightly-2026-09-01` 工具链；`msrv` job 同步安装该 nightly 并验证 1.100 系列可构建。
- `repository` 统一声明为 `https://github.com/spapient/llm`：写在 `[workspace.package]` 中，并由 `llm` 与 `cutile` 各自以 `repository.workspace = true` 继承（workspace 级字段不会自动下发，成员不写就等于没有）。

- 仓库升级为 Cargo workspace：根 `Cargo.toml` 声明 `[workspace] members = ["crates/*"]`、`resolver = "3"`，元数据与 lint 策略上提到 `[workspace.package]` / `[workspace.lints]`，两个成员改为 `.workspace = true` 继承。
- CI 的 clippy / test / msrv / doc 命令统一改为 `--workspace`，此前根包存在时这些命令只覆盖 `llm`，漏掉 `crates/` 下的成员。
- `.gitignore` 的 `/target/` 改为 `target/`，避免子目录构建产物进入版本库。

### Removed

- `op::layout` 的 `LayoutNF` / `Order` 与 `LayoutSpec::instantiate`：正规形需要槽位的两级形状，而构建期只有 tile 取值，旧实现里的 `instantiate(tile, 64)` 中的 `64` 是编出来的。布局现在只保留闭集与声明式；等槽位形状与片段布局声明到位（P3）再加回正规形与判定。
- `llm-xtask`（仓库任务 `check-graph`）与 CI 的 `graph` job：注册表检查与依赖方向判定失去自动执行者；P0-3 起由 `tests/architecture.rs` 在 `cargo test` 里恢复自动执行。
- `docs/architecture/layers.toml`：没有消费方的人工注册表（每个成员的 `(layer, rank)` 与 `owns`）已删除，分层与依赖方向只保留在 `docs/ARCHITECTURE.md` 的层表里（P0-3 起由测试执行）。
- `crates/cutile/Cargo.lock` 与 `crates/cutile/target/`：workspace 只保留根锁文件与根构建目录。
- 仓库根 `cutile.toml`：与代码默认值逐字重复的第二事实源，发布包里并不存在。
- `plan.dir` 与 `log.level`：没有任何消费方，已从键闭集移除（`CUTILE_PLAN_DIR` / `CUTILE_LOG` 一并失效）。
- `cutile::host` 模块（`src/host.rs`，152 行 + 4 个测试）：`Tensor::partition` 与网格推导全仓零调用，也没有对外 API 落地。DESIGN §4.4 改为"P0 未实现"，ADR 0003 补修订段，`tensor` / `exec` / `ARCHITECTURE.md` 的引用同步。
- `cutile` 里没有输入的死代码：`layout::check` / `Verdict`（判定要两个正规形，而片段尚不声明布局）、`LayoutNF::{bytes, elements}`、`cap::unmet_prims`（只被测试调用）。`DESIGN §6.2` 与 §12.2 的 `@repack` 预算相应改写为构建期存储契约。

### Fixed

- **设备档案探测的不是它自己声明的那个编译器**：`toolchain::for_profile` 只有 `Emit::Cuda` 才 `probe(meta.tool)`，其余一律 `detect()`（宿主 `cc`/`clang`/`gcc`）。于是 `ascend` 声明 `ccec` 却拿宿主 C 编译器去编——`doctor` 的"实测"一栏报"未知"（看着像没探测），`bench` 真编出来的却是 gcc 的产物并按 910B 报数。现在只有便携 C 走 `detect()`（它的登记值 `cc` 与 `*` 本来就是"按 cc → clang → gcc 探测"），其余按 `[meta].tool` 探测且**不退回宿主编译器**：新增 `toolchain::probe_declared` 按空白切命令与固定前缀参数（`metal` 的 `xcrun metal` 因此可探），身份保留声明原文，报告能与档案逐字对账。`doctor` 与 `for_profile` 共用这一条，`ascend` 现在与 `nvidia` 一样如实说"没有编译器"。副产物：装了 Xcode 的机器上 metal 的"实测"从"未知"变成 `xcrun metal@32023.922`。**验证**：`cutile` 167 条（`toolchain` 新增两条：带前缀参数的声明可探且身份保留原文、设备档案只探自己声明的编译器；`doctor` 的 ascend 用例补上"身份不能是宿主编译器"断言）、`llm-cli` 88 条全绿；`doctor --op gemv_i4 --device ascend` 仍是"本机没有该设备"，只是这一栏不再自相矛盾。
- **`reduce` 片段把归约退化成"取首元素"，且只有 `gemv` 的 `&v, 1, 1` 恰好正确**：`ascend/reduce.cpp` 与 `intel/reduce.cl` 都写成 `($0) = ($1)`，不引用 `$func` / `$op` / `$axis`，对 `softmax`（`GMAX` 最大值）与 `reduce_*`、`rmsnorm` 的多元素归约会**静默返回首元素**且编译期不报错。两处都改成与 `cpu` / `nvidia` 同源的 `$func($1, $2, $3)`（长度双通道与遍历封在 `$func` 里）。`intel` 侧同时补上 `[compile].preamble`：`sub_group_reduce` 早先只是"文本上像归约"，空 preamble 下真的调用 `$func` 无处定义；现在挂了 `_cutile_sum_f32` / `_cutile_max_f32` / `_cutile_exp_*` 全套软件接口（`exp`/`log`/`sqrt`/`sin`/`cos`/`pow` 按 OpenCL C 的 gentype 重载写）。`prim` 新增用例 `a_reduction_fragment_must_fold_through_func` 钉住这条语义——按 `impl` 分组的文本同构判定看不到它（`scalar_loop` / `shfl.bfly` / `sub_group_reduce` 是三个不同的 impl 名）。**验证**：`llm-prim` 20 条、`llm-prof` 57 条全绿；`intel` 的 preamble 另以 `clang -x cl -cl-std=CL2.0 -fsyntax-only` 单独过了一遍语法门。
- **`ascend` 的 `quant` 片段硬编码 4 位，i2 结构性缺席**：片段里写死 `0x0F` 掩码与 `-8` 零点，忽略模板注入的 `$cpb` / `$width` / `$zero`，于是 `gemv_i2` / `gemm_i2` 在 `ascend` 上不是"等级不足"而是"dt 缺 i2"。现在掩码按位宽派生（`(1 << $width) - 1`），与 `cpu/quant.c`、`nvidia/quant.cu` 同源，档案的 `scalar_dequant` 同步补 `i2`。**验证**：`prof` 档案装载（`llm-prof` 57 条）与 `cutile` 协商用例全绿；`prof` 侧新增 `a_dequant_signature_covers_both_packed_widths` 钉住这条（档案改回 i4-only 即失败）；`doctor --op gemv_i2 --device ascend` 的九条需求现在只报 `L0 < L2`，不再出现 `dt 缺 i2`。
- **`ascend` 档案缺 `[compile]`**：`doctor` 打印"参数 无"，且 `@reduce` 的 `$func` 在设备源码里无处定义——这是条定时炸弹：归约片段一改对，编译期立刻炸。现在补 `flags = ["-O2"]` 与 preamble（`_cutile_sum_f32` / `_cutile_max_f32` / `_cutile_exp_sum_f32` / `_cutile_exp_norm_f32` 及软件数学接口，与 `cpu` 同源；`exp` 走位精确实现，其余 libm）。核架构参数（`dav-c220`）当时刻意没写：本机没有 `ccec`，猜一个拼写进档案只会让"参数"这一格从空变成错的；它随语法门与执行后端一并补上（见本版接下来两条），并在档案注释里标明"未经真机核对"。`[mem]` / `[peak]` / `[simt]` 三段同样留空并在文件里写明理由（前两者是实测值，没有真机不写数字；昇腾的 lane 交换由向量单元承担，不是 SIMT，写 `group = 1` 会把它错报成单 lane 设备）。**验证**：`doctor --op gemv_i4 --device ascend` 的参数栏从"无"变为 `-O2`（设备语言与核芯名随后补齐），其余结论不变（仍全 L0、仍无可行变体——这三条修复不动证据等级，只让片段与声明对得上）。

- **`ascendc` 的语言门永远是关的，理由还写成了"没有开源前端"**：`gate::GATES` 里没有 `Ascendc` 这一行，`language_gate` 对它恒返回 `Absent{本机没有 ascendc 的语法前端（接了 cuda / opencl）}`——把"没登记"说成了"没有这种东西"，而 ccec 本来就带语法前端（`-fsyntax-only`）。现在表里多一行 `Emit::Ascendc`：`ccec -xcce --cce-aicore-arch=dav-c220 -fsyntax-only`（`-xcce` 让 ccec 按设备语言解析这份翻译单元，核芯名跟 `id = "910b"` 走——换核芯要改这一行）。表行的解析器同时改成 `front(spec)`：`clang` 行仍指本机 clang 前端（`clang` → 名字里带 clang 的 `cc`），其余按 `[meta].tool` 的写法原样探测、**不退回宿主编译器**——拿 `cc` 去解析 AscendC 只会把"没装 CANN"报成一条 C 语法错误，即把缺席伪装成拒绝（`Rejected` 比 `Absent` 严厉得多）。**验证**：`cutile` 新增两条（表行的三条判据逐项钉住；本机没有 `ccec` 时门是 `Absent` 且点名 `ccec`，不是 `Rejected`），`cutile` 170 条全绿；`doctor --op gemv_i4 --device ascend` 的语言门一行从"本机没有 `ascendc` 的语法前端（接了 cuda / opencl）"变为"本机没有 `ascendc` 的语法前端：`ccec` 不在本机"。
- **`ascendc` 没有执行后端：搬来一台 910B 也跑不了**：`device::spec(Emit::Ascendc)` 恒 `None`，`emit = "ascendc"` 只决定 `.cpp` 后缀——产出里没有 `__aicore__`、没有 `GM_ADDR` / `__gm__`、没有 AscendCL，所以即使 ccec 与 NPU 都在场，见证也只会落回"本机没有该设备"。现在 `engine::backends` 里有 `AscendBackend`（与 `CudaBackend` 同形，注册进同一张进程级表）：可用性要求 `ccec` 与 `npu-smi info -l` **都在**（`detect_cce` / `ascend_device` 各自探测、合取才叫可用——缺的是编译器还是设备，报告里读得出来：装 CANN 还是换机器）；生成的是**单源** `.cpp`——`extern "C" __global__ __aicore__ void cutile_witness(GM_ADDR, GM_ADDR, int)` 核函数 + `AscendC::GetBlockIdx()`（一条见证一个 AI Core，与 `blockIdx.x` 同位）+ 见证函数前置 `__aicore__` + AscendCL host 驱动（`aclInit` / `aclrtMalloc` / `<<<>>>` / `aclrtMemcpy`），回读口径与 CUDA 版逐字一致（位模式 + `witness N`），否则同一个 `driver::readback` 在两台设备上会读出两种东西。编译参数全取自档案 `[compile].flags`，只有一处分拣：`-l` 类挪到源文件之后（ccec 把它们转给链接器，排在源前就是 `undefined reference to aclInit`）；`ascend.toml` 的 `flags` 因此补成 `["-xcce", "--cce-aicore-arch=dav-c220", "-O2", "-lascendcl"]`。`lower` / `harness` 那几条生产态入口明确报 `Unsupported`：P0 片段只到标量 C 语义，编成 AscendC 只会得到"能跑但和 CPU 一样慢"的核函数，`DataCopy` / `Mmad` 的落点属 P3。等级仍是 `L0`——这条路径还没在任何 910B 上跑过。**验证**：`llm-engine` 新增三条（单源文本的形状：核函数前缀 / `GM_ADDR` / `__gm__` / `GetBlockIdx` / AscendCL / `<<<>>>` / preamble 进翻译单元 / 每条见证都真的带 `__aicore__`；可用性 = 编译器 ∧ 设备且注册表与后端判定一致；缺 `ccec` 时错误点名 `ccec` 而不是一条 C++ 语法错误），`llm-engine` 157 条、`llm-prim` 20 条、`llm-prof` 57 条全绿。

- **`ascend` 拿不到 `L2` 的最后一段：`verify` 的语义门在昇腾上是空的**。上一条修完执行后端，见证能在 NPU 上跑了，但 `verify` 里 `Lane::ready` 只认 `c` / `cuda` / `metal` 三种 `emit`——`ascendc` 落到 `unsupported_emit` 那句"内核执行路径尚未接通"，于是后端注册了也走不到，形状矩阵永远得不出"与参考对拍通过"，`L2` 这条证据在昇腾上结构性地不可能。现在 `Lane::Ascend` 把它接进同一张表（与 `Cuda` / `Metal` 并列，可用性只由"本机有没有这台设备"决定），新增的 `engine::ascend_codegen` 把 `.k` 模板拼成**单源** AscendC 程序：内核与档案 preamble 里的 `_cutile_sum_f32` 一族一律加 `__aicore__`（`@reduce` 的 `$func` 落在 preamble 里，只给内核加属性等于让 AI Core 调宿主函数，ccec 会拒），`extern "C" __global__ __aicore__ void cutile_launch(GM_ADDR …)` 形参与 `<<<1, nullptr, stream>>>` 单块发射同 CUDA 的 `<<<1,1>>>` 同位，缓冲实参补回 `__gm__` 元素类型，host 走 `aclrtMalloc` / `aclrtMemcpy` / `aclrtSynchronizeStream`，回读与 CUDA 逐字一致（位模式逐行）——同一个 `driver::readback` 因此不会在两台设备上读出两种东西。AscendCL 的调用守卫 `ACL_GUARD` 收成一处，见证程序与语义门共用（两处各写一套时，新加一个 API 调用漏掉的那处会以"回读全是 0"的形式失败）。仍是 `L0`、仍缺真机：`lower` / `benchmark_harness` 与 `Source::Pair` 的生产态入口继续如实 `Unsupported`（P0 片段只到标量 C，`DataCopy` / `Mmad` 属 P3），没有并行内核时报告里的通道写"语义门"而不是"并行"。**验证**：`llm-engine` 新增一条（对 `gemv_i4` 拼出的程序逐项断言：核函数前缀与 `GM_ADDR` 形参、内核与 preamble 的 `__aicore__`、`__gm__ const` 补类型、`<<<1, nullptr, stream>>>`、往返拷贝、位模式回读），`cutile` 新增两条（`ascendc` 走设备通道；整条 `verify_one` 对 ascend 报的是 `NO_DEVICE` 而不是先撞上 `L0 < L2` 的协商结论——通道判据先于协商，缺机器与声明不足是两件要修不同东西的事），`cutile` 171 条与 `llm-engine` 148 条（另 `tests/schedule.rs` 14 条、文档用例 3 条）全绿。

## [0.1.0] - 2026-11-11

### Added

- 工程创建。
