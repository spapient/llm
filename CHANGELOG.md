# Changelog

本文件的格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- **`checkpoint` 成员（目录 `crates/checkpoint`）：检查点文件**。`train::Checkpoint` 只记"恢复所需的事实"，但"这些事实怎么落盘、版本怎么认、条目偏移怎么校验"此前没有主人。现在 `Format`（魔数 `LLMC` + 版本，`check` 只认当前版本）与 `Snapshot`（步数 / 并行度 / 层数 / 批行数 + `weights::Header`）负责这一段：`encode` / `decode` 是纯字节的往返，条目表在解码时仍交给 `weights::Header::new` 重新校验（重名、区间重叠、起点早于数据段），截断、版本不符与尾部多余字节都以 `invalid` 带上偏移报出。`train::Checkpoint::snapshot(&weights::Header)` 把内存事实组合成快照，删掉 `checkpoint` 会让 `train` 编译失败；层表把它放在 L6 最左（`checkpoint` → `train`），依赖 `weights` / `tensor` / `diagnostic`，依赖边 79 → 83。新增 6 条单元测试（往返、魔数、版本、截断、尾部多余、零层数）与 `train` 侧一条快照往返用例。
- **autograd 从"只给一个逆序"升级为反向图调度**：`Tape` 现在按 `Requirement`（`None` / `Backward` / `Grad`）维护需求闭包——`record` 从输入推出需求，`record_with(.., None)` 是显式 detach，`retain` 只允许把**还没被引用**的叶子升级为梯度终点（已被引用即 `invalid`，因为父节点的需求在记录时就定了，事后改会让剪掉的输入悄悄漏梯度）；每个节点同时记深度（输入最大深度加一）。`Tape::backward(loss)` 返回 `Schedule`：只排 loss 可达的节点，顺序为"深度降序、同深度编号降序"，既是合法的反向拓扑序（输出总比输入深），也是并行提示（同深度互不依赖）；`endpoints` 只列可达的 `Grad` 变量，于是没参与 loss 的参数不会进调度、也不会被分配梯度缓冲。`Schedule::account()` 给出 `Account`：按可达步骤的输入位置数出扇出（同一变量在一次调用里出现两次就是两份贡献），`settle` 在梯度未收满、步骤已结算或步骤不属于本调度时返回 `invalid`，收满时返回因此可执行的变量——"谁的梯度已齐"不再靠调用方自觉。实现是稠密 `u32` 索引的平行数组（需求 / 深度 / 生产者）加显式栈遍历：无哈希、无递归，两万层前向照常排。数值仍不在本层：梯度与前向激活归运行时缓冲，反向公式归 `op` 的反向表。所有权表补 `Node` / `Requirement` 与 `Schedule` / `Account` 两行。`Tape::clear` 保留容量地清空，训练循环每步复用一张图时稳定态不再分配。新增 22 条单元测试（含随机图逐条不变量对拍、两万层深链不栈溢出、乱序与重复结算被拒）与一条标量引擎跑完整反向再对解析梯度的用例（`d/dx (x+y+xy) = y+1`、`d/dy = x+1`）。
- **签名对齐的第四段：片段声明自己实现哪条签名**：同一原语在档案里可以有多条签名（`dot` 的 `dp4a` 与标量 `fma`），片段层每个原语却只有一段文本；此前"这段文本写的是哪一条"只写在注释里，`needs`↔`provides` 与 `provides`↔`witness` 都已可判定，唯独 `provides`↔片段靠人读。现在每段片段的**第一行**是机器可读的 `/* cutile-sign: <prim>/<impl> */`（四台设备共 36 段片段补齐），`prim::Fragment::parse` 在装载期解析并拒绝"缺声明 / 原语名与文件名不一致 / `impl` 为空"；`cap::fragment_gaps` 与 `negotiate` 用同一张需求表、同一个 `select`，逐变体逐需求比对"协商会挑中的实现"与"片段声明实现的实现"，`Catalogue::validate` 把差异变成装载期错误（写错签名是 `invalid`，缺文本是 `missing`）。于是"能协商"与"能生成"不再可以同时为真却指向不同的算法。新增测试 `every_fragment_declares_the_implementation_negotiation_picks`（四台设备逐条对齐）与 `a_fragment_declaring_another_implementation_fails_the_load`（把 `nvidia/dot.cu` 改成声明 `dp4a` 即装载失败），`prim` 侧另有四条解析用例。等级不变：这条判据只说"文本与签名同构"，不构成设备证据。
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
- **架构契约**：`docs/ARCHITECTURE.md`（七层表 + 依赖规则 + 所有权表 + 新增成员三步）、`docs/adr/0001..0004`（分层序、单一主人、kernel 与运行时类型、占位策略）；每成员至少一条边、全部 `publish = false`、仓库只有一个 `target/`；P0-3 起由 `tests/architecture.rs` 在 `cargo test` 里执行。
- **16 个占位成员改成接口形状**（ADR 0004）：`diagnostic`（`Error`/`Code`/`Result`）、`testkit`（`approx_eq`/`device_enabled`/`scratch_dir`）、`mem`（`Arena`/`Region`）、`tensor`（`Tensor`/`DType`/`Shape`/`Strides`）、`storage`（`Layout`/`Order`/`remap`）、`exec`（`Stream`/`Graph`/`Launch`/`Executor`）、`tokenizer`（`Tokenizer`/`Token`/`Bytes`）、`weights`（`Header`/`Entry`/`Reader`）、`kv`（`Page`/`Table`/`Cache`）、`autograd`（`Tape`/`Var`）、`comm`（`Group`/`Reduce`/`Collective`）、`proto`（`Envelope`/`Kind`/`Version`）、`data`（`Batch`/`Split`/`Dataset`）、`nn`（`Layer`/`LayerSpec`/`Linear`，层定义必须引用 `op` 已声明的算子）、`shard`（`Topology`/`Partition`，rank 布局与并行组）、`train`（`Schedule`/`Step`/`Checkpoint` + 恢复拓扑校验）、`engine`（`Sequence`/`Scheduler`，准入即占页）、`serve`（`Route`/`Endpoint`/`Response`）。
- `tool` 成员（目录 `crates/tool`）：`Report` / `join_names`，把人可读输出从组装点里抽出来，避免三处各写一套排版。
- `benches` 成员（目录 `crates/benches`）：`Case` / `Baseline` / `Judgement` / `Direction`，把“方向 / 相对预算 / 回退与改进”写进类型，供后续 bench job 直接判定。
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

### Changed

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
- `README.md` 的成员表与目录树同步：`cutile` 职责改为“设备无关 kernel DSL”，并链接 `crates/cutile/DESIGN.md`。

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

## [0.1.0] - 2026-11-11

### Added

- 工程创建。
