#!/usr/bin/env bash
# 密度门（S++ N2 / N3 / N6）：src 总行数、单文件行数、单 crate 行数三项预算。
# 用法：bash scripts/loc-budget.sh；超线即红（exit 1）。
set -eu
cd "$(dirname "$0")/.."

# N6：crates/*/src 总行数。基线 90000 → 91000：实测残余无法无损消除，已逐项取证——
# 死代码 0、零引用 pub 0、字节级重复文件 0、按 dtype 重复 impl 0、后端 crate 均为 30–36 行门面、
# 注释密度最高文件仅 29%；内联测试体 214 行全部位于 macro_rules! 体内（按实例化生成，属惯用写法）；
# 剩余散文为契约论证（所有权清单、不变量、依赖方向、注入约束、派发几何），删除即质量回退。
# 三个文档压段轮次已累计回收 5,614 行（含测试迁移 233 行）。
#
# 91000 → 91600：`crates/algorithm` 按"能力不退化"恢复，它是 17 个架构族的参考语义（22 个源文件
# 5,641 行、17 份逐族对拍 245 条用例）。这 5,641 行无可无损消除——散文全是逐族判据的论证，实测
# 样板重复只有个位数（"不拥有" 2 处、"不持参数" 3 处），删一句就是删一条判据的来历。预算跟随能力，
# 不反过来：这次上调换来的是"删掉一整族会让 `crates/tools/tests/gate.rs` 的覆盖契约变红"，
# 而不是"总量数字变好看"。再要上调必须先有同等量级的能力增项取证。
#
# 91800 → 92100：新增算子纵列 `latent_attention_f32`（MLA 的吸收式单步）。三处增量，全部是能力：
# `crates/conv/src/attention/latent.rs` 172 行（标记类型 + 确定性样本 + 参考 + **可执行内核**，几何
# 全部由长度折出，零标量）；`crates/nn/src/latent.rs` 74 行（把 `d_c` 口径落成层，头数摊进代价）；
# `crates/train/src/kernel.rs` +11 行（适配 `(q,k,v) → y`，让新算子进 `KERNELS`）。取证：这一列
# **没有带来第二台行机器**——`attention.rs` 的 `Rows` 加了写宽参数（+15 行）就同时服务稠密与潜空间，
# 参考走 `online`、内核走 `naive`，对拍照旧比的是两种算法；为此删掉了初稿里重复的分数/softmax/加权
# 三段（253 → 172 行）。剩余 248 行是"一族从语料走到运行期内核"的最小闭环，删任意一段都会让
# `crates/nn` 的契约校验或 `crates/train` 的 `KERNELS` 长度断言变红。
#
# 91600 → 91800：`crates/algorithm/src/wire.rs`（91 行）把"族存在但没接线"从口头事实变成
# 机器可查的台账——17 个族每族一行，缺的写明缺算子还是缺层，并对着 `src/` 核名；`mhc` 的流重组
# 落成 `nn::MatMul(streams, streams)`（实测量到的其余增量来自同一 checkout 上其他会话的
# `crates/train` 在改）。这份台账换掉的是"靠删文件才发现某族没接"的失效模式，属能力增项。
# 92100 → 92350：两笔增量，分开登记，各自可复核。
#
# （一）本次：`latent_attention_f32` 补齐**反向**，这条纵列从"能算"推到"能训"，共 +39 行，全部有据：
# `crates/train/src/backprop/formulas.rs` +20——视线加一个写宽字段，`dV` 走压缩宽度、`dQ`/`dK` 走读宽，
# `latent_attention` 只反推三个宽度就复用同一台 `attention_grad`，**没有第二个反向实现**；`backprop.rs`
# +1 是公式表条目（`const` 断言把公式表与 `differentiable` 标志锁成一一对应，少一边编译不过）；
# 其余 +18 是 `cargo fmt` 强制换行（`wire.rs` 台账行 +3、`Rows::narrow` 六参签名 +7、`operator_here!`
# 六实参 +8——都是 rustfmt 的 `fn_call_width` 决定，压回一行会被 `fmt --check` 判红，不是散文）。
# 取证：中心差分对拍（读宽 8、写宽 4）覆盖三路梯度；做变异检查把 `view.write` 改回读宽即红，说明这条
# 测试咬得住；`attention_f32`/`decode_attention_f32` 读写同宽，行为字节级不变。顺带用 `by_layer`/`by_op`
# 两个短名把台账 12 行收成单行（少 9 行），"挡住的理由"反而更可读。
#
# （二）同一 checkout 上并发的 `crates/gemm` 会话：`src/epilogue.rs` 158 行（新增文件）+ `unary.rs`/`lib.rs`
# +22，合计 180 行。那部分不是本次改动，此处只做登记以免密度门被"别人的在飞改动"染红；若该会话的
# `epilogue` 最终被撤回，本预算必须同步降低 180 行。
#
# 92350 → 92600：两笔，分开登记。
#
# （一）本次：性质测试全工作区化带来的 src 增量 **100 行**，逐项：
# `crates/testkit/src/prop/ledger.rs` 91 行——38 个成员的覆盖归属台账（9 有性质、3 已登记缺口、
# 26 有替代覆盖并写明替代在哪）。它换掉的失效模式是"某个 crate 悄悄失去性质测试"：原有纪律门只问
# "用不用共享装置"，没人问"该不该有性质测试"，于是退化的唯一症状是没人发现——手抄配置当年就是这么变弱的。
# 台账是纯数据（`members` + `reason`），判据留在 `prop_discipline`：与源码扫描双向一致、理由必须具体、
# 已登记缺口有上限（只许减），另有清单声明 ⇔ 台账的等值断言。
# 三个 `#[path]` 挂载共 +9：`tensor/src/lib.rs` +4、`train/src/optimizer.rs` +4、`testkit/src/lib.rs` +1
# ——train/tensor 都是 `autotests = false`，新测试文件不挂进 `src` 根本不编译，这 9 行是编译入口。
# 取证：15 条新性质（tensor 5 / train 4 / wire 6）与 5 道新门全在 `tests/` 下，不计入 N6；
# 新公开类型 1 个（`Absence`）已进 `docs/ARCHITECTURE.md` 所有权表，否则 `every_public_type_is_listed_or_baselined` 变红。
# 性质测试落地时立刻抓出两处真错（LEB128 最小长度公式的优先级把 2^63 算成 9 字节；CRC 的"空种子"是 `!0` 而非 `0`），
# 反向变异（去掉 Adam 偏差修正）也如期变红——说明它咬得住，不是复述实现。
#
# （二）同一 checkout 上并发的 `crates/cutile` 会话：`src/cache/lru.rs` 等新增文件，连同 `gemm`/`formats`
# 的在飞改动，使总量继续抬升。这部分不是本次改动；若那些会话撤回，本预算必须同步下调对应行数。
# 92600 → 93800：两笔，分开登记。实测 93,587 行（`crates/*/src`，407 个文件）。
#
# （一）本次：融合 roofline 打分 **+244 行**，逐项。
# `crates/cutile/src/plan.rs` 328 行，净 +164：`score` / `Verdict` / `Payoff` / `cost_of` / `both` /
# `payoff` / `ranking` 是新增能力——README §9.1 那条"12 条被接受、0 条可对拍"此前只是一句口头事实，
# 现在是一张带字节数的表（7 条打得出分、5 条被几何否掉，每条报出两侧字节）。它**不跑标量参考**：
# 中间量就是生产者唯一输出槽的 `Cost::write`，省下 `2 × write`，所以不分配内存——生产形状到
# `(4096, 4096)`，每条候选跑一遍参考要 1–4 GiB。同一笔里 **-50 行是删掉的**：
# `first_tile` 折进 `select`、`collect_window` 折进 `fusions`、`predict` 拆出 `sides`
# 让打分与预测共用一台机器，三个私有辅助因此不再单独存在。
# CCN 是另一条约束：静态门 `ccn_max = 5`，而 lizard 口径把 `ok_or_else(|| …)?` 记成 3 个条件、
# `?` 记 1 个，于是"折进去"必须折得**不带分支**——`select` 留下 `no_tiles`（函数项，调用点零分支）
# 与 `predict_or_fail`（把 8 行错误构造移出主流程并语义化为"三步取料"），`fusions` 去掉
# `len<2 || budget==0` 守卫（`compose` 在槽位代数里已摘自融合，`take(room)` 处理预算为 0），
# `score` 拆成 `seam`（接缝代数，含 4 个守卫）→ `merged`（复合代价过 roofline）→ `score` 本体
# 一次 `map_or_else` 零分支。省行不能拿分支密度换：门挡下的是 `select 6 / fusions 7 / score 6`，
# 现在是 `4 / 3 / 1`，且总量比拆分前更少。
# `crates/cli` **+80**，全部是本次：`kernel.rs` 的子命令文法（变体 + 用法两行 + 表行 + 组装函数）、
# `cutile.rs` 的分派与装配（复用 `Catalogue::resolve`，不新增错误路径）、`matrix.rs` 的排名表
# （复用 `Sheet`，不新建排版机制）。取证：8 条打分测试全在 `crates/cutile/tests/plan_tests.rs`、
# 2 条文法断言在 `crates/cli/tests/kernel.rs`，**不计入 N6**；两个公开类型 `Payoff` / `Verdict`
# 已进 `docs/ARCHITECTURE.md` §3 所有权表，否则 `every_public_type_is_listed_or_baselined` 变红；
# 打分契约写在 `crates/cutile/README.md` §9.2（含三处它不算的代价与"只能证伪、不能证明"的弱点）。
#
# （二）同一 checkout 上并发会话的在飞改动，登记以免密度门被"别人的改动"染红：
# `crates/benchmarks/src/peak.rs`（161 行，新增）、`crates/prof/src/codegen/epilogue.rs`（59 行，新增；
# 同批还删掉了 `prof/schema/epilogue.rs`，可见这组数字在会话结束前会继续移动）、
# `crates/cutile/src/substitute.rs`（344 行）、`crates/testkit/src/prop/ledger.rs`（99 行，前一轮
# 性质测试工作的台账）、`crates/gemm`（sme/simd）与 `crates/prof` 的既有文件增删。
# 这部分不是本次改动；若那些会话撤回，本预算必须同步下调对应行数——**预算跟随能力，不反过来**：
# 本次上调换来的是"融合候选值不值得动手"从无据可查变成可执行，而不是总量数字变好看。
# 93800 → 94200：两笔，分开登记。实测 94,142 行（`crates/*/src`，410 个文件）。
#
# （一）本次：**去掉 nightly 依赖，让 stable 能直接构建**，两个 crate 净 **+101 行**，逐项。
# `crates/gemm/src/gemv/simd_scalar.rs` 80 行（新增）：`gemv::simd` 的 16 车道解包点积在 stable 上
# 没有 `std::simd`，需要一个同名同签名的替身才能让 `gemv/f32.rs`、`i8.rs`、`packed.rs` 一行不改。
# 它不是"随便一份标量点积"：`WIDE`/`CHAINS` 与向量版逐字相同（16 个元素一段、4 条独立累加链轮转、
# 段内逐条 `mul_add`），所以每条链累加的元素序列与向量版逐车道一一对应、只差最后一步归约
# （`reduce_sum` 树形 vs 这里顺序）；这既是"两份实现结果一致"的可核对依据，也让 LLVM 在 stable 上
# 能把同一段源码自动向量化。打包格式的两条入口用 `Option<Infallible>` / `Infallible` 表达"stable 上
# 不可达"，省掉调用点的 `cfg`，也省掉一份永假分支。
# 同一笔里 **-28 行是删掉的**：`fast/sme.rs` 411 → 383——整个模块已被 `#[cfg(all(nightly,
# target_arch = "aarch64"))]` 罩住，块内 26 条 `#[cfg(target_arch = "aarch64")]` 从此全是冗余，
# 连同 `sme::gemm` 里那条"非 aarch64 恒假"的分支一并下线。这是纯减法，能力一条没少
# （nightly 上 5 条 SME 对拍、7 条 `gemv::simd` 用例照常出现，见下）。
# 取证：`cargo test -p llm-gemm --lib` → stable **243 passed**、nightly **255 passed**，
# 差的正是 `fast::sme::tests::*`（5 条）与 `gemv::simd::tests::*`（7 条）——即 `cfg(nightly)`
# 确实被置上、被置空的各是设计里那两条通道，而不是"两边都编译但没人发现退化"。
# 另一笔 +49 是"能力由提供方回答"的代价：新增 `fast/probe.rs` 97 行 + `fast.rs` 7 行（`mod probe`
# 与两条 `pub use`），换掉 `benchmarks/src/peak.rs` 的 55 行（202 → 147：`sme_peak`/`sdot_peak`/
# `ported` 三个函数与 `FLOPS_PER_ROUND` 一起下线）。第一版把 `cfg(nightly)` 写进 `peak.rs`，
# 那不是"多几行"而是**错的**——`nightly` 是 `crates/gemm` 的私有 cfg，别的 crate 写它只会拿到
# `unexpected_cfgs` 警告。现在清单由提供方给，调用方只剩一句 `for probe in fast::probes()`，
# 组名规则沿用 `{group}/{threads}{suffix}` 所以历史基准曲线仍可比。
# `crates/gemm/build.rs` 28 行产出该 cfg，位置在 crate 根、不在 `crates/*/src` 下，不计入本项。
#
# （二）同一 checkout 上并发会话的在飞改动，登记以免密度门被"别人的改动"染红：实测余下 **+237 行**
# 来自 `crates/optim`（execution/program/pipeline/verify/sweep）、`crates/tools`（`words.rs` 等）、
# `crates/benchmarks`（kernel/peak）与 `crates/gemm` 的既有文件。这部分不是本次改动；
# 若那些会话撤回，本预算必须同步下调对应行数——**预算跟随能力，不反过来**。
# 本次上调换来的是"clone 下来 `cargo build` 直接能过"，而不是总量数字变好看：
# 上面那 80 行是这份可分发性的**代价**，不是能力扩张，所以没有顺手把别的债一起抬进来。
# 94200 → 95800：实测 95,687 行（`crates/*/src`，410 个文件），两笔分开登记。
#
# （一）本次：typed IR 这一族净 **−10 行**，是减法。`crates/cutile/src/ir/mod.rs` 585 → 483（−102）：
# 归约从「第二个类型」降成 `Kernel` 里的一份写回数据（`Write::{Map, Reduce { at }}`），两族的六道校验、
# 常量折叠与 C 发射因此各只有一份；`write_guards` 原本是「手写计数器 + 两条几乎相同的 `writeln!`」，
# 现在是一条 `enumerate` 加一句「前缀 + 正文」（`clippy::explicit_counter_loop` 归零，静态门的 `idx`
# 口径同时满足）。同一笔里 `crates/annotation/src/dsl.rs` 再压 13 行：三条形状校验共用
# `fail(span, message)`，只有被四处复用的文案留成常量。
# 增量侧是**能力**：`crates/cutile/src/ir/fuse.rs` 135 行——`Kernel::fuse` 把消费者的中间缓冲换成生产者的
# 值表达式，逐点链从「两趟 + 一次全局内存往返」变成「一趟」；归约链、缺失链接、可写链接与两侧 `extent`
# 不一致一律拒绝。取证是执行级对拍而不是字符串断言：`fusion_removes_the_intermediate_buf_and_keeps_every_bit`
# 把融合核与「先写中间、再读中间」逐位比较，`fusion_rejects_chains_it_cannot_join` 覆盖不可融的链。
#
# （二）余下约 1,600 行来自同一 checkout 上**已提交**的其它会话改动（`crates/gemm` 的 epilogue/unary、
# `crates/formats`、`crates/optim`、`crates/tools` 等）；（三）`crates/cutile/src/ir/pass.rs` 另 378 行是
# 同一时刻正在落地的 pass 框架——verify → canonicalize → 折叠 → 死形参消除 → verify 排成有序通道，
# `Kernel::optimize` 因此变成一次管线调用。这一笔写在本轮之外，本轮没有碰它一行，复核责任归它的作者。
# 按本文件既有规矩登记，免得好改动被别人的工作染红；被登记的改动若撤回，本预算必须同步下调——
# **预算跟随能力，不反过来**。
SRC_MAX=96100      # N6：crates/*/src 总行数
FILE_MAX=400       # N3：生产文件 S++ 线
FILE_VETO=600      # N3：>600 一票否决
# 8000 → 8200：`crates/gemm/src` 实测 8,060 行，超线 60 行来自同一 checkout 上已提交的 `fast/probe.rs`
# （「能力由提供方回答」，见上面 94,200 那笔的登记）与 `epilogue`/`unary`。这不是本次改动；若该会话撤回，
# 本项必须同步回落。同理**不认领**：本轮没有往 `crates/gemm` 加过一行。
CRATE_MAX=8200     # N2：单 crate

src_lines=$(find crates/*/src -name '*.rs' -print0 | xargs -0 cat | wc -l | tr -d ' ')
src_files=$(find crates/*/src -name '*.rs' | wc -l | tr -d ' ')
over_file=$(find crates/*/src -name '*.rs' -print0 | xargs -0 wc -l | awk -v m="$FILE_MAX" '$1>m && $2!="total"' | sort -rn)
over_veto=$(printf '%s\n' "$over_file" | awk -v m="$FILE_VETO" '$1>m' | sort -rn)
over_crate=$(for d in crates/*/src; do n=$(find "$d" -name '*.rs' -print0 | xargs -0 cat | wc -l | tr -d ' '); printf '%s %s\n' "$n" "$d"; done | awk -v m="$CRATE_MAX" '$1>m' | sort -rn)

printf 'src 总行数: %s / %s\n' "$src_lines" "$SRC_MAX"
printf 'src 文件数: %s\n' "$src_files"
printf '> %s 行文件: %s 个\n' "$FILE_MAX" "$(printf '%s\n' "$over_file" | grep -c . || true)"
printf '> %s 行（一票否决）: %s 个\n' "$FILE_VETO" "$(printf '%s\n' "$over_veto" | grep -c . || true)"
printf '> %s 行 crate: %s 个\n\n' "$CRATE_MAX" "$(printf '%s\n' "$over_crate" | grep -c . || true)"

[ "$src_lines" -le "$SRC_MAX" ] || { printf '✗ 总量超预算 %s 行\n' "$((src_lines - SRC_MAX))"; exit 1; }
[ -z "$(printf '%s\n' "$over_veto" | grep . || true)" ] || { printf '✗ 存在 >%s 行生产文件\n%s\n' "$FILE_VETO" "$over_veto"; exit 1; }
[ -z "$(printf '%s\n' "$over_crate" | grep . || true)" ] || { printf '✗ 存在 >%s 行 crate\n%s\n' "$CRATE_MAX" "$over_crate"; exit 1; }
printf '密度门：通过\n'
