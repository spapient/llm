# 测试 / 质量门禁评分基线（tools crate）

> 快照日期 2026-11-24，基于当前 workspace revision 逐件取证。本文把「S++ / 10 分」拆成可测子项，
> 逐项回链源码与可复现命令；只给当前证据支持的档位，未测基线的能力保持**待校准**，不签发满分。
> 标尺沿用 `docs/COMPARISON-llm-llmu-burn.md` §16.1：9–10 成熟完备、7–8 主体完整、5–6 端口骨架、
> 3–4 最小可用、1–2 仅声明、0 缺失。

## 1. 结论先行

- `crates/tools` 是 L1 门禁基础设施：只依赖 `diagnostic`（+dev `testkit`），运行期无 TOML 解析器，
  配置由 `build.rs` 编译期生成 OnceLock 表。
- 静态规则族（词法 / 复杂度 / 禁止形态 / 命名 / 测试剔除 / 行数 / 文档链接）与性能判定族
  （统计 / roofline / 基线回退 / 协议卡）均在位并有单测 oracle；配置闭集、棘轮结清、CI 门覆盖是强项。
- **当前不签发 10 分**。距满分的可枚举缺口：无 severity 级别列、不解析 `#[...]` 属性语义、
  分析器自身无覆盖率下限、性能耗时与误报率未建量化基线。这些项在补齐测量之前档位记待校准。
- 源码内部同构重复已收敛（#49 错误工厂、#50 遍历骨架、#51 链接 oracle）；继续抽泛型层在当前证据下
  无收益，强行抽象属 `PROMPT-S++-AUDIT.md` §9 反模式 #2（平铺冒充高内聚）。

## 2. 基线证据（可复现）

| 项 | 实测值 | 取证命令 / 回链 |
| --- | --- | --- |
| 源文件 | 15（含 `lib.rs`） | `find crates/tools/src -name '*.rs' \| wc -l` |
| 生产行 | 2715 | `wc -l crates/tools/src/*.rs` |
| 最大文件 | `lex.rs` 408、`card.rs` 400，均 <600 | 同上；N3 线 `>600` 否决 |
| 公开项 | 125（含 `pub use` 重导出） | `grep -rhE "pub (fn\|struct\|enum\|trait\|const\|use)" crates/tools/src/*.rs` |
| 测试 target | 4：`gate`（挂 `ratchet` 子模块）、`card`、`lex`、`words` | `crates/tools/Cargo.toml` `[[test]]` |
| 测试用例 | gate 15（ratchet 子模块 11 + gate 4，1 红为并行存量债，非 tools）、card 39、lex 27、words 10，共 91 | `cargo test -p llm-tools --test <target>`；`--list` 核定 |
| 运行期依赖 | 仅 `diagnostic`；`testkit` 为 dev；`diagnostic` build feature 供构建期读表 | `Cargo.toml` |
| 重复率门 | jscpd `threshold: 0`，`minLines: 5`、`minTokens: 80` | `.config/.jscpd.json` |
| 静态阈值 | ccn 5 / nloc 60 / file 600 / test 800；Call 预算 0 | `.config/lints/vocabulary.toml` `[gate]`，`gate::limit` |
| CI job | 14：fmt、clippy、test(nextest)、doc、metal、asan、msrv、nightly、dup(jscpd)、gates、deny+machete、coverage、bench 回归、fuzz | `.github/workflows/ci.yml` |
| 覆盖率下限 | cutile 行 75 / 函 70；wire 68 / 64；dtype 88 / 88；weights 70 / 64；**tools 行 88 / 函 85**（本地基线行 94.5% / 函 94.9%） | ci.yml `coverage` job |

### 规则面清单（13 族）

| # | 族 | 入口 | 判定面 |
| --- | --- | --- | --- |
| 1 | 词法 | `lex::parse` | 注释 / 字符串 / 字符字面量词法层剔除；位置 1-based；括号闭集 |
| 2 | 复杂度 | `ccn::analyze` | 圈复杂度 + 有效行数（lizard 1.25 口径） |
| 3 | 禁止形态 | `call::scan` | `unwrap` / `expect` / `panic!` / `unreachable!` |
| 4 | 命名 | `words::scan` | 概念拼写收敛；外来 API 整名 / 成员名豁免 |
| 5 | 测试剔除 / 挂载 | `strip::{strip_tests,submodules,test_modules}` | 内联与 `#[path]` 挂载的 cfg(test) 条目识别与递归传播 |
| 6 | 配置门 | `gate::Gate` | 四阈值 + 存量棘轮 + 闭集键校验 + stale 结清 |
| 7 | 文件行数 | `dup::file_lines` | 生产 600 / 测试 800 双限 |
| 8 | 文档链接 | `docs::links` | 相对链接存在性；外部协议 / 纯锚点 / 空路径 / 未闭合剔除 |
| 9 | 统计 | `stats` | 均值 / 标准差 / 分位数 / bootstrap 区间 |
| 10 | roofline | `bench::{Roofline,Score,Bottleneck}` | 占峰值百分比 + 带宽 / 算力瓶颈侧 |
| 11 | 基线回退 | `regress::{Case,Baseline,Judgement}` | 方向 / 相对预算 / Pass-Regressed-Improved 三态 |
| 12 | 协议卡 | `card::{Protocol,Card,Standing}` | 样本下限 / 三态结论 / 摘要缺测记 OPEN |
| 13 | 转义 | `escape::quote` | JSON 字符串转义 |

## 3. 八维评分

状态图例：● 证据落地可跑；◐ 骨架 / 部分实现；○ 缺失或仅声明。档位为当前快照相对值，不是能力上限。

| 维度 | 10 分判据 | 当前证据 | 状态 | 档位 | 待校准 / 测量动作 |
| --- | --- | --- | --- | --- | --- |
| A 规则覆盖面 | 静态与性能规则族齐全，每条规则有明确白名单与判据 | 静态 7 族 + 性能判定 4 族；缺注解规则、缺 severity 分级 | ◐ | 7 | 枚举待新增规则清单；每条补判据与白名单后再评 |
| B 判定正确性 / oracle | 每族有正反 oracle；误报率 / 假阴可量化 | 各族有单测；docs 边界已补（#51）；豁免白名单人工维护，误报计数未测 | ◐ | 7 | 建 oracle 用例库，统计假阳 / 假阴命中数；测量命令待定 |
| C 配置化 / 闭集安全 | 阈值 / 白名单 / 档位全部声明化，拼错键构建期报错 | 编译期生成表；`gate::unknown_keys` 闭集；棘轮 + stale 结清 | ● | 9 | 无；新增级别键时复用同一闭集校验 |
| D 泛型 / 声明化密度 | 同构 ≥3 份（≥5 行 / ≥80 token）全部泛型 / 宏 / 表驱动 | 错误工厂、遍历骨架已收敛；7 处短名 match 属闭集安全且未达同构体量 | ◐ | 7 | 跑 jscpd threshold 0 复核；新增同构先判体量再决定是否抽 |
| E 规模 / 密度 | 无超线文件；公开项密度 ≤80 行 / 项 | 15 文件 2715 行，最大 408；公开项密度 ≈21.7 行 / 项 | ● | 9 | 公开项逐个核调用方（N11），死导出即删 |
| F 分析器性能 | 单次分析耗时有基线与回归门；热点复杂度可证 | 非热路径；`ccn::holder` 为 O(F·P) 线性归属，未 benchmark | ○ | 5 | 先 micro-bench 取单文件分析耗时基线；再决定是否上区间归属 |
| G CI / 证据完备 | 本地与 CI 同一命令；覆盖率 / 重复 / 许可证 / 回归全进门 | 14 job 覆盖 fmt/clippy/test/doc/asan/msrv/nightly/coverage/bench/fuzz/deny/jscpd/architecture/name；tools 覆盖率下限已挂（行 88 / 函 85） | ● | 9 | 下限随基线刷新上调；其余族反向用例补全后再评 |
| H 测试分层 | 单测试文件 ≤800；夹具复用；每族反向验证（注入违规即红） | 4 target 91 例，平铺 tests/；部分族缺反向注入用例 | ◐ | 7 | 逐族补「注入违规 → 门变红」反向用例；统计夹具复用率 |

**等权参考分**（八维算术平均，仅作基线锚点，不作交付结论）：(7+7+9+7+9+5+9+7)/8 ≈ **7.5**。
对照 `COMPARISON` §16.3 维度 19「测试 / 质量门禁」llm 9 / Burn 8——该分是仓库级门禁成熟度；
本表是 tools 组件自身的细粒度基线，口径不同，不直接替换。

## 4. 差距清单（按优先级）

### G1 — 分析器自身覆盖率下限（已落地）

- 基线：tools 91 例集成测试，本地干净基线行覆盖 94.5%（2356/2494）、函数覆盖 94.9%（224/236），
  取证命令 `cargo llvm-cov test -p llm-tools --all-features -- --skip every_source_passes_the_static_gate`。
- 落地：ci.yml `coverage` job 增 `Coverage floor (tools)`：
  `cargo llvm-cov nextest -p llm-tools --all-features --locked --fail-under-lines 88 --fail-under-functions 85`。
- 口径：tools 纯 Rust、无平台探测分支，下限留单次样本波动余量（行 −6.5、函 −9.9 点），只防整体回退；
  `every_source` 全仓门的红绿由存量债负责，不在本覆盖率门口径内。
- 复核：本机 `--fail-under-lines 88 --fail-under-functions 85` 退出码 0，过线。

### G2 — severity / 优先级分级（能力增量，跨模型）

- 现状：结果只有「硬失败」与「报告」两态；`Over` 无级别列。
- 10 分线：命中带级别（error / warn），级别进编译期配置与闭集校验，编排层按级别决定阻断还是报告。
- 前置拍板：warn 在 CI 是阻断还是仅报告；哪些规则默认 error、哪些默认 warn。
- 度量：分级后统计各规则级别分布；误报率必须先有基线（B 维度），否则 warn 会变成静默放宽。

### G3 — 注解驱动规则（新能力）

- 现状：`lex` 把 `#` 与 `[...]` 当普通 token；`strip::attr_at` 能识别属性形态，但不解析属性语义。
- 10 分线：可枚举的属性白名单 + 违规判据 + oracle；新增规则走声明表，不写第二套按名分派。
- 前置拍板：扫哪些属性、什么算违规、豁免清单。
- 度量：规则数与命中样本；每条规则附正反 oracle。

### G4 — 分析器性能基线（待校准，不吹提速）

- 现状：`ccn::holder`（`ccn.rs:271`）对每个条件位置线性扫全部函数范围，O(F·P)；门禁非热路径，
  无耗时基线。`close` 行号不单调，不能直接按行二分；归属误判会改变 CCN 判定，风险高。
- 10 分线：有单文件分析耗时基线与回归门；热点若改区间归属，须先证明归属口径不变。
- 度量：先 micro-bench 取 F·P 规模下的耗时；收益不显著则保持线性扫描，注释说明保留理由。

## 5. 本轮不做的事

- **不抽统一 Finding 泛型层**：`Call / Hit / Finding / Link` 字段与语义不同（Kind、found/canon、
  nth/ccn/nloc、target/line），强抽会丢信息或被迫做宽 enum；tools 是 L1，不能反向依赖 L2 prof 的宏。
- **不拆 tools 文件**：15 文件各族内聚，最大 408 行；再拆只会增文件数，不符 N2/N4。
- **不放宽任何阈值**：阈值只收紧；新增档位必须进 `[gate]` 闭集并经构建期校验。
- **不把计划测试 / 未来算子写成当前已通过**：G2–G4 在落地前均记待校准；G1 已落地并复核过线。

## 6. 风险与回滚

- 并行在途改动（nn / cutile / annotation / train / device 等）触发的 gate 红项不属于 tools；
  提交前 `git status --short` 划边界，只认领 tools 文件。
- 快照机制可能把未提交改动收进 HEAD；新文件写盘后立即 `git add`，避免被 untracked 清理删除。
- 本基线数字随并发改动漂移；以 `§2` 命令的当场输出为准，文档修订时刷新快照表。

## 7. 取证命令附录

```bash
# 规模
find crates/tools/src -name '*.rs' | wc -l
wc -l crates/tools/src/*.rs
grep -rhE "pub (fn|struct|enum|trait|const|use)" crates/tools/src/*.rs | wc -l

# 测试（按 target）
cargo test -p llm-tools --test gate
cargo test -p llm-tools --test card
cargo test -p llm-tools --test lex
cargo test -p llm-tools --test words

# 重复率 / 命名 / 密度
npx --yes jscpd
bash scripts/name-lint.sh
bash scripts/loc-budget.sh

# 覆盖率基线与下限复核（红用例为全仓存量债，取 tools 自身覆盖时跳过）
cargo llvm-cov test -p llm-tools --all-features -- --skip every_source_passes_the_static_gate
cargo llvm-cov nextest -p llm-tools --all-features --fail-under-lines 88 --fail-under-functions 85
```
