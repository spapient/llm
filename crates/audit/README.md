# llm-audit

内核审计：`doctor` / `verify` / `bench` / `cache gc` 的**计算**（DESIGN §16 的 P0 交付物）。

前三条命令共用同一条流水线：声明层 → 协商 → tile 取值 → `cutile::build` 的拼接与 emit →
编译执行。任何一步失败都带上下文返回 `Err`，不写"跳过"。

## 这里有什么

| 项 | 说明 |
|---|---|
| `doctor` / `Doctor` | 把流水线每一步摊开：声明头部（含生产形状）、档案头部、证据（依据 / 设备执行 / 自检条数 / 失败原因）、逐变体判定、plan、布局声明、逐层存储（档案容量 + 构建期契约比较过的声明用量）、emit 摘要。协商用的是 `Evidence` 的结论，不是档案里的 `grade` |
| `verify` / `VerifyReport` | 逐设备跑见证 + 形状矩阵（最小合法 / 多行非方形 / 生产形状），与标量参考对拍；设备跑不到的见证先跑本机自检，非 C 设备再如实报“缺设备”；`witnesses` 与 `self_checked` 相加等于该设备全部声明；`negotiations` 逐设备给出协商结论（见下行） |
| `Negotiation` | 一台设备的协商结论：`Head`（命中降级链首档）/ `Fallback`（第 k 档）/ `NoPath`（链上无可行变体，带首选缺的原语）/ `Unknown`（没有档案）。只读声明与档案，本机缺设备时也有答案——`VerifyReport::head_hits()` 就是 DESIGN §16 的分数 |
| `Fragment` | 一份生成片段的 §16 口径：`lines` + `FRAGMENT_LIMIT`（400）/ `FRAGMENT_GOAL`（240），`within_limit()` 就是“片段 ≤ 400 行”这条判据；`doctor` 的 emit 摘要与 `verify` 的结论行都带着它 |
| `bench` / `Bench` | 计时（min 与 p50）、吞吐换算、达成率口径（工作集 vs FAST 容量）；计时前先跑证据，`Bench::proven` 就是这次协商用的见证条数，没有证据就拒绝计时 |
| `Performance` | §16 的性能判据：达成率连同它的尺子一起给（roofline 实测 ÷ 档案峰值），`is_measured` 在口径无效或没有峰值时为假（按 §12.3 视为未测量），`needs_specialisation` 就是“某设备 < 70%”。`Bench::performance` 是唯一出处，`bench` 对未测量也打印结论 |
| `cache_gc` | 按 LRU 把缓存回收到上限以内，缺省上限 `CACHE_LIMIT`（2 GiB）；删哪些条目（含与上限无关的残留）由 `cutile::cache` 决定 |
| `why` / `CacheWhy` / `CacheState` | `doctor --why <key>`：把缓存里的键与本次构建的键逐字段比较、只列出真正不同的项；`CacheState` 先把"半截落盘"与"从未落盘"分出来，否则没有 `source` 的目录会因为差异为空被说成命中 |
| `witness` | 见证执行器（`Evidence` 同时记设备执行与自检各自的编译参数：证据的适用条件是「片段 + 工具链 + 参数」）：`run_all` 把档案里**每一条已声明**的见证在本机执行一遍（`Scope::Device`，不按 `grade` 筛）；`self_check` 用 host C 前端跑**设备执行跑不到**的那些（`Scope::Host`），不改写等级。前端是「注解 + trait + const 泛型」：`#[frontend(emit = .., ext = .., tag = ..)]` 生成 `Language`（语言 / 后缀 / 缓存标签，且 `ext` 与 `Emit::ext()` 在编译期比对），手写 `Device`（`available` / `program::<OUT>` / `run`），`Frontend: Language + Device` 把两半绑死，分发是一张 `const FRONTENDS` 表——没有 `match`，也没有 `dyn`。两个范围不重叠且合起来覆盖每一条声明。`establish` 把两者合成一份 `Evidence`（`Basis` / 设备执行条数 / 自检条数 / 两套编译参数 / 失败原因 / `Proof`），`doctor` / `verify` / `bench` 共用同一份结论。**语言门**是第三种结论（`LanguageGate`）：设备执行跑得到的设备免开（`Covered`），其余设备交给**它自己语言的解析器**（`Front` + `const GATES`：`cuda` 走 `clang -x cuda --cuda-host-only`、`opencl` 走 `clang -x cl`），只解析不执行，`ascendc` 没有开源前端就如实缺席（`Absent`）；被自己的语言拒绝（`Rejected`）记进 `Evidence::failure`——那是证据缺陷，不是“本机没有该设备” |

- **没有编译器也能跑**：产物已落盘时 `verify` / `bench` / 见证按 `bin/<hex>.key` 里的身份直接复用，不重新编；只有“需要重新编而本机没有 `cc`”才以 `unsupported` 报“没有这份产物”（`--repeat` 变了就是这种情况）。

报告字段全是数据：没有 `println!`，也不拼装人读排版。

## 这里**不**有什么

- **命令行文法**：在 `llm-cli::cutile` 的 `CutileCommand`。
- **排版与退出码**：排版在根包的 `src/bin/cutile.rs`（L7 入口，不定义新类型）；本 crate 的失败一律是
  `diagnostic::Error`，退出码由 `Code::exit_code()` 给出（`2` 输入不合法 / `3` 缺能力 / `4` 缺东西 /
  `5` 读写 / `70` 内部不变量被破坏）——"设备没这条能力"与"参数写错"因此可区分。
- **配置分层**：在 `llm-config`（`Settings` / `Key` / `App`），本 crate 只读。
- **机制**：协商、拼接、emit、编译执行、缓存都在 `cutile`；数据装载在 `Catalogue`。
  本 crate 决定"跑哪些形状、重复几次、怎么算达成率"，不决定"产物长什么样"。

## 命令

```bash
cargo run --bin cutile -- doctor --device cpu
cargo run --bin cutile -- doctor --device cpu --why 83a3ecdb31cd1225
cargo run --bin cutile -- verify --devices cpu
cargo run --bin cutile -- bench  --repeat 20   # 形状缺省取算子的生产形状
cargo run --bin cutile -- cache gc             # 缺省上限 2 GiB
```
