# llm-audit

内核审计：`doctor` / `verify` / `bench` / `cache gc` 的**计算**（DESIGN §16 的 P0 交付物）。

前三条命令共用同一条流水线：声明层 → 协商 → tile 取值 → `cutile::build` 的拼接与 emit →
编译执行。任何一步失败都带上下文返回 `Err`，不写"跳过"。

## 这里有什么

| 项 | 说明 |
|---|---|
| `doctor` / `Doctor` | 把流水线每一步摊开：声明头部（含生产形状）、档案头部、逐变体判定、plan、布局声明、逐层存储（档案容量 + 构建期契约比较过的声明用量）、emit 摘要 |
| `verify` / `VerifyReport` | 逐设备跑见证 + 形状矩阵（最小合法 / 多行非方形 / 生产形状），与标量参考对拍；非 C 设备先跑见证自检，再如实报“缺设备” |
| `bench` / `Bench` | 计时（min 与 p50）、吞吐换算、达成率口径（工作集 vs FAST 容量） |
| `cache_gc` | 按 LRU 把缓存回收到上限以内，缺省上限 `CACHE_LIMIT`（2 GiB）；删哪些条目（含与上限无关的残留）由 `cutile::cache` 决定 |
| `why` / `CacheWhy` / `CacheState` | `doctor --why <key>`：把缓存里的键与本次构建的键逐字段比较、只列出真正不同的项；`CacheState` 先把"半截落盘"与"从未落盘"分出来，否则没有 `source` 的目录会因为差异为空被说成命中 |
| `witness` | 见证执行器：把 `grade ≥ L2` 的供给编成可执行文件并逐条对拍；`self_check` 把每条**已声明**的见证按 host C ABI 跑一遍，结论记作 `Scope::Host`，不改写等级 |

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
