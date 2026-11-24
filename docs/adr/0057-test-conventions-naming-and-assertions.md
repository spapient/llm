# 0057：测试工具链口径——命名分层、内联/外置边界、断言分工与覆盖率取舍

- 状态：已采纳（nextest 双 profile、testkit 结构化 diff 断言入口、覆盖率摘要留痕已落地；
  存量源码红账——src 下划线文件名、复杂度/行数越线、存量克隆——属并发在途债，由各静态门
  棘轮报出，不在本 ADR 配置层清绿）
- 日期：2026-11-24
- 相关：[`ARCHITECTURE.md`](../ARCHITECTURE.md)（分层与所有权）、
  [`scripts/name-lint.sh`](../../scripts/name-lint.sh)（文件名/符号形状门）、
  [`.config/nextest.toml`](../../.config/nextest.toml)（测试运行器 profile）、
  [`crates/testkit/src/lib.rs`](../../crates/testkit/src/lib.rs)（共享夹具模块表）、
  [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)（coverage / test job）、
  [`docs/GATE-RUBRIC-2026-11-24.md`](../GATE-RUBRIC-2026-11-24.md)（八维标尺）

## 背景

满分改造把上一轮评审扣分项逐条复核后，发现其中一部分是**刻意的纪律立场**（被误记为缺陷），
另一部分才是真正缺的设施。本 ADR 固化最终口径：哪些判据由脚本/测试可执行地守住，哪些取舍
只在文档里说明理由。实测基线（`cargo` / `find` / `grep` 可复算）：

- 测试布局：193 个内联 `#[cfg(test)]` 文件；280 个外置 `tests/` 文件，分布在 23 个 crate。
- 命名：`tests/` 下 191 个文件名含下划线（如 `attnres_tests.rs`、`codec_tests.rs`）；
  `src/` 下 65 个文件名含下划线（存量红账，name-lint 零容忍门报出，非本批新增）。
- 错误处理：`tests/` 内 4371 处 `.unwrap()` / `.expect()` / `panic!`；testkit 生产夹具模块
  （`fixture` / `compare` / `hash` / `rng` / `gating` / `device` / `prop`）零 unwrap/expect；
  门禁与跨层对拍模块（`architecture` / `boundary` / `contracts` / `mirror` / `ownership` /
  `e2e`）的 `expect` 都带明确消息，语义是「前置条件必须成立」的断言，不是忽略错误。

## 决策

### 1. 命名：官方基线 + src 自定义收紧，tests 分层豁免

- **符号命名**（src 与 tests 同一套，遵循 Rust 官方）：类型/枚举/trait/结构体
  UpperCamelCase；函数/方法/变量/模块 snake_case；全局常量 SCREAMING_SNAKE_CASE。这条由
  `scripts/name-lint.sh` 的符号后缀/泛名门 + clippy 共同守。
- **src 文件名：零下划线，是本仓比官方更严的自定义收紧。** Rust 官方允许模块文件名
  snake_case 且带下划线（官方书 `front_of_house.rs` 即为例）；本仓要求限定词一律下沉到目录
  （`kernel/softmax.rs` 而非 `kernel_softmax.rs`），`src` 文件名匹配 `*_*.rs` 即命中命名门，
  零容忍、不留存量基线。
- **tests/ 文件名：允许下划线，豁免 src 那套下沉规则。** 测试文件按被测主题聚合，普遍是
  `<topic>_tests.rs` 形态；把同样的目录下沉规则套到测试文件上，只会增加文件跳转而不改善
  可读性。豁免只针对**文件名**，测试内的符号命名仍按上一条官方规则。
- 两套口径并存是有意的分层，不是门的漏扫；`name-lint.sh` 注释里登记了这条边界。

### 2. 内联 vs 外置测试边界

- **内联 `#[cfg(test)]`**：与被测单元强耦合、需要访问私有项的白盒单测；测试代码住在源码旁，
  `super` / `crate` 仍指向本 crate。
- **外置 `tests/`**：走公共 API 黑盒对拍、跨层契约、设备见证、性质测试；由 `src/` 里的
  `#[cfg(test)] #[path = "../tests/…"] mod …` 钩子挂载（testkit 即此形态，`autotests = false`），
  或作为独立集成测试 target。
- **夹具不反向依赖被测 crate 之外的生产层**；共享夹具收口在 testkit，各成员只作 dev-dependency
  引用，夹具不进任何生产二进制。

### 3. 断言三分工，不许用错判据

| 断言入口 | 判据 | 适用 |
|---|---|---|
| `testkit::assert::assert_eq`（similar-asserts） | `PartialEq` 精确相等，失败给行级 diff | 集合 / 结构体 / 字节片等结构化相等对比 |
| `testkit::compare`（`assert_close` / `assert_within_rel` / `approx_eq`…） | 逐元素容差 / 整批相对误差 / 方向余弦 | 浮点对拍；**数值禁止用精确相等** |
| `testkit::hash::assert_table_matches_bitwise` | 位级 CRC-32 参考 | 权重 / 二进制快照逐字节锁定 |

`assert_eq` 是 std `assert_eq!` 的诊断增强替代，判定语义仍是 `PartialEq`，不引入任何浮点容差。
存量 4154 处 `assert_eq!` 不强制批量替换；新写的大集合/结构对比优先走 `testkit::assert::assert_eq`。

### 4. 快照：保留手写位级快照，不引 insta；编译错误契约走展开断言，暂不引 trybuild

本仓快照是权重字节、HTTP 响应、CLI 输出这类**二进制/逐字节**锁定，由 witness/evidence/hash
装置逐字节或 CRC 对拍，期望值进档案、每条见证在本机算得出档案写的值。这类快照需要的是
位级可控与可追溯，`insta` 的结构快照 `cargo insta review` 批量接受流程收益有限，迁移 280 个
外置测试成本高、风险大，故不引入。后续若出现大量复杂结构快照迭代场景，再单开 ADR 评估。

编译错误契约同理：全仓只有 `annotation` 一个 proc-macro crate，非法注解的错误路径由展开文本
断言（生成的 token 流含 `compile_error!`，指在出错 span）与 `audit.rs` 把守。`trybuild` 式
「独立 crate 编译失败 + 错误信息快照」暂不引入——宏表面小、错误契约已被可执行断言覆盖，新增
dev-dep 与失败用例目录的收益不抵维护成本；待宏表面扩大或需要对错误文案做快照回归时再评估。

### 5. unwrap / expect 纪律：分层成文，不进 deny lint

- 生产夹具（testkit 被运行时复用的模块）零 unwrap/expect，错误沿返回值传播。
- 门禁/对拍代码的 `expect` 必须带消息，表达「前置条件必须成立」；失败即契约破坏，消息写清
  是哪条前提。
- `tests/` 语境下的 unwrap/expect 是测试前置条件断言，允许存在；**不**对 tests 侧加
  `disallowed_methods` deny——4371 处存量一次性 deny 会引爆全仓，且测试语境本就允许短路失败。
  纪律以成文约定维持，不翻译成一条会误伤的 lint。

### 6. 覆盖率口径：nextest 采集 + 分 crate 下限 + 摘要留痕，doctest 不入覆盖数字

- 采集与主测试运行器一致走 nextest（lib/集成/二进制并行隔离）；硬下限按 crate 分档
  （`cutile` / `wire` / `dtype` / `weights` / `tools` 的行/函数下限见 ci.yml coverage job）。
- 主采集后出一份 text 整体覆盖率摘要进 job 日志，便于观察回退趋势；lcov 留档 14 天。
- **doctest 覆盖不并入硬门口径**：`cargo llvm-cov --doc` 采集文档示例覆盖仍是不稳定特性
  （需 nightly，cargo-llvm-cov issue #2）；文档示例的**正确性**由独立 `doc` job 的
  `cargo test --doc -D warnings` 断言，覆盖率数字以 lib/集成测试为准。
- **新增 crate 下限必须先有实测基线**，不许拍脑袋设虚高/虚低阈值；本机无 GPU，Metal/CUDA
  分支跳过，阈值口径按可采集到的分支定。扩面登记走 coverage job 的 floor step。

### 7. flaky 立场：不重试，修根因

nextest 两个 profile `retries = 0`。重试会把确定性失败藏进偶发通过；偶发失败由 JUnit XML
记录、回查根因，不靠重试刷绿。CI profile 与本地 profile 的差别只在留痕与超时终止策略，不在
重试。

### 8. 变异测试：手动/可选定位，不进阻断门

变异测试（cargo-mutants）跑一遍要对每个变异体编译执行全量测试，耗时远超普通 CI；业界通行
做法是按需手动跑或独立非阻断 job，不把它合并进 PR 阻断门禁。定位见 ADR 落地脚本与文档入口。

## 后果

### 正面

- 命名/布局/断言/覆盖率的边界都有成文口径与可执行门，新成员知道该用哪套判据、夹具收口在哪。
- CI 留痕（JUnit、lcov、text 摘要）齐备，失败与覆盖率回退可追溯；设备/探针类用例有分类超时宽限。
- 测试依赖（proptest、similar-asserts）收口在 testkit 的 dev 夹具图，不进生产二进制。

### 负面

- tests 文件名下划线不受目录下沉约束，偶有同名主题文件偏多时需靠目录/前缀人工组织。
- 存量 src 下划线、复杂度/行数、克隆红账要靠后续批次逐模块清，配置层不能让它们凭空变绿。
- 不重试意味着一次偶发抖动会让 CI 变红，需要人工回查——这是刻意用可用性换 flaky 可见性。

### 中性

- similar-asserts 传递依赖 `similar` / `console`，均 MIT OR Apache-2.0，在 cargo-deny
  license allow list 内；只进 dev 构建图。
- doctest 覆盖留待 `--doc` 稳定后再评估并入；当前正确性已由 doc job 把守。

## 备选方案（已评估并否决）

- **pretty_assertions 替代 similar-asserts**：前者 2018 edition、输出基于整段 Debug；后者
  2024 edition、行级 diff、依赖更轻。测试诊断场景取 similar-asserts。
- **CI 开 `retries = 1` 消偶发红**：否决，违背反重试立场，会掩盖确定性失败。
- **引入 insta 统一快照**：否决，见决策 4；本仓快照以位级锁定为主，收益不抵迁移成本。
- **引入 trybuild 做编译失败快照**：否决，见决策 4；proc-macro 仅 annotation 一处，错误契约
  已由展开文本断言与 audit 覆盖，宏表面扩大后再评估。
- **tests 侧 unwrap/expect 进 clippy deny**：否决，4371 处存量会引爆，测试语境本就允许短路。
- **变异测试进 PR 阻断门**：否决，执行时间不可接受；保留手动/可选入口。
- **把 doctest 覆盖并进 stable coverage job**：否决，`--doc` 不稳定且需 nightly，口径先分开。

## 参考

- Rust 命名规范：<https://doc.rust-lang.org/style-guide/advice.html>
- 模块与文件路径：<https://doc.rust-lang.org/reference/items/modules.html>
- 测试组织：<https://doc.rust-lang.org/book/ch11-03-test-organization.html>
- nextest profile / JUnit / 超时：<https://nexte.st/book/registry-config.html>
- cargo-llvm-cov：<https://crates.io/crates/cargo-llvm-cov>
- similar-asserts：<https://docs.rs/similar-asserts/2.0.0/similar_asserts/>
