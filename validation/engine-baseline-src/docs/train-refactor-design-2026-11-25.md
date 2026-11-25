# `crates/train` 第一轮实现设计与验证计划

日期：2026-11-25　范围：仅数据采样路径；本轮不代表整个 crate 已完成。

## 1. 不变核心与可变差异

- 不变核心：`Batch` 借用语义、`Dataset` 对象安全契约、`Loader` / `Chunks` 公共路径、采样种子与顺序、
  所有公开错误类别和既有测试覆盖。
- 可变差异：顺序 / 洗牌 / 加权三种采样策略，继续由既有 `Sampling` 和 `Rng` 表达，不新增配置源或业务分支。
- `geometry/shapes.rs` 暂不改：其归一化输出视图牵涉 `geometry/kernels.rs`，后者同样超过 300 行。
  几何部分必须另做完整设计，不能用局部豁免或删注释凑第一轮指标。

## 2. 正交分层

- 数据定义层：`Batch`、`LoaderConfig`、`Sampling` 和 `Policy` 只承载状态，不读取环境或配置源；
  其中 `LoaderConfig` / `Sampling` 归 `rng.rs`，与采样策略同源。
- 核心内核层：`dataset.rs` 内的 `Sampler` 持有策略状态、索引表、游标和共享 epoch，执行确定性索引重建。
- 执行入口层：`Loader::with_config` 负责构造期校验、对象安全取批与诊断，不复制采样算法或几何检查。

`Sampler` 的策略与生命周期独立于 `Loader`，但不需要独立文件。最终不新增 `sampler.rs`，避免文件下限与
碎片化问题；`dataset.rs` 同时受 300 行上下限约束，配置上移后保持职责清晰且恰好 300 行。

## 3. 复用路径

1. 首轮 `Sampler::from_config` 与 epoch `reset` 共用 `rebuild(epoch)`，消除洗牌/加权初始化的第二份实现。
2. `Policy` 只保留策略差异，`epoch` 上提到 `Sampler` 共享，避免两个策略变体各维护一份游标。
3. `LoaderConfig` 保留显式的 `rows = 1 / cols = 1` 默认值；`stride` 与 `Batch::new` 复用既有校验和 `Batch::exact`。
4. 保留 `draw_round` 的 `clear + extend` 容量复用，以及现有指针/容量断言，不改变随机调用次数或顺序。
5. 复用既有 `getters` / `builders` / `validation_rules` / `Rng`；不新增第三方依赖、全局状态或配置镜像。

## 4. 扩展方式

- 新增采样策略：增加 `Sampling` / `Policy` 配置变体及 `rebuild` 中的一条策略行，Loader 与 Batch 不动。
- 新增采样算法：只改 `dataset.rs` 的索引重建，不改变数据借用、批装配和对象安全接口。
- 后续归一化输出视图：先完成几何调用链设计，再同时修改声明与全部调用点，禁止局部参数豁免。

## 5. 量化目标

| 项目 | 基线 | 本轮验收 |
|---|---|---|
| 文件行数 | `dataset.rs` 404、`rng.rs` 222 | `dataset.rs` / `rng.rs` 均不超过 300 |
| 函数长度 / CCN | 仓库静态门零命中 | 每个函数不超过 30 行，CCN 不超过 5，控制流嵌套不超过 3 层 |
| 重复率 | jscpd 5.4.3：35 行 0.46%、291 tokens 0.59% | 两项均低于 0.5%，且不得新增克隆 |
| 文件 / 依赖 | 13 个生产文件、无新增依赖 | 不新增生产文件、不新增依赖 |
| 测试 | 246 passed / 0 failed / 1 ignored | 不减少用例、覆盖点与断言强度 |
| 缓冲复用 | 现有测试已断言索引指针和容量不变 | 首轮与 reset 共用重建逻辑后持续通过 |

功能密度使用同功能、同测试的物理行数与公共组件承载位置衡量，不宣称 1/10 或未经实测的性能加速。
本轮只承诺去除重复初始化与重复状态，不宣称采样吞吐提升。

## 6. 验证计划

1. `cargo test -p llm-train`：覆盖正常、空流、零几何、溢出、非法权重、跨 epoch、恢复与反向路径。
2. `cargo test -p llm-train --doc`：全部 rustdoc 示例。
3. `cargo clippy -p llm-train --all-targets --no-deps -- -D warnings`：无参数豁免、无新告警。
4. `cargo fmt --all --check`、`git diff --check`。
5. 静态门与逐函数扫描：函数长度、CCN、控制流嵌套和文件行数。
6. `npx --no-install jscpd crates/train/src --min-lines 5 --min-tokens 50 --reporters console`：
   重复行与重复 tokens 均低于 0.5%。
7. 单独复跑 `weighted_draws_one_batch_per_declared_batch`，用指针和容量确认缓冲复用。
8. 最终统一执行 `scripts/ci.sh`，覆盖格式、全特性 Clippy、nextest、doc-test、依赖卫生、架构、命名、密度与基准判定；
   工具缺失或其它 crate 阻断均逐项记录并标为未完成，不以弱化命令替代，不作全绿声明。
