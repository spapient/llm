# 文档

仓库只保留**四类活文档**，每一类回答一个不同的问题；其余写作（一次性审计、对标评分、任务分解、
backlog 矩阵）不随代码长期存放——结论要么落成代码与测试，要么落成一条 ADR。

## 一、读什么

| 文档 | 回答什么 | 由谁保证不漂移 |
|---|---|---|
| [`../README.md`](../README.md) | 怎么编译、怎么跑、目录长什么样 | 人工；示例命令随 CI 冒烟 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 谁属于哪一层、允许依赖谁、每个类型归谁 | [`crates/testkit/src/architecture.rs`](../crates/testkit/src/architecture.rs)：读 `cargo metadata` 与源码，逐条核对（断言分在 `src/` 下的 `layer` / `ownership` / `boundary` 等文件，与 crate 单测一起跑） |
| [`lld.md`](lld.md) | 数学建模与算子判据：`E*` 出口、规则 `0.x`、形状与精度口径 | 代码注释按编号引用（`E6`/`E7`、规则 `0.3`、§`10.10`、§`16`），编号因此冻结 |
| [`adr/`](adr/) | 为什么这样而不是那样：一份决策一个文件，只留仍被代码按编号引用的那几条 | 引用点就是判据——删一条 ADR 必须同一次提交改完全部引用方 |
| `crates/<name>/README.md`、`src/lib.rs` 的 `//!` | 一个成员**怎么实现**：拥有什么、不拥有什么、依赖谁、被谁依赖 | crate 文档测试（`--workspace` 的 doctest）与各成员自带断言 |
| [`../CHANGELOG.md`](../CHANGELOG.md) | 版本与行为变更 | 发布时追加 |

它们不重叠：README 说**是什么**，`ARCHITECTURE.md` 说**允许什么**，crate 文档说**怎么实现**，
ADR 只说**为什么这样而不是那样**。
改机制读 crate 文档；加成员读 `ARCHITECTURE.md`；换决策写 ADR。

## 二、写作规则

1. **一份事实只写一次。** 成员数、依赖边数、注册算子数、模块数只在 `ARCHITECTURE.md` 出现，
   别处只引用不复述——数字被抄成两份，就有一份一定会过期。
2. **能守住的写进测试，守不住的写进 ADR。** 依赖方向、所有权、闭集、注解与注册表镜像都有测试；
   "为什么先做 CPU 再做设备"这类判断没有测试可写，才归 ADR。
3. **删除也是维护。** 一次性报告的寿命是当轮评审；它的结论要么已经进了代码与测试，要么应该
   补一条 ADR。把历史报告留在 `docs/` 只会让读者以为它仍然成立。
4. **引用用相对路径，删文档时同一次提交改完引用方。** 断链比没有链接更坏。
5. **判据写可复算的量。** "性能更好"不是判据；"`max_rel_err = 0`、达成率 ≥ 70%"才是。

## 三、ADR 索引

只留**仍被产品代码按编号引用、且承载代码读不出的取舍**的决策：已被 `architecture.rs` /
`contracts.rs` 或 crate 文档守住的口径不再单独成文（本轮由 49 个文件、48 个编号收敛到 15 条、随后追加 0052/0053，现存 17 条）。正文里出现的旧编号
（0021、0028、0049 之类）是被取代的历史前提，文件已删除、结论已落在代码与 `ARCHITECTURE.md` 里。

| 主题 | 决策 |
|---|---|
| 边界与所有权 | [0025](adr/0025-runtime-backend-abstraction.md) 运行时后端三件套 · [0027](adr/0027-kernel-registry-and-fragment-contract.md) 注册表与片段契约 · [0035](adr/0035-unsafe-ffi-boundary.md) unsafe / FFI 白名单 · [0050](adr/0050-operator-base.md) 算子基座 · [0051](adr/0051-execution-backend-base.md) 执行基座 · [0052](adr/0052-execution-platform-contract.md) 六平台执行合同 |
| 机制与数据 | [0005](adr/0005-annotations-as-contracts.md) 注解即契约（无运行期注册表） · [0017](adr/0017-kernel-ast-and-ir-freeze.md) AST 冻结、无 IR · [0018](adr/0018-dtype-single-source.md) 精度单一事实源 · [0043](adr/0043-program-scalar-slots.md) 标量槽 · [0044](adr/0044-bindings-metadata.md) 绑定编号 · [0045](adr/0045-u-weight-container-v2.md) `.u` 容器 v2 · [0046](adr/0046-batch-flat-buffer-geometry.md) 扁平批缓冲几何 · [0053](adr/0053-typed-operators.md) 类型化算子入口 · [0054](adr/0054-operator-coverage-spp.md) 算子覆盖骨架族与能力表 · [0056](adr/0056-cutile-prof-dsl-ir-spp.md) prof 真内核与 cutile DSL/IR 裁决标尺 |
| 配置 | [0022](adr/0022-build-time-corpora.md) 语料构建期解析 · [0023](adr/0023-settings-to-cli.md) 设置策略上移 `cli` · [0048](adr/0048-runtime-config-file.md) 运行时配置文件 |

新决策按下一个四位编号追加，文件名 `<编号>-<kebab-标题>.md`，首行是 `# <编号>：<一句话结论>`，
正文只写**背景 / 决策 / 后果 / 替代方案**四段；与既有决策冲突时，改的是**旧文件的状态**，不是新文件的措辞。
一条决策被代码或测试完全接管后，**删掉它的文件并改完引用方**——留下来的是仍然会被人问"为什么"的那几条。
