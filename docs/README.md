# 文档

仓库只保留**四类活文档**，每一类回答一个不同的问题；其余写作（一次性审计、对标评分、任务分解、
backlog 矩阵）不随代码长期存放——结论要么落成代码与测试，要么落成一条 ADR。

## 一、读什么

| 文档 | 回答什么 | 由谁保证不漂移 |
|---|---|---|
| [`../README.md`](../README.md) | 怎么编译、怎么跑、目录长什么样 | 人工；示例命令随 CI 冒烟 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 谁属于哪一层、允许依赖谁、每个类型归谁 | [`crates/testkit/tests/architecture.rs`](../crates/testkit/tests/architecture.rs)：读 `cargo metadata` 与源码，逐条核对 |
| [`lld.md`](lld.md) | 数学建模与算子判据：`E*` 出口、规则 `0.x`、形状与精度口径 | 代码注释按编号引用（`E6`/`E7`、规则 `0.3`、§`10.10`、§`16`），编号因此冻结 |
| [`adr/`](adr/) | 为什么这样而不是那样 | 只追加、不改写；被取代的决策在原文件标注 |
| `crates/<name>/README.md`、`src/lib.rs` 的 `//!` | 一个成员**怎么实现**：拥有什么、不拥有什么、依赖谁、被谁依赖 | crate 文档测试（`--workspace` 的 doctest）与各成员自带断言 |
| [`../CHANGELOG.md`](../CHANGELOG.md) | 版本与行为变更 | 发布时追加 |

三者不重叠：README 说**是什么**，`ARCHITECTURE.md` 说**允许什么**，crate 文档说**怎么实现**。
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

| 主题 | 决策 |
|---|---|
| 分层与所有权 | [0001](adr/0001-layered-dependency-order.md) 显式层表 · [0002](adr/0002-single-owner-per-type.md) 一个概念一个主人 · [0003](adr/0003-kernel-and-runtime-types.md) kernel 与运行时类型分离 · [0004](adr/0004-placeholder-policy.md) 占位占接口形状 · [0038](adr/0038-crate-merge-policy.md) crate 合并节奏 · [0019](adr/0019-packaging-merge.md) 单消费者吸收 · [0035](adr/0035-unsafe-ffi-boundary.md) unsafe/FFI 边界 |
| 内核与算子 | [0005](adr/0005-annotations-as-contracts.md) 注解即契约 · [0007](adr/0007-semantic-cost-layer.md) 代价来自求和表达式 · [0013](adr/0013-dense-projection-operator.md) 第二条算子只加数据 · [0017](adr/0017-kernel-ast-and-ir-freeze.md) AST 与无 IR · [0026](adr/0026-gemm-generic-skeleton.md) GEMM 泛型骨架 · [0027](adr/0027-kernel-registry-and-fragment-contract.md) 注册表与片段契约 · [0031](adr/0031-kernel-macro-derived-table.md) 适配表由宏派生 · [0032](adr/0032-assembly-family-descriptor.md) 装配族描述符 · [0041](adr/0041-annotation-macro-retention.md) 注解宏保留 · [0050](adr/0050-operator-base.md) 算子基座 |
| 精度与权重 | [0018](adr/0018-dtype-single-source.md) 精度单一事实源 · [0012](adr/0012-quantized-weights.md) 分组与宽度同源 · [0010](adr/0010-borrow-vs-load.md) 借出与载入 · [0015](adr/0015-pack-weight-container.md) `pack` 容器 · [0036](adr/0036-reader-loader-mmap-source.md) Reader/Loader 与 mmap · [0045](adr/0045-u-weight-container-v2.md) `.u` 容器 v2 · [0047](adr/0047-2bit-quantized-lane.md) 2-bit lane |
| 模型与训练 | [0008](adr/0008-layer-floor-from-semantics.md) 层下界由语义推出 · [0009](adr/0009-real-weights-to-model-floor.md) 真实权重到下界 · [0011](adr/0011-directional-floor-and-absence.md) 方向分账与"无法计价" · [0014](adr/0014-real-weights-execution.md) 真实权重执行 · [0037](adr/0037-optimizer-ownership-and-param-role.md) 优化器与 `ParamRole` |
| 运行时与服务 | [0016](adr/0016-wire-frame-v2.md) 帧 v2 · [0024](adr/0024-shared-wire-base.md) `wire` 字节码闭集 · [0025](adr/0025-runtime-backend-abstraction.md) 后端抽象 · [0028](adr/0028-device-data-vs-mechanism-boundary.md) 设备数据与机制 · [0029](adr/0029-admitter-port.md) `Admitter` 端口 · [0030](adr/0030-request-field-descriptor.md) 请求字段描述符 · [0034](adr/0034-local-closed-set-macro.md) 本地闭集宏 · [0043](adr/0043-program-scalar-slots.md) 标量槽 · [0044](adr/0044-bindings-metadata.md) 绑定编号 · [0046](adr/0046-batch-flat-buffer-geometry.md) 扁平批缓冲 · [0051](adr/0051-execution-backend-base.md) 执行基座 |
| 配置与依赖 | [0006](adr/0006-config-mechanism-vs-identity.md) 机制与身份分离 · [0020](adr/0020-settings-layer-absorption.md) 设置层吸收 · [0021](adr/0021-toml-layer-into-diagnostic.md) TOML 层归属 · [0022](adr/0022-build-time-corpora.md) 语料构建期解析 · [0023](adr/0023-settings-to-cli.md) 配置策略上移 · [0039](adr/0039-parser-dependency-policy.md) 解析器依赖策略 · [0048](adr/0048-runtime-config-file.md) 运行时配置文件 |
| 汇总 | [0049](adr/0049-refactor-summary-2026-11.md) 2026-11 全仓重构 |

新决策按下一个四位编号追加，文件名 `<编号>-<kebab-标题>.md`，首行是 `# <编号>：<一句话结论>`，
正文只写**背景 / 决策 / 后果 / 替代方案**四段；与既有决策冲突时，改的是**旧文件的状态**，不是新文件的措辞。
