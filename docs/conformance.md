# 文档一致性记录

状态: 拟定
范围: `docs/` 下四份规范文档（`architecture.md`、`engineering.md`、`design.md`、本记录）及其定义的工作区规则的一致性状态
读者: 工程师、评审者、维护者、AI 编码代理

---

本记录是一致性判定的唯一场所。每份规范文档在自身中定义其判定词汇；本记录记录当前判定及其背后的证据。

文档的存在不构成一致性的证据。判定为「已验证」MUST 引用已执行的检查或已核对的源码。

本记录随 2026-10-10 的文档合并更新：原先记录在 14 份顶层文档与 19 份 ADR 上的判定，已按其主题归属映射到合并后的 `architecture.md`、`engineering.md`、`design.md` 的对应章节；判定结论与证据沿用合并前经执行的检查。2026-10-10 的对抗性修正重测了静态门，并清除了黑盒测试迁移后失效的计数（见「本轮重测证据」）；通过/失败汇总未在本轮重跑，凡引用具体通过/失败数字者均以其所属章节的日期快照为准。性能相关的判定保持「拟定」，因为没有执行任何优化或前后对比。

## 判定词汇

- **拟定（Proposed）** —— 要求已规定但尚未与实现对验。
- **已验证（Verified）** —— 源码审阅与相关检查支持该要求。
- **不一致（Non-conforming）** —— 存在已知违规。
- **受阻（Blocked）** —— 验证需要不可用的代码、工具或环境。

## 本轮重测证据（2026-10-10）

环境：macOS 27.0.1（build 26A434，arm64）、`rustc 1.99.0 (b940084d7 2026-09-28)`、git HEAD `978c57ee`。

- `bash scripts/check.sh` —— 报告 `All static quality checks passed.`（`cargo fmt --check`、`cargo check --workspace`、`clippy --workspace --all-targets -- -D warnings`、`checknaming.sh`、`checksrctests.sh`、`checkthresholds.sh`、`cargo doc` 全部通过），exit 0。
- `bash scripts/checksrctests.sh` —— `inline-test ratchet: 0 file(s) still pending migration.`，exit 0；`scripts/srctestbaseline.txt` 仅含一个换行，故为**空基线**，棘轮禁令绝对。
- `cargo test --workspace --locked -- --list` —— 枚举 **736** 个测试用例（含 doctest）；该命令只编译并列测试，不运行测试体。本记录不据此重报通过/失败总数。

## 当前判定

### architecture.md

| 章节 / 主题 | 判定 | 证据 |
| --- | --- | --- |
| 架构边界 | 已验证 | 全部 19 个 `crates/*` 成员 crate 与根 `llm` package 已实现（工作区共 20 个 Cargo package，其中 19 个为成员 crate）。解析后的工作区依赖图无环（`cargo check --workspace --all-targets` exit 0；`cargo metadata` exit 0——若有环将解析失败），且 `scripts/check.sh`（fmt、`clippy --workspace --all-targets -- -D warnings`、`rustdoc -D warnings`）干净。 |
| crate 归属与依赖图 | 已验证 | 清单与 `cargo metadata` 显示生产边无环且方向正确；根 `llm` package 无生产依赖，其 dev-dependencies 覆盖全部 19 个成员 crate（仅测试用），`grep crates/*/Cargo.toml` 未发现指向 `cli`/`serve` 的产品边，`cli`/`serve` 不是任何生产计算 crate 的依赖。逐 crate 的实际生产边见 `architecture.md` §13.2。 |
| 工作区布局 | 已验证 | 根 `Cargo.toml` 是 package + workspace 根（无 lib/bin）；跨 crate 集成测试在根 `tests/`（41 个套件），基准在根 `benches/`（7 个目标）；`README.md`、`LICENSE`（`LICENSE-APACHE`、`LICENSE-MIT`）与 `clippy.toml` 位于仓库根；`criterion` 在根声明一次。`cargo bench --workspace --no-run` 构建所有基准目标。 |

### engineering.md

| 章节 / 主题 | 判定 | 证据 |
| --- | --- | --- |
| 工作流与执行协议 | 已验证 | 其余 crate（P1–P12）均按此工作流实现：阅读契约、计划、实现、执行范围化验证、记录 ADR；合并后的验证运行了 `scripts/check.sh` 与 `scripts/test.sh`。 |
| 错误处理标准 | 已验证 | 每个 crate 拥有 `thiserror` 错误枚举（如 `clierror`、`trainerror`、`serveerror`、`opterror`、`infererror`）；工作区拒绝 `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`，且 `cargo clippy --workspace --all-targets -- -D warnings` 通过，因此可恢复失败被返回而非 panic。 |
| 命名标准与机械门 | 已验证 | `bash scripts/checknaming.sh`（禁用同义词、常量/宏大小写、豁免条款检查，覆盖 `crates/*/src` 去掉 `#[cfg(test)]`）exit 0；`bash scripts/checkthresholds.sh` 对 `DS3` 与 `A4` 均 exit 0。8 个构造动词已收敛（`architecture::build`→`newmodel` 等），`nn::init` 工厂规范化为 `fill*`/`init*`，两处 `DS3` 生产违规重构到 80 行以下。`bash scripts/check.sh` 报告 "All static quality checks passed"（本轮重测，见「本轮重测证据」）。 |
| 公开 API 设计标准 | 已验证 | 每个 crate 都有公开 API；`RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` 干净，每个文档化的公开项均可解析。 |
| 测试策略与质量门 | 已验证 | 黑盒迁移完成：所有 `crates/*/src` 零内联 `#[cfg(test)]` 模块（`rg -l '#\[cfg\(test\)\]' crates/*/src` 为空；19/19 成员 crate 已迁移）。41 个根 `tests/` 套件仅调用公开 `pub` API，`benches/` 保留 7 个基准目标。`scripts/srctestbaseline.txt` 仅含一个换行（空基线），`scripts/checksrctests.sh` 的棘轮禁令绝对（本轮重测：`0 file(s) still pending migration.`，exit 0）。`bash scripts/check.sh` 干净。 |
| 黑盒契约优先测试 | 已验证 | 决策、门与诊断缝均就位。19 个成员 crate 已迁移（0 `#[cfg(test)]`，基线为空），由 `scripts/migrateinline.py`、`scripts/migratecrates.py`、`scripts/finalizecrate.py` 驱动。缝在 `infer`、`train`、`checkpoint`、`serve`、`cli` 中接线，`EnvFilter` 订阅者由 `cli::runcli` 与 `httpserver::serve`/`serveone` 安装；`tests/diagnostics.rs` 证明无需重编的运行期切换（同一 `loadcheckpoint` 调用在 `DEBUG` 订阅者下发出 "checkpoint loaded"，在 `WARN` 下抑制）。 |
| 集成测试与基准整合 | 已验证 | 每个 crate 局部集成套件移到仓库根 `tests/`（14 个移动文件，合并 `common/`），`optim` 基准移到根 `benches/`；根 `llm` package 声明仅测试的 dev-dependencies。`scripts/check.sh` 与 `scripts/test.sh` 通过（本轮重测 `check.sh` exit 0；通过/失败汇总未在本轮重跑），`cargo bench --workspace --no-run` 构建 7 个目标。其「crate 局部覆盖现仅内联单元测试」的后果已被黑盒测试决策取代。 |
| 性能工程标准 | 拟定 | 每个被基准的 crate 在仓库根有可见基线：`benches/{cpuops,kernels,metadata,autograd,nn,models,optim}.rs`，共享工作负载在 `crates/bench`。尚未执行任何优化或前后对比，故不声称改进。 |

### design.md

| 章节 / 主题 | 判定 | 证据 |
| --- | --- | --- |
| 层 0：lib / tensor / api | 已验证 | `lib`、`tensor`、`api` 已按文档化契约实现；根 `tensorcontract`、`apicontract`、`crosscrate` 套件通过。 |
| 层 0：张量布局契约 | 拟定 | 该契约首次成文于设计文档（不再是指向其它文档的占位），依据是对 `crates/tensor/src` 的源码核对；尚未作为独立规范单独验证。 |
| 层 0：api 计算契约扩展 | 已验证 | `api` 暴露冻结的 `kernel<T>` 以及增量的 `binarykernel`/`weightedkernel`/`gatherkernel`/`compute` 契约；`crates/cpu` 实现它们，根 `cpucontract` 套件从 crate 外部触发它们。 |
| 层 1：cpu | 已验证 | `crates/cpu` 实现 `backend`、`executor` 与算子模块（`matmul`、`attention`、`embedding`、`normalization`、`rotary`、`softmax`、`activation`、`elementwise`）；单元测试现由仓库根 `tests/cpuunits.rs` 经 `pub` 接口触发（`crates/cpu/src` 零内联 `#[cfg(test)]`），契约测试在根 `tests/cpucontract.rs`，`scripts/check.sh` 干净。基准在仓库根：`benches/cpuops.rs`（执行器级）、`benches/kernels.rs`（原始算子）、`benches/metadata.rs`（张量元数据）。 |
| 层 1：存储 dtype 与计算 precision | 已验证 | `dtype::f64` 与 `impl element for f64` 存在且 `sizeinbytes() == 8`；根 `tests/tensorunits.rs` 与 `tests/tensorcontract.rs` 覆盖 element/dtype 宽度不变量与 `f64` 视图往返；`clippy -D warnings` 与 `cargo doc -D warnings` 干净。 |
| 层 1：autograd | 已验证 | `crates/autograd` 实现图、变量与所支持算子的 VJP；根 `autograd` 套件对每个算子用有限差分参考检查，`cargo bench --no-run` 构建 `benches/autograd.rs`。 |
| 层 2：nn | 已验证 | `crates/nn` 实现参数、模块与层集；根 `nnlayers` 套件用 `testkit` 的独立标量参考检查每一层。 |
| 层 2：models（模型支持契约） | 已验证 | `crates/models` 实现因果 LM 契约与 Llama config/weights/block/model；根 `modelforward` 套件钉住确定性前向 logits。模型支持契约首次成文于设计文档，依据是对 `crates/models/src` 的源码核对。 |
| 层 2：tokenizer | 已验证 | `crates/tokenizer` 实现词表、合并、特殊词元、对话模板、编码/解码与引擎；根 `tokens` 套件通过。 |
| 层 2：data | 已验证 | `crates/data` 实现记录、数据集、打包、批次、采样器与加载器；根 `pipeline` 套件端到端组合它们。 |
| 层 2：format 与 checkpoint（检查点格式契约） | 已验证 | `crates/format` 与 `crates/checkpoint` 实现解析/校验与原子 save/load/兼容性；根 `artifact` 与 `roundtrip` 套件通过。检查点格式契约首次成文于设计文档，依据是对两个 crate 源码的核对。 |
| 层 3：optim | 已验证 | `crates/optim` 在拥有的张量上实现 AdamW、动量 SGD、梯度裁剪与 LR 调度；根 `optim` 套件通过，整合的 `benches/optim.rs` 可构建。 |
| 层 3：context | 已验证 | `crates/context` 实现取消、预算、指标、执行上下文与 worker；根 `context` 套件通过。 |
| 层 3：train（训练契约） | 已验证 | `crates/train` 实现配置、损失、梯度累积、步进、评估与恢复；根 `lossdecrease`、`trainingresume`、`cancellation` 套件通过。经 `nn`/`models` 的反向是声明的限制。训练契约首次成文于设计文档，依据是对 `crates/train/src` 的源码核对。 |
| 层 3：infer（推理契约） | 已验证 | `crates/infer` 实现引擎、请求/生成状态、prefill/decode、KV cache、采样、停止与流式；根 `generation`/`isolation` 套件通过。推理契约首次成文于设计文档，依据是对 `crates/infer/src` 的源码核对。 |
| 层 4：cli | 已验证 | `crates/cli` 实现解析、分发与中央退出码表；根 `cli` 套件覆盖解析、错误映射与退出码，`cli train` 对声明的限制返回文档化退出码 4。 |
| 层 4：serve | 已验证 | `crates/serve` 实现 std-only HTTP/1.1 适配器、路由、准入控制与 deadline→cancellation 映射；根 `service` 与 `server` 套件（含真实 loopback socket）通过。 |
| 基础设施：testkit、bench | 已验证 | 两者均为依赖叶子且非生产依赖；`testkit` 仅提供夹具/断言/参考实现/微型模型，`bench` 提供共享确定性工作负载；根基准目标消费它们并成功构建。 |

### conformance.md（本记录）

| 主题 | 判定 | 证据 |
| --- | --- | --- |
| 一致性记录本身 | 拟定 | 判定随 2026-10-10 文档合并由 33 份文档映射到 4 份文档；判定与证据沿用合并前已执行的检查，并由 2026-10-10 对抗性修正补记静态门重测（见「本轮重测证据」）。每当某文档的主题被实现、测量或发现违反其规则时，本记录 MUST 更新。 |

本记录 MUST 在文档主题被实现、测量或发现违反其自身规则时更新。
