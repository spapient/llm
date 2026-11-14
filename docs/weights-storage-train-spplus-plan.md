# weights / storage / train 升 S++：基线证据、差距清单与落地切片

> 测量口径：`cargo check`、`cargo clippy --all-features -- -D warnings`、`cargo test`、
> `cargo llvm-cov`、`tokei`、`tool` 静态门（`crates/tool/tests/gate.rs`）、`cargo deny`。
> 事实回链到当前工作树；未读到的行为标 `unknown`，不进无条件结论。

## 1. 基线（2026-11-14 工作树，实测）

| 工具 | 结果 | 证据 |
|---|---|---|
| 构建 / clippy 严格 | 干净，0 告警 0 错误 | `cargo check -p llm-weights -p llm-storage -p llm-train` 退出 0；`clippy -D warnings` 无 warning/error |
| 测试 | 239 通过，0 失败 | weights 171 单测 + round_trip 10；train 50；storage 8 |
| 行覆盖（llvm-cov） | 行 87.15% / region 85.55% / 函数 83.28% | 三 crate 合并 `--summary-only` |
| 代码量（tokei） | 40 文件，11 330 代码 / 1 615 注释 / 1 589 空行 | 三 crate `src` |
| 静态门 | `ccn ≤ 5`、`nloc ≤ 60`、文件 ≤ 899、生产零 `unwrap/expect/panic`、`debt` 空表 | `lints/vocabulary.toml` `[gate]`；`gate` 测试绿 |
| 依赖 | weights 运行时零第三方；storage/train 同 | 各 `Cargo.toml` |

覆盖薄弱文件（行覆盖，需补）：`zip.rs` 39.66%、`crc.rs` 60.66%、`seg.rs` 84.97%、
`reader.rs` 83.04%、`json.rs` 86.40%、`safetensors.rs` 88.68%。

体量失衡：超大 `snapshot/mod.rs` 893、`gguf.rs` 892、`dsa.rs` 873、`safetensors.rs` 847、
`store/mod.rs` 786、`bin.rs` 695；近空 `dtype.rs` 7、`testutil.rs` 22、`codec.rs` 34、`crc.rs` 89。

## 2. 外部生产消费面（精确到 `weights::` 限定路径）

crate 外真实引用：`Header` / `Entry` / `Reader` / `Loader`、`safetensors::{Archive,File}`、
`json`（fuzz 边界）、`crc`、`Archive`（storage 文档）。其余容器内部件只被 weights 自身与
`round_trip` 集成测试使用。

## 3. 六维度差距与证据

### 3.1 类聚与模块职责

- **已成立**：`Entry/Header/Reader/Loader` 契约自洽；`storage` 只做物理布局；`train` 只记恢复事实。
- **差距 1（真重复，跨 crate）**：`train::snapshot` 自造了第二套 `Sink`（Count/Vec/Slice/File/Sealed）、
  `Cursor`、LEB128 编解码与非规范拒绝判定，而 `weights::sink/leb` 已有同一套，只是 `pub(crate)`
  借不出来。审计 §P2-7 已记账。
- **差距 2（错层）**：纯字节码基础件（CRC/LEB/Sink）位于 L5 权重层；L6 为取 CRC 向上借权重依赖。
- **差距 3（扁平混杂）**：weights 34 个文件平铺六类关注——契约、字节码、LLPK 容器、外部格式适配、
  量化族、后量子证明；lib 文档有分组但无边界强制。
- **处置**：ADR 0024 抽 L1 `wire`；weights 内部按六类在 lib.rs 注释与模块顺序上归组（不新增 src 子目录）。

### 3.2 证据、判据纪律

- **已成立**：头部重名/重叠/越界、布局字节数、快照条目表逐条比对、非规范 LEB 拒绝、CRC 对拍均有负例。
- **差距**：快照缺字节级黄金向量（audit §11.4 待办）；薄弱覆盖文件里的错误分支少有人钉。
- **已修**：`store` 三处 `checked_div(..).unwrap_or(0)` 改为 `batches()` 显式校验——行宽为 0 或不能整除返回 `Code::Invalid`，不再把维度错误静默成 0。
- **处置**：补黄金向量；错误分支补负例。

### 3.3 静态强制门禁

- **已成立**：ccn/nloc/文件行数/禁用形态/空 debt 由 `tool` 构建期门强制；架构成员/边/方向/所有权
  由 `tests/architecture.rs` 与 cargo metadata 双向比对。
- **差距**：新增 `wire` 成员必须同步层表、边集（75→78，`+dev` 仍 11）、所有权基线；覆盖率与重复率
  目前不在强制门内。
- **处置**：随 S1 更新三处文档；评估把行覆盖下限与 jscpd 重复率纳入门内阈值。

### 3.4 测试质量

- **基线**：行 87.15%。目标：核心容器/编解码路径 ≥ 95% 行，薄弱文件先补到 ≥ 90%。
- **处置**：薄弱文件补错误分支与边界用例；快照往返加黄金向量；容器往返保留现有 10 条端到端用例。

### 3.5 性能工程

- **已成立**：计数 Sink 与写出共用同一编码路径，长度零漂移；CRC 八路查表；rANS 8-lane。
- **差距**：仓库集中测量容器 `benches` 尚未覆盖容器编解码、`storage::remap`、CRC 分段喂入；
  部分编码路径存在可消除的重复分配。
- **处置**：在 `benches` 补 criterion 基准（预热/采样沿用其既有配置）；热路径去冗余分配；不换算法，
  压缩率受熵约束只承诺不劣于。

### 3.6 广度与生产成熟度

- **已成立**：零第三方运行时依赖、`deny(unsafe_code)`（mmap 为唯一 feature 隔离岛）、错误边界统一
  `diagnostic::Error`；safetensors 整缓冲/头部两种后端；原子落盘 + 临时文件清理。
- **差距**：适配格式覆盖有限；文档与代码漂移条目需收口；fuzz 只接了 json/safetensors。
- **处置**：随能力演进补适配；逐条清文档漂移；不引入新运行时依赖。

## 4. 裁剪纪律（重要）

`docs/weight-container-optimal-plan.md` 第 179 行明确"不因研究标签放弃格量化、五轴选码、谱隙账本、
图拓扑"——`lattice/axes/ledger/ir/scale/svd/ntt/keccak/golay/dsa/envelope/patch/mode/zip` 是
已完成的 P4/P5 与 D1–D5 能力，**不作为死代码删除**。删除只针对编译器 + 测试双重证实无生产消费者、
且不在方案能力表内的件；每删一处保留可回退记录。

## 5. 落地切片（每片结束保持全绿、可单独回退）

| 切片 | 内容 | 验证门 |
|---|---|---|
| S1 | **已落地**：抽 `wire` L1（crc/leb/sink）；weights 薄转发；snapshot 删重复 `Sink/Cursor/varint` 改复用；同步架构表/边/所有权基线；移除 weights 孤儿 dev 依赖 testkit | 239 用例不丢（wire 11 + weights 160 + 往返 10 + train 50 + storage 8）；clippy 0；架构门 8 绿；静态门 36 绿；deny 干净；行覆盖 87.09%（基线 87.15%，持平） |
| S1b | **已落地**：wire LEB 全泛型收口——写端标记族 `Uint`（u8/u16/u32/u64/usize，`as_u64()` 单点归约）+ 单一泛型 `write<N: Uint>`；读端私有 `raw()` 累积引擎 + 单一泛型 `read<N: TryFrom<u64>>`；删除 `write_u64/write_u32/read_u64/read_u32/read_varint` 全部宽度包装；LEB 定长 10 字节栈缓冲一次 `put`；CRC 八路查表；热函数补 `#[inline]`；循环仅限串行依赖链并注释保留理由；`benches/wire_codec` criterion 基准带 `Throughput`（Bytes/Elements）出吞吐 | wire 12 用例不丢；clippy 0；架构门 8 绿；基准（release，100 样本/5s，中位数[95%CI]）：CRC 4KiB 1.526µs（2.48 GiB/s）、编码 8 值 3.13ns（2.52 Gelem/s）、解码 22.26ns（355 Melem/s，较原始 35.8ns 降约 38%）、密封 1KiB 455.5ns（2.07 GiB/s）；热路径无堆分配 |
| S2 | 模块体量再平衡：近空件并入同族，超大件按职责拆分；不新增 src 子目录 | 文件行数 ≤ 899、ccn/nloc 门绿。切片 S2a 已落地：`dtype.rs`（Wire 别名垫片）并入 lib 再导出，`codec.rs`（Codec 闭集）并入 `seg.rs`，weights 文件 30 → 28，160 用例不丢、clippy/架构门/契约/静态门/deny 全绿 |
| S3 | 测试与覆盖加固：薄弱文件、快照黄金向量、错误分支负例 | 行覆盖不低于基线并向 ≥95% 收敛 |
| S4 | 性能：benches 补容器/remap/CRC 基准，热路径去冗余分配 | criterion 出数；行为用例不变 |
| S5 | 成熟度收口：文档漂移、所有权/词汇基线复核、fuzz 边界补齐 | 架构门、词汇门、deny 全绿 |

## 6. 不承诺项（避免把度量写成宣传）

- 压缩率受香农熵约束，只承诺不劣于当前；CRC、mmap、原子写是 OS/硬件语义，不存在"指数级"空间。
- 覆盖阈值、基准采样参数在 S3/S4 落地时按实测标定，先用基线值占位。
