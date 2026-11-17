# 机制代码口径分账基线（M1-C1）

> 状态：M1-C1 基线快照，2026-11-17 工作树。分层与排除规则沿用
> [`llmu-parity-optimization-design.md`](llmu-parity-optimization-design.md) §5.3、
> [`ARCHITECTURE.md`](ARCHITECTURE.md) §一。本口径**只用于机制代码密度对照**，不得与
> 含测试的粗口径行数混算。

## 1. 两条口径不可混用

| 口径 | 数值 | 含义 | 用途 |
|---|---|---|---|
| 含测试粗口径 | 76,839（收口复核实测，2026-11-17 工作树） | 全仓 Rust 总行数，含内联/外部测试、注释、工具成员 | 规模量级参照，**不作密度倍数分母** |
| 机制代码口径（本文件） | 见 §3 表 | 排除测试、工具成员、构建脚本、生成片段与 vendor 后的生产机制 Rust | 与 burn / candle / cubecl 等同口径机制行对照 |

密度倍数（"约 1/15"）只在两条口径各自内部计算；不接受用总行数或含测试行数凑倍数。

## 2. 排除规则（与 `scripts/mechanism_lines.sh` 一一对应）

统计范围：`crates/*/src/**/*.rs` 与根包 `src/**/*.rs`。以下不计入机制分母：

1. **成员级测试目录**：任意路径段为 `tests/` 的文件（集成测试；根包 `tests/` 同属此类）。
2. **工具成员整体**：`annotation`（编译期注解）、`fuzz`（模糊测试契约）、`benches`（集中测量容器）。它们不参与运行时分层，也不进机制行数分母（ARCHITECTURE §一表注）。
3. **构建脚本**：crate 根 `build.rs` 不在 `src/` 内，扫描目录天然排除；`src/` 下名为 `build.rs` 的模块是正常机制文件，照常计入。
4. **生成片段与 vendor**：`target/` 不进树；仓内无 `vendor/`；`ops/*.toml`、`devices/*.toml`、`lints/vocabulary.toml` 是编译期语料，不是 Rust 机制行。
5. **内联 `#[cfg(test)]` 模块**：目录级脚本尚未逐行扣除；严格口径由 `tool::strip_tests`（基于词法树剔除测试条目、保留行号）在审计/门禁侧再扣一次。本快照表注此偏差。
6. **注释与空行**：当前计数保留注释行与空行；tokei Code 列式的"去注释去空行"细粒度扣除留给后续审计器，不在本脚本内二次裁剪。

## 3. 分层归属与基线表

分层沿用 ARCHITECTURE §一 / design §3.1：

| 层 | 成员 | 文件 | 机制行（目录级） |
|---|---|---:|---:|
| L1 基础设施 | `diagnostic` `testkit` `dtype` `wire` `tool` | 28 | 5,564 |
| L2 数据层 | `prof` `op` `prim` | 55 | 10,686 |
| L3 内核 | `cutile` | 25 | 7,090 |
| L4 运行时 | `tensor` `storage` `tokenizer` | 23 | 3,068 |
| L5 模型与训练 | `weights` `autograd` `data` `nn` `shard` | 53 | 11,353 |
| L6 调度与服务 | `train` `engine` `serve` `backend` `cli` | 42 | 11,041 |
| L7 入口 | `llm` bin（`src/`，含 `src/bin/cutile/`） | 7 | 1,911 |
| **合计** | | **233** | **50,713** |

> 50,713 为目录级口径（含内联测试体、注释与空行）；扣除内联测试体后的严格机制行以
> `tool::strip_tests` 跑出的结果为准，回填时在本表加一列"严格机制行"。工具成员
> （`annotation` 1,641 / `fuzz` 849 / `benches` 测量容器）未计入上表。

## 4. 可复跑

```bash
scripts/mechanism_lines.sh          # 分层表 + 合计（人读）
scripts/mechanism_lines.sh --csv    # 机器可读 CSV：layer,files,lines
```

- 脚本不读 `target/`、不解析 Cargo 元数据；成员→层映射硬编码在 `layer_index`，新增成员时必须先在 ARCHITECTURE 分层表落位，再同步脚本。
- 口径变更（如接入 `strip_tests` 严格扣除、去注释空行）必须同时改本文件 §2 与脚本注释，避免规则与计数漂移。
- 对标框架分母按同一份排除规则采集；C/C++ 框架（llama.cpp）只在"机制等价职责"口径上对照，不拿全文件行数直接除。

## 5. 未决

- 内联 `#[cfg(test)]` 体的逐行扣除尚未接进脚本（依赖 `tool::strip_tests` 作为可调用计数入口）；当前表为目录级上界。
- 各框架同口径机制行分母未采集；M3/M4 出口在同口径下回填实际倍数，未重测前不写倍数结论。
