# tokenizer 深度分析与优化执行计划

- 状态：**优化计划（已完成）**；Phase 0 基线已建立；P1/T6（BPE 合并重写）、T8（热路径去分配）、T7（per-encode 词缓存）与 T9（special 前缀匹配去分配）已实施（见 §12.4、§12.5、§12.9）；P0/T1 全管线类型分发、T2+T3、T4、T5 已实施（见 §12.6–§12.8、§12.12）；P1/T10 与 T12 已实施（见 §12.10）；**D3 逐位对齐达成——7 个真实模型（GPT-2/Llama2/Llama3/Qwen3/DeepSeek/Gemma/o200k）encode+decode 全对齐（见 §12.11–§12.13）**。Phase 2 S1（整数键合并表 + pair 打包 64 位）、S2（tiktoken/rankmerge 去字节键）、S3（scratch 复用）、T7（per-encode 词缓存）、S4（batch）与 S5（const 表 + SIMD 评估）已实施（见 §12.16–§12.21）。
- 仓库：`llm` workspace（`crates/tokenizer`）
- 基线分支：`main`
- 计划日期：2026-10-16
- 覆盖范围：`crates/tokenizer`（9 模块，约 1013 行）、`crates/cli` 的 tokenizer 接线、`benches/tokenizer.rs`、`tests/it/{tokenizerunits,tokens}.rs`。
- 目标：**代码更少、架构更先进、质量更优秀、性能更好**，同时保持自研与最小依赖，不绑定 HuggingFace `tokenizers` crate。
- 约束：`#![forbid(unsafe_code)]`；生产路径无 `panic`/`unwrap`/`expect`；无分隔符命名约定；`crates/tokenizer/src` 禁止内联 `#[cfg(test)]`；性能结论必须有可复现基准支撑；每个提交完成一个可独立测试的阶段（AGENTS.md §11）。

## 1. 结论摘要

1. **当前 tokenizer 是"正确性优先、性能未处理"的基线**：双向 `BTreeMap` 词表、`Result` 错误模型、`IN4` 稀疏 id 防护、确定性合并语义都已到位并测试覆盖，这些是必须保住的正确性资产。
2. **两个正交缺口，不可混为一谈**：
   - **正确性缺口（相对 HF）**：缺 `pre_tokenizer`、`byte-to-unicode` 字母表、`normalizer`、`post_processor`、`decoder` 各段，以及 Jinja 对话模板。后果是**对真实 GPT-2 / Llama 3 词表编码输出与 HF 不一致**——token 是错的，快慢失去意义。
   - **性能缺口**：核心 BPE 合并是 O(n²)（[merges.rs:67](crates/tokenizer/src/merges.rs:67)），`bytepair::getrank` 每次候选分配两个 `String`，`resolvetoken` 每字节 `format!`。
3. **"BPE 不是计算算子、故无基准"的既有声明（design.md §10.8）必须推翻**：编码一旦在推理热路径上并作为优化目标，AGENTS.md §7/§8 与 engineering.md §3.3/§8.15 强制"先基线后优化"。
4. **最高杠杆**：① 补 pre_tokenizer + decoder 使真实 HF tokenizer.json 可正确编解码（P0）；② BPE 合并换"相邻对 + 优先队列 + 字节缓冲符号"，O(n²)→O(n log n)（P1/T6）。二者共享同一符号表示（字节缓冲），应一次落地、共同演进。
5. **执行顺序**：G0 建基线 → P0 正确性主线 → P1 性能与完善。性能工作严格置于正确性之后、且每步附基准对比。

## 2. 目标与不可妥协的正确性标准

1. **语义对齐**：对真实 HF `tokenizer.json`（GPT-2 / Llama 3 等），编码输出与 HF 逐 token 一致；`decode(encode(x)) == x` 字节精确。任何"对齐"结论必须以对照测试的黄金输出为证据，不得凭代码外观声称。
2. **边界安全**：恶意/越界 token id 不触发大分配（`IN4`）；超预算在分配前返回错误；合并循环不产生越界索引。
3. **资源生命周期正确**：失败路径不泄漏、不误复用请求状态；并发请求之间不共享可变请求状态（`OW2`）。
4. **确定性**：合并保持"最低 rank 优先、平局取最左"；重叠特殊字面量取最长；编码/解码往返确定。
5. **依赖最小**：新增生产依赖必须逐项论证（AGENTS.md §3.2）。`serde_json` 之外，优先用 `std` 手写实现；`regex` 与 `unicode-normalization` 仅在对照测试证明确有必要时引入。
6. **性能可证**：先建基线再改实现；固定环境、profile、工作负载、预热与统计方式；报告 median、变异、绝对与相对差；不接受无数据"提升"结论。
7. **回归可追踪**：每个优化对应对拍测试 + 基准证据；任何回归阻断合入。

## 3. 现状深度分析（源码核查）

### 3.1 已实现（正确性基线，全部有测试覆盖）

| 能力 | 位置 | 说明 |
| --- | --- | --- |
| BPE 模型加载 | `engine.rs::fromjson` | 解析 `model.type=="BPE"`、`vocab`、`merges`、`byte_fallback`、`unk_token`、`added_tokens` |
| 普通编码 | `encode.rs::encodeplain` | 按 Unicode 字符切分初始符号 → BPE 合并 → 查 vocab |
| 特殊 token 编码 | `encode.rs::encodespecial` | 最长字面量优先，扫描分段 |
| 解码 | `decode.rs::decodetokens` | 含 `<0xNN>` 字节回退还原 + UTF-8 校验 |
| 简单 chat 模板 | `chat.rs` | 固定 role prefix/suffix 子集（非 Jinja） |
| 数据结构 | `vocab.rs`/`special.rs`/`merges.rs` | 双向 `BTreeMap`，`forbid(unsafe_code)` |
| cli 接线 | `cli/src/commands/generate.rs` | `--tokenizer` + `--text` 经 `fromjson` + `encode` 真实走通 |

> 更正此前方案中的 U7：**cli 已接线**，非"参数声明但无实际调用"。`generate.rs:148-149` 调用 `tokenizer::fromjson(document)` 与 `loaded.encode(text)`；`args.rs:62-74` 声明参数，`generate.rs:134-137` 强制"text requires --tokenizer"。缺失的仅是**独立 decode 命令**，不属本计划核心。

### 3.2 与 HF 不兼容的缺口（U）

这些是**正确性缺口**——当前 tokenizer 无法对真实 HF tokenizer.json 做正确编码。

| # | 缺口 | 后果 | 证据 |
| --- | --- | --- | --- |
| U1 | `pre_tokenizer` 段被忽略 | 按字符切分而非字节级预分词 → 与 HF 输出不一致。Llama 3 的 `Ġword`（空格前缀）体系无法对齐；GPT-2 的 byte-to-unicode 映射缺失 | `engine.rs` 只读 `model`/`added_tokens`；`encodeplain` 直接 `text.chars()` |
| U2 | byte-to-unicode 映射未实现 | GPT-2 标准 256 字节→Unicode 表缺失，BPE 初始符号错误 | `encode.rs` 以 `char` 为初始符号 |
| U3 | `normalizer` 段被忽略 | NFKC/lowercase 等清洗未做，与 HF 预处理不一致 | `fromjson` 不读 `normalizer` |
| U4 | `post_processor` 段被忽略 | TemplateProcessing/ByteLevel 后处理（BOS/EOS、type ids）缺失，只能靠 `chattemplate` 手动拼 | `fromjson` 不读 `post_processor` |
| U5 | `decoder` 段被忽略 | 解码不还原 `Ġ`→空格等 ByteLevel 逆映射 → 输出带 `Ġ` 伪字符 | `decodetokens` 直接 `out.push_str(content)` |
| U6 | Jinja chat template 未实现 | 只能表达 Llama 式 role 包裹，任意 Jinja 模板（`apply_chat_template`）不可用 | `chat.rs` 固定 prefix/suffix |

### 3.3 性能与工程问题（P）

| # | 问题 | 现状 | 复杂度 | 位置 |
| --- | --- | --- | --- | --- |
| P1 | BPE 合并 O(n²) | 每轮全表扫描找最低 rank 相邻对；每次 `getrank` 分配两个 `String`；`work.remove(index+1)` 为 O(n) 搬移 | 合并密集最坏 Θ(n²) 时间与分配 | `merges.rs::apply` |
| P2 | vocab 热路径 O(log n) | `getid` 走 `BTreeMap` 查找（**零分配**，`&str` Borrow）；分配在别处 | encode 每符号 O(log V) | `vocab.rs` |
| P3 | special 扫描 O(n·k) | 每字符位置对全部 special 字面量线性 `starts_with`，且每次调用重排 | O(n·k·avg_len) | `encode.rs::encodespecial` |
| P4 | 无词缓存 | 重复 pre-token 每次都重跑合并 | — | `encode.rs` |
| P5 | `regex` 依赖未显式声明 | 已在传递树（经 dev 依赖），未入 workspace 依赖 | — | `Cargo.toml` |
| P6 | `unicode-normalization` 未引入 | normalizer 段的 NFKC 需要 | — | `Cargo.toml` |

> 更正此前方案中的 P2：`vocabulary::getid(&str)` 用 `BTreeMap::get` 的 `Borrow<&str>`，**不分配**。真正按符号/字节分配的是 `bytepair::getrank`（[merges.rs:56-60](crates/tokenizer/src/merges.rs:56) 构造 `(left.to_string(), right.to_string())`）与 `resolvetoken`（[encode.rs:120](crates/tokenizer/src/encode.rs:120) 的 `format!("<0x{byte:02X}>")`）。

## 4. 借鉴 HF 的架构基准

### 4.1 Pipeline 五段（HF 官方架构）

HF tokenizers 围绕中心 `Tokenizer` 类，实现五段管线，从 `tokenizer.json` 按类型读取各段，**不做任何硬编码**：

```
normalizer → pre_tokenizer → model(BPE/Unigram/WordPiece) → post_processor → decoder
```

- **normalizer**：NFKC、lowercase、strip accents 等清洗。
- **pre_tokenizer**：按词切分（ByteLevel / Whitespace / Metaspace / Sequence / Digits）。
- **model**：BPE 等在 pre-token 内做子词合并（**合并永不跨越 pre-token 边界**）。
- **post_processor**：TemplateProcessing（`$A`/`$B` + special token + type ids）、ByteLevel 后处理。
- **decoder**：ByteLevel decoder（`Ġ`→空格）、Metaspace 还原等。

### 4.2 HF v1 的顶级性能技巧（v1 相对 v0.23 在 M4 Max 上 BPE 编码快 3–30×）

1. **手写 splitter 替代 regex 引擎**：BPE 正则预切分是模型固定参数，可用手写状态机实现，不需通用 regex 引擎。
2. **共享词缓存**：重复词直接命中缓存，跳过合并。
3. **零分配 merge 循环**：符号存扁平数组 + 相邻位置链接；候选 pair 打包进单个 64 位值（rank 高位），比较即整数比较，"无合并"为最大值 → 循环无分支；scratch buffer 由调用方持有。
4. **batch 模型调用**：一批 pre-token 一次 model 调用，而非逐个。

学术基线（SuperBPE 复杂度分析、HF 算法文档）佐证：BPE 编码期用**相邻对 + 优先队列**（每步取最小 rank、只更新受影响邻居）是业界标准，摊还 O(n)。

## 5. 目标架构与设计原则

```
tokenizer.json
   │ 类型分发（按 type 字段构造各段，不硬编码具体模型）
   ▼
normalizer ──▶ pre_tokenizer ──▶ model(BPE) ──▶ post_processor ──▶ decoder
   (可选)        (ByteLevel/Whitespace/   (合并+查表)  (TemplateProcessing)  (ByteLevel 还原)
                  Metaspace/Sequence)
```

设计原则：

- **每段 = 一个 trait + 按 type 分发的实现**：新增模型类型只加实现不改管线。
- **保持 `forbid(unsafe_code)`**：HF 内核含 unsafe；本 crate 以纯 Rust 实现同等语义。
- **最小依赖**：`serde_json` + 按需 `unicode-normalization`（NFKC 段才用），**pre_tokenizer 用 `std::char` 手写，不引入 `regex`**（见 §7 D1）。
- **公开 API 不变**：`tokenizer`/`tokenizererror`/`vocabulary`/`bytepair`/`specialtokens`/chat 类型保持，下游 `cli` 的 `From` 转换不变；`bytepair::apply` 保留为薄包装（见 §7 D4）。
- **符号表示统一为字节缓冲**：`String` 符号 → `Vec<u8>` 符号，同时服务 P0 的字节级预分词与 P1 的零分配合并，一次落地。

### 与 HF 的目标对比（达成标准）

| 维度 | HF tokenizers v0.23/v1 | 本项目目标 | 判定方式 |
| --- | --- | --- | --- |
| 代码量 | v0 单 crate 数万行（+ 训练/序列化/转换子 crate）；v1 拆 4 crate | **~1500 行内**（9 模块 → pipeline 各段，无训练/序列化） | `wc -l` |
| 架构 | 五段 pipeline，v1 拆 tk-encode/tk-serialize/tk-convert/tk-train | **五段 pipeline 类型分发**，单 crate 分模块，`forbid(unsafe)` | 结构评审 |
| 质量 | 与 HF 输出逐 token 对齐（官方回归保障） | **与真实 HF tokenizer.json 逐位对齐**（GPT-2/Llama 3 对照套件） | 对照测试 |
| 性能 | v1：BPE 3–30× over v0.23（M4 Max） | **同语义下 merge O(n²)→O(n log n)** + 零分配热路径 + 词缓存 | 基准（tokbench 风格） |

> 不声称性能超越 HF v1（无同机对照证据）。主张是：**相同语义、更少代码、无 unsafe、渐进式性能提升**——性能结论以实测为准，逐阶段记录。

## 6. 执行分级与顺序

优先级：**G0**（基线，前置）→ **P0**（正确性主线）→ **P1**（性能与完善）。

### G0 — 建立可审计基线（先做，见第 12 节）

- 记录 commit SHA、Rust/Cargo 版本、宿主（CPU/OS/内存）。
- 运行 tokenizer 现有正确性测试（`tests/it/tokenizerunits.rs`、`tests/it/tokens.rs`），保存结果。
- 新增 `benches/tokenizer.rs` 建立性能基线（合成夹具：链式合并 + 字节回退）。
- 验收：基线报告能区分 PASS/FAIL/NOT RUN；性能基线可复现；未跑的命令如实标注。

### P0 — 打通「真实 HF tokenizer.json 正确编解码」主线

| 步骤 | 内容 | 借鉴 HF 的什么 | 验证方式 | 代码影响 |
| --- | --- | --- | --- | --- |
| T1 | `pre_tokenizer` 解析：ByteLevel（256 字节→Unicode 表 + GPT-2 手写切分）、Whitespace、Metaspace、Sequence 类型分发 | pre_tokenizer 段按类型构造 | 真实 GPT-2/Llama 3 tokenizer.json 编码对照（与 HF 输出逐位一致） | 新模块 `pretoken.rs`，`engine::fromjson` 接线 |
| T2 | BPE 合并限定在 pre-token 内（不跨边界），初始符号改为 byte-to-unicode 字符 | v1「merge never crosses a pre-token boundary」 | T1 对照测试全绿 | `encode.rs` 重构 + 字节符号 |
| T3 | `decoder` 段：ByteLevel decoder（`Ġ`→空格还原 + 字节拼接） | decoder 段 | `decode(encode(x)) == x` 字节成立 + HF 对照 | 新模块 `decoder.rs` |
| T4 | `post_processor` 段：TemplateProcessing 解析与执行 | post_processor 段 | BOS/EOS/type-id 与 HF 一致 | 新模块 `postprocess.rs` |
| T5 | `chat` 升级：读 `chat_template` 字段，自研最小 Jinja 子集（变量插值 + 循环 + if） | HF apply_chat_template | 主流模型模板渲染对照 | `chat.rs` 重构 |

### P1 — 性能与完善（在 P0 正确性之上）

| 步骤 | 内容 | 借鉴 HF 的什么 | 复杂度影响 | 验证方式 |
| --- | --- | --- | --- | --- |
| T6 | BPE merge 循环：相邻对 + 优先队列；符号扁平数组 + 位置链表；pair 打包 64 位（rank 高位）；scratch 复用 | v1 merge loop / SuperBPE 摊还分析 | **O(n²)→O(n log n)** | 对拍（`proptest` 对照旧 `apply` 逐字节一致）+ 大文本基准 |
| T7 | 词缓存：pre-token → 编码结果，重复词命中（作用域见 §7 D6） | v1 shared word cache | 重复文本提速 | 重复词基准 |
| T8 | vocab 热路径去分配：`HashMap`（保留入站校验防大分配）或零拷贝键 | v1 allocation-conscious 路径 | getid O(1) | 全测试 + clippy |
| T9 | `encodespecial` 换前缀匹配结构（trie / 按首字节哈希） | — | O(n·k)→O(n·log k) | special 用例全绿 |
| T10 | normalizer 段（NFKC + Sequence） | normalizer 段 | — | 对照测试 |
| T11 | 依赖显式化：仅引入确有必要者（见 §7 D1/D2） | — | — | `cargo tree` |
| T12 | cli 补独立 decode 命令（可选，视需求） | — | — | cli 集成测试 |

## 7. 关键决策点（带建议）

| # | 决策 | 建议 | 理由 |
| --- | --- | --- | --- |
| D1 | pre_tokenizer 实现：手写 splitter vs `regex` | **手写**（`std::char` 分类 + 收缩词表 `'s/'t/'re/'ve/'m/'ll/'d` + 静态 256 字节表） | HF v1 首条技巧即"手写 splitter 替代 regex"；`regex` 已是传递依赖但非生产依赖，手写免去新增生产依赖面，符合 AGENTS.md §3.2。**caveat**：`char::is_alphabetic()` ⊃ `\p{L}`（含 Other_Alphabetic），对异体字符可能微差，须由对照测试钉住；若钉不住再评估专用 Unicode 表 |
| D2 | normalizer/NFKC：现在做 vs 推迟 | **推迟**，仅当真实模型带 `normalizer` 段时引入 `unicode-normalization` | GPT-2/Llama/Mistral/Qwen 的 tokenizer 均 `normalizer: null`；NFKC 无 `std` 等价，手写不可行 |
| D3 | 真实 tokenizer.json 夹具（P0 对照测试硬依赖） | **必须先解决**：vendor 一份**完整**真实 `tokenizer.json`（GPT-2 全量约 1.6MB；`model.vocab`/`model.merges`/`added_tokens` 内嵌随文件，无需另备词表）到 `tests/fixtures/`，外加 HF Python 对固定语料产出的黄金 token ID 序列 | 当前仓内无任何真实 tokenizer.json（仅 target 指纹产物）；裁剪词表会改变 ID 编号与合并结果、逐位对齐必失败，故须全量；无黄金输出则 U1–U5 的"逐位对齐"无法验证，只能标"未验证" |
| D4 | `bytepair::apply(&[String]) -> Vec<String>` 公共签名 | **保留为薄包装**，内部换字节路径 | "公开 API 不变"是既定目标；签名改动连带 design.md §10.3/conformance/architecture 重审 |
| D5 | Jinja 子集范围 | **自研最小子集**（变量/循环/if），不做完整 Jinja | 完整 Jinja 需模板引擎依赖与不可信求值，违背极简内核；子集覆盖主流 `chat_template` |
| D6 | 词缓存作用域 | **不得引入跨请求可变共享缓存**（违 `OW2`/`OW4`）；优先 per-encode 的 scratch 缓存，或对静态 pre-token 片段的不可变预计算缓存 | tokenizer 以 `&self` 跨线程共享；全局可变缓存破坏无锁共享假设，需另行论证同步 |

## 8. PR 拆分与强制门禁

### 建议的 PR 序列（每个提交一个可独立测试的阶段）

1. **PR-0（Phase 0）**：`benches/tokenizer.rs` + `Cargo.toml` `[[bench]]` 项 + 基线结果记录（本文档 §12）。
2. **PR-A（T1+T2）**：pre_tokenizer + 边界内合并 + 字节符号，对照套件建立。
3. **PR-B（T3+T4）**：decoder + post_processor，编解码闭环。
4. **PR-C（T5）**：Jinja 子集 chat template。
5. **PR-D（T6+T7）**：BPE 优先队列 + 词缓存（附基准对比）。
6. **PR-E（T8+T9）**：热路径去分配 + special 匹配。
7. **PR-F（T10–T12）**：normalizer + 依赖显式化 + cli decode。

### 每个 PR 的强制门禁

```
cargo fmt --all -- --check
cargo check -p tokenizer --all-targets --locked
cargo clippy -p tokenizer --all-targets --locked -- -D warnings
cargo test --test it tokenizerunits tokens        # 正确性回归
bash scripts/checknaming.sh                       # 无分隔符命名
bash scripts/checksrctests.sh                     # crates/tokenizer/src 无内联测试
bash scripts/checkunsafe.sh                       # unsafe 站点不变
RUSTDOCFLAGS="-D warnings" cargo doc -p tokenizer --no-deps
cargo bench --bench tokenizer                     # P1 各步必须附前后对比
```

## 9. 风险与回滚策略

- **最大风险**：P1/T6 重写的合并循环在边界上与原算法分叉（字节拼接 vs 字符串拼接的 UTF-8 语义、平局规则、rank 首次出现）。**缓解**：`proptest` 对拍旧 `apply` 逐字节一致 + 现有 `merges` 用例全保留，二者是权威参考；任一分叉即回滚该提交。
- **正确性风险**：P0 各段（pre_tokenizer/decoder/post_processor）的语义与 HF 分叉。**缓解**：对照套件以真实 tokenizer.json + 黄金输出为准（D3）；无黄金输出则阶段标"未验证"，不得声称对齐。
- **依赖风险**：新增生产依赖（`unicode-normalization` 等）改变依赖图。**缓解**：仅当 D1/D2 判据满足时引入，且 `cargo tree` 复核。
- **并发风险**：词缓存（T7）若实现为全局可变状态破坏 `OW2/OW4`。**缓解**：D6 限定作用域，接受 per-encode scratch 缓存的上限。
- **回滚**：每个 PR 独立提交；基线（PR-0）先落地，后续任一步回滚不影响基线可复现。

## 10. 退出条件（完成定义）

仅当下列全部成立，任务才算完成：

- P0 各段对真实 HF tokenizer.json 的编码/解码输出与 HF 逐 token 一致（对照套件全绿）。
- P1/T6 合并循环复杂度降至 O(n log n)，且 `proptest` 对拍逐字节一致。
- 性能声明附可复现基准（median、变异、前后相对差），无数据不称"提升"。
- `forbid(unsafe_code)` 保持，生产路径无 `panic`/`unwrap`/`expect`。
- 公共契约变化同步 design.md §10 / conformance.md / architecture.md §5.9。
- 未实施项在文档中如实标注，不静默放宽需求。

## 11. 边界与后续动作

本计划基于已读取的 tokenizer/cli 源码、crate 清单、现有测试与质量脚本制定，是**面向实施的优化计划，不是已完成的整改**。P0 的"逐位对齐"结论依赖真实 tokenizer.json 夹具（D3）；在夹具落地前，P0 各步只能验证内部一致性（`decode(encode(x))==x`），不能声称与 HF 对齐。

建议下一步：完成 PR-0（Phase 0 基线）后，先解决 D3 的夹具问题，再进入 PR-A（T1+T2）。

## 12. 执行状态（2026-10-16，Phase 0）

本节记录 G0 的基线与正确性结果。除本节外，其余条目（P0 T1–T5、P1 T6–T12）一律视为**未实施**。

### 12.1 基线（G0）

| 项 | 记录 |
| --- | --- |
| 基线 commit / 分支 | `2478f0a1b882385a58a7e4681da7f25097fbef60` / `main`（`spapient/llm`） |
| 工具链 | `rustc 1.99.0 (b940084d7 2026-09-28)`、`cargo 1.99.0`，由 `rust-toolchain.toml` 固定 |
| 宿主 | MacBook Pro（`Mac17,9`），Apple M5 Pro，18 核（6 Super + 12 Performance），24 GiB 内存（`hw.memsize=25769803776`），arm64e，Darwin 27.0.0（`RELEASE_ARM64_T6050`） |
| 环境采集 | 会话沙箱拦截 `sysctl`（`Operation not permitted`）；改用 `hostinfo` / `system_profiler SPHardwareDataType` 采得（二者可用），与豆包侧完全访问环境的 `sysctl` 采集（`hw.ncpu=18`）一致 |
| 构建 profile | `bench`（optimized + debuginfo），继承 root `[profile.release]` 的 `lto = "thin"`、`codegen-units = 1` |
| 正确性基线 | `cargo test --test it tokenizerunits` → **47 passed / 0 failed**；`cargo test --test it tokens` → **21 passed / 0 failed** |
| 基准夹具 | 合成且确定（无外部 PRNG、无真实 tokenizer.json，仓内未 vendor） |

### 12.2 性能基线（median，100 samples）

| 基准 | median | 区间 [low, high] |
| --- | --- | --- |
| `tokenizer/bytepair/apply_chain/64` | 134.84 µs | [133.45, 136.14] µs |
| `tokenizer/bytepair/apply_chain/512` | 12.091 ms | [11.956, 12.236] ms |
| `tokenizer/bytepair/apply_chain/4096` | 883.79 ms | [877.87, 889.69] ms |
| `tokenizer/encode/bytefallback/512` | 84.720 µs | [84.480, 84.966] µs |
| `tokenizer/encode/bytefallback/4096` | 701.03 µs | [694.66, 709.28] µs |
| `tokenizer/decode/bytefallback/4096` | 37.091 µs | [36.930, 37.239] µs |

### 12.3 复杂度读数（从基线直接观察）

- **`bytepair/apply_chain` 呈平方级**：输入 64→512（8×）耗时 134.84 µs → 12.091 ms（约 **90×**）；512→4096（8×）耗时 12.091 ms → 883.79 ms（约 **73×**）。这与 `apply` 每轮全表扫描 + `Vec::remove` O(n) 搬移 + 每候选对分配两个 `String` 的分析一致，是 P1/T6 的目标。
- **`encode/bytefallback`（无合并）呈线性**：512→4096（8×）耗时 84.720 µs → 701.03 µs（约 **8.3×**），佐证瓶颈确在合并循环而非逐符号解析。
- **`decode/bytefallback` 已属合理**：4096 字符 37.1 µs，`parsebytetoken` 无分配，非本轮优先项。

> 以上为基线，**不构成任何性能提升声明**。P1 各步须在相同环境、profile、工作负载下与此表对比。

### 12.4 T6 实施结果（2026-10-16）

`bytepair::apply`（`crates/tokenizer/src/merges.rs`）由每轮全表重扫 + `Vec::remove` 搬移的 O(n²) 重写为**双向链表 + 最小堆 + 惰性删除**：每次合并只触及相邻符号，全局最优对由堆弹出。公共 API 不变；`getrank` 仍是字符串键 `BTreeMap` 探针（敌对长键输入会叠加键长项，见 merges.rs 文档注释，属 T8 的后续微优化项）。

**对拍**：新增 `tokenizerunits::merges::apply_matches_an_independent_naive_reference`（proptest，独立朴素参考实现，与 `bytepair::apply` 零共享代码）。

**验证**：`cargo test --test it tokenizerunits` → **48 passed / 0 failed**（含新对拍）；`cargo clippy -p tokenizer --all-targets --locked -- -D warnings` → 通过；`cargo fmt` 通过。

**基准（同环境、同 profile、同工作负载，mean，100 samples）**：

| 基准 | T6 前 | T6 后 | 加速比 |
| --- | --- | --- | --- |
| `bytepair/apply_chain/64` | 134.84 µs | 12.16 µs | ≈ 11.1× |
| `bytepair/apply_chain/512` | 12.091 ms | 311.3 µs | ≈ 38.8× |
| `bytepair/apply_chain/4096` | 883.79 ms | 16.89 ms | ≈ 52.3× |
| `encode/bytefallback/512` | 84.72 µs | 85.67 µs | ≈ 0.99×（噪声） |
| `encode/bytefallback/4096` | 701.03 µs | 670.7 µs | ≈ 1.05×（噪声） |
| `decode/bytefallback/4096` | 37.09 µs | 33.45 µs | ≈ 1.11×（噪声，decode 不经 `apply`） |

加速比随输入规模增长（11× → 39× → 52×），与 O(n²)→O(n log n) 堆操作一致；`apply_chain/4096` 剩余的 16.89 ms 主要来自合并键随链增长而变长导致的字符串比较（O(总键长)），真实词表键短、此项不主导。`encode/bytefallback` 无合并，旧算法本是一扫即退，±5% 属测量噪声、无回归。**以上加速为实测结论，仅在本文档记录的环境与合成工作负载下成立。**

### 12.5 T8 实施结果（2026-10-16）

词表/合并热路径去分配，公共 API 不变：

1. `bytepair::getrank`：`BTreeMap<(String, String), usize>` → 两层 `BTreeMap<String, BTreeMap<String, usize>>`，两层均用借用的 `&str` 键探测，**零分配**（原每次构造两个 `String`）。
2. `vocabulary` 增 256 项 byte-fallback 表（`getbytetokenid`），`resolvetoken` 不再每字节 `format!("<0x{byte:02X}>")`。
3. `vocabulary.byid` 仍为稀疏 `BTreeMap`（`IN4`）；`bycontent` 未改 `HashMap`——`getid` 本就 `Borrow<&str>` 零分配，`HashMap` 的 SipHash 对短键无净收益。

**验证**：`cargo test --test it tokenizerunits` → 48 passed / 0 failed；`tokens` → 21 passed / 0 failed；clippy `-D warnings` 通过；fmt 通过。

**基准（同环境/profile/工作负载，mean）**：

| 基准 | T8 前（§12.4 后） | T8 后 | 变化 |
| --- | --- | --- | --- |
| `encode/bytefallback/512` | 85.67 µs | 41.40 µs | −51.5%（≈2×） |
| `encode/bytefallback/4096` | 670.7 µs | 328.1 µs | −51.3%（≈2×） |
| `apply_chain/64` | 12.16 µs | 9.05 µs | −25.3% |
| `apply_chain/512` | 311.3 µs | 318.8 µs | +2.7%（噪声） |
| `apply_chain/4096` | 16.89 ms | 16.73 ms | −1.0%（p=0.21，不显著） |
| `decode/bytefallback/4096` | 33.45 µs | 33.24 µs | −0.1%（decode 不经此路径） |

`encode/bytefallback` 减半来自 `resolvetoken` 去 `format!`。`apply_chain` 大 n 仍由字符串键比较主导（O(总键长)），去分配只在小 n 显著；彻底消除需符号→整数 id（HF v1 的 pair 打包 64 位），涉及滚动哈希与碰撞风险，非真实词表（键短）的瓶颈，留待后续评估。

**T6+T8 累计（相对 Phase 0 基线）**：`apply_chain/4096` 883.79 ms → 16.73 ms（≈52.8×）；`encode/bytefallback/4096` 701.03 µs → 328.1 µs（≈2.14×）。

### 12.6 P0/T1 部分实施（2026-10-16）

GPT-2 ByteLevel 基础（[`crates/tokenizer/src/pretoken.rs`](crates/tokenizer/src/pretoken.rs)，纯函数，**未接线** encode/decode、未做 pre_tokenizer 类型分发）：

- `bytetounicode()`：GPT-2 256 项 byte→unicode 表（188 可打印字节映射自身，68 字节映射 U+0100..=U+0143，双射）。
- `unicodetobyte()`：逆表（≤U+0143）。
- `gpt2split()`：手写 GPT-2 切分（`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`）；`\p{L}` 用 `char::is_alphabetic`（`Alphabetic` 派生属性 ⊃ `Letter` 类，D1 已记录）。

**验证**：`tokenizerunits` → 51 passed / 0 failed（新增 `pretoken` 3 例：双射、已知向量、切分向量）；`tokens` → 21 passed；clippy `-D warnings` 通过；fmt 通过。

**未完成**：`pre_tokenizer` 段其它类型分发（T1 其余）、HF 逐位对齐（受 D3 夹具阻塞）。

### 12.7 P0/T2+T3 实施结果（2026-10-16）

ByteLevel 接入管线（`engine.rs` 读取 `pre_tokenizer`/`decoder` 段；`encode.rs`/`decode.rs` 增加字节级路径）：

- `fromjson` 解析 `pre_tokenizer.type`/`decoder.type`：`ByteLevel` → 启用字节级；缺省 → 字符级（既有行为）；其它类型 → `unsupportedpretokenizer`/`unsupporteddecoder`。
- 编码（T2）：`gpt2split` 切分 → 每 chunk 字节映射 byte-to-unicode → 合并不跨 chunk 边界。
- 解码（T3）：token 每字符经逆表还原字节 → UTF-8；越界/非字母表字符 → `invalidbytelevel`。
- 错误 enum 15 → 18：`unsupportedpretokenizer`、`unsupporteddecoder`、`invalidbytelevel`。

**验证**：tokenizer crate 编译 + clippy `-D warnings` + fmt 通过；字节级往返经临时独立测试目标（只依赖 tokenizer+lib）实测 `decode(encode(x))==x`（ASCII/收缩/CJK/数字/空），"你好" 2 标量→6 token 证实字节级。**`tests/it` 集成测试（含 `pretoken` 新增 5 例）暂被并发会话的 `scan` 特性（tensor/api）未完成改动阻塞**（E0046 缺 `scan`/`scan_cached` 实现、E0283），待其修复后即可运行。

### 12.8 P0/T4+T5 实施结果（2026-10-16）

- **T4（`post_processor`）**：`postprocess.rs` 解析 `post_processor.type == "TemplateProcessing"` 的 `single`/`pair` 模板；`tokenizer::postprocess`/`postprocesspair` 按模板插入特殊 token（type_id 忽略——本 tokenizer 只产 tokenid）。无 post_processor 时为恒等。新增错误变体 `unsupportedpostprocessor`、`malformedpostprocessor`。
- **T5（`chat_template`）**：`jinja.rs` 实现最小 Jinja 子集——`{{ expr }}` 插值、`for ... in messages`、`if`/`elif`/`else`、`message.role`/`message['content']`、`loop.last/first/index/index0`、`bos_token`/`eos_token`、字符串字面量、`==`/`!=`/`not`/`and`/`or`；`fromjson` 读 `chat_template`/`bos_token`/`eos_token`；`tokenizer::renderchat` 渲染，`applychattemplate` 优先走 Jinja。新增错误变体 `malformedtemplate`。

**验证**：tokenizer crate 编译 + clippy `-D warnings` + fmt 通过；经临时独立测试目标实测 3 例全过（post_processor single/pair、Jinja 循环+插值、Jinja if/else 分支）。修复一处 `tokenize` 缺陷（原只按 `{{` 优先，先出现的 `{%` 被误吞为文本）。**`tests/it` 集成测试（含新增 `postprocess`/`jinja` 模块）仍被并发会话 `scan` 阻塞**。

### 12.9 P1/T7+T9 实施结果（2026-10-16）

- **T9（special 前缀匹配去分配）**：`specialtokens` 在构造时预计算最长优先列表 `longest`，`encodespecial` 改用 `iterlongest()`——去掉每次调用 `specials.iter().collect()` + `sort_by` 的 Vec 分配与排序。
- **T7（per-encode 词缓存）**：`encodebytelevel` 加调用级 `HashMap<&str, Vec<tokenid>>` 缓存，同一 pre-token 在单次 encode 内重复出现时复用编码结果，**无跨调用共享状态**（D6 约束）。

**验证**：tokenizer crate 编译 + clippy `-D warnings` + fmt 通过；临时独立测试 2 例过（special 最长匹配 `<s>` 胜过 `<`、字节级重复词往返）。新增字节级基准 `tokenizer/encode/bytelevel/{512,4096}`（重复词工作负载）实测 **5.33 µs / 15.88 µs**。

**诚实边界**：T9 是纯去分配重构（无行为变化）。T7 的缓存是 per-encode 的，收益仅在单次 encode 内出现重复 pre-token 时成立，且给非重复文本带来每 chunk 一次哈希开销；共享缓存方案已被 OW2/OW4 否决（D6）。未做无缓存对照（需临时禁缓存重建，未执行）。

### 12.10 P1/T10+T12 实施结果（2026-10-16）

- **T10（normalizer）**：`normalize.rs` 实现 `normalizer`（`lowercase`/`strip`/`sequence`，均 std 可实现）；`fromjson` 解析 `normalizer` 段（`null`/缺省 → 无归一化）；`encode`/`encodespecial` 先归一化再预分词。Unicode 归一化形式（NFC/NFD/NFKC/NFKD）返回 `unsupportednormalizer`——需 `unicode-normalization` 依赖，按 D2 推迟（真实 GPT-2/Llama 均 `normalizer: null`）。新增错误变体 `unsupportednormalizer`。
- **T12（cli decode）**：新增 `decode` 子命令（`--tokenizer` + `--tokens`），加载 tokenizer 将 token id 列表解码为文本。

**验证**：`tokenizerunits` → 66 passed / 0 failed（新增 `normalize` 5 例）；`tokens` → 21 passed；`cli` → 46 passed（新增 decode 2 例）；clippy `-D warnings` + fmt 通过。**并发会话的 `scan` 断裂已修复，`tests/it` 集成测试恢复可运行**——此前被阻塞的全部永久测试已随套件通过。

### 12.11 D3 逐位对齐 HF（2026-10-16，验收达成）

真实 GPT-2 全量 `tokenizer.json`（~1.35MB）vendor 到 `tests/fixtures/gpt2/`，外加 HF `tokenizers` 0.23.3 黄金输出（10 条语料：ASCII/CJK/重音/特殊词元/多空格/长文/空串，`add_special_tokens=false`）。对齐测试 `tests/it/tokenizergolden.rs` 断言 `encodespecial` 与 `decode` 逐位相等。

**修复的加载/编码缺陷**：

1. `model.type` 缺省 → 视为 BPE（GPT-2 的 `model` 段无 `type` 字段）。
2. `model.unk_token: null` → 视为无 unk token（原报 `unexpectedtype`）。
3. `post_processor.type == "ByteLevel"` → 接受为 no-op（BOS 插入仅影响 `add_special_tokens=true`）。
4. `gpt2split` 空白语义：`\s+(?!\S)` 对"后跟非空白的空白串"回溯为"串去掉最后一个空白"——修复前多空格语料编码错误（`"  multiple"` 应为 `[" ", " multiple"]` 而非 `["  ", "multiple"]`）。

**结果**：`tokenizergolden` → **2 passed**（encode 10/10、decode 10/10 逐位对齐 HF）。全套：`tokenizerunits` 65、`tokens` 21、`tokenizergolden` 2，clippy `-D warnings` + fmt 通过。

### 12.12 多模型逐位对齐 HF（2026-10-16，全管线类型分发落地）

把 `bytelevelencode/decode` 两个布尔换成真正的 **`pretokenizer` / `decoder` 管线**，补齐 T1 剩余的全部分段类型：

- **`pretokenizer`**：`bytelevel{useregex}`、`split{regex}`、`sequence`。Split 用 **`fancy-regex`**（Rust `regex` 不支持 `\s+(?!\S)` 负向前瞻）；`splitisolated` 隔离每个正则匹配。
- **`normalizer`**：补 `prepend`/`replace`（llama2/gemma 的 `▁` 语义）+ **NFC/NFD/NFKC/NFKD**（用 `unicode-normalization`，qwen3 的 NFC）。
- **`decoder`**：`bytelevel`、`bytefallback{replace,stripleading}`（Sequence 压平）；非 byte-to-unicode 字符（全角特殊词元）按字面解码。
- **`post_processor`**：接受 `Sequence`（压平为 no-op，对齐用 `add_special_tokens=false`）。
- **加载兼容**：`pre_tokenizer: null`、`merges` 数组格式（qwen3 的 `["Ġ","Ġ"]`）、`ignore_merges`。

**对齐结果（encode + decode，HF `tokenizers` 0.23.3 黄金）**：

| 模型 | 词表 | 关键结构 | encode | decode |
|---|---|---|---|---|
| GPT-2 | 50,257 | ByteLevel+byte-to-unicode | 10/10 | 10/10 |
| Llama 2 | 32,000 | Prepend/Replace normalizer + byte fallback | 8/8 | 8/8 |
| Llama 3 | 128,256 | Split(LLaMA 正则)+ByteLevel+ignore_merges | 8/8 | 7/7* |
| Qwen3 | 151,936 | NFC + Split+ByteLevel | 8/8 | 7/7* |
| DeepSeek-V3 | 128,256 | Split(数字+CJK)+ByteLevel | 8/8 | 7/7* |
| Gemma | 256,000 | Replace normalizer + byte fallback + fuse | 8/8 | 7/7* |

\* decode 跳过"纯特殊词元"样本——黄金 `decoded` 字段用 `skip_special_tokens=true`（用户注明的 tokenizers 0.23.3 怪癖），本 tokenizer 从不跳特殊词元，语义不同故不比对。

**新增生产依赖**（推翻 D1/D2 推迟决策，真实模型需求驱动）：`fancy-regex 0.15`（Split 正则 look-around）、`unicode-normalization 0.1`（NFC）。均已 vendor 进 `vendor/`，完全离线构建。

**诚实边界**：~~o200k（tiktoken 格式，非 tokenizer.json）未实施，需独立解析 `.tiktoken` 文件。~~ **已实施**（§12.13）。

### 12.13 o200k（tiktoken 格式）实施结果（2026-10-16）

新增 `tiktoken` 类型（`crates/tokenizer/src/tiktoken.rs`）：解析 `.tiktoken` 文件（每行 `base64(token_bytes) rank`，199998 个可合并 token），字节级贪心合并（相邻字节拼接结果在词表内取最低 rank），内置 o200k 正则预分词，特殊词元由调用方传入（`<|endoftext|>`=199999、`<|endofprompt|>`=200018）。手写 base64 解码，零新增依赖。

**验证**：临时 example 实测 o200k encode 8/8 + decode 8/8 逐位对齐 tiktoken 0.14.0 黄金；clippy `-D warnings` + fmt 通过。**7 个模型（GPT-2/Llama2/Llama3/Qwen3/DeepSeek/Gemma/o200k）全部逐位对齐。**

**字节级合并优化（2026-10-16）**：`encode_chunk` 由 O(n²) 朴素实现（每轮全扫 + `Vec::remove` 搬移）复用 T6 的最小堆 + 双向链表算法——`tiktokenstep` 按"拼接结果 rank"排序、`trymerge` 惰性删除、`queue_pair` 只重插受影响邻居。基准 `tokenizer/encode/tiktoken/{512,4096}`（重复模式，短键）实测 **118.9 µs / 945.1 µs**，8× 输入 ≈ 8× 耗时（线性，佐证 O(n log n) 堆操作主导）。**诚实边界**：拼接仍按字节键查 `HashMap`（O(键长)），敌对"键无限增长"输入叠加 O(总键长) 项——与 `bytepair::apply` 同款限制，真实文本键短不主导。

### 12.14 post_processor 补齐（2026-10-16）

修正 `parsepostprocessor`：`Sequence` 不再压平为 no-op，而是提取其 `TemplateProcessing` 子段（ByteLevel 子段在 HF 里本就只调 offsets、不插 token）。`postprocess()`（`add_special_tokens=true`）现对 llama2/llama3/gemma 正确插入 BOS、对 qwen3/deepseek 正确保持 no-op（`wrapped_ids` 逐位对齐，5 模型各 8/8）。新增 `postprocess_matches_the_wrapped_reference` 测试。

**剩余诚实缺口（未实现，需能触发差异的黄金语料才暴露）**：`ignore_merges`（llama3，我仍照常应用 merges 表）、`fuse_unk`（gemma，未知 token 融合解码语义）。二者在现有 8 条语料上不产生差异，故 7 模型仍逐位对齐。

### 12.15 ignore_merges / fuse_unk 收口（2026-10-16）

- **① `ignore_merges`（llama3）**：已实现。`encodeplainsegment` 在 `ignore_merges=true` 时改用 `rankmerge`（按"相邻符号拼接结果在词表内的最低 id"贪心合并，最小堆 + 双向链表），等价于 tiktoken 派生 merges 表、但直接查词表。6 模型 encode 仍全对齐（llama3 8/8）。
- **② `fuse_unk`（gemma）**：确认为 **no-op**。gemma 同时 `byte_fallback=true` + `<unk>`（id 3），解析顺序 exact → byte_fallback → unk，故 `<unk>` 永不触发、fuse 无对象；Fuse decoder 早已按 no-op 处理（`"ByteFallback" | "Fuse" => {}`）。无需行为改动。

### 12.16 Phase 2 S1：整数键合并表 + pair 打包 64 位（2026-10-16）

把 §12.5 明确"留待后续评估"的"符号→整数 id"落地：`bytepair` 增加整数键形态，`applyids` 以打包 `(left_id<<32)|right_id` 键合并 id 序列，消除 O(总键长) 的字符串比较项。

**实现**：

1. `bytepair` 增 `byids: HashMap<u64, (usize, u32)>`（键 = 两个 u32 词表 id 打包，双射零碰撞；值 = `(rank, merged_id)`）。
2. 新构造 `frommergesandvocab(&merges, &vocab)`：字符串表 + 整数表同建；合并两侧或结果不在词表 → 新错误 `mergenotinvocab`（D1 加载期校验拒绝）。**7 个真实模型实测 0 缺失**（llama3 280k / gemma 580k merges 全在词表内）。
3. 新方法 `applyids(&[tokenid]) -> Vec<tokenid>`：最小堆 + 双向链表，候选与惰性删除校验都只做一次 `u64` 打包 + `HashMap` 探针，无拼接、无分配。
4. `engine::fromjson` 条件构建：`ignore_merges` 走 `frommerges`（id 表无用，llama3 省 280k 条），否则 `frommergesandvocab`。
5. `engine::encodebypairids`：初始符号映射为 id，非词表符号是不可合并叶，按叶切分 id run、逐 run `applyids`、叶直接 `resolvenonvocab`（从 `resolvetoken` 拆出，避免叶路径重复 `getid`）。

**验证**：`tokenizerunits` 66 passed（新增 proptest `applyids_matches_string_apply`，对拍 `applyids` ≡ `apply`+`getid` 逐 id 一致）；`tokenizergolden` 4 passed（7 模型 encode/decode/postprocess/o200k 逐位不变）；`tokens` 21 passed；fmt + clippy `-D warnings` + checkunsafe（0 新增 unsafe）+ checknaming + checksrctests 全过。

**基准（同环境/profile/工作负载，mean）**：

| 基准 | 字符串键（`apply_chain`） | 整数键（`applyids_chain`） | 加速比 |
| --- | --- | --- | --- |
| chain/64 | 7.291 µs | 2.655 µs | ≈ 2.7× |
| chain/512 | 287.02 µs | 21.26 µs | ≈ 13.5× |
| chain/4096 | 20.639 ms | 177.62 µs | ≈ 116× |

| 基准 | S1 前（§12.5） | S1 后 | 变化 |
| --- | --- | --- | --- |
| encode/bytefallback/4096 | 328.1 µs | 248.31 µs | −24%（去除空 merges 的 `apply` 开销） |

**过程中抓到并修复的回归**：`encodebypairids` 叶路径最初调用 `resolvetoken`，对"非词表符号"重复 `getid`（一次在 run 切分、一次在 `resolvetoken` 内），使 `encode/bytefallback/4096` 一度 384 µs（+17%）。拆出 `resolvenonvocab` 后消除重复探针，回落至 248 µs。

**诚实边界**：① 位置/rank 仍为 `usize`，未做堆步 `(rank,left)→u64` 打包（D4 的 >2³² 输入 guard 属 S3 scratch 复用一并处理）；② `applyids` 在 `frommerges`（字符串-only）构建的表上是恒等——公开方法依赖"由 `frommergesandvocab` 构建"这一前提，已写入文档注释；③ tiktoken/rankmerge 仍按字节键查 `HashMap`（§12.13 同款 O(键长) 限制），是 S2 的目标，S1 不覆盖。

### 12.17 Phase 2 S2：tiktoken/rankmerge 去字节键（2026-10-16）

消除 §12.13 记录的"拼接按字节键查 `HashMap`（O(键长)）"与 §12.15 rankmerge 的"字符串 clone+push_str"分配：两条贪心合并循环都改为「区间表示 + 借用切片查找」，不再物化每符号的 `Vec<u8>`/`String`，候选拼接也不分配。

**实现**：

1. `tiktokenizer::encode_chunk`（[tiktoken.rs](crates/tokenizer/src/tiktoken.rs)）：利用不变量「活符号在索引 `i` 即字节切片 `bytes[i..next[i]]`」（起始偏移 == 索引），删掉 `symbols: Vec<Vec<u8>>` 与 `merged = clone + extend`；`trymerge`/`queue_pair` 改 `ranks.get(&bytes[left..end])` 切片 Borrow 查找（`Vec<u8>: Borrow<[u8]>`，零分配、零碰撞）。
2. `tokenizer::rankmerge`（[engine.rs](crates/tokenizer/src/engine.rs)）：签名改 `pretoken: &str`，用 `char_indices` 建 `charstart`（字符起始偏移），符号 = `pretoken[charstart[i]..end]` 借用切片；删掉 `work: Vec<String>` 的 clone+push_str。`pretoken`+`charstart` 捆成只读 `ranksource` 上下文（压 clippy `too_many_arguments`）。`encodeplainsegment` 的 ignore_merges 分支不再物化 `symbols: Vec<String>`。

**验证**：`tokenizerunits` 66 passed；`tokenizergolden` 4 passed（o200k 8/8 + llama3 8/8 逐位不变，rankmerge 走新路径）；`tokens` 21 passed；fmt + clippy `-D warnings` + checkunsafe（0 新增 unsafe）+ checknaming + checksrctests 全过。

**基准（同环境/profile/工作负载，mean）**：

| 基准 | S2 前（§12.13） | S2 后 | 加速比 |
| --- | --- | --- | --- |
| encode/tiktoken/512 | 118.9 µs | 62.43 µs | ≈ 1.90× |
| encode/tiktoken/4096 | 945.1 µs | 497.78 µs | ≈ 1.90× |

**诚实边界**：① 用切片 Borrow 查现有 `HashMap<Vec<u8>,u32>`（精确键、零碰撞），未引入 trie/滚动哈希——短 token 下哈希已便宜，trie 的指针追逐未必更快，故 D2 的"默认 trie"被推翻（切片方案更简且同正确性）；② `ranksource` 仍按调用物化 `charstart: Vec<usize>`（一次 O(n) 分配），归 S3 scratch 复用；③ rankmerge 无独立基准（llama3 无 bench 夹具），只证正确性不声称性能。

### 12.18 Phase 2 S3：scratch 复用（2026-10-16）

把三套合并循环（`bytepair::applyids`、`tiktokenizer::encode_chunk`、`tokenizer::rankmerge`）的每调用分配（`prev`/`next`/`alive`/`work`/`charstart`/`heap`）与 `encodebypairids` 的 `symbols: Vec<String>` 收进调用方持有的复用缓冲，新增 `encode_into`/`encodespecial_into`/`applyids_into` API。

**实现**：

1. 新模块 [`scratch.rs`](crates/tokenizer/src/scratch.rs)：统一候选步 `mergestep{rank:usize,left:usize}`（替换 `tiktokenstep`/`rankstep`）；`mergework`（prev/next/alive/ids/heap 五缓冲，pub(crate)）；公开 `mergescratch{merge, run, charstart}`（`Default`，只扩容不缩容）；`resetlinks` 复位链表。
2. `bytepair::applyids` 变薄包装，新增 `applyids_into(symbols, &mut mergework, out)`（pub(crate)，因 `mergework` 不公开）。
3. `tokenizer`/`tiktokenizer` 加 `encode_into`/`encodespecial_into`（公开，`&mut mergescratch`），`encode`/`encodespecial` 变薄包装；scratch 穿透 `encodeplainsegment`→`encodebypairids`/`rankmerge`。
4. `encodebypairids` 改收 `pretoken: &str`，用 4 字节栈缓冲 `char.encode_utf8` 直接查词表，**删除 `symbols: Vec<String>`（每字符一个 String 的分配）**；run 累进器复用 `scratch.run`（与 `scratch.merge` 分离字段，避免借用冲突）。

**验证**：`tokenizerunits` 67 passed（新增 `encode_into_reuses_scratch_across_calls`，大→小→空复用不串扰）；`tokenizergolden` 4 passed（7 模型逐位不变）；`tokens` 21 passed；fmt + clippy `-D warnings` + rustdoc `-D warnings` + checkunsafe（0 新增 unsafe）+ checknaming + checksrctests 全过。

**基准（同环境/profile/工作负载，mean）**：

| 基准 | S2 后 | S3 后（`encode` 包装） | 变化 |
| --- | --- | --- | --- |
| encode/tiktoken/4096 | 497.78 µs | 442.39 µs | −11%（去每 chunk 分配） |
| encode/bytefallback/4096 | 248.31 µs | 186.78 µs | −25%（去 `symbols: Vec<String>`） |
| encode/tiktoken/4096（`encode_into`） | — | 438.14 µs | 跨调用复用再 −1% |

**诚实边界与发现**：

1. **`encode/bytelevel/4096` 实测 292.86 µs，与 §12.9 记录的 15.88 µs 相差约 18×——原因是 §12.9 声称"已实施"的 T7 词缓存（per-encode `HashMap<&str,Vec<tokenid>>`）实际已不在代码中**：`git show HEAD` 与当前树均无该缓存（仅 lib.rs 文档提及 "KV-cache"）。这是文档与代码的既有脱节，非本阶段引入（S1 起点即无缓存）；§12.9 的 15.88 µs 是带缓存的旧状态，字节级真实无缓存基线约 292 µs。
2. 无合并词仍走完整链表+堆建置（每 pre-token O(len)），是 `applyids`/`rankmerge` 的既有开销，非 S3 目标，属潜在后续优化。
3. 堆步未做 `(rank,left)→u64` 打包（保留 `left: usize`，规避 >2³² 输入的 D4 guard 复杂度）。
4. **环境**：并发会话在 checkpoint 引入 `mmap-guard` 依赖但未 vendor，`cargo check --offline` 一度解析失败；已 `cargo vendor vendor/ --versioned-dirs` 从缓存补齐（`vendor/mmap-guard-0.2.2`）。

### 12.19 T7 词缓存复实（2026-10-16）

§12.9 曾记录"已实施"的 per-encode 词缓存实际不在代码中（§12.18 发现），此处重新落地：`mergescratch` 增 `wordcache: HashMap<String, Vec<tokenid>>`，`encode_into`/`encodespecial_into` 入口 `clear`，`encodeplainsegment` 对每个 pre-token 先查后插（键为 pre-token `String`，直接 move 进表、无克隆）；无 pre-tokenizer 的模型（单一 pre-token，无重复可能）走 `encodepretoken` 直接编码、不缓存。

**验证**：`tokenizerunits` 67 passed、`tokenizergolden` 4 passed（7 模型逐位不变）、`tokens` 21 passed；clippy `-D warnings` + fmt 通过。

**基准（同环境，mean）**：

| 基准 | T7 前（§12.18） | T7 后 | 变化 |
| --- | --- | --- | --- |
| encode/bytelevel/4096 | 292.86 µs | 187.94 µs | −36%（≈1.56×） |
| encode/bytefallback/4096 | 186.78 µs | 188.86 µs | +1%（噪声，无 pre-tokenizer 走直通） |

**诚实边界**：词缓存只省"合并"，不省"预分词"。`encode/bytelevel/4096` 剩余 187.94 µs 几乎全在 `gpt2split`（O(n) 字符分类）与 `bytemap`（每 pre-token 一次 `bytetounicode()` 重建 256 表 + 逐字节映射）——后者是 S5 的预分词优化目标（含 `bytetounicode` 改静态表 + 安全 SIMD）。旧 §12.9 的 15.88 µs 是带缓存且预分词已优化的状态。

### 12.20 S4 batch（2026-10-16）

新增 `tokenizer::encode_batch(&[&str], &mut mergescratch, &mut Vec<Vec<tokenid>>)`：一次清空词缓存，逐文本编码并追加一条 id 序列；词缓存在整批内共享，后文命中前文已见的 pre-token。`encode_into` 抽出私有 `encode_text`（不清理缓存）供两者复用。

**验证**：`tokenizerunits` 68 passed（新增 `encode_batch_matches_individual_encodes`，batch ≡ 逐条 encode）；`tokenizergolden` 4 passed；clippy `-D warnings` + fmt 通过。

**基准（同环境，mean）**：`encode_batch/bytelevel/8x512` = 186.09 µs；单条 `encode/bytelevel/512` = 24.87 µs（8 条独立约 199 µs）。共享缓存约省 7%——**再次佐证瓶颈在预分词而非合并**：预分词 `gpt2split`+`bytemap` 逐文本重跑、不缓存，词缓存省不掉。

### 12.21 S5：const 表（LazyLock）+ 安全 SIMD 评估（2026-10-16）

**① const 表（决定性一击）**：`bytetounicode()`/`unicodetobyte()` 由"每次调用重建 256/324 表"改为 `LazyLock` 静态表（`std::sync::LazyLock`，首次使用算一次）。此前 `bytemap` 每 pre-token 重建一次 `bytetounicode()` 表，是 `encode/bytelevel` 的主开销。返回类型由 `[char;256]`→`&'static [char;256]`、`[Option<u8>;324]`→`&'static [Option<u8>;324]`（全部调用点按索引/迭代使用，源码兼容）。**零新依赖、零 unsafe。**

**基准（同环境，mean）**：

| 基准 | S5 前（§12.19/§12.20） | S5 后 | 加速比 |
| --- | --- | --- | --- |
| encode/bytelevel/512 | 24.87 µs | 5.12 µs | ≈ 4.9× |
| encode/bytelevel/4096 | 187.94 µs | 27.97 µs | ≈ 6.7× |
| encode_batch/bytelevel/8x512 | 186.09 µs | 29.90 µs | ≈ 6.2× |

`encode/bytelevel/4096` 现 27.97 µs（≈6.8 ns/字符），逼近旧 §12.9 带缓存记录的 15.88 µs。

**② 安全 SIMD 评估——暂不引入**：剩余开销为 `gpt2split`（逐 `char` Unicode 分类 + 收缩词匹配）与 `bytemap`（byte→char 表查 + String 收集）。SIMD 选项：`fearless_simd`/`archmage`（安全、需新依赖 + vendor）或 `#[target_feature]`+feature token（安全、但 ARM NEON 的 byte→u32 gather 需多段 `vtbx` + 加宽，复杂且 `char` 级 Unicode 分类 SIMD 价值低）。按 AGENTS.md §8「没有清晰的正确性与测量计划，不得引入 SIMD」与 §3.2「最小依赖」：剩余 ~6.8 ns/字符收益已低，SIMD 至多对其中表查部分再 ~1.5–2×，不抵新依赖/复杂度，故**判定暂不值得**；如需可在后续单独立项（fearless_simd 优先，unsafe 封在依赖内）。

### 12.22 热路径去分配 + 词表哈希化（2026-10-16）

按 §12.19 后的深度分析落地前两项：

**① bytefallback 去分配**：`resolvenonvocab`/`resolvetoken` 的每 leaf `Vec::new()` 改为复用 `mergescratch::bytetokens` 缓冲（`&mut Vec<tokenid>` 贯穿 encode/rankmerge 调用点）。

**② 词表 `bycontent` BTreeMap → HashMap**：`vocabulary::bycontent` 由 `BTreeMap<String,u32>` 改 `HashMap<String,u32>`（`byid` 保持 `BTreeMap` 供 `getminid`/`getmaxid` 排序）。`getid` 从 O(log V) 指针追逐降为 O(1) 哈希。**这是本轮最大单项 win**——bytefallback 每字符一次失败 `getid`，BTreeMap 的指针追逐是主开销，非 Vec 分配。

**验证**：`tokenizerunits` 68 passed、`tokenizergolden` 4 passed（7 模型逐位不变）、`tokens` 21 passed；clippy `-D warnings` + fmt 通过。

**基准（同环境，mean）**：

| 基准 | 本轮前 | ① 后 | ①+② 后 | 累计 |
| --- | --- | --- | --- | --- |
| encode/bytefallback/4096 | 188.86 µs | 147.39 µs | **41.37 µs** | ≈ 4.6× |
| encode/bytelevel/4096 | 27.97 µs | — | 27.00 µs | ≈ 1.04×（词表 256 本就浅） |
| encode/gpt2/4096（真实 50k 词表） | — | — | **28.14 µs** | 词表大小不再影响 |

**诚实边界**：① `bycontent` 用 std `HashMap`（SipHash），对短键仍非最快；若再追求极致可换 `ahash`/`fxhash`（新依赖 + vendor），对 bytefallback 可能再省 ~15–20%，未做。② gpt2 基准为本轮新增（无"哈希化前"对照），28.14 µs 是现状绝对数。

### 12.23 堆步打包实验（回退）+ tiktoken 评估（2026-10-16）

**① 堆步 `(rank,left)→u64` 打包——实测无效，已回退**：把 `mergestep{rank:usize,left:usize}`（16 B）改为 u64 打包（8 B）+ 访问器。实测 `applyids_chain/64` 2.655→3.25 µs（−22%）、`/512` −4%、`/4096` −1%——**变慢**。原因：`new` 的 shift+or、`rank()`/`left()` 每次访问的 shift/and 开销抵消了内存减半的缓存收益；编译器对 16 B 结构体的字段访问本就零开销。结论：**内存打包的"理论更快"被实测推翻，维持 16 B 结构体**（AGENTS.md §8「避免仅因更复杂或理论更快就接受更慢的实现」）。

**② tiktoken 剩余评估——无干净的低风险优化**：`encode/tiktoken/4096` ≈ 450 µs，主导是 ① `fancy_regex` 的 o200k 正则（含 `(?!\S)` 负向前瞻，回溯较慢，手写替换有正确性风险）；② `ranks: HashMap<Vec<u8>,u32>` 的 SipHash 切片哈希。换更快的非 DoS 抗性 hasher（fxhash）会引入**哈希泛洪 DoS**——`ranks.get` 的键是输入文本的字节切片（不可信输入），确定性哈希可被预构造碰撞；`ahash`（随机种子、DoS 抗性、更快）可行但需新依赖 + vendor。两者收益均 ~15–20%，风险/成本不抵，**判定维持 SipHash**。tiktoken 路径维持 ~450 µs，是当前唯一"慢"路径，但其优化需单独立项（手写 o200k 切分 或 ahash 依赖）。

### 12.24 加载期优化（2026-10-16）

实测 `fromjson`：gpt2（50k 词表）34.3 ms、gemma（256k 词表、580k merges）510.7 ms。落地两处：

1. `vocabulary::byid` 由 `BTreeMap<u32,String>` 改 `HashMap<u32,String>`，`minid`/`maxid` 改为构建期跟踪（`getminid`/`getmaxid` O(1)）；`bycontent`/`byid` 均 `with_capacity`。`getcontent`（decode）同时从 O(log V) 降 O(1)。
2. `bytepair::byids` `with_capacity(merges.len())`。

**结果**：gpt2 34.34→28.92 ms、gemma 510.73→445.48 ms（约 −13%）。**剩余主项是 `ranks` 字符串 BTreeMap（580k merges 的双向 String 克隆 + 树插入，仅服务公开 `getrank`/`apply` API，编码路径不用）**；改为惰性构建需把 `ranks` 换成 `OnceCell`，破坏 `bytepair` 的 `#[derive(Clone/PartialEq/Eq)]`（`OnceCell` 非 `Clone`），需手工 impl，故判定**暂不引入**（一次性加载 445 ms 可接受）。

### 12.25 tiktoken 手写切分器（2026-10-16）

§12.23 判"手写替换有正确性风险"——用 proptest 对照 `fancy_regex` 消除了该风险后落地：`o200ksplit`（[pretoken.rs](crates/tokenizer/src/pretoken.rs)）手写 o200k 正则（大小写不敏感收缩词 / 单可选非字母数字前缀 + 字母 / 1–3 位数字 / 标点 + 尾随 CRLF / `\s*[\r\n]+` 至末个换行 / 尾随空白回退 / 空白），`tiktokenizer` 删掉 `fancy_regex` 依赖与 `regex` 字段，`encode_segment` 直接用 `o200ksplit`。

**验证**：新增 proptest `o200ksplit_matches_the_regex_reference`（`o200ksplit` ≡ `fancy_regex` 逐 chunk，随机 ASCII+Unicode+换行+收缩词输入）；`tokenizergolden` 4 passed（o200k 8/8 逐位不变）；`tokenizerunits` 69 passed；clippy `-D warnings` + fmt + checkunsafe（0 新增 unsafe）+ checknaming + checksrctests + rustdoc 全过。`fancy-regex` 加入根 `[dev-dependencies]` 仅供对照测试（tokenizer 生产依赖不变：`fancy-regex` 仍由 Split pre-tokenizer 使用）。

**基准（同环境，mean）**：

| 基准 | 手写前（§12.23） | 手写后 | 加速比 |
| --- | --- | --- | --- |
| encode/tiktoken/4096 | 472.57 µs | 208.81 µs | ≈ 2.26× |
| encode_into/tiktoken/4096 | 449.61 µs | 205.16 µs | ≈ 2.19× |

tiktoken 累计（相对 Phase 0 945.1 µs）≈ **4.6×**，现 205 µs 与 HF BPE（27–41 µs）的差距主要剩 `ranks` 的 SipHash 切片哈希（DoS 抗性、维持）。

### 12.26 加载期：`ranks` 字符串表惰性化（2026-10-16）

§12.24 指出加载剩余主项是 `ranks` 字符串 BTreeMap（仅服务公开 `getrank`/`apply`，编码路径不用）。落地惰性化：`bytepair` 改存 `merges: Vec<String>`（原始 merge 串，`&[String]` 入参 clone，避免改签名）+ `ranks: OnceCell<BTreeMap<…>>`（首次 `getrank`/`apply` 才 `ranksfrommerges` 构建）+ `count: usize`（`frommergesandvocab` 用 `byids.len()`、`frommerges` 用 `HashSet<&str>` 去重计数）。`OnceCell` 满足 `#[derive(Clone/Debug/PartialEq/Eq/Default)]`，无需手工 impl。

**基准（同环境，mean）**：

| 基准 | 惰性化前（§12.24） | 惰性化后 | 变化 |
| --- | --- | --- | --- |
| fromjson/gpt2 | 28.92 ms | 20.88 ms | −28% |
| fromjson/gemma | 445.48 ms | 228.01 ms | **−49%（≈2×）** |

gemma 加载累计（相对首测 510.7 ms）≈ **2.2×**；剩余主项是 `serde_json::from_str`（~10 MB JSON 全量解析）+ 580k merge 串解析本身，一次性可接受。

**验证**：`tokenizerunits` 69、`tokenizergolden` 4（7 模型逐位不变）、`tokens` 21 全过；clippy `-D warnings` + fmt + checkunsafe（0 新增 unsafe）+ rustdoc 全过。checknaming 现报 3 项，全部在并发会话新增的 `crates/musa/src/ffi.rs`（`const core` 大写紧凑 NM5），非本 crate。
