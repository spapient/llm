# 0045：自有 `.u` 容器 v2——引用去重、中性条目流导入与现场度量

- 状态：已采纳（实现随本轮落在 `weights`；回传主树后并入门禁基线）
- 日期：2026-11-17
- 相关：[0015](0015-pack-weight-container.md)（LLPK v1 字节布局与能力基线）、[0036](0036-reader-loader-mmap-source.md)（Reader/Loader 双契约）；旧基线 `.u`（LLMU v4）机制取证见 [`../lld.md`](../lld.md) §E8.2，对照口径见 [`../llmu-parity-optimization-design.md`](../llmu-parity-optimization-design.md)

## 背景

ADR-0015 的 LLPK v1 解决了"读写对称、全精度 wire 闭集、显式量化、对齐、两级 CRC、Merkle 证明、rANS"，但相对旧基线 `.u` 与业界容器（safetensors / GGUF / Burn Recorder）仍有四个结构性缺口：

1. **无跨段去重**：严格相等的张量逐份重复占字节，没有引用代表段的机制；旧 v4 的 REF 共享在退役代码里，未迁回。
2. **无外部条目归一通路**：写端只能手工拼 `Segment`；safetensors 之外的 GGUF/bin 条目没有到中性段流的映射，更没有 CLI 把外部文件落成本有容器。
3. **无现场度量**：体积比、参数数、无损标志、共享命中是旧 `deploy` 账本的能力，v1 容器打开后现算不出来。
4. **CLI 缺容器命令面**：旧 v4 的 `import` / `deploy` / `verify` 链路不在当前树；现有 `verify --op` 是算子校验，词面不能混用。

本轮裁决把容器升级为自有 **`.u` 容器 v2**：在 0015 既有骨架上演进，不另起平行 crate、不退役重写；对标旧 v4 的 REF/计量/CLI 口径，并把适配面扩到 safetensors / GGUF / bin。

## 决策

### 1. 格式身份：`.u` 容器 v2，不向后兼容 v1

- 文件后缀 `.u`；魔数 `u32 LE = 0x0000_752E`（字节 `[0x2E, 0x75, 0x00, 0x00]`，ASCII `.u\0\0`），`VERSION = 2`。
- 读端拒绝：魔数不符、版本高于本端（`Unsupported`）、版本低于本端（`Invalid`）、flags 非零预留位、`align_log` 超 `2^16`。v1 LLPK 字节不由 v2 读端解析；仓内无交付的 v1 产物，不负担双格式回读。
- 固定头沿用 0015：magic 4 + version 2 + flags 4 + align_log 1 = 11 字节；flags `bit0=GLOBAL_CRC`、`bit1=SEG_CRC`，其余预留必须 0；尾部全局 CRC-32 4 字节。

### 2. 演进既有骨架，不新增平行抽象

复用并只在 `weights` 内演进：`Writer`（流式 Sink 落盘、不整份进内存）、`Archive`/`Stored`（读视图）、`Segment`（写端段）、`Codec`（`Store=0` / `Rans=1`）、`Quant`（显式仿射方案）、`Proof`（手写 SHA-256 + Merkle 包含证明）、两级 CRC、对齐零填充、LEB128、XOR 补丁与双模式（`lifecycle`）。

- 模式/补丁从 `archive` 抽出到独立 `lifecycle` 模块：`llm.mode` 保留名段、`opt.` 状态剥离、`diff`/`join` 自逆 XOR；`mode`/`patch` 薄转发不变。
- 零生产外部依赖、`deny(unsafe_code)`；mmap 的 unsafe 仍只在 `storage` feature 岛（ADR 0036/0035）。

### 3. 引用段（REF 去重）：哈希预检 + 逐位比对，引用图打开期校验

描述符新增双形态，公共头部（名字 / dtype / shape）两种形态共用一份编码：

| 形态 | code | 承载字段 |
|---|---|---|
| `FORM_INLINE` | 0 | codec、量化方案、帧长 LEB u64、段级 CRC-32；数据区写自己的存储帧 |
| `FORM_REF` | 1 | 量化方案、代表段下标 LEB u32；不写 codec / 帧长 / CRC，存储帧与校验都归代表段 |

写端去重规则：

- 键 = `(dtype code, shape, 字节内容摘要)`；哈希命中后**必须再逐位比对**元数据（`same_value`：dtype/shape/quant 一致，名字不参与）与字节相等才签发引用；哈希碰撞只错过复用，绝不错配。
- 引用段与代表段名字本就不同（两个条目共享一份字节），名字不参与取值比对。

读端 `validate_refs` 单色 DFS 校验，以下一律 `Invalid`：下标越界、自引用、代表段本身是引用、取值元数据不一致、引用成环。`resolve_refs` 先收集回填决议再落地，把代表段的 codec/区间/长度/CRC 回填到引用段视图；对外仍是完整段视图。

### 4. 中性条目流导入：按扩展名分派，不静默转换

`import` 模块把外部源归一到既有 `Segment` 列表，不新增第二份条目类型：

| 源 | 通路 | 映射口径 |
|---|---|---|
| `.safetensors` | `safetensors::File`（只读头部，按条目 seek+read） | 条目 name/dtype/shape/字节直映；不支持的 wire 由 `Segment::new` 拒绝 |
| `.gguf` | `Gguf::parse` 整份解析 | 标量 `F32`/`F16` 直映；GGML 块量化（Q4_K/Q5_K/Q6_K/Q8_0 等）显式 `Unsupported`，不反量化、不把块字节塞标量段伪装 |
| `.bin` / `.pt` | `Bin::parse`（ZIP_STORED + pickle 栈机） | 既有 storage 九类直映；读不通在 bin 解析层显式失败 |

未登记后缀显式 `Unsupported`。导入侧不挂量化配对：标度/零点仍是源格式自己的条目，配对推断留在既有 `Quantized` 通路。

### 5. 度量打开后现算，不把账本写进文件

`Archive::metrics()` 在打开容器后按现场字节与段表现算，不解析文件内保存的任何度量字段：

- `entries`：段表行数（含引用段）；`params`：Σ `numel`（饱和）；
- `v_artifact`：文件总字节（含头部、对齐填充、两级 CRC）；`v_fp16`：Σ `numel × 2`（FP16 参考体积）；
- `ratio = v_artifact / v_fp16`，分母为 0 记 0；
- `lossless`：所有段都不带量化方案，且 codec ∈ `Store`/`Rans`；带量化或未识别 codec 不计无损；
- `ref_hits`：被去重写成引用的段数；`ref_saved_bytes`：引用段逻辑字节之和（未重复落盘部分，饱和）。

不输出 Roofline / 设备数字，不替量化条目下无损结论；24× 之类的体积门必须跑真实素材实测，格式层只提供口径，不出历史证人结论。

### 6. 保留名段与 CLI 命令面

- 保留名：`llm.mode`（生命周期模式）、`llm.vocab`（词表段占位，字节容器；字节 → `VocabSource` 解析仍属 M4-1/D13，本刀不接 tokenizer）；`opt.` 前缀为优化器状态命名空间。
- CLI 表驱动新增三条命令，复用既有文法表模式，不复制第二份解析：
  - `import --in <源> --out <file.u>`：分派适配通路 → 中性段流 → `Writer` 流式落盘；
  - `inspect --in <file.u>`：条目清单 + 现场度量报告；
  - `check --in <file.u>`：逐段 `load` 解码并逐位还原，触发段/全局 CRC，输出解码条目数；坏文件按 `Code` 口径落非零退出码。
- 旧 `verify --op` 算子校验保留不动；容器校验命令取名 `check`，避免词面冲突。

## 后果

### 正面

- 严格相等张量只存一份字节，引用段继承代表段的 codec/区间/CRC；去重在写入期完成，读取期按代表段视图借用。
- 外部三种主流格式到自有容器有统一归一通路；不可映射类型/块量化在边界显式失败，没有静默降级路径。
- 体积比、参数数、无损标志、共享命中与节省字节在打开后即可现算，口径与旧 v4 `deploy` 对齐但不依赖写入端自报。
- 容器能力沿既有模块增长，条目类型/读取契约/量化方案不另起平行抽象；新增 `import`/`lifecycle` 两个模块各管一段装配。

### 负面

- v1 LLPK 字节不被 v2 读端兼容；仓内虽无交付 v1 产物，外部旧文件需走一次性转换（当前未提供 v1→v2 解码器）。
- GGUF 块量化只登记拒绝，不做反量化；要落低位宽权重需后续码本/格码方案与校准素材。
- 度量是打开期现算的瞬时视图，不写回文件；跨进程对比需重新打开容器。

### 中性

- `llm.vocab` 只是预留段名，本刀不解析词表；模式/补丁语义沿用 0015 三修订口径。
- 引用去重的键含字节摘要，相等性以逐位比对为准；摘要只用于预检，不参与正确性判定。

## 未决（Stage 3，按素材与后端门禁推进，本轮不硬做）

- **码本 / 格码低位宽压缩**：I8 codes + F16 books、E8/Golay/SVD 格码本一类方案；需校准数据与训练侧码本微调，容器只预留 codec 码位与描述符扩展位，本刀不实现编码算法。
- **分片多文件清单与显式随机访问索引**：对接 `shard` 拓扑；大文件 mmap 后端沿用 0036 feature 岛。
- **词表段解析**：`llm.vocab` 字节 → `VocabSource` 归 tokenizer 装配点（D13 / M4-1）。
- **真实权重端到端生成、HTTP 后端、同机对照基准**：受 M2 推理几何、真实素材、设备后端阻塞（见 backlog 矩阵 M4-1…3）。
- **strided / block-sparse 布局**（含 MoE 专家表）：描述符预留布局位，本轮只落 dense packed。

## 备选方案

- **新建平行 crate 承载 v2**：拒绝。两 crate `publish=false`、消费体重叠，`Entry`/`Segment` 是同一抽象；0015 修订已撤销过独立 `pack` crate，本轮不重复悬空。
- **照搬退役 v4 `storage/store.rs` + `cert.rs`**：拒绝。v4 生产者已退役、当前树不可复跑；本轮按现行 `Writer`/`Archive` 骨架重新实现 REF 与计量，只继承口径（逐位校验、FP16 分母、无损标志），不搬历史代码。
- **以 safetensors 作为唯一写格式**：拒绝。safetensors 无版本/特性位、无对齐保证、无容器级校验与去重；自有容器要承担训练快照与部署包双重职责。

## 参考

- ADR-0015：LLPK v1 字节布局、wire 闭集、仿射量化、对齐与两级 CRC
- `docs/lld.md` §E8.2：旧 v4 REF 去重、V_FP16/V_artifact/ratio 口径、`import`/`deploy`/`verify` CLI 与历史证人（历史数字不回填为本轮结论）
- `docs/llmu-parity-optimization-design.md`：不退化清单与对标口径
