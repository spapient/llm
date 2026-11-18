# M4-1 剩余 + M4-2 深度分析与优化（2026-11-18）

- 范围：M4-1 剩余的「`.u` v2 `llm.vocab` 段 → `VocabSource` → `Vocab` → Tokenizer 注入 + 指纹对拍」；M4-2 的「真实 HTTP 传输层（替换桩路由）」，卡 D8。
- 依据：`batch-backend-serving-design.md` §4（既有设计，本稿只**定形其开口决策**，不另立平行方案）、`weights/src/archive.rs:750`、`tokenizer/src/vocab.rs`、`serve/src/{router,request,protocol,lib}.rs`。
- 优化原则对齐 `implementation-task-plan.md` §2：单事实源、闭集表、零生产依赖、复用优先。

## 1. M4-1 现状与缺口

**已可复用（零新增类型）**：

| 件 | locator | 作用 |
|---|---|---|
| `VOCAB_NAME = "llm.vocab"` | `weights/src/archive.rs:750` | 词表段保留名，与权重同 Reader/Loader 通路按名取段 |
| `VocabSource` | `tokenizer/src/vocab.rs:61` | 中性值对象 `{family, pieces, special, scores, merges}`，无格式键名 |
| `VocabFamily` | `vocab.rs:13` | 5 族 + `requires_merges()`/`requires_scores()` **声明式附件判定** |
| `Vocab::try_new` | `vocab.rs:100` | 闭合双向查表/编号范围/族附件不变量 |
| `Fingerprint` | `fingerprint.rs` | FNV-1a `{family, vocab_size, digest}`，`fold_merges`/`fold_scores` 并入附件 |
| `seg_crc` | `archive.rs:738` | 段级 CRC，词表段沿用同一字节完整性口径 |

**缺口**：段内编码格式未定（design §4.2 "由后续刀定"）、适配器未接、指纹对拍未接。

## 2. M4-1 优化设计：自描述中性布局 + 单份表驱动解析

**核心减码点**：5 个族共用一个解析器，附件块是否读取**由 `VocabFamily::requires_merges/requires_scores` 两个 const fn 声明式决定**，不写 5 份族解析；格式键名不进入 tokenizer（适配器在 weights→serve 装配点完成映射）。

段内布局（镜像 `protocol.rs` 固定头 + 载荷 + CRC 风格，小端）：

```text
magic(u32) | version(u16) | family(u8) | vocab_size(u32) | special_count(u32)
| pieces:  vocab_size × (len(u32) + bytes)
| special: special_count × (len(u32) + bytes)
| [Bpe] merges:   count(u32) + count × (len_l(u32)+l | len_r(u32)+r)
| [Unigram] scores: count(u32) + count × f32
| digest(u64)          ← 期望 Fingerprint.digest（FNV-1a，语义对拍用）
| crc32(u32)           ← 段字节完整性（沿用 seg_crc 口径）
```

- **单份解析器**：`parse_vocab(bytes) -> VocabSource`。头部读 `family/vocab_size/special_count`；词片/特殊共用同一个 `len` 前缀字节序列读取助手；附件块只在 `family.requires_merges()` / `family.requires_scores()` 为真时读，各读一个块。**没有按族名 `match` 的扇出**。
- **指纹对拍（构造期，不静默降级）**：适配器产出 `VocabSource` → `Vocab::try_new` 得 `Vocab::fingerprint()`；与段尾 `digest` 比对，不符 `Code::Invalid`。`digest` 与段 CRC 是两层：CRC 守字节、digest 守「词表与模型预期」语义一致。
- **归属裁决（沿用 design §4.2 已定）**：解析适配器放装配点；`weights` 不依赖 `tokenizer` 类型，只按名给段字节；`Completion` 持 `Box<dyn Tokenizer>`，不持词表字节副本。

## 3. M4-2 HTTP：深度分析 + D8 裁决

**已可复用（HTTP 只是薄适配器，硬活全已零依赖落地）**：

| 件 | locator | 作用 |
|---|---|---|
| `Router::call(method, path, body)` | `router.rs:58` | 方法+路径匹配、404/重复挂载，无 socket |
| `Request::parse` | `request.rs:36` | 最小行文法（id/prompt/max_new），零 JSON 依赖 |
| `Response::frame` / `Envelope/Frame/Decoder` | `lib.rs:142` / `protocol.rs` | 二进制帧 + CRC，读写两端已闭环 |
| `MAX_PAYLOAD` | `protocol.rs` | 载荷上限，协议常量非配置项 |

**D8 三方案对比**：

| 方案 | 依赖 | 代码量 | 与 D5/ADR-0042 |
|---|---|---|---|
| axum/tokio | 破零生产依赖 | 少（框架代劳） | 需 D5 例外 ADR；与"传输与端点分离"可共存但引入生态面 |
| 手写完整 HTTP/1.1（keep-alive/chunked/TLS） | 零 | 爆炸 | 大量边界代码，非本路线需要 |
| **std-only 薄适配器**（推荐） | 零 | ~150–250 行 | 与 D5 零依赖、ADR-0042 分离原则完全一致 |

**裁决建议 = std-only 薄适配器**：`std::net::TcpListener` + HTTP/1.1 最小子集——

1. 读请求行 `METHOD SP PATH SP HTTP/1.1` + `Content-Length`（只解析这一个头）+ 主体；
2. 交给 `Router::call(method, path, body)`（路由/404/请求文法已在位）；
3. 响应 `Response::frame()` 字节包进最小响应 `HTTP/1.1 {status}\r\nContent-Length: N\r\nConnection: close\r\n\r\n` + body。

关键点：**HTTP 层不重写路由、请求文法、帧格式、CRC、上限**——它只是「socket 字节 → `Router::call` → `Response::frame` → socket 字节」的薄胶水。gRPC 未来在同一条 `Endpoint`/`Router` 上再挂一个适配器即可，不动端点模型。

**明确排除（首刀）**：TLS（前置代理终结）、keep-alive/chunked（`Connection: close` + 只认 `Content-Length`）、多线程调度（沿用 `State` 的 `Mutex` 现状，单 accept 循环先保证正确）。

## 4. 减码/配置化收口（两处接缝同一纪律）

- 字节进 → 中性值对象 → 既有判据（`Vocab::try_new`/`Fingerprint`/`Router::call`/`Response::frame`），**不新增格式键名、不新增解析器依赖、不新增平行类型**。
- 唯一新代码是两段薄胶水：词表段解析适配器（一份，表驱动）+ HTTP accept 循环（一份，std::net）。两者都指向既有事实源，不复制第二份映射。

## 5. 门禁与负例

- M4-1：容器内词表段往返对拍；指纹不符 → `Invalid`；编号越界/族附件缺失或多余 → `Invalid`；空词表/特殊 token 越界负例。出口必须**零外部词表依赖、零新生产依赖**。
- M4-2：桩路由契约对拍全绿（不替换 `Router` 语义）；请求侧错请求 400、未命中 404、超 `MAX_PAYLOAD` 拒绝；并发下批次与页账一致；`cargo deny` 仍全绿。

## 6. 待拍板的两个决策

1. **词表段内格式**：自描述中性布局（推荐，零依赖、单源） vs 复用外部词表格式（tokenizer.json/sentencepiece，引入外部解析语义）。推荐前者。
2. **D8 HTTP 选型**：std-only 薄适配器（推荐，符合 D5） vs 引入 axum/tokio（需 D5 例外 ADR）。推荐前者。

以上两决策一经确认，即可按 `batch-backend-serving-design.md` §4.3 的刀 4-1 / 4-3 直接动工；本稿不自行翻转 D8，也不预写实现。
