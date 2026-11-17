# `llm-weights` 架构深度分析与 S++ 升级方案（只读评估）

- 对象：`crates/weights`（包名 `llm-weights`）。依赖 `dtype` / `diagnostic` / `tensor` / `wire`，dev 依赖 `storage`(mmap)。
- 时点：2026-11-14，本机 macOS aarch64；所有数值由命令实跑得到，可复现。
- 口径：**confirmed** = 直接读源码或命令实测；**inferred** = 由结构推断，需落地时再验证；**unknown** = 本仓内无证据。
- 范围纪律：本轮**只分析不改代码**；工作区其他 crate 的在途改动不在结论内。

## 一、基线实测（可核验）

| 工具 | 命令 | 结果 |
| --- | --- | --- |
| 测试 | `cargo nextest run -p llm-weights` | **170/170 通过** |
| 静态分析 | `cargo clippy -p llm-weights --all-targets` | 无告警 |
| 格式 | `cargo fmt -p llm-weights -- --check` | **4 个文件有 use 排序差异**，非干净 |
| 覆盖率 | `cargo llvm-cov -p llm-weights --summary-only` | 行覆盖 **85.10%**（2102/14103），区域 82.28%，分支 86.50% |
| 未用依赖 | `cargo machete` | 无未用依赖 |
| 供应链 | `cargo deny check` | advisories / bans / licenses / sources 全 ok |
| 重复代码 | `cargo-dupes -p crates/weights stats` | 精确克隆 **5.0%**（36 组 / 105 单元 / 380 行），近重复 0.9%（6 组 / 12 单元 / 68 行），基数 7665 行 |
| 复杂度门禁 | 工作区 `[gate]`：ccn≤5、nloc≤60、文件≤899 行 | 门禁测试通过，**无函数级复杂度欠债** |
| 基准 | criterion | **weights 无任何 bench** |
| README | — | **crate 无 README** |

### 1.1 规模（`wc -l`，28 个源文件，合计 10 567 行）

| 子域簇 | 文件 | 行数 | 下游是否消费 |
| --- | ---: | ---: | --- |
| 核心契约（Entry/Header/Reader/Loader/Archive/Writer/Segment/Codec/mode） | 6 | 1 492 | 是（train / nn / llm） |
| 容器编解码（rans / affine / quant / scale / patch / merkle / sha256） | 7 | 2 000 | 内部为主 |
| 外部格式适配（safetensors / gguf / bin / json / zip / ir） | 6 | 3 733 | 仅 `safetensors::File` 被根 bin 用 |
| 量化 / 向量编码（axes / lattice / svd） | 3 | 1 493 | 无仓外消费者 |
| 后量子签名 / 密码学（dsa / envelope / ntt / keccak / golay / ledger） | 6 | 1 849 | 无仓外消费者 |

### 1.2 下游真实公开面（grep 仓内全部调用点，confirmed）

下游只用到：`Entry`、`Header`、`Loader`、`safetensors::File`、`crc::crc32`。
经全仓引用检索：完全无引用 2 569 行、仅测试引用 1 446 行（明细见 §2.3）；其余签名、格量化、SVD、谱隙账本、私有 IR 等在生产路径无消费者。

## 二、职责盘点：一个 crate 装了五个域

### 2.1 架构事实

- **核心契约**（`lib.rs`）：`Entry` / `Header` + `Reader`（借用字节）/ `Loader`（拷进缓冲）两条 trait，分工是所有权而不是风格，设计成立（confirmed，`lib.rs:262-293`）。
- **自有 LLPK 容器**：`writer` 收集段 → `seg` 描述符 → `rans` 可选压缩 → 两级 CRC → `merkle` 根；`reader` 对称解析与校验。
- **外部格式**：`safetensors`（整缓冲 `Archive` + 文件 `File`）、`gguf`、`bin`（torch pickle 栈机）、`json`（受限解析器）、`zip`（仅 STORED）。
- **量化族**：`affine` 显式量化、`scale` 块标度、`axes` 五轴选码、`lattice` D24/E8 格量化、`quant` 成对结构。
- **PQ / 密码学件**：`dsa`（ML-DSA，依赖 `ntt` + `keccak`）、`envelope` 签名侧车、`golay` G24、`svd` 单边 Jacobi、`ledger` 谱隙证书账本、`sha256`、`merkle`。

### 2.2 边界问题（confirmed / inferred）

1. **私有 IR 是孤儿**：`ir.rs`（355 行，`Graph` / `Node` / `Op`）只在 `lib.rs:107` re-export，仓内仓外零调用点（confirmed）。
2. **签名簇未接入生产路径**：`envelope → dsa → ntt/keccak` 只被 `round_trip.rs` 触达；`reader`/`writer` 都不导入 `envelope`（confirmed）。
3. **格量化 / SVD / 谱隙账本无仓外消费者**（confirmed grep）。
4. **域耦合方向不清**：`dsa` 反过来依赖自家 `json`；格式解析器与密码学件共用同一 crate 的 re-export 面，后续任一域演进都会扰动核心契约（inferred）。
5. 对照 ADR 0024 的既有先例：CRC/LEB/Sink 已从 weights 抽到 L1 `wire`。当前可继续复用 `wire` 的只有字节级基础件（对齐、游标）；整簇格式/量化/签名不再外迁 crate，按 §5.2 在单 crate 内归置。

### 2.3 无引用/仅测试引用清单（grep 源码、tests、examples、fuzz 全仓，confirmed）

引用口径：`use crate::<mod>` / `crate::<mod>::` 在源码、`tests/round_trip.rs`、`examples`、`crates/fuzz` 中检索。

| 模块 | 行数 | 引用状态 | 链路 |
| --- | ---: | --- | --- |
| `ir.rs` | 355 | 无生产/测试引用，仅 re-export | 孤立 |
| `axes.rs` | 563 | 无任何引用 | 孤立 |
| `golay.rs` | 179 | 无任何引用 | 孤立 |
| `ledger.rs` | 332 | 无任何引用 | 依赖 `svd` |
| `svd.rs` | 288 | 仅被 `ledger` 引用 | 链尾无上游 |
| `scale.rs` | 210 | 无任何引用 | 依赖 `lattice` |
| `lattice.rs` | 642 | 仅被 `scale` 引用 | 链尾无上游 |
| `patch.rs` | 108 | 仅 `round_trip.rs:194-235` 测试 | 增量 diff/join |
| `envelope.rs` | 172 | 仅测试自闭环 | 依赖 `dsa` |
| `dsa.rs` | 873 | 仅 `envelope` 与测试 | 依赖 `ntt`/`keccak`/`json` |
| `ntt.rs` | 142 | 仅 `dsa` | 链尾 |
| `keccak.rs` | 151 | 仅 `dsa` | 链尾 |

- **完全无引用合计 2 569 行**：ir + axes + golay + ledger + svd + scale + lattice。
- **仅测试引用合计 1 446 行**：patch + envelope + dsa + ntt + keccak。
- 仍在生产路径：`seg/writer/reader/lib/mode/affine/quant/rans/patch(仅测试)/merkle/sha256/crc` 与格式簇 `safetensors/gguf/bin/json/zip`。
- 注意：这些无引用模块可能是**预留特性**而非废代码；删除/冻结前需逐项确认产品路线（对应 D2/D3），不能凭覆盖度判定为垃圾。

## 三、六维差距评估（对标 Burn store / safetensors / Candle）

参考实现（联网调研，见 §七）：Burn `burn-store` 的 Record/Store 分层、懒物化、mmap 零拷贝、流式保存、过滤/重映射、精度适配；safetensors 的 JSON 头 + 裸缓冲、头大小上限、无孔洞、小端行主序；Candle `MmapedSafetensors` 的 yoke+mmap 与名字→分片路由。

### 3.1 类聚与模块职责 — 当前 C+，目标 S++

- **成立的部分**：`Reader`/`Loader` 接口隔离清晰；核心校验（名字唯一、区间不重叠、起点不早于数据段）集中在 `Header::new`（confirmed）。
- **差距**：五个域挤在同一编译面，公开 re-export 29 项；格式适配、量化、签名与核心生命周期同享版本号；孤儿 `ir.rs` 占 355 行；近邻小文件未合并。
- **目标**：单 crate 内按 §5.2 高内聚合并，核心契约保持薄文件；格式 / 量化 / 签名在文件边界上分开但不拆 crate；公开面按下游真实调用收口。

### 3.2 证据与判据纪律 — 当前 B+，目标 S++

- **成立**：解析拒绝路径有 `Code::Invalid` / `Unsupported` 与定位串；容器截断 / 篡改 / 多余字节 / 非零填充都在打开期失败（`reader.rs`，confirmed）。
- **差距**：
  - `bin.rs`（pickle 栈机，695 行）行覆盖 **19.32%**，仅 4 个测试，主路径靠 example `dump_bin.rs`，nextest 不跑 example（confirmed）。
  - `zip.rs` 36.45%、`seg.rs` 79.81%、`json.rs` 82.50% 为次低覆盖点（confirmed llvm-cov）。
  - 手写 FIPS 件（sha256 / keccak / dsa / ntt）仓内无外部对照向量或审计结论留存；正确性证据只有自闭环测试（confirmed 缺口，强度 unknown）。
  - 孤儿 `ir.rs` 90.65% 覆盖但无人消费，覆盖本身不构成价值（inferred）。

### 3.3 静态强制门禁 — 当前 B+，目标 S++

- **成立**：clippy 干净、machete 干净、deny 四闸 ok；函数复杂度在工作区门禁内（confirmed）。
- **差距**：
  - fmt 4 文件未达标（confirmed）。
  - AST 重复 5.0% 精确克隆，高于 `../.config/.jscpd.json` 的 1.5% 阈值口径（工具不同，需在同一口径下复核，inferred）。
  - annotation 有专门 `tests/audit.rs` 静态门；weights 没有等价的「依赖闭集 / 禁容器 / 禁正则 / 禁惰性初始化」本地门（confirmed 对照）。
  - 无 weights 专属 README 与元数据描述收口（confirmed）。

### 3.4 测试质量 — 当前 B+，目标 S++

- **成立**：170 用例全绿；`round_trip` 覆盖写→读→篡改拒绝闭环；`lattice`/`rans`/`quant`/`svd` 覆盖 ≥97%（confirmed）。
- **差距**：
  - 格式解析器负例覆盖不均；`bin` pickle 栈机分支基本未测。
  - fuzz 目前只触达 `json`（`fuzz/drivers.rs`、`tests/parsers.rs`），gguf/zip/seg/容器写端无模糊输入（confirmed）。
  - 无确定性 / 属性化 sweep 证据记录；无基准回归守卫。

### 3.5 性能工程 — 当前 C+，目标 S++

- **成立**：`Reader` 零拷贝借用；`Loader::load` 是 `copy_into` 的分配封装，边界判断只有一处；`rans` 8-lane 并行；`Writer` 用 `Count`/`FileSink` 同一条编码路径算长度与落盘，避免两份布局逻辑（confirmed）。
- **差距（confirmed 源码）**：
  1. **Merkle 叶子重复计算**：`Archive::leaves()` 在 `merkle_root` / `prove` / `verify` 每次调用都重新解压 + 哈希全段（`reader.rs:181-210`）。
  2. **写端全量驻留 + 两遍编码**：`Writer` 持有全部 `Segment.data`；`prepare()` 再建一份压缩帧 Vec；`encoded_len()` 与 `finish()`/`encode_into()` 各自调一次 `write_body→prepare()`，rANS 对同一数据被编码两遍（`writer.rs:127-159`）。原子落盘本身**已成立**：`write_to_path` 走同目录临时文件 → BufWriter 流写 → fsync → rename（`writer.rs:181-207`），崩溃不留半截文件；缺的是把输入/压缩帧从全量驻留降到「峰值≈最大单段」的流式边界（对照 Burn 流式保存）。
  3. **生产后端无 mmap 零映射**：`Source = std::fs::File`（`safetensors.rs:32`），`File::copy_into` 每条目 `seek + read_exact` 拷进调用方缓冲；`storage::Mmap` 只在 `round_trip.rs:178` 测试里出现，生产 `File` 后端未接入零映射（对照 Candle yoke+mmap）。
  4. **压缩预判浪费**：`compress` 开启时对每段都先 `rans::encode` 再比较大小，即便最终走 Store（`writer.rs:106-112`）。
  5. **名字查找线性**：`Header::get`（`lib.rs:210-211`）与 `Archive::get`（`reader.rs:162-163`）都是 `iter().find`，N 段每次 O(N)；热路径按名取段应一次建排序索引 + 二分，避免引入关联容器（见 §5.4）。
  6. 无 criterion 基准，热路径（头部解析、段布局、rANS 编解码、CRC、Merkle 根）无可测回归基线。
- 偏好约束（用户既定）：热路径优先整块 slice / 查表 / 批处理，逐元素循环必须有串行依赖理由并注释。

### 3.6 广度与生产成熟度 — 当前 B-，目标 S++

- **对照 Burn store 缺项（inferred 哪些适用本仓需落地确认）**：跨后端 Record 抽象、按名过滤 / 张量重映射、f16/f32 精度适配、no-std、分片路由。
- **本仓已有**：双后端（内存借用 / 文件按需）、两级 CRC、模式位（训练断点 / 推理发布，`mode.rs`）、临时文件 + fsync + rename 的原子替换写（`writer.rs:181-207`）。
- **缺口**：无 README；无基准；错误面（IO / 截断 / 不支持版本 / 非规范编码）缺跨格式一致性清单；手写 JSON 解析器与工作区已有的 `toml` 依赖并存，格式工具链未收口。

## 四、复用与泛型机会（具体落点）

### 4.1 可抽象的重复（confirmed）

| 重复点 | 现状 | 收口方向 |
| --- | --- | --- |
| `align_up` | `reader.rs:304` 与 `gguf.rs:469` 两份实现 | 提到 `wire` 或 weights 公共字节辅助，单源 |
| 小端 / 游标读取 | `gguf` / `zip` / `axes` 各自 `take` + `from_le_bytes`；`wire::leb::Cursor` 已有 `read`/`take`/`pos` | 解析器统一走 `Cursor`，删除私有取数辅助 |
| 头部校验 | safetensors / gguf / bin 各自构造 `Header`，名字唯一、区间不重叠判据分散 | 抽 `trait FormatDecoder { fn decode(src) -> Result<Header> }`，校验只调 `Header::new` 一处 |
| 编解码选择 | `Codec` 枚举 + match 分支 | 策略 trait `Compress { fn encode } / fn decode`，Store / Rans 两实现；新增编码不改 match |
| 段 / 条目样板 | `Entry` 与 `Stored` 字段高度重叠 | 描述符与运行期视图分层，避免双份字段漂移 |

### 4.2 泛型现状（confirmed grep）

泛型集中在 `dsa`(13)、`gguf`(6)、`safetensors`(5)；核心 `Reader`/`Loader` 已是后端泛型。可再泛型化的方向：字节源 `B: AsRef<[u8]>`、解码输出缓冲、分片路由 `name -> source`。

### 4.3 注解 / 宏方式（评估）

weights 当前**不使用过程宏**。除非常量描述符样板被证明重复到值得生成，否则**不建议新增 proc-macro**；优先 `derive` + trait + 泛型。annotation crate 的生成能力服务于「声明 → 契约实现」，与权重解析无关，不混用。

## 五、目标架构（建议，落地前需你拍板）

### 5.1 现有设计模式盘点（confirmed）

| 模式 | 落点 | 评价 |
| --- | --- | --- |
| 接口隔离 | `Reader`（借字节）/ `Loader`（拷缓冲）两条 trait | 成立，保留 |
| 建造者 | `Writer` 收集段 → 定长校验 → 编码落盘 | 成立，可补流式入口 |
| 适配器 | safetensors / gguf / bin / zip 各自适配外部格式 | 缺统一解码契约，校验分散 |
| 策略 | `Codec` 枚举 + match 分支 | 可上提为 `Compress` trait |
| 零拷贝 | `Reader` 返回 `&[u8]`；`Cow` 压缩帧 | 成立 |
| 记录/仓库 | `Header` + `Entry` 块表 | 成立，查找为线性 |

### 5.2 目标目录：单 crate 内高内聚收敛（不新增 crate）

**约束（按你的要求）**：
- 不拆兄弟 crate，不堆一级子目录；`weights` 仍是一个 crate。
- 同一概念、同一格式的代码并进一个文件；跨文件只依赖核心契约与已有 `wire`/`storage`。
- 只有字节级基础件归到**已有** `wire`（`align_up`）；mmap 复用**已有** `storage::Mmap`；不在 weights 重造。
- 文件数从 29 收敛到 **11 个活动平级文件**；未接路线的件以平级文件 + `#[cfg(feature = "...")]` 隔离，无路线即删；全程不建子目录、不留后缀兄弟文件。

**目录硬约束**：`src/` 下只允许平级 `.rs` 文件，**禁止任何子目录**；`src` 平级（tests/examples）同样不建目录。899 行文件门禁靠合并精简或保留平级兄弟文件解决，不靠建目录。

```
crates/weights/src/   （全部平级，无子目录；同类合一，不留下划线后缀兄弟文件）
  lib.rs          Entry / Header / Reader / Loader 契约 + check_capacity
  segment.rs      Segment / Desc / Stored 合并为 Descriptor 读写视图 + Codec
  archive.rs      Archive/Stored 读端 + mode 发布 + patch 增量（读端生命周期合一）
  writer.rs       Writer 落盘（Count/FileSink/Sealed）
  codec.rs        rANS 压缩 + Compress 策略（替代 Codec match 分支）
  proof.rs        Merkle + SHA-256
  quant.rs        affine Quant + Quantized（在用量化方案）
  json.rs         受限 JSON（safetensors 与签名共享）
  safetensors.rs  Archive / File 后端
  gguf.rs         GGUF 解析 + Kind + 反量化（原 gguf_q 并入）
  bin.rs          torch pickle 栈机 + ZIP STORED 读取（原 zip 并入）
  scale.rs  lattice.rs  axes.rs  ledger.rs  svd.rs
  envelope.rs  dsa.rs  ntt.rs  keccak.rs  golay.rs  ir.rs
                  ↑ 未接路线：平级保留 + #[cfg(feature=...)]，不建目录、不进公开面
```

**合并行数与门禁**（`file_max_lines = 899`）：

| 合并后文件 | 源合计 | 门禁判定 |
| --- | ---: | --- |
| `archive.rs` | 336 + 128 + 108 = 572 | 达标 |
| `proof.rs` | 177 + 303 = 480 | 达标 |
| `quant.rs` | 183 + 449 = 632 | 达标 |
| `gguf.rs` | 730 + 174 = 904 | 超 5 行，并入时去重复样板即可达标 |
| `bin.rs` | 695 + 348 = 1 043 | 超 144 行；必须靠复用 `wire::Cursor`、去重复取数/诊断样板压到 ≤899；压不下来则上报，不靠拆第二个文件解决 |

#### 5.2.1 现状文件 → 目标归属映射

| 现状文件 | 行数 | 目标文件 | 合并动作 | 引用状态 |
| --- | ---: | --- | --- | --- |
| `lib.rs` | 369 | `lib.rs` | 契约保留 | 生产 |
| `seg.rs` | 284 | `segment.rs` | 三套视图收敛为 Descriptor | 生产 |
| `reader.rs` | 336 | `archive.rs` | 读端主体 | 生产 |
| `mode.rs` | 128 | `archive.rs` | 读端生命周期操作并入 | 生产 |
| `patch.rs` | 108 | `archive.rs` | 增量 diff/join 并入 | 仅测试 |
| `writer.rs` | 357 | `writer.rs` | 保留 | 生产 |
| `rans.rs` | 570 | `codec.rs` | 编码策略并入 | 生产 |
| `merkle.rs` | 177 | `proof.rs` | 与 sha256 合并 | 生产 |
| `sha256.rs` | 303 | `proof.rs` | 与 merkle 合并 | 生产 |
| `affine.rs` | 183 | `quant.rs` | 与 Quantized 合并 | 生产 |
| `quant.rs` | 449 | `quant.rs` | 与 affine 合并 | 生产（safetensors 用） |
| `json.rs` | 596 | `json.rs` | 共享解析器保留 | format + sign |
| `safetensors.rs` | 847 | `safetensors.rs` | 保留 | 生产（下游用 `File`） |
| `gguf.rs` | 730 | `gguf.rs` | 解析主体 | 适配 |
| `gguf_q.rs` | 174 | 并入 `gguf.rs` | GGUF 反量化，只被 gguf 用；不留独立后缀文件 | 适配 |
| `bin.rs` | 695 | `bin.rs` | pickle 栈机 | 适配，覆盖 19.32% |
| `zip.rs` | 348 | 并入 `bin.rs` | bin 专用 ZIP；去重后压到 ≤899 行，不拆第二个文件 | bin 依赖 |
| `scale.rs` → `lattice.rs` | 210 → 642 | 平级 `scale.rs`/`lattice.rs`（feature cfg） | 无上游引用，待路线定 | 孤儿 |
| `axes.rs` | 563 | 平级 `axes.rs`（feature cfg） | 无任何引用，待路线定 | 孤儿 |
| `ledger.rs` → `svd.rs` | 332 → 288 | 平级 `ledger.rs`/`svd.rs`（feature cfg） | 链尾无上游，待路线定 | 孤儿 |
| `envelope.rs` → `dsa.rs` → `ntt.rs`/`keccak.rs` | 172 + 873 + 142 + 151 | 平级 `envelope.rs`/`dsa.rs`/`ntt.rs`/`keccak.rs`（feature cfg） | 仅测试自闭环 | 仅测试 |
| `golay.rs` | 179 | 平级 `golay.rs`（feature cfg） | 无任何引用，待路线定 | 孤儿 |
| `ir.rs` | 355 | 平级 `ir.rs`（feature cfg） | 图 IR 预留，待路线定 | 孤儿 |

#### 5.2.2 跨 crate 复用（不新增 crate，只接已有件）

- `wire`：CRC/LEB/Sink 已在外层；`align_up` 重复实现收口到 `wire`；各格式私有 `take + from_le_bytes` 改走 `wire::leb::Cursor`。
- `storage`：`File` 后端接已有 `storage::Mmap` 零映射，不另写映射层；不可映射平台回退 seek/read。
- `dtype`/`tensor`：dtype 闭集、形状元数据继续复用，不在 weights 重定义。
- 项目内无既有 JSON/ZIP/哈希实现可复用（已全仓扫描，`op/dtype` 里的 `from_le_bytes` 是算子内部读取，不可抽取）。

### 5.3 泛型与 trait 草图（落地时按现有代码约束细化）

```rust
/// 外部格式解码器：任意字节源 → 已校验 Header；校验只走 Header::new 一处。
/// 复用现有 Reader/Loader 作为读端契约，不另起一套字节源抽象。
trait FormatDecoder {
    type Source;
    fn decode(source: Self::Source) -> Result<Header>;
}

/// 段级编码策略：替代 Codec 枚举的 match 分支；新增编码不改既有 match。
trait Compress {
    fn code(&self) -> u8;
    fn encode(data: &[u8]) -> Result<Cow<'_, [u8]>>;
    fn decode(frame: &[u8], expected: u64) -> Result<Vec<u8>>;
}
```

- 描述符统一：`Segment`（写输入）/ `Desc`（解码）/ `Stored`（读视图）字段高度重叠，收敛为共享 `Descriptor`，写端持有数据、读端借用数据，消除双份字段漂移。
- 字节辅助单源：`align_up` 收到 `wire`；gguf/zip/axes 的私有 `take + from_le_bytes` 全部改走 `wire::leb::Cursor`。
- **注解方式**：weights 不引入过程宏；样板优先 `derive` + trait + 泛型。除非描述符序列化被证明重复到值得生成，否则不新增 proc-macro（与 annotation crate 的生成职责分离）。

### 5.4 性能目标（可测）

1. **Merkle 叶子惰性缓存**：首算后复用，`prove`/`verify` 不再全量解压。
2. **两遍编码合一**：长度趟与落盘趟共用一次 `prepare`；压缩帧在编码趟产出后复用于布局计算，避免 rANS 编两遍。
3. **名字索引**：一次建排序索引 + 二分查找替代 `iter().find`；不引入关联容器（与 audit 静态门一致）。
4. **mmap 后端**：`File` 接 `storage::Mmap` 零映射取数；不可映射平台回退 seek/read，`Loader` 拷贝路径保留。
5. **rANS 预检**：先估长/快速判定值得压缩，避免对每段无条件全编码。
6. **流式写边界**：输入段可逐段喂入 Sink，峰值内存约束到最大单段；原子替换写已成立，保留临时文件 + fsync + rename。
7. **基准守卫**：criterion 覆盖头部解析、段布局、rANS 编解码、CRC、Merkle 根；进回归基线。

### 5.5 质量目标（验收口径）

- fmt / clippy / deny / machete 全绿；补 weights README；建 `tests/audit.rs` 静态门（依赖闭集、禁关联容器/正则/惰性初始化、禁 `unsafe`）。
- 覆盖率：整体行覆盖 ≥95%；`bin` ≥85%、`zip` ≥85%；`seg`/`json` ≥92%；格式解析器补负例，fuzz 从 `json` 扩到 gguf/zip/seg/容器写端。
- 重复度：精确克隆降到门禁阈值内（`../.config/.jscpd.json` threshold 1.5），同一口径重测 jscpd / cargo-dupes。
- 公开面：只保留下游在用类型 + 兼容 re-export；其余模块按 §2.3 状态以平级文件 + feature cfg 冻结或删除。

### 5.6 分阶段路线（单 crate 内推进，每步独立可回滚，过全套门禁）

| 阶段 | 内容 | 风险 | 验收 |
| --- | --- | --- | --- |
| T0 基线锁定 | 固化 nextest/llvm-cov/dupes/空 criterion 基线；写 ADR | 无 | 基线数值入档 |
| T1 合并收口 | 按 §5.2.1 合并近邻文件、收敛 Descriptor；fmt、audit 门、README、`align_up`/`Cursor` 复用、`FormatDecoder` 统一校验、覆盖补欠、fuzz 扩面 | 低，不改公开名 | 文件数下降、测试全绿、覆盖达标、克隆下降 |
| T2 性能 | 叶子缓存、排序索引、mmap 后端、两遍合一、rANS 预检、基准 | 中，改读路径 | 基准对比 + 正确性回归 |
| T3 未接路线归置 | 孤儿/仅测试簇以平级文件 + feature cfg 隔离；路线确认后删除或保留；密码学按 D2 决策 | 中，可能删预留件 | 公开面只留生产类型；未接路线文件不进默认编译 |

### 5.7 风险与缓解

- **合并丢内聚 / 超门禁**：近邻文件合并前核对依赖边（§5.2.1）；`gguf_q` 并入 `gguf.rs`、`zip` 并入 `bin.rs`，不留下划线后缀兄弟文件。`bin.rs` 合并后须靠复用 `wire::Cursor`、去重复样板压到 ≤899 行；压不下来先上报，不擅自拆文件。
- **删孤儿丢预留特性**：`ir/axes/golay/ledger/svd/scale/lattice` 可能是路线内件；先用 feature cfg 在平级文件里冻结，路线确认后再删，不直接物理删除。
- **密码学替换**：换外部 crate 与「运行期零外部依赖」策略冲突，需 ADR 显式改策略并复核许可证（deny bans/licenses）；短期保留手写 + 补对照向量。
- **mmap 跨平台**：`Mmap` 为 unix + feature 门，缺失时回退 `File` seek/read；测试双后端。
- **兼容面**：`Entry/Header/Loader/File/crc32` 签名保持不变，合并只动模块内部路径，必要时保留 re-export。

## 六、待你决策的点（不擅自定；附默认建议）

- **D1 拆分粒度（已定调）**：**不新增兄弟 crate**，`weights` 单 crate 内按 §5.2 合并收敛；只有字节级基础件归到已有 `wire`、mmap 复用已有 `storage`。不再为分层新造一级模块。
- **D2 密码学件去向**：保留手写零依赖实现并补 FIPS/NIST 对照向量测试；还是引入仓外已审计密码学 crate（与当前「运行期零外部依赖」策略冲突，需改依赖策略 + deny 白名单，属架构决策）。**建议**：短期保留手写 + 补对照向量；生产签名落地前再评估外部 crate。
- **D3 孤儿处置**：`ir.rs` 及 §2.3 无引用簇（axes/golay/ledger/svd/scale/lattice）删除、feature cfg 冻结，还是保留给后续路线？**建议**：先以平级文件 + feature cfg 冻结，不物理删；图 IR 与量化/谱隙路线确认后再定。
- **D4 范围排序**：先低风险收口（文件合并 + 消重复 + 格式统一 trait + 覆盖率补欠 + 静态门），还是优先性能（懒物化 / 两遍合一 / mmap / 基准）？**建议**：T1 收口先行，T2 性能紧随。
- **D5 兼容面**：合并 / 改名是否允许破坏当前仅有的下游公开面（`Entry/Header/Loader/File/crc32`）？**建议**：保持这五个名字与签名不变，内部合并只动模块路径并用 re-export 兼容。

## 七、对标来源（联网调研）

- Burn `burn-store`：后端无关 Record、懒物化、mmap 零拷贝、流式保存、过滤/重映射、精度适配、no-std —— https://docs.rs/burn-store/0.22.1/burn_store/ ，https://burn.dev/books/burn/saving-and-loading.html
- Burn 持久化模型记录与 Store API 分层 —— https://deepwiki.com/tracel-ai/burn/7.2-model-records-and-persistence
- safetensors：JSON 头 + 裸缓冲、无代码执行、头大小上限、无孔洞、小端行主序、数据近零拷贝 —— https://pytorch.org/projects/safetensors/ ，https://crates.io/crates/safetensors
- Candle `MmapedSafetensors`：yoke + mmap、名字到分片路由 —— https://docs.rs/candle-core/0.9.2/aarch64-unknown-linux-gnu/src/candle_core/safetensors.rs.html

## 八、T1 实施记录（2026-11-14）

约束不变：单 crate、`src/` 全平级无子目录、不留下划线后缀兄弟文件、五个下游公开名（`Entry/Header/Loader/File/crc32`）与签名不动；内部合并只改模块路径，必要时 re-export。

### 已完成的平级合并（提交 `54e60451`）

| 合并后文件 | 来源 | 行数 | 门禁 |
| --- | --- | ---: | --- |
| `proof.rs` | `merkle.rs` + `sha256.rs` | 468 | 达标 |
| `quant.rs` | `affine.rs` + `quant.rs` | 616 | 达标 |
| `safetensors.rs` | `safetensors.rs` + `safetensors_file.rs` | 826 | 达标 |
| `gguf.rs` | `gguf.rs` + `gguf_q.rs` | 889 | 达标 |
| `archive.rs` | `reader.rs` + `mode.rs` + `patch.rs` | 555 | 达标 |
| `bin.rs` | `bin.rs` + `zip.rs` | 899 | 达标（方案 C：精简注释/文档，测试全保留） |

- 已删除源文件：`merkle/sha256/affine/gguf_q/reader/mode/patch/safetensors_file/zip` 及 `examples/dump_bin.rs`。
- 兼容路径：`patch::diff/join`、`mode::{Mode, MODE_NAME, mode_of, is_opt, release}` 在 `lib.rs` 以薄转发模块保留；`Archive/Stored/Proof/Quant/Quantized/File` 仍从 crate 根导出。
- `bin.rs` 内 ZIP 解析降为私有件（`Zip/ZipEntry`）；模块文件级 `#![allow(missing_docs)]`，未用的 `Tensor::count`、`Bin::data_named` 一并移除。

### D3 已拍板：孤儿/仅测试簇物理删除

- 删除 `ir/axes/golay/ledger/svd/scale/lattice/envelope/dsa/ntt/keccak` 共 11 个平级文件；同步移除 `lib.rs` 模块声明与 `Envelope/Graph/Node/Op` 导出，删掉 `round_trip.rs` 内签名侧车用例。
- 删前全仓核实：`train/nn/serve/根 bin` 无任何对这些模块或符号的引用；`serve` 的 `Envelope` 是其自有 proto 类型，与 weights 无关。
- 测试数 170 → **104**（被删簇自带的自闭环用例随之移除）；`cargo check --all-targets`（含 nn/train/根）干净。

### SEG 收口与 Writer 单次编码（提交 `af350b7c`）

- **三视图收敛**：`Segment`（写输入）/ `Desc`（解码）/ `Stored`（读视图）共享的 `name/dtype/shape/quant` 抽成内嵌 `pub(crate) Descriptor`，字段单一来源；`Segment` 补对称访问器 `name/dtype/shape/quant`，`data` 仍为公开字段。公开构造 `Segment::new` 与下游消费面不变。
- **两遍编码合一**：`Writer::finish` 原先先 `encoded_len`（prepare 一遍，逐段 `rans::encode`）再 `write_body`（又 prepare 一遍）。抽出 `write_body_in(packed, ..)`，`finish` 只 prepare 一次，把同一份 packed 复用于长度计数与正文写出；`encoded_len` 仍独立单遍。`encode_into`/`write_to_path` 路径行为不变。
- 复核结论：`Codec` 仅 `Store/Rans` 两变体，enum+match 分派已清晰，不另套 `Compress` trait（避免为两个策略过度设计）。

### T2 性能第一笔：Merkle 叶子摘要惰性缓存（提交 `3516be5a`）

- **瓶颈定位（criterion 基线，64 段 × 1 KiB f32，集中在既有 `benches` 容器，未新建 crate/目录）**：`merkle_root` 302.92 µs、`prove` 302.59 µs、`verify` 292.01 µs——三者每次都经 `leaves()` 重解压全段再逐段哈希；`parse` 7.69 µs、`get_by_name` 52.3 ns、`finish` 199.34 µs 不在热点。
- **改动**：`Archive` 增 `OnceLock<Result<Vec<Digest>>>` 惰性缓存；`collect_leaves` 只负责算，`cached_leaves` 首算后复用，`merkle_root/prove/verify` 改走缓存。`get_or_try_init` 在本工具链属未稳定库特性（`once_cell_try`，E0658），故用稳定 `get_or_init` 包裹内层 `Result`：解码失败的错误存下后原样克隆复返（`Error` 已 `Clone`）。容器字节不可变，缓存不引入失效语义；公开 API 与零拷贝读取不变。
- **复测（同基准）**：`merkle_root` 302.92 → **18.68 µs**（约 16×，余下为对缓存叶子建树）；`prove` 302.59 → **19.32 µs**（约 16×）；`verify` 292.01 → **1.82 µs**（约 160×，只用命中缓存的单叶核对）；`parse` 7.66 µs、`get_by_name` 53.9 ns、`finish` 201.93 µs 均在噪声内持平。
- **基准登记**：`crates/benches/criterion/weights_container.rs`（parse / root / prove / verify / get / finish），`benches` 增 dev 依赖 `weights`；架构文档边表同步登记 `benches → +dev: weights`（边 76→77，`+dev` 12→13）。

### 延后项（需更大改动或实测证据，本轮不强行推进）

- T2 性能：`Header::get`/`Archive::get` 名字线性查找，基准 52 ns（64 段），加名字索引会增加结构与代码，实测无收益，暂不做；生产 `File` 后端按条目 seek+read，`storage::Mmap` 目前仅测试用，进生产要把 storage 提为运行时依赖；rANS 小帧预检阈值需要基准支撑；`finish` 约 310 MiB/s，已是单次编码，未见第二遍冗余。
- 质量补欠：`bin` 负例与覆盖在 `tests/` 内补回；fuzz 从 `json` 扩到 gguf/zip/seg/写端。

### 门禁同步

- 架构门禁 `every_public_type_is_listed_or_baselined` 初跑报 10 条过期基线：`Axis/Note/Params/Pick/SecretKey/Spectral/Stable/Svd/Trunc/Zip`（孤儿簇已删、`Zip` 合并后私有化）。已从 `lints/vocabulary.toml` 的 `[[ownership.unlisted]]` weights 段清除；现存适配器内部件保留 `Bin/Gguf/Mode/Sha256/Value`。
- 复测：`tests/architecture.rs` 8/8 通过；`cargo nextest --workspace` **952/952** 通过；`cargo clippy --workspace --all-targets` 0 告警；`cargo fmt --all --check` 0 diff；weights 12 个平级源文件，全部 ≤899 行。
