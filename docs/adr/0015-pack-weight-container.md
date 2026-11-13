# 0015：自家权重容器 `pack`——存储与传输一体的版本化字节格式

- 状态：已采纳（归属已修订，见下）
- 日期：2026-11-12

> **修订（2026-11-12）**：本 ADR 原决策第 1 条"新建独立 crate `pack`"已被后续架构评审调整。
> 理由：两个 crate 均 `publish=false`、消费方高度重叠、`Entry` 与 `Segment` 是同一类抽象，独立 crate
> 造成 pack 悬空。现把容器实现整体并入 `weights`，各模块直接平铺在 `crates/weights/src/` 根
> （不设子目录；为避免与外部量化 `quant.rs` 同名，容器的显式仿射量化模块命名为 `affine.rs`，
> 类型名仍为 `Quant`），独立 `crates/pack` 已删除；格式名、魔数、字节布局与全部能力
> （21 种 Wire、rANS、Merkle、原子写、两级 CRC）保持不变。下文关于"独立 crate"的表述仅作历史记录。
>
> **再修订（2026-11-12，P4 落地）**：图拓扑与格量化已迁移并增强——`ir.rs`（8 闭集算子、
> 边只能指向更早节点保证无环）、`lattice.rs`（D24/E8/3×E8 + rank/unrank，E8 壳计数对拍）、
> `scale.rs`（对数域定点把 scale 区从 `4n` 压到 `8+n`，攻克旧版"生产不启用"根因）、
> `golay.rs`（G24 `[24,12,8]`，向真 Leech Λ24 的码层基础，未宣称达到 Λ24 密度）。
> 124 测试全过、clippy 零警告、全 workspace 零失败。

## 背景

[ADR 0010](0010-borrow-vs-load.md) 与 [ADR 0012](0012-quantized-weights.md) 把 `weights` 的定位
钉成"**只读**的 safetensors 适配层"：`Entry/Header` 自证合法，`Reader/Loader` 两条契约分离。
这套抽象对"读懂一个文件"是干净的，但它覆盖不了一个权重容器在真实训练与服务里要承担的全部职责。
对照 safetensors、Burn 的 Recorder 与 GGUF，现有口径有八个结构性缺口：

1. **只读不写**：`weights` 没有 Writer，无法产出权重；读写不对称。
2. **类型闭集窄**：`tensor::DType` 只有 5 种，容器无法承载全精度（INT1–FP64、F8、更宽整数）。
3. **量化靠文件名猜**：分组从 `_scales` 之类的后缀推断，没有显式方案描述符，脆弱。
4. **无对齐**：safetensors 不保证数据起点对齐，无法做 GPU 直传零拷贝，也无法干净地 mmap。
5. **无完整性校验**：权重文件本身没有 checksum，挡不住静默比特翻转（`checkpoint` 有 CRC，但那是
   训练快照，不是通用权重容器）。
6. **无版本/特性位**：格式无法演进，读端无法协商能力。
7. **无压缩/分块预留**：per-tensor codec、随机访问索引没有落点。
8. **布局单一**：只有 dense packed，没有为 strided/block-sparse（MoE）留出口。

目标是一个**自家容器**：在工程质量上对标并超越 Burn 的 Recorder 与 GGUF，且默认零外部第三方依赖、
`deny(unsafe_code)`。本 ADR 只定 **P1 容器核心**；mmap、文件后端、压缩、网络流式按阶段门禁推进。

## 决策

### 1. 独立 crate `pack`，格式名 `pack`

新建 `crates/pack`（包名 `llm-pack`，crate 名 `pack`），workspace 成员。`weights` 保持
"格式中立的 safetensors 适配层"不动，两者职责不重叠：

| crate | 职责 |
|---|---|
| `weights` | 读 safetensors：字节 → `Header`，`Reader/Loader` |
| `pack` | 自家容器的**读写对称**：版本化头、全精度描述符、对齐、checksum、（后续）压缩/分片 |

文件后缀 `.pk`，魔数 `u32 = 0x4B50_4C4C`（小端字节即 ASCII `"LLPK"`）。固定 4 字节魔数让
"读错文件/读到半截流"在第一时间被拒。依赖仅 `diagnostic`、`tensor`，无第三方 crate。

### 2. 字节布局（全小端）

```
magic      u32 LE   "LLPK"
version    u16 LE   当前 1
flags      u32 LE   位域，预留位必须为 0
align_log  u8       对齐边界 = 2^align_log，范围 0..=16（1 B .. 64 KiB），默认 6（64 B）
count      LEB128   段数（u32）
<count 个段描述符>  紧凑 LEB 编码，无填充
<数据区>            每段先按 align 填充（必须填 0），再写该段数据
global_crc u32 LE   覆盖从 magic 到数据区末尾的全部字节（含填充）
```

端序固定小端、不设端序字段：与 `proto`/safetensors 一致，跨端序是消费方的事，不拿一个字段换
两套读路径。`flags` 只有两个定义位：`bit0 = GLOBAL_CRC`、`bit1 = SEG_CRC`，默认都置 1；关闭时
对应 CRC 字段仍占位存 0，保证布局固定。任何非零预留位都被拒绝。

### 3. 段描述符与全精度 wire 类型

段（`Segment`）描述符按 canonical LEB128 编码：

```
name      LEB 长度 + UTF-8（非空）
dtype     LEB u16，wire 类型 code
shape     LEB rank + rank 个 LEB u64 轴长
codec     LEB u8，P1 仅允许 0（store）；其余 code 报 Unsupported
quant     1 字节 present 标志；为 1 时跟显式量化方案（见 §4）
data_len  LEB u64，存储字节数；store 时必须等于 shape×dtype 推出的字节数
seg_crc   LEB u32，该段逻辑字节的 CRC-32（不含对齐填充）；关闭时为 0
```

**wire 类型闭集独立于 `tensor::DType`**，覆盖全精度：

- 布尔：`Bool`
- 无符号整数：`U1 U2 U4 U8 U16 U32 U64`
- 有符号整数：`I1 I2 I4 I8 I16 I32 I64`
- 浮点：`F8_E4M3 F8_E5M2 F16 BF16 F32 F64`

每个 wire 类型有稳定 code、位宽与类别（bool/uint/int/float）。容器**解析不拒绝**任何闭集内类型，
即使当前 `tensor` 还没有对应元素——"能存任意精度"是容器的职责，"能不能被某个内核消费"是上层的
职责。`to_tensor_dtype()` 只对能映射的返回 `Some`，其余由上层按 `Unsupported` 处理。

### 4. 显式量化方案，取代文件名猜测

`quant present = 1` 时写一个自描述方案，P1 定义一套仿射（affine）方案：

```
scheme    u8      0 = affine（x ≈ scale · q + zero）
axis      LEB u32 分组轴；per-tensor 写 rank（标度不沿任何真实轴）
block     LEB u32 组宽（元素数）；per-tensor 写 0
flags     u8      bit0 = 有无 zero-point
scale_ty  LEB u16 标度的 wire 类型（浮点）
zp_ty     LEB u16 零点的 wire 类型（仅在有 zero-point 时写）
```

分组、零点、标度类型因此是**文件里的事实**，不再靠 `_scales` 后缀推断。P1 只做描述符的完整
往返与结构校验，反量化数学与内核对接在后续阶段；但描述符一次定到位，避免日后改格式。

### 5. 对齐与 canonical 填充

数据区每个段的起点 = `align_up(prev_end, 2^align_log)`，填充字节**必须为 0**。读端重算每个起点并
校验填充区全 0：非零填充被当作非规范字节拒绝。段的绝对偏移**不进描述符**——它由"数据区起点 +
对齐规则 + 各段长度"确定性推出，读端顺序重算。这消除了"存一份偏移、再算一份偏移"的漂移，也让
同一份逻辑容器只有一种字节形态。随机访问在 P1 靠顺序重算（段数量级几千，成本可忽略）；显式索引
段留给 mmap/分块阶段。

### 6. 两级 CRC-32，复用既有口径

CRC-32 采用与 `checkpoint` 完全相同的 IEEE 802.3 反射多项式（`0xEDB8_8320`）八路查表实现，
新 crate 内独立一份同口径代码：

- **段级**：`seg_crc` 覆盖该段逻辑字节（不含填充），定位损坏到具体张量。
- **全局**：`global_crc` 覆盖从 `magic` 到数据区末尾（含规范化填充），截断、改一个比特、尾部
  多余字节都在打开期被拦。

两级都可经 `flags` 关闭，但默认开启；关闭只影响"是否要求校验通过"，字段位置不变。

### 7. 读写对称与双射门禁

- `Writer::push(name, dtype, shape, bytes)` 收集段，`Writer::finish(align)` 先算头再产出字节，
  额外内存只到写缓冲，不缓冲"整份文件大小"的副本。
- `Archive::parse(bytes)` 校验魔数、版本、flags、对齐、填充、每段长度、两级 CRC，给出段查找与
  借用读取。P1 只做内存 `Archive`；文件后端（seek+read）与 mmap 是后续阶段。
- 双射由测试钉死：`decode(encode(x)) == x` 且 `encode(decode(b)) == b`；LEB 拒绝非规范编码
  （0 的多字节、末尾组全 0、超 u64），任何"还能更省却没这么写"的字节都被拒绝。

## 后果

- 权重容器第一次具备**写入端**与版本化能力，读写对称；全精度 wire 闭集让 INT1–FP64、F8 都有
  稳定落点，且不强迫 `tensor` 立刻扩类型。
- 显式量化方案取代 `weights` 的文件名推断，分组/零点成为可校验事实；对齐填充让 GPU 直传与未来
  mmap 有了零拷贝基础。
- 两级 CRC 把静默损坏、截断、错位的发现点提前到打开期，而不是训练若干步后才表现为 loss 异常。
- 默认零第三方依赖、`deny(unsafe_code)`；mmap 所需的 `unsafe` 被隔离到一个**默认关闭、
  feature-gated 的独立小 crate**，主容器的安全口径不变。
- 代价：P1 只有内存后端，超大文件仍需后续的文件/mmap 后端；wire 类型里当前无法映射到
  `tensor::DType` 的段，只能存储/传输，不能直接进内核。

## 未决（后续阶段，按门禁推进）

- **P2**：文件后端（只读头部 + 按段 seek/read）、feature-gated `pack-mmap` 零拷贝映射。
- **P2**：per-tensor 压缩 codec（store 之外的 deflate/zstd/lz4 与自家 scx 对接），`codec` 字段已预留。
- **P3**：显式索引段与 range 随机访问、断点续传，对接 `proto` 帧做网络流式传输。
- **P3**：strided/block-sparse 布局（含 MoE 专家表）。
- **P4**：与 `weights`（safetensors）双向转换、分片多文件清单（与 `shard` 拓扑对齐）。

## 再修订注记（2026-11-12，P5 完成）

- 独立 `pack` crate 已撤销、容器整体并入 `weights` 且模块平铺根目录；mmap 的 unsafe 隔离到 `storage`
  的 feature-gated 模块，不再有独立 crate。
- P5 落地自研 Jacobi SVD、强类型 `ledger` 账本、五轴 `axes` 选码，并应要求把外部格式提前并入：
  `gguf.rs`（GGUF v3 六类反量化）、`zip.rs`+`bin.rs`（零依赖 ZIP/pickle 读取，九类 storage，零拷贝）。
- 真机对拍而非 mock：torch 2.9.1 生成的 2 个固定文件与 20 个随机 fuzz 文件共 100 个张量，
  解析字节与 `torch.load` 逐字节一致；weights 152 测试、全 workspace 测试零失败，clippy 零警告。
- 旧版 `legacy.rs` 经取证是自有 v2 容器而非 PyTorch pickle，故 bin 读取为零依赖新写，不属逐行迁移。

## 三修订注记（2026-11-12，P6 完成）

- `mode.rs`：模式写在保留名段 `llm.mode`，不新增二进制机制；`opt.` 命名空间为优化器状态，
  `release` 过滤发布，旧文件无标记按推理兼容。
- `patch.rs`：逐字节 XOR 自逆且零额外标记——同名等长 XOR、否则全量；diff/join 同构共用
  `rebuild`，XOR 大量零经 rans 压小。新增函数全部满足静态门 ccn≤5。
- 对外命名：外部格式统一称 bin，不带 PyTorch 前缀。
- 说明：`llm-tool` 的全仓库静态门（gate）当前对大量历史 crate 的既存函数报 ccn/行数技术债，
  非本阶段引入；本阶段新增的 mode/patch 已主动合规，未在本次范围重构历史 crate。
