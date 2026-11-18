# 0036：装载源沿用 Reader/Loader 双契约，MmapSource 实现 Reader（草案）

- 状态：草案（读取契约与映射源已在位并过测；serve 装配根分流 2026-11-18 回填，最终接受以看板为准）
- 日期：2026-11-17
- 相关：[0010](0010-borrow-vs-load.md)（借出/载入双契约的既有裁决）、[0035](0035-unsafe-ffi-boundary.md)（unsafe/FFI 边界与唯一映射岛规则）；设计取证见 [`../llmu-parity-optimization-design.md`](../llmu-parity-optimization-design.md) §4.1 M1-A、[`../backend-runtime-design.md`](../backend-runtime-design.md) §1.2、§4.3

## 背景

ADR 0010 把权重读取拆成两条互不混淆的契约：`Reader::read(&self, entry) -> Result<&[u8]>` 把字节**借**出来（零拷贝，生命周期绑定在 `&self`），`Loader::copy_into(&self, entry, &mut [u8])` 把字节**拷进**调用方缓冲；`Loader::load` 是按条目长度分配后转调 `copy_into` 的封装，长度检查只有一处 `check_capacity`。0010 的未决项里挂着一条："仍然没有 mmap"——真要内存映射需要先决定是否放宽 `unsafe_code = deny`，那是独立决策。

M1-A 轨道要接只读映射装载源：复用 `storage::Mmap`（全仓唯一生产 unsafe 岛，ADR 0035），在 weights 侧提供一个从映射字节解释头部、按条目借用切片的源。动工前需要裁决：这个源挂在哪条契约上，要不要在两条既有 trait 之上再抽一个平行的 `Source` 父 trait，以及非 Unix 目标如何收口。

## 决策

### 1. 沿用 Reader/Loader，不立平行 Source 父 trait

不新增第三条聚合性的 `Source` 父 trait。读取面仍由 0010 的双契约承担：

- 能把整段字节留在原地、按条目借出切片的后端实现 `Reader`；
- 需要按条目把字节拷进调用方缓冲的后端实现 `Loader`。

两者的分歧（零拷贝借用 vs 分配拷贝）必须在**类型**上可见：不在 `Reader` 上加 `load` 默认方法，也不在 `Loader` 上补一个返回 `&[u8]` 的借出方法（0010 §1 的口径照旧）。抽一个 `Source` 父 trait 把两条契约收口到一起，会把"这个后端是不是真的零拷贝"藏进动态分发后面，重新引入 0010 已经否决的混淆。

### 2. MmapSource 实现 Reader

`weights::MmapSource` 持有的是 `storage::Mmap`（拥有型只读映射）与一份解析校验过的 `Header`：

- `open(path)`：先 `Mmap::open` 拿到只读映射，再在映射切片上 `Archive::parse` 校验头部；头部不过即报错，不继续解释数据段。
- `read(entry)`：先按 `header.data_start()` 从整份映射裁出数据段，再走与内存归档共用的 `entry_slice` 边界算术，返回映射上的借用切片。

它**不实现 `Loader`**：映射字节已经在地址空间里，再按条目拷进缓冲等于抹掉 mmap 的收益；需要拥有型字节的调用方走 `Loader::load` 封装那条路径，与映射源无关。

### 3. File 保持 Loader

磁盘后端 `safetensors::File` 维持 0010 §2 的形态：持有 `std::fs::File` 句柄，`copy_into` 做 `seek + read_exact`，边界由 `check_capacity` 与条目区间保证。本轮不为它补借出方法，也不把它改成映射后端。

### 4. Unix + feature 双门控，非 Unix 整块裁掉

`MmapSource` 与其模块、导出均以 `#[cfg(all(feature = "mmap", unix))]` 双门控：

- `mmap` 是 weights 的 optional feature，透传 `storage/mmap`；不开 feature 时映射源不参与编译。
- 非 Unix 目标整块剔除，**不提供**假装映射的空实现（返回空切片、整份读入冒充映射之类）。缺失类型在消费侧直接编译报错，与 storage 侧 `Mmap` 的收口口径一致。

### 5. unsafe 边界不越岛

weights lib 保持 `#![deny(unsafe_code)]`。`MmapSource` 只持有 `Mmap` 与 `Header`，不创建/解引用裸指针，不写第二处映射逻辑：唯一 `unsafe` 留在 `storage::Mmap` 这个既有岛内（ADR 0035 §1、§2）。映射的只读、私有映射、`Drop` 归还等不变量由 storage 岛的头注释与类型状态保证，weights 侧只消费其只读视图。

### 6. Archive::load 的 Cow 二分保持不变

`Archive::load` 既有的返回口径不退化：Store 段返回 `Cow::Borrowed`（借用归档字节），Rans 压缩段返回 `Cow::Owned`（解压分配）。映射源只让 Store 段经映射借用；压缩段在任何后端下都必须解压成 Owned，不允许在映射路径上把压缩字节冒充借用。

## 备选方案

- **抽平行 `Source` 父 trait 统一两条契约**：否决。会掩盖后端是零拷贝还是分配拷贝的类型分歧，与 0010 §1"分歧点必须在类型上可见"直接相悖；动态分发还会让长度检查与边界判断的归属变模糊。
- **MmapSource 同时实现 Reader 与 Loader**：否决。Loader 路径强制按条目拷贝，映射源的存在意义被抵消；双实现也会让调用方看不出哪条路是零拷贝。
- **非 Unix 提供空映射占位（返回空切片或整份读入）**：否决。假装映射属于静默错误：上层以为拿到了借用，实际语义是拷贝或空数据，排错路径很长。缺失类型应在编译期暴露。
- **在 weights 侧新写一份映射逻辑**：否决。裸指针/`from_raw_parts` 越过安全边界的操作必须收在 storage 既有岛（0035），weights 不新增第二处 unsafe。

## 后果

### 正面

- 装载后端的选择落在类型上：`Reader` 实现即借出，`Loader` 实现即拷贝；新增后端按已有接口归类，不需要先改 trait 形状。
- 映射源的越界面收敛：weights 侧无裸指针，审计时只需复核 storage 既有 mmap 岛一处。
- 非 Unix 目标不背映射语义的包袱：门控剔除后编译报错点明确，不会出现"能编译但映射是空的"。

### 负面（代价）

- 同一个文件在两种后端下要走不同调用约定（`read` 借出 vs `load` 载入），装配点必须先认清后端类别；混用会在类型检查期暴露，而不是运行期悄悄拷贝。
- `mmap` feature 显式开启，不开的 target 拿不到映射源；需要在基线文档里写清 feature 与平台口径。

### 取舍

契约清晰度与安全边界收敛优先于接口统一的便利；本 ADR 不规定映射源之上的装配与驻留缓冲接线（那是 M1-A3 与 serve 轨道的事）。

### 重议条件

- 出现第二类只读映射语义（可写映射、共享映射、设备侧映射）时，复核是否仍归 storage 岛，必要时回本 ADR 扩边界白名单。
- 若未来需要跨平台映射抽象（如 Windows 映射视图），先裁决它属于 storage 岛还是新边界模块，再动 weights 接口。

## 不做

- 不在本轮改 serve 装配与模型宿主：驻留缓冲接线归 weights/storage 公共件与另一轨道，本 ADR 只定读取契约归属。
- 不新增 workspace 成员，不引外部映射 crate；不在 weights 生产代码引入第二个 unsafe 岛。
- 不为 GGUF 等其它格式补装载实现（格式适配另案，接口形状沿用本 ADR 的双契约）。

## 修订

- 2026-11-17：初稿，回应 M1-A2 装载源归属；双契约沿用、MmapSource 实现 Reader、File 保持 Loader、双门控与 unsafe 不越岛落档。
- 2026-11-18：serve 装配根落地。`serve::CpuModel::from_path` 在装配点按后端类别做类型可见分流：`#[cfg(all(feature = "mmap", unix))]` 走 `MmapSource`（`Reader`）→ `Params::from_reader_stack`，回退 `File`（`Loader`）→ `Params::from_stack`；两条路径的栈、参数与模型构造一致，差别只在字节来源，未新增平行 `Source` 父 trait。`serve` 出 optional `mmap` feature 透传 `weights/mmap`；真实文件装配→前向对拍在默认与 `mmap` 两种构建下均通过。§1/§3/§4 口径在装配点兑现。
