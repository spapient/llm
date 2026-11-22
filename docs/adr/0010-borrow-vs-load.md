# 0010：借出还是载入——权重文件的两条读取契约

- 状态：已采纳
- 日期：2026-11-12

## 背景

[ADR 0009](0009-real-weights-to-model-floor.md) 在"未决"里写了一句 **"`Reader` 已经把接口留好了"**，
意思是将来换 mmap 或按条目懒读不需要改契约。这句话不成立：

```rust
pub trait Reader {
    fn read(&self, entry: &Entry) -> Result<&[u8]>;   // 返回值的生命周期绑定在 &self 上
}
```

文件后端交不出这个借用。要在"不装 `unsafe`、不引外部依赖"的前提下从 `std::fs::File` 返回
`&[u8]`，只有两条路：mmap（`memmap2` 是外部依赖，自己写 mmap 是 `unsafe`），或者把字节读进
`self` 里再借出去（那就等于整份读入，正好是要避开的事）。所以 `Reader` 只能服务"字节已经在内存
里"的后端：`Archive<'a>` 与测试夹具。

同时组装点仍然是：

```rust
let bytes = std::fs::read(path)?;                  // 7B 权重十几 GB
let archive = weights::safetensors::Archive::parse(&bytes)?;
```

"能读真实文件"这一条因此只对小文件成立——真实模型的第一条不会是 32 KiB 的夹具，而是十几 GB。

## 决策

### 1. 把"借出"与"载入"拆成两条契约

| trait | 语义 | 谁能实现 | 唯一必须实现的方法 |
|---|---|---|---|
| `weights::Reader` | 把字节**借**出来 | 内存切片、mmap、测试夹具 | `read(&self, entry) -> Result<&[u8]>` |
| `weights::Loader` | 把字节**拷进**调用方缓冲区 | 文件、压缩包、远端分块 | `copy_into(&self, entry, &mut [u8]) -> Result<()>` |

`Loader::load(&self, entry) -> Result<Vec<u8>>` 是默认方法（按 `entry.len()` 分配后转调
`copy_into`），因此两种后端都不必重写它，长度检查只有一处（`check_capacity`）。

**不给 `Reader` 加一个 `load` 默认方法**：那样"零拷贝"与"分配"会混进同一个 trait，每个实现都要
写两遍，而且默认实现会掩盖"这个后端是不是真的零拷贝"。两条 trait 的分歧点必须在类型上可见。

### 2. 文件后端只读头部，数据段按条目读

`crates/weights/src/safetensors.rs` 新增 `File`：

```rust
impl File {
    pub fn open(path: impl AsRef<Path>) -> Result<Self>;   // 只读 8 字节前缀 + 头部 JSON
    pub fn header(&self) -> &Header;
    pub fn header_len(&self) -> usize;                     // 打开时实际读了多少字节
    pub fn file_len(&self) -> usize;                       // 文件总长度
}

impl Loader for File { /* seek + read_exact，边界由 check_capacity 与条目区间保证 */ }
```

自证合法的口径**一个字都没放宽**：`Archive::parse` 用切片长度核对"头部 + 条目声明的数据段是否
正好覆盖整个输入"，`File::open` 用 `metadata().len()` 核对同一件事。被截断与尾部多余字节都在
**打开期**失败，而不是等到读某个条目时崩在别处。

`parse_header(text, data_start)` 与 `split_prefix(prefix)` 是两者共用的唯一实现——两个后端解出的
头部必须逐字段相等，这条由测试
`both_backends_produce_the_same_header_and_the_same_bytes` 钉住。

### 3. 组装点用 `File`，并读一条样例当证据

`llm status --weights <file>` 改成：

- 打开只读头部，输出 `只读头部 X / Y 字节`（`X` 与数据段大小无关）；
- **按需读一条样例**（最小的那条条目），证明"头部说合法"之外字节能真取出来；
- 样例超过 `SAMPLE_LIMIT = 1 MiB` 时如实报"未读"——样例是证据，不是替用户读张量。

挑最小的条目而不是第一条：真实权重里第一条往往是嵌入矩阵，读它等于把整份文件搬进内存。

## 后果

- 打开一个大文件的内存占用从 `O(文件大小)` 降到 `O(头部大小)`。测试用 1 GiB 稀疏文件钉住这条：
  `header_len() == 8 + json.len()`，`file_len() == 8 + json.len() + 1 GiB`。
- `Archive` 一行未改（除复用 `parse_header`）：内存后端仍然零拷贝，`Reader` 的语义不变。
- 头部分配仍以文件自报的长度为准，因此 `header_len > file_len` 必须**先**判：否则一个坏掉的 8 字节
  前缀就能让 `open` 替它分配一块任意大的内存。这条被 `a_prefix_that_promises_more_than_the_file_holds_is_rejected`
  覆盖。
- `llm status` 的错误语义更早、更具体：文件不存在是 `io`（退出码 5），截断/多余字节是 `invalid`
  （退出码 2），两者都在"层清单"之前失败。
- 代价：`File` 持有 `std::fs::File` 句柄，因此它是 `Debug` 但不是 `Clone`/`PartialEq`；需要可比较
  的场景用 `header()`（本身就是 `PartialEq`）。

## 未决

- **仍然没有 mmap**：mmap 的收益是"零拷贝访问张量数据"，与"能不能打开大文件"是两件事，后者已经
  由按条目读解决。真要 mmap 需要先决定是否放宽 `unsafe_code = deny`，那是一个独立决策。
- **GGUF**：不带 safetensors 头部，但只要是"字节 → `Header` + `Loader`"就能接进来，本层类型不变。
- **`SAMPLE_LIMIT` 是常量**：只有在出现"用户想调"的真实需求时才进 `llm-config` 的键闭集——按
  ADR 0006/0008 的口径，没有消费方的配置项不加。
