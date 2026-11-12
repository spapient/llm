# 0009：真实权重文件 → 层栈 → 模型下界

- 状态：已采纳
- 日期：2026-11-12

## 背景

[ADR 0008](0008-layer-floor-from-semantics.md) 把"这一层至少要搬多少、算多少"做成了可断言的
`nn::Bounds`，但它有两个悬空的前提：

1. **层的清单是人手写进去的。** `llm status` 里那两个 `Linear` 是组装点自己摆的；真实模型有多少层、
   每层多宽，代码里没有任何来源。于是"模型下界"只能演示，不能对任何真实文件成立。
2. **`weights` 一个字节都读不出来。** 它拥有 `Header` / `Entry` / `Reader` 三个类型与它们的不变量
   （名字唯一、区间不重叠、起点不早于数据段），却没有任何解析器——`Header::new(Vec::new(), 0)` 是
   它唯一被调用的方式。"权重结构"这一层因此只对测试夹具成立。

这两条合起来意味着：整条推理链路的第一段（文件字节 → 张量元数据）在代码里是不存在的。任何关于
"模型每 token 至少要多快"的结论都无法用真实文件复现。

## 决策

### 1. safetensors 头部解析放在 `weights`，并且零依赖

格式本身很小：8 字节小端头部长度 + 该长度的 UTF-8 JSON + 数据段。于是：

- `crates/weights/src/json.rs`：手写的受限 JSON 读取器。接受完整 JSON 文法，但**只产出**头部用得到的
  几种值，小数与指数直接判错（头部的形状与偏移都是整数），重复键直接判错，报错带字节位置。
- `crates/weights/src/safetensors.rs`：`Archive<'a>` = 已校验的 `Header` + 数据段切片，并 `impl Reader`。
  它做的是**自证合法**：`shape × dtype` 推出的字节数必须等于声明的区间长度；数据段必须正好被条目覆盖；
  条目字段只允许 `dtype` / `shape` / `data_offsets`；`dtype` 只在 `{F32, F16, BF16, I8}` 闭集内映射。

`U8` 之类没有对应 `DType` 的名字被**明确拒绝**（`Code::Unsupported`）而不是当成 `I8`：int4 权重在
safetensors 里以 `I8`/`U8` 存储并在 `__metadata__` 里写分组信息，把两者混为一谈会让解包因子先错一次。

### 2. `nn::Stack`：层栈 + 总下界，未覆盖条目必须列出

```rust
pub struct Stack { layers: Vec<Linear>, uncovered: Vec<String> }

impl Stack {
    pub fn new(layers: Vec<Linear>) -> Result<Self>;          // 声明
    pub fn from_header(header: &Header) -> Result<Self>;      // 权重条目
    pub fn layers(&self) -> &[Linear];
    pub fn uncovered(&self) -> &[String];
    pub fn floor(&self) -> Result<Bounds>;                    // Bounds::plus 累加
}
```

四条约束：

1. **总下界只有一份算法**：`floor()` 就是 `Bounds::ZERO.plus(..)` 的折叠，没有第二个流量公式。
2. **秩 2 的条目才是一次投影**，维度按 `[out, in]`（与 `Linear` 一致）；名字去掉 `.weight`。
3. **其余条目进 `uncovered`，不静默丢弃**：真实权重里还有范数、偏置、位置编码。把它们忽略掉，
   下界会只算半张模型，而数字看起来仍然漂亮。
4. **重名与零宽拒绝**：层名重复（例如同时有 `a` 与 `a.weight`）与 `[0, n]` 都在构造期报错，
   因为下界归因与后续参数表都按名字对齐。

`Model` 这个名字留给 `op`（算子语义表）；本层的类型叫 `Stack`，避免"一个类型一个主人"被同名破坏。

### 3. 组装点给出一条可执行的路径

```bash
cargo run -p llm -- status                                  # 声明层：两个投影
cargo run -p llm -- status --weights model.safetensors      # 真实文件：条目 → 层栈 → 总下界
```

- 输出把"读到了什么"（条目数、数据段字节、载荷字节）与"算出了什么"（层数、下界、未覆盖条目）分开，
  这样读者能判断下界覆盖了什么；
- 退出码走 `Code::exit_code`（io = 5、unsupported = 3、invalid = 2），不再一律 `FAILURE`。

### 4. 依赖方向不变，不新增成员

`nn → weights` 是同层靠左的一条边（L5 表内 `weights` 在 `nn` 之前），因此不需要调整分层，也不需要
把解析器塞进 `nn`。

## 后果

- `weights` 从"三个类型 + 三组测试"变成"能读真实文件的一层"，`Archive::read` 返回的是文件里的字节
  切片本身（不复制）。
- 一个真实的 safetensors 文件现在能走完整条链路：`Archive::parse` → `Stack::from_header` →
  `Bounds::plus`，`llm status` 是这条链路的可执行证据；`cargo test` 里同时锁住了 happy path、
  "未覆盖条目"与"文件不存在是 io 错误而不是 panic"。
- 解析器的严格性是可回归的：区间与形状不符、数据段多一个字节、未知字段、重复键、非整数、超长头部
  都有用例，报错里能直接看到差多少字节。
- 代价：多了一个手写 JSON 读取器（约 400 行 + 测试）。它不对外承诺"通用 JSON"，只服务头部；将来若
  需要通用能力，应当先评估引入依赖，而不是把这个读取器扩成半个 `serde_json`。

## 未决

- **只有 safetensors 一种格式**：GGUF 的分块与量化描述不同，将来再写一个"字节 → `Header`"，本层类型不变。
- **秩 2 不等于投影**：嵌入表、位置编码也可能是秩 2。要真正区分需要权重命名规则或模型元数据，
  目前按秩分类并在 `uncovered`/`layers` 上如实呈现，不做语义猜测。
- **没有批与切分**：下界按"一次调用"给；批大小、TP/PP 分片会改变流量，等 `shard` 能给出真实
  拓扑之后再乘。
- **没有 mmap**：`Archive` 借用调用方给的字节切片，`llm status` 用 `std::fs::read` 一次读入；
  大模型必须换成 mmap 或按条目懒读（`Reader` 已经把接口留好了）。

上面“`Reader` 已经把接口留好了”这句已被 [ADR 0010](0010-borrow-vs-load.md) 推翻：`Reader` 返回借用，
文件后端交不出这个借用，所以真正需要的是另一条“拷进调用方缓冲区”的契约。现在 `llm status` 走
`safetensors::File`（只读头部 + 按条目读），整份读入只剩 `Archive` 那条内存后端在测试里使用。
