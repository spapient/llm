# 0011：下界按方向分账、按文件说格式，并允许"无法计价"

- 状态：已采纳
- 日期：2026-11-12

## 背景

ADR 0008 把层的下界从"手写常数"收敛成"算子语义的投影"，ADR 0009 又把它接到真实权重文件上。
方向是对的，但三处口径在真实模型上都会给出**看起来合理、实际是错的**数字，或者干脆算不出来：

1. **一个 `bytes` 把两个方向混在一起。** 设备的读上限与写上限不是同一个数：只流式读权重的
   GEMV 可以跑过拷贝上限。合成一个总量之后，"这一层卡在哪一侧"再也无法判断，而"把读挪成写、
   总量不变"也会看起来等价——融合、量化、布局的收益因此无法被检查。
2. **量化格式被钉死成一个全局常量。** `Linear::OP = "gemv_i4"` 于是 `Stack::from_header` 把
   **任何**秩 2 条目都按 int4 计价，哪怕文件里写的是 `F32`。`[4096, 4096]` 的 f32 权重有 64 MiB，
   按 int4 报出来是 10.5 MB——差 6.4 倍。这不是"更松的下界"，而是声明与文件不符。
3. **一层算不出来，整张模型就归零。** `floor()` 全有或全无：LoRA 的 `[4096, 16]` 条目（16 不是
   量化分组的整数倍）或一张还没有算子的 f32 权重，会让 300 层之和变成一个 `Err`。真实模型里
   这种"合法但没有可计价形状"的组件是常态，不是异常。

## 决策

### 一、`Cost` / `Bounds` 按方向分账

```rust
pub struct Cost  { pub flops: f64, pub read: u64, pub write: u64 }
pub struct Bounds { read: u64, write: u64, flops: f64 }

impl Bounds {
    pub fn of_op(op: &str, m: usize, n: usize) -> Result<Self>; // 仍然唯一入口
    pub fn of_ops(ops: &[String], m: usize, n: usize) -> Result<Self>; // 组合层 = 各算子之和
    pub fn read(&self) -> u64;
    pub fn write(&self) -> u64;
    pub fn bytes(&self) -> u64;      // 派生合计，只为打印与兼容旧口径
    pub fn moves_within(self, other: Self) -> bool; // 每个方向都不更高
}
```

`einsum::Formula::cost` 按操作数自己的 `Access` 累加两个方向，因此流量口径仍然只有一处。

`moves_within` 是"融合省了多少"的可检查形式：复合组件的下界只有在读、写、算三项上都不高于
子组件之和时才谈得上更省。**反过来，总量相同但把读挪成写不落于其中**——那不是省流量，只是
换了一个瓶颈。

### 二、格式是层的事实，来自文件

```rust
pub enum WeightFormat { F32, F16, Bf16, I8, Packed4, Int4 { group: usize } }

impl WeightFormat {
    pub fn of_dtype(dtype: DType) -> Self;      // 只有元素类型能决定的部分
    pub fn operator(self) -> Option<&'static str>; // 没有算子就是 None，不借别的格式顶上
}
```

- `Stack::from_header` 从 `Entry::dtype()` 取格式，不再有全局默认的 int4。
- int4 的**分组**从算子语义表读（`op::model(Linear::OP).group`），不在 `nn` 里另写一个 32：
  分组与算子声明不一致时 `operator()` 返回 `None`，而不是套一个别组的算子。
- `U4` 只给出 `Packed4`（"4 位打包，分组未声明"）。分组不在元素类型里，任何具体数字都只能是
  猜的；要计价必须显式构造 `WeightFormat::int4(group)`。
- 没有算子的格式仍然是**合法的层声明**：`Linear::with_format` 成功，`spec.ops()` 为空。把它做成
  构造错误反而会逼调用方"随便挑一个格式"让构造通过。

### 三、`bounds` 进契约，允许缺席

```rust
pub type BoundsResult = Result<Option<Bounds>>; // 名字由 nn 导出，调用方不必依赖 diagnostic

pub trait Layer { fn bounds(&self) -> BoundsResult { Ok(None) } } // 其余三个访问器不变

pub struct Floor { total: Bounds, unpriced: Vec<Unpriced> }
impl Stack {
    pub fn floor(&self) -> Floor; // 不再返回 Result
}
```

三种状态刻意分开：`Ok(Some)` = 有下界；`Ok(None)` = 层合法但没有可计价的形状（注解层没给
`in`）；`Err` = 有算子却算不出来（形状没有合法内核，或格式没有算子）。`Stack::floor` 把后两种
都转成 `Unpriced { name, reason }`，其余层照常相加。

`#[layer]` 注解新增可选 `in = <宽度>`：给了它，生成的 `bounds` 与 `Linear` 走同一条
`Bounds::of_ops`；不给就沿用契约的默认值，而不是生成一个"返回空"的实现把这件事藏起来。

### 四、编排不进 `nn`

删除 `Bounds::calls_per_second_ceiling`：它是 `cutile::plan::predict`（`max(flops/峰值,
bytes/带宽) × (1+unmasked)`）的半个副本，且全仓零调用方。`nn` 只给数对，而且每个方向各给一份；
roofline 与峰值只属于 `cutile` / `prof`。

## 后果

1. **旧数字逐字节不变**：`Bounds::bytes()` 与改造前相等，`gemv_i4(4096, 4096)` 的合计仍是
   `4096² / 2 + 4096·4 + 4096·128·4 + 4096·4`，测试逐项钉住；新增的是 `read`（权重 8 MiB +
   输入 16 KiB + 缩放 2 MiB）与 `write`（16 KiB）两个分项。
2. **f32 权重不再被报成 int4**：`Stack::from_header` 遇到 f32 条目会把它列成"无法计价"并给出
   原因。真实文件因此暂时算不出模型下界——这是如实报告，不是回归：等浮点投影算子在 `op` 里
   有声明与语义表，同一行代码就会开始给数，`nn` 不需要改。这条空缺是刻意留下的，不是待修的 bug。
3. **一层失败不再清零整张模型**：LoRA、f32 权重、没有 `in` 的注解层都只让自己进清单。
4. **`Stack` 的公开面变化**：收 `Box<dyn Layer>`，`layers()` 返回 `&[Box<dyn Layer>]`，
   `floor()` 返回 `Floor`。`llm status` 相应把层/模型下界写成"读 … / 写 …"，并单列一行
   "无法计价"及其原因。
5. **下一步（未做）**：`Model::cost` 的 `(m, n)` 只够描述一条投影。要加 RMSNorm / Embedding /
   Attention，必须先把形状上下文抬进算子语义表，否则每个新层都会被逼着伪造一对 `(m, n)`——
   那正是"第二份公式"的开始。
