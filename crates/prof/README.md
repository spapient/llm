# prof —— 档案层（设备档案 + schema）

设备能力档案的存放地：每个文件描述"这台设备能做什么、证据到什么等级"。
**设备的一切差异都写在这里，代码里没有任何 `if vendor == …`**：存储层物理名、输出语言、
工具链版本、执行单元及其可访问的存储层、每条能力区间，全部是 `devices/*.toml` 的闭集字段。
`cutile` 只做机制（探测 → 协商 → 选择 → 拼接 → emit），通过本 crate 取档案，所以新增设备
**只改数据，不改代码**。

## 目录

```
crates/prof/
├── Cargo.toml
├── README.md
├── devices/
│   ├── ascend.toml      # P0 只有 L0：单元、形状与见证已声明，真机执行待补
│   ├── cpu.toml         # 真机后端：L2（与参考实现对拍通过）
│   ├── intel.toml
│   └── nvidia.toml      # P0 只有 L0/L1：见证已声明，未执行的能力视为不具备
└── src/
    ├── atom.rs          # 能力词汇表：16 原语、位集、Need/Provide、AttrKey 闭集（含 exec）
    ├── lib.rs           # 内嵌（BUILTIN）、目录装载（load）、目录位置（dir）
    └── prof.rs          # 档案 schema：解析、闭集字段、[units] 一致性、见证校验
```

## 一份档案写什么

| 段 | 内容 | 语义定义在 |
|---|---|---|
| `[meta]` | `vendor` / `id` / `emit` / `tool` / `tool_ver` / `checked` | `prof::prof` |
| `[fingerprint]` | 探测命中的键（`arch`、`cc`、`soc`…）及其取值模式 | `prof::prof`、`cutile::probe` |
| `[space]` | 三级存储的物理名（`GM` / `FAST` / `REG`） | `prof::prof`、`cutile::space` |
| `[units.<name>]` | 执行单元名 → 它能访问的存储层 | `prof::prof` |
| `[mem]` / `[simt]` / `[peak]` / `[compile]` | 容量、SIMT 形状、峰值、编译参数 | `prof::prof`、`cutile::space`（容量与 bank）、`llm-audit`（峰值与 `flags`） |
| `[[provides]]` | 每个原语至少一条；同一原语可用不同 `impl` 表达不同签名：形状区间、位宽、存储、`exec`、实现名、证据等级（`L0`–`L3`）、`witness` | `prof::atom`、`cutile::cap` |

`[[provides]]` 里除闭集字段（`prim` / `impl` / `m` / `n` / `k` / `dt` / `space` / `grade` / `witness`）
之外的键，都必须是 `AttrKey` 闭集里的一格——`exec` 就是其中之一：

```toml
[units.cube]             # 单元名由数据定义，代码不认识任何具体名字
space = ["FAST", "REG"]  # Cube 只能访问 L1/L0

[[provides]]
prim  = "dot"
impl  = "mmad"
dt    = ["f16", "bf16"]
space = ["FAST", "REG"]
exec  = ["cube"]         # 加载期校验：cube 必须访问得到上面声明的每一层存储
```

写错一个名字（`exec = ["cube0"]`）或让单元访问它到不了的存储层，`Profile::validate` 会在
加载期报错并列出闭集，而不是等真机跑错。

同一原语可以写**多条** `provides`，用不同的 `impl` 区分互不覆盖的签名：`dot` 的 `dp4a`
（i8×4）与 `fma`（标量 f32）就是两种能力。合成一条只能靠少写 `dt`/`lanes` 来伪装——
而 `covers` 是集合包含，少写字段就等于声称"也支持"。协商（`cap::negotiate`）逐条比对：
先取能**覆盖**需求的那条，再看证据等级，都不覆盖时选差异最少的那条做诊断（同一个
原语下的 `impl` 不得重名，否则无法区分谁实现了什么）。

`impl` 不只是标签：片段层每段文本的第一行要写下它实现的那条签名（`/* cutile-sign: dot/fma */`），
`cutile` 装载期逐条比对"协商会挑中哪条"与"片段声明写了哪条"，不一致就报错。因此档案里补一条
签名等于同时承诺"有一张与它同构的片段文本"——少这一半，`provides` 与片段可以各自自洽却指向
不同的算法（`dp4a` 的文本被当作 `fma` 用），而这种错在构建期之外没有任何地方能发现。

`[peak]` 的 `value` 与 `src` 必须一致：`src` 只能是数值字面量与 `+ - * /`、括号组成的表达式，
装载期会求值并与 `value` 比对，不一致或含变量名/单位都会在解析期报错。量纲与物理含义写在
TOML 注释里，峰值数字因此只有一个事实源（DESIGN §12.1）。

证据等级的含义：`L3` 真机性能达标、`L2` 与参考实现对拍通过、`L1` 仅源码可生成、`L0` 未验证。
协商只使用等级 ≥ `L2` 的能力，因此**档案里的乐观声明不会变成错误结果**。

`witness` 是一个可执行的判据：`{ file, func, expect }`。`file` 是**相对片段层根**的路径（仓库里根是
`crates/prim`，即 `cpu/witness.c`；设置 `data.root` 时是 `<data.root>/prim`），`func` 是统一 ABI
`int func(float *out)` 的入口，`expect` 是按顺序写出的期望值。

```toml
witness = { file = "cpu/witness.c", func = "witness_quant", expect = [-4.0, 3.5] }
```

`profile.validate` 只静态校验（文件可读、函数名在源码里、期望值非空且不超上限）；真正的证据来自
`cutile verify`：它把每条 `grade ≥ L2` 的见证编成可执行文件并执行，值不对就整台设备失败。
声明了见证的 `L0`/`L1` 供给走另一条路——**host 自检**：见证的 ABI 是 host C，与片段语言无关，
所以自检跳过档案的 `[compile].preamble` 与 `flags`，直接编译执行 `<vendor>/witness.<ext>`。

校验与等级解耦：**声明了见证就校验它**，即使这条供给还是 `L0`/`L1`。这样加速器可以先有证据
文本（`nvidia/witness.cu`、`intel/witness.cl`、`ascend/witness.cpp` 已经登记在 `prim::WITNESS`），
文件名或函数名写错会在装载期暴露，而不是等升 `L2` 才发现。自检则更进一步：它按 host C ABI
真正编译执行这些见证，期望值写错会当场失败（结论记作 `Scope::Host`）。但自检**不**说明设备
具备该能力，因此不改写等级：`nvidia` / `intel` / `ascend` 仍停在 `L0`/`L1`，协商继续把它们的
每条提供视为不具备。

## 新增设备 / 新增执行单元

1. 在 `devices/` 下写 `<name>.toml`（键必须属于闭集，写错字段会在解析期报错）；
2. 在 `src/lib.rs` 的 `BUILTIN` 里补一行 `include_str!`（否则发布出去的二进制里没有它）；
3. `cargo test -p prof` —— `builtin_matches_disk` 会把"磁盘有、内嵌无"直接判为失败；
4. `cargo run --bin cutile -- doctor --op gemv_i4 --device <name>` 看协商结果（`units:` 那一行就是本档案的执行单元）；
5. 有真机时 `cargo run --bin cutile -- verify --op gemv_i4 --devices <name>` 对拍并回填 `grade` / `witness`。

## 什么不放进来

- **算子声明与模板**（`ops/*.{toml,k}`）：在 `crates/op`。
- **原语片段**（`prim/<vendor>/<prim>.<ext>`）与见证源码：在 `crates/prim`（本 crate 只在装载期检查
  见证文件是否存在）。
- **机制**：协商（`cap`）、选择（`plan`）、拼接（`substitute`）与 emit（`emit`）、探测（`probe`）都在 `cutile`；
  依赖方向固定为 `llm → cutile → {prof, prim}`、`prim → prof`，本 crate 不依赖 `cutile`。
- 构建与工具链配置：见根目录的 `Cargo.toml`、`clippy.toml`、`rustfmt.toml`。
