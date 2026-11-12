# op —— 声明层

算子声明、模板与算子语义的存放地，也是 `cutile` 唯一的领域内容入口。**写算子只改这里**：本 crate 不含
任何协商、片段拼接或 emit 逻辑，因此新增算子是"加两个文件 + 一页语义 + 两行注册"，不动 `cutile` 一行代码。

## 目录

```
crates/op/
├── Cargo.toml
├── ops/
│   ├── gemv_f32.toml    # 声明：稠密 f32 投影（骨架 / 轴 / 需求 / 布局 / 降级链）
│   ├── gemv_f32.k       # 模板：分块点积，未量化
│   ├── gemv_i4.toml     # 声明：weight-only int4 解码 GEMV
│   └── gemv_i4.k        # 模板：C 风格文本 + {{插值}} + @原语 槽位
└── src/
    ├── lib.rs           # 注册表（REGISTRY）与它的两张投影（BUILTIN / model）、目录装载（load）、目录位置（dir）
    ├── decl.rs          # `<name>.toml` 的解析与轴展开：OpDecl / Variant / Axis / TileCombo
    ├── layout.rs        # 布局构造子闭集（row / col / swz / pad）与声明式签名
    ├── sch.rs           # 调度模板闭集（direct / pipe / ws / pingpong）与降级链的两条判据（单调性、掩盖）
    ├── tpl.rs           # 模板词法：文本 / {{槽位}} / @原语(...) 与行号；借用正文，装载期只切一遍
    ├── model.rs         # 算子语义的类型：样本 Sample、语义表 Model（生产形状 / 输出类型 / Cost）
    ├── sig.rs           # `.k` 入口签名的解析与样本对齐（ABI 的唯一事实源）
    ├── gemv.rs          # 一条 GEMV 骨架 + 权重格式契约（Format / Element / OperandSpec / Cache）
    ├── gemv_f32.rs      # 稠密格式：`Dense<B>` 的表达式、位宽、块宽与解包循环
    └── gemv_i4.rs       # int4 格式：`Quantized<G>` 的表达式、位宽、粒度与解包循环
```

同一个算子的两个文件必须**同名成对**（`<name>.toml` ↔ `<name>.k`），缺模板会在装载时报错并指出两个路径。
**解析就在本 crate**（`decl` + 三张闭集 + `tpl`）：`cutile` 只消费 `OpDecl`，它的 `layout`/`sch` 是
`pub use` 转发路径——闭集与解析各只有一份，改声明语法不必动机制层。

## 两个文件各写什么

| 文件 | 内容 | 语义定义在 |
|---|---|---|
| `<name>.toml` | `[op]` 名字/骨架/摘要、`[axes]` 轴候选、`[needs]` 算子级能力签名、`[layouts]` 布局契约、`[[variants]]` 有序降级链与 `extra_needs` | `op::decl`、`op::layout`、`op::sch`、`prof::atom` |
| `<name>.k` | 模板正文：`{{槽位}}` 插值、`@原语(...)` 替换点 | `op::tpl`（词法）、`cutile::substitute`（拼接）、`cutile::emit`（成文） |

闭集（骨架 5 种、原语 16 个、布局构造子 4 个、调度模板 4 个）与书写规则见
[DESIGN §4](../cutile/DESIGN.md)。声明与模板的**解析**在 `op::decl`（并在这里做闭集校验），
`cutile` 只接住 `OpDecl`——所以 schema 只有一份，改动它不会出现第二处需要同步的地方。

语义（`src/<name>.rs`）是声明的第二份实现：一条**求和表达式**（`EXPR`，如 `nm,m,ng,n->n`）加每个
操作数的位宽/方向，就能让 `einsum` 推出 roofline 要的 `(flops, bytes)`——本 crate 不再写任何流量常数；
`sample` 给确定性的输入，`reference` 是标量参考（片段只与它对拍）；`shape` / `dtype` 声明生产形状与
输出类型——审计的形状矩阵、`bench` 的缺省形状、`verify` 的对拍阈值都取自它们，而不是在命令层
再写一份方阵或“每个元素 4 字节”。

## 一份骨架，多条精度

`gemv_i4` 与 `gemv_f32` 的代价、样本、校验、遍历是同一段代码，写在 `gemv.rs`；每个格式只写自己的
**契约**（`impl gemv::Format`）：权重元素类型（`u8` / `f32`）、位宽、形状粒度、派生轴（量化有
`g = m / GRAIN`，稠密没有）、标度取值与内核本体。骨架按格式**单态化**，所以 `Model` 里仍是三个
`fn` 指针，热路径上没有任何 `match op` 或枚举分发。

哪些东西进类型、哪些留在数据里，是按内核自己怎么收划的：稠密的块宽 `Dense<32>` 直接出现在
`chunks` 上（进热循环）；量化的声明粒度 `Quantized<32>` 决定派生轴除数与样本里的标度个数（冷路径）；
**实际分组大小**留在数据里，因为模板本身就把它当 `int gsz` 实参收——训练侧按 8 元素分组调用同一份
`forward` 不需要第二份单态化副本。

`MODEL` 由 `#[annotation::model(name = "..", shape = [..], dtype = "..")]` 生成，标在
`type 名字 = <具体格式>;` 上：`cost` / `sample` / `reference` 写成 `<名字 as Format>` 的实例，`group`
取 `Format::GRAIN`，另加"生产形状是粒度整数倍"的编译期断言与一条"注册表里的语义就是这份 `MODEL`、
样本与 `.k` 签名一致"的测试。写错格式名或形状，`cargo check` / `cargo test` 就失败，不必等到审计。

## 三处自证：格式、注解、注册表

注解只负责"让它被求值"，规则本身写在类型上，因为规则属于契约、不属于宏：

- **格式自证**（`Format::CHECK`，编译期）：`GRAIN > 0`，且派生轴的除数就是 `GRAIN`
  （`gemv::derived_matches_grain`）。两处都拿同一个常数算分组——`cost` 用派生轴除数造轴、`sample`
  用粒度算每行标度个数；写歪一个数不会让任何一步失败，只会让代价与样本说的是两种语义。
  生成物里的 `const _: () = <别名 as Format>::CHECK;` 就是这条规则的求值点（不被引用的关联常量不报错）。
- **注解 ↔ 输出类型**（`__model_check`，测试期）：注解的 `dtype` 说"回读按什么解释"，`.k` 的可写输出
  说"内核写的是什么"，`signature.output_type()` 把后者读出来逐项比较。两者分家时不该由 `verify`
  用一个错宽度的缓冲去对拍。`cutile::backend::harness` 另有一道同源闸：非 `float` 的输出在拼任何 C
  之前就被拒，诊断带算子名与缓冲名（而不是 C 编译器的一句 `incompatible pointer type`）。
- **注解 ↔ 注册表**（`tests/contracts.rs::the_registry_and_the_model_annotations_are_mirrors`，文本门禁）：
  标了 `#[model]` 必须在 `REGISTRY` 里有同名条目；注册表的 `model:` 必须写 `..::MODEL`，不能是手写的
  `Model { .. }` 字面量。语义表只有一个生产者，注册表与注解两侧互为镜像，没有豁免节。

内核实参不在这里手写：`op::sig` 从 `.k` 的入口函数解析形参名、类型与可写性，样本只按名字提供缓冲
与标量取值，实参顺序与输出缓冲名都由签名派生。签名、样本与模板任一漂移，装载期或 `verify`/`bench`
的拼接之前就会报错，诊断带算子名与形参名。

模板里的 `@原语(key: value)` 是签名的另一半：这些命名参数会原样拼进片段文本，说的就是"本次生成用哪条
签名"。因此 `key` 若属于能力属性闭集（`sat` / `op` / `axis` / `kind` / `accuracy` / `group` / `lanes` /
`src` / `dst` / `stages` / `exec`），它必须在该变体的 `[needs]` / `extra_needs` 里声明、且取值被声明
允许——否则设备按声明的值被协商验证，代码却按模板的值生成。闭集之外的命名参数（如 `along:`）是片段级
提示，协商不检查它们，留给片段。

属性键还必须属于**该原语**的签名（`prof::atom::Prim::attr_keys`）：`sat` 只对 `cvt` 成立，
`lanes` 只对 `dot` / `shuffle` / `broadcast` 成立，搬运是 `src` / `dst` / `stages`，`exec` 对所有
原语成立。写在别的原语上（`store.sat`、`mma.lanes`）在装载期报错并给出该原语允许的键——协商
只比对需求里出现的键，这类键既不会被检查，也不会被片段消费，只会让声明看起来更具体。

`cutile` 通过这些函数指针工作，因此它的代码里不出现任何算子名。要取**某一条**算子请用
`builtin_named(name)`，不要用 `builtin()[0]`：注册表从一条变两条时，按位置取会静默换人。

表达式里有一类轴必须声明成**派生轴**：`gemv_i4` 的 `scales[n, g]` 中 `g = m / 32`，它只影响流量
（这块张量确实只有 `n · m/32` 个元素），不能参与运算次数。`Extents::group` 用一条推导来表达这件事，
`Formula::axes` 因此能把每条轴标成批 / 自由 / 规约 / 分组四类角色。

## 现有的两条算子

| 算子 | 输入格式 | 求和表达式 | 用途 |
|---|---|---|---|
| `gemv_i4` | 4 位打包 + 每 32 元素一个标度 | `nm,m,ng,n->n` | weight-only 量化推理（带宽受限） |
| `gemv_f32` | f32（无标度） | `nm,m,n->n` | 稠密权重；真实 safetensors 最常见的形态 |

两条算子**只差数据**：同一副 `gemm` 骨架、同一批 CPU 原语（9 条 L2 见证）、同一条 lowering 流水线；
`cutile` 与 `prim` 在加入 `gemv_f32` 时一行未改（ADR 0013）。它们的代价不能互相顶替——同一份
`[4096, 4096]` 权重，f32 的读下界是 int4 的 8 倍，所以"格式 → 算子名"是 `nn::WeightFormat` 的映射，
不是一个全局默认值。

## 新增一个算子

1. 在 `ops/` 下写 `<name>.toml` 与 `<name>.k`；
2. 在 `src/<name>.rs` 写语义：`EXPR`（求和表达式）+ 操作数（位宽/方向）、`sample` / `reference`，
   并以 `#[annotation::model(..)]` 标注生成 `MODEL`（不要手写 `Model { .. }`）；样本里的缓冲名与
   标量名必须和 `.k` 签名一致，实参列表不在这里写；
3. 在 `src/lib.rs` 的 `REGISTRY` 里补一条 `Registration`（名字 + 两段 `include_str!` + `MODEL`）：
   声明、模板与语义表是同一条注册，`BUILTIN` 与 `model()` 都是它的投影，因此不存在
   "内嵌里有、语义表里没有"的第二份事实；
4. `cargo test -p llm-op` —— `builtin_matches_disk` 会把"磁盘有、内嵌无"直接判为失败；
5. `cargo run --bin cutile -- doctor --op <name> --device cpu` 看协商与 emit 结果，
   `cargo run --bin cutile -- verify --op <name> --devices cpu` 与标量参考对拍。

## 什么不放进来

- **设备档案**（`devices/*.toml`）：在 `crates/prof`，描述"设备能做什么"，与算子无关。
- **原语片段**（`prim/<vendor>/<prim>.<ext>`）与见证源码：在 `crates/prim`，描述"具体指令怎么写"。
- **机制**（协商、片段拼接、emit、编译与对拍）：在 `cutile`，本 crate 不依赖它——依赖方向固定为
  `llm → cutile → op → einsum`；`op` 的依赖只有 `einsum`（代价语义）、`diagnostic`（错误类型：
  `invalid` / `missing` / `io`，与 `Code::exit_code` 的退出码同一套）与同层的 `prof` / `llm-config`，
  不出现任何机制代码。命令行的参数文法也不在这里，在 `llm-cli::cutile`。
- 构建与工具链配置：见根目录的 `Cargo.toml`、`clippy.toml`、`rustfmt.toml`。
