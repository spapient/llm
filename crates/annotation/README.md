# `llm-annotation`

编译期注解层：把"一份声明"写成属性宏，由编译器检查，而不是手写 trait 样板再靠人复核。宏只生成
调用点所在 crate 的代码，不拥有任何领域类型——生成的实现指向契约路径（`nn::Contract`、
`op::model::Model`），因此本 crate 对它们没有运行期依赖。

## 两道注解

| 注解 | 键 | 生成的东西 | 绑定的契约 |
|---|---|---|---|
| `#[spec(..)]` | `name`、`out`、可选 `in`、`ops` | 一份 `impl ::nn::Contract`（含懒构造的 `ContractSpec`）+ 一条校验测试 | `nn::Contract` / `nn::ContractSpec` |
| `#[spec(..)]`（运行时形态） | 每个值写成 `$构造参数`；可选 `role` | 注入私有 `spec` 字段 + 构造宏 `__layer_spec_<类型>!(...)`；不生成校验测试 | 同上 |
| `#[model(..)]` | `name`、`shape`、`dtype`、可选 `skeleton` | `MODEL` 常量、粒度/形状编译期断言 + 一条注册表对拍测试 | `op::model::Model` / `gemv::Format` |

运行时形态用于名字或宽度由构造参数算出的层：注解只能标命名字段 `struct`，`in` 不在它的键表里
（运行期宽度无法交给静态的 `in`，下界由层自己实现）。

## 结构

文法与 `proc_macro` 解耦是这一层的核心决定：`proc_macro::TokenStream` 只在展开期可用，规则直接
写在它上面，每条错误路径都无法被测试。

四个生产文件平铺在 `src/`，单向依赖 `syntax ← decl ← render ← lib`，没有子目录、没有循环。

| 层 | 文件 | 职责 | 可测性 |
|---|---|---|---|
| 文法 | `src/syntax.rs` | token 模型、游标、逐键取值、目标类型判据、全部错误路径 | 纯函数，不接触 `proc_macro` |
| 声明 | `src/decl.rs` | 三种形态的键表与解析产物（层 / 运行时 / 语义表） | 表驱动，加键 = 加一行 |
| 渲染 | `src/render.rs` | 声明 → `impl` / 常量 / 校验测试的文本拼接 | 纯字符串拼接 |
| 适配 | `src/lib.rs` | `TokenStream` ↔ token 模型、错误映射回 span；`Streams` 依赖倒置 | 字符串替身实现同一 trait |

测试在 `tests/`：`fixture.rs`（注解文本 → token 模型的最小词法器，分层夹具）、`decl_tests.rs`、
`render_tests.rs`、`adapter_tests.rs`、`audit.rs`（独立的 `[[test]]`，读源码与 manifest 的静态门禁）。

## 判据与证据

- **闭集只有一份事实源。** `DTYPES` / `SKELETONS` 是词汇表的投影：与
  有阈值的 `DType` 集合、骨架分支表逐项相等，由仓库级契约测试核对；每个调用点
  生成的测试再把注解字符串与词汇表变体对拍。
- **错误指在写错的 token 上。** 未知键、重复键、缺必填键、非字面量、转义序列、零宽轴、空算子集、
  泛型参数、非 `struct`/`enum` 被标注项，都产出指向具体位置的 `compile_error!`，而不是笼统地指
  在被标注类型上。
- **失败也保留被标注项。** 解析或生成失败时，原被标注项原样留在输出里，避免次生错误埋掉真正原因。
- **骨架表一行一条。** 十二条骨架的差异（模块名、`group`、是否泛型单态化、是否带粒度断言）集中在
  `SKELETONS`；闭集与分支表一一对应，漏一行先在本 crate 的测试里红掉。
- **确定性。** 同一注解两次展开得到相同文本；不维护运行期注册表，注册表仍是 `llm_op::REGISTRY` 一张事实源。

## 性能与边界

- 文法是游标**单趟线性扫描**：名字/闭集只在数组上线性包含，不引入关联容器、正则或惰性初始化
  （`tests/audit.rs` 静态门禁守住这条）。
- 运行期依赖**为空**（`tests/audit.rs::runtime_dependency_surface_is_empty` 守住）：不引入
  `syn` / `quote` / `proc-macro2`，也不依赖 `dtype` 基座——`MODEL` 由 `Operator` 投影得到。
- 过程宏在编译期运行，输入只是一小段属性；`proc_macro` 适配层只在编译器展开期可用，不参与运行期
  计时。纯文法与渲染逻辑由单元测试与确定性种子回归覆盖。

  展开期成本（`--release`、每形态 200k 次、`black_box`、5 轮中位数；只计**渲染**，夹具词法器
  不计入 —— 它只存在于测试里）：

  | 形态 | 改造前 | 改造后 | 变化 |
  |---|---|---|---|
  | 层 `#[spec]` → 文本 | 324.7 ns | 158.6 ns | −51.1% |
  | 运行时 `#[spec]` → 文本 | 478.8 ns | 189.0 ns | −60.5% |
  | 语义表 `#[model]` → 文本 | 507.2 ns | 495.3 ns | −2.3% |
  | 层：parse + render | — | 435.4 ns | — |
  | 层：lex + parse + render | 927.8 ns | 741.3 ns | −20.1% |

  `#[model]` 几乎不动，因为它的文本量最大且绝大部分是固定前缀，本轮去掉的中间 `String` 在其中占比
  很小；层与运行时形态的文本里内插点密集，收益才显出来。

  下降来自三件事：热路径只分配**一个**输出缓冲（`String::with_capacity` + `write!` 直写，不再
  `format!` 后复制）；`Quoted` 借用视图让字面量零分配嵌进格式串；`comma` 闭包让调用点只描述
  "每一项长什么样"，不再先攒 `Vec<String>` 再 join。这三条由
  `tests/audit.rs::rendering_allocates_only_its_output_buffer` 静态钉住 —— 在这三个渲染函数体内
  出现 `format!` / `.to_string()` / `.to_owned()` / `String::new()` 即失败，因为本 crate 是
  `#![forbid(unsafe_code)]`，挂不了计数分配器，计时门在 CI 上又会抖。

  第二轮量的是**容量预算**。预留量原先按短样例拍的，实测最长的层要 1940 B、语义表要 3102 B，都超过
  预留 —— 于是"只分配一次"其实一直是"分配 + realloc + memcpy"。按实测上界重定（`STATIC_ROOM`
  1536→3072、`MODEL_ROOM` 3072→4096；`RUNTIME_ROOM` 512 维持，实测只用到 337 B）：

  | 形态 | 重定前 | 重定后 | 变化 |
  |---|---|---|---|
  | 长层（34 字符名、4 轴、6 算子） | 528.2 ns | 420.8 ns | −20.3% |
  | 语义表（`gemm_f32` 整张表） | 552.5 ns | 464.1 ns | −16.0% |

  同一轮还把 `unique` 的临时集合改成按 `holes` 的上界预留，消掉每份运行时声明两次 realloc；A/B 实测
  23.0 vs 22.4 ns，落在噪声里 —— 这条是结构正确性（"只分配一次"这句自述现在真的成立），不是提速项。

  预留量也不是拍的：`tests/render_tests.rs::one_render_allocates_exactly_once` 用 `capacity()` 断言
  "渲染完容量仍是预留值"。`String::with_capacity(n)` 精确给出 `n`，写超一次就翻倍增长，所以这条断言
  把三个常量钉死 —— 把预留量调回旧值立刻红（1775 B 与 3105 B 两个夹具都反向验证过）。

## 状态

API 对内稳定，尚未发布到 crates.io（`publish = false`）：发布意味着 MSRV 与 semver 承诺，本仓库
尚未作出。依赖方向与所有权见 [ARCHITECTURE.md](../../docs/ARCHITECTURE.md)。
