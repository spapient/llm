# prim —— 片段机制

`prim` 只做**机制**：片段长什么样、从哪读、内嵌成哪几张表、怎么按原语查出来。片段**文本**
一行都不在这里——它住在六个设备 crate 里，每台设备一个 crate。

| | 在哪 | 是什么 |
|---|---|---|
| 机制 | 本 crate（`fragment` / `layout` / `builtin` 三个文件） | 片段与供给签名的形状、装载、内嵌表 |
| 文本 | `crates/{cpu,cuda,opencl,metal,musa,rocm}/<prim>.<ext>` | 一台设备的片段与见证源码 |
| 能力区间 | `crates/prof/devices/<vendor>.toml` | 这台设备有哪些原语、精度、形状、证据等级 |

`src/lib.rs` 是 64 行的薄入口：只写 crate 文档、`mod` 声明与 `pub use`，没有逻辑。拆成六个
crate 之后，`lib.rs` 里既没有片段表也没有 `include_str!`，发布包与源码树都通过同一套表取文本。

## 目录

```
crates/prim/src/
    lib.rs        薄入口：mod 与 pub use
    fragment.rs   片段是什么：Fragment / FragmentSet、签名声明行
    layout.rs     片段从哪来：Layout::builtin() / Layout::at(root)、见证查找
    builtin.rs    编译进来了什么：BUILTIN / WITNESS / DEVICES / ROOTS
crates/cpu/       bar.c  commit_wait.c  cvt.c  dot.c  load.c  load_async.c  quant.c  reduce.c  store.c + witness.c
crates/cuda/      同上，.cu
crates/opencl/    同上，.cl
crates/metal/     同上，.metal
crates/musa/      同上，.cu
crates/rocm/      同上，.cpp（AscendC）
```

每个设备 crate 都有 `src/lib.rs`（`ROOT` / `FRAGMENTS` / `DEVICE` 三个 `pub const`，`ROOT` 指向
自己的目录）、`build.rs`（一行）、以及一份生成到 `OUT_DIR` 的 `tables.rs`。加一台设备的成本因此
是"放一个目录 + 一行 `build.rs`"，而不是往 `prim` 里再塞一个 vendor 分支。

## 目录约定：vendor ≠ crate 名

`vendor` 是**档案里的键**，不是目录名——它是内嵌表的键，也是标签里 `witness.file` 的前缀，
所以历史名字留下了：

| crate | vendor | ext | emit | 备注 |
|---|---|---|---|---|
| `crates/cpu` | `cpu` | `c` | `c` | 本机执行 + host 自检 |
| `crates/cuda` | `nvidia` | `cu` | `cuda` | |
| `crates/opencl` | `intel` | `cl` | `opencl` | |
| `crates/metal` | `metal` | `metal` | `metal` | 唯一有镜像配方的设备 |
| `crates/musa` | `musa` | `cu` | `cuda` | 同一门语言，不同工具链 |
| `crates/rocm` | `ascend` | `cpp` | `ascendc` | |

"片段层根"`<root>` 就是那个 crate 的 `env!("CARGO_MANIFEST_DIR")`（`prof::Device` 的 `ROOT`）：
`Layout::builtin()` 用六台设备内嵌的源码根，`Layout::at(dir)` 用磁盘上的目录树——前者给发布包
（没有源码树），后者给开发机与 `cargo test`。装载逻辑只有一份，两者共用。

## 片段与供给签名

片段文件名必须是 `Prim` 闭集里的名字（`dot` / `quant` / ...），扩展名必须是这台设备的 `ext`。
第一条非注释行是**签名声明**：

```c
/* cutile-sign: dot/fma */
```

它说"我是哪条供给"。形状与解析在 `prof::sign`，`prof::codegen` 在构建期、`fragment` 在装载期
读的是同一个解析器，不存在两份正则。

同一台设备的同一原语可以有**多段文本**，靠 `impl` 区分：`dot/fma` 与 `dot/dp4a` 同住 `dot.c` …
放在不同文件里会与"文件名即原语"冲突，所以按 `impl` 分在档案的 `provides` 里、按声明行分在
文本里。`FragmentSet` 就是这张 `(原语, impl) → 文本` 的表，`get` 取"这条原语有没有默认实现"，
`get_impl` 取"这条实现有没有文本"。档案协商会挑中 `fma`、文本却只写了 `dp4a` 时，装载期就说
"这条签名没有文本"——不必等到构建期，更不会静默用错算法。

声明行是**元数据**：拼接只取它之后的文本（`Fragment::body`），它既不进生成的源文本，也不进
缓存指纹。因此给已有片段补这一行不会改动已落盘的产物，`verify` 打印的 `片段 N 行` 与缓存键
逐字不变。

六台设备各 9 段（P0 模板用到的 9 个原语），档案里共 102 条 `provides`：文本只写这 9 个，其余
签名停在 `L0`，只有见证没有文本。`cutile` 的
`every_device_fragment_set_splices_the_p0_template` 把这条约定变成判据：台架上的档案只要能
协商出变体，它的片段就必须能把 P0 模板拼出来——参数个数或命名参数写错不必等到构建期。

## 见证源码

`<vendor>/witness.<ext>` 是"这个设备真的能跑"的最小证据。函数统一为 `int func(float *out)`：
返回写入的元素个数（≤ 8），值按 f32 读回；档案里同时写清楚入口函数与期望输出：

```toml
[[provides]]
prim    = "dot"
grade   = "L2"
witness = { file = "cpu/witness.c", func = "witness_dot", expect = [70.0] }
```

`file` 相对片段层根（即 `crates/cpu/witness.c`）。`grade ≥ L2` 的供给**必须**有见证；
`cutile verify` 先把每一条见证编成可执行文件并执行，值与 `expect` 不符就整台设备失败。
`L0`/`L1` 不要求见证，但**一旦声明就会校验**：文件名可读、入口函数在源码里、期望值非空。
设备执行跑不到的见证还会走**本机自检**：见证的 ABI 是 host C，与片段语言无关，所以自检
跳过档案的 `[compile].preamble` 与 `flags`，把文件编成可执行程序逐条对拍——期望值写错会
当场失败。结论记作 `Scope::Host`：自检只证明源码与期望值自洽，不证明设备能跑，因此不改写
任何等级。

两个范围**不重叠、合起来覆盖每一条声明**：设备在本机时 `run_all` 跑全部声明（`cpu` 与
`metal` 都属于这种，后者靠 Metal 工具链判定），自检一条不跑；设备不在本机时全部进自检。
`cpu` 的那几条此前两头落空——设备执行只挑 `L2`，自检又只跑非 C 设备，写下的证据从未被执行。

`metal` 的见证因此必须在 MSL 里成立：出参写成 `thread float *out`（MSL 没有默认地址空间，
`float *out` 编不过）。`cc` 只因为它是纯 C 才放行，`xcrun metal` 不放——自检走 host C 前端时
会前置一行 `#define thread`，设备前端不经过这一行。

同一件事对另外三台设备也成立：`crates/cuda/witness.cu` 与 `crates/opencl/witness.cl` 由 clang 的
`-x cuda` / `-x cl` 前端各解析一遍（`crates/rocm/witness.cpp` 由 CANN 的 `ccec -xcce` 前端解析）。
解析器比 host `cc` 严格：`double` 在没开 `cl_khr_fp64` 的 OpenCL C 里是错误，而 `cc` 照收。所以
写见证时按**它自己那门语言**的规则写，不要只按"`cc` 能过"写。

**一条签名一条见证**：两个 `impl` 不能指向同一个入口函数。见证证明的是"这条签名写的是什么
算法"，共用会让"每条签名都有见证"退化成"每条都指向同一份证据"；`ascend` 的 `quant` /
`load` / `load_async` 因此各有两份函数。`profile.validate` 把这条规则做成装载期判据。

五台加速器的见证覆盖整个原语闭集（`nvidia` 17 条、`intel` 17 条、`ascend` 20 条、`metal` 17 条、
`musa` 17 条）：除了 P0 用到的 9 个，还包括张量核 `mma`、`shuffle` / `broadcast` / `scan` 与
`exp2` / `rcp` 的标量等价实现。这些签名没有片段文本（P0 模板不调用它们），等级停在 `L0`。

`cpu` 的 `witness.c` 同样是 14 条：P0 用到的 9 个（`L2`）、`atomic` 与 `mma` / `scan` / `exp2` /
`rcp`（`L0`）。`shuffle` / `broadcast` **不**声明——`[simt].group = 1` 的单线程没有可交换的
第二条 lane，`prof` 的装载期判据会直接拒绝这种声明。闭集的空白格要么有见证，要么有数据
解释，不留"忘了写"。

`witness.<ext>` 与片段同目录同扩展名，因此**装载片段时按名字跳过它**——否则目录扫描会把它
当成一个名字非法的片段。发布机上没有源码树，所以见证源码由各设备 crate 的 `build.rs` 编进
`WITNESS`（与 `BUILTIN` 分开：见证不是片段）。

## 构建期：四张表

`BUILTIN`（片段）、`WITNESS`（见证源码）、`DEVICES`（每台设备的原语枚举 + 入口映射表 + 可选镜像）、
`ROOTS`（源码根）由 `builtin.rs` 合成，四张表按下标同源（`the_tables_line_up_by_index` 守着顺序）。
每个设备 crate 的那一份由它自己的 `build.rs` 一行生成：

```rust
fn main() { prof::codegen::device("rocm"); }
```

生成器住在 `crates/prof/src/codegen`（`prof` 的 `build` feature 之后），因为六台设备都依赖 `prof`
——让生成器依赖 `prim` 会绕成环。生成与校验走同一趟，数据错在构建期就失败：

1. 文件名必须是 `Prim` 闭集里的名字；
2. 片段第一行的签名声明 `/* cutile-sign: <prim>/<impl> */` 必须在档案里有对应 `provides`；
3. 那条供给必须带见证（映射表指向的是镜像里的一个入口）；
4. `meta.vendor` 必须等于 `build.rs` 里写的那个设备名，扩展名必须等于 `meta.emit.ext()`。

`DEVICES` 里的 `Device::prims` 是**原语枚举 + 映射表**：这台设备有哪些原语、每个原语在镜像里的
入口叫什么。入口名来自档案里 `(原语, impl)` 那条供给的见证函数，所以它一定能在那份见证源码里
找到（`every_entry_is_a_function_in_the_witness_source` 把这条约定变成判据）。

### 源码调试 / 设备镜像发布

| 模式 | 怎么开 | 带什么 | 谁用 |
|---|---|---|---|
| 调试 | 默认 | 片段文本 + 见证源码（`Device::image` 为 `None`，`status` 为 `Source`） | 开发与 `doctor`：运行期照旧 emit 后编译 |
| 发布 | `--features device-image` | 额外把能编的设备编成镜像并嵌入（`metal` → `metallib`） | 发布包：不依赖源码树与设备编译器 |

没有镜像**不是错误**：`Device::status` 是闭集，`NoRecipe`（这门语言还没有配方）、
`Skipped { reason }`（工具链不在本机 / 编译失败，附第一行 stderr）各说各的原因。发布流水线要
"缺镜像即失败"时显式要求：

```bash
cargo build -p llm --features device-image                        # 带镜像（feature 逐层透传 cutile → prim → 六个设备 crate）
CUTILE_REQUIRE_IMAGE=metal cargo build -p llm --features device-image   # 缺 metal 镜像直接失败
```

镜像里目前是**见证**编译出的设备侧库（一份完整翻译单元、每条签名一个入口），不是片段：片段是
语句块，`$0..$3` 的绑定由算子模板的 `@prim(...)` 给出，脱离模板编译只能得到与真实调用无关的
代码。这也是为什么只有 `metal` 有配方——MSL 收普通函数，CUDA / OpenCL / AscendC 的镜像只收
`__global__` / `__kernel__` 入口，把 host 见证编进去只会得到空产物。

编译不传档案的 `[compile].flags` / `preamble`：见证是 host C ABI（`int func(float *out)`），
与片段语言无关，`verify` 的本机自检因同一个原因跳过它们。

## 新增设备 / 更换一段实现

1. 新建设备 crate（`crates/<name>`）：`Cargo.toml` 依赖 `prof`，`build.rs` 一行
   `fn main() { prof::codegen::device("<vendor>"); }`，`src/lib.rs` 用 crate 宏导出 `ROOT` /
   `FRAGMENTS` / `DEVICE`（照抄 `crates/cpu/src/lib.rs` 即可）；
2. 在 crate 根写片段（只写这台设备真正支持的，其余留空），第一行必须是
   `/* cutile-sign: <prim>/<impl> */`，`impl` 取档案里 `provides.impl` 的那一个；
3. 写 `witness.<ext>`（每条签名一个 `int func(float *out)` 入口），并在
   `crates/prof/devices/<vendor>.toml` 里登记 `provides`——`meta.vendor`、`meta.emit.ext()`、
   `<vendor>` 三者必须一致，否则构建期报错；
4. 在 `crates/prim/src/builtin.rs` 给四张表各补一行（顺序任意，`the_tables_line_up_by_index`
   会守住同源）；
5. `cargo test -p llm-prim` —— `builtin_matches_disk` 对拍内嵌与磁盘，`every_device_directory_has_a_row`
   保证新 crate 一定进表；
6. `cargo test -p llm-prim --features device-image` 有工具链时验证镜像（`metal` 会断言 `MTLB` 魔数
   与符号数）；
7. `cargo run --bin cutile -- verify --op <op> --devices <vendor>` 有真机时对拍并回填档案的 `grade` 与 `witness`。

换一段实现只动 `<vendor>/<prim>.<ext>` 与档案里那条供给；`prim` 一行不用改。

## 什么不放进来

- **设备能力区间**（哪个原语、什么精度、什么形状、什么证据等级）：在 `crates/prof/devices/`。
  片段只说"这条能力怎么写"，不说"这台设备有没有"。
- **算子声明与模板**：在 `crates/op`。
- **机制**：模板词法与拼接（`lower/substitute`）、emit（`lower/emit`）、缓存与编译（`cache`、`backend`）、
  探测与协商（`probe`、`cap`）都在 `cutile`。依赖方向固定为
  `cutile → prim → {六个设备 crate, prof, diagnostic}`，设备 crate 只依赖 `prof`（生成器在
  `prof::codegen`，所以不会绕成环），本 crate 不依赖 `cutile`。
- 构建与工具链配置：见根目录的 `Cargo.toml`、`clippy.toml`、`rustfmt.toml`。
