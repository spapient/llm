# prim —— 片段层（原语实现 + 见证源码）

每台设备上每个原语"具体怎么写"的文本集合：`<vendor>/<prim>.<ext>`。
**这一层不含机制**：拼接、emit、编译、执行都在 `cutile`，本 crate 只保证片段能被找到、
文件名落在原语闭集内、内嵌副本与磁盘一致。

```
crates/prim/
├── Cargo.toml
├── build.rs              # 编译期：目录 → 表（BUILTIN / WITNESS / DEVICES）+ 可选设备镜像
├── README.md
├── cpu/                  # vendor = cpu，扩展名 = c
│   ├── load.c            # 9 段片段，文件名即原语名（Prim 闭集）
│   ├── load_async.c
│   ├── commit_wait.c
│   ├── store.c
│   ├── quant.c
│   ├── dot.c
│   ├── bar.c
│   ├── cvt.c
│   ├── reduce.c
│   └── witness.c         # 见证源码（L2 证据），不是片段，装载时跳过
├── nvidia/               # 扩展名 = cu（emit = cuda）：同样 9 段 + witness.cu
├── intel/                # 扩展名 = cl（emit = opencl）：同样 9 段 + witness.cl
├── ascend/               # 扩展名 = cpp（emit = ascendc）：同样 9 段 + witness.cpp
├── metal/                # 扩展名 = metal（emit = metal）：同样 9 段 + witness.metal
└── src/
    ├── lib.rs            # builtin 查表 / load 目录装载 / dir 与 layer_root / 片段解析
    ├── device.rs         # Device / Image / Status：源码 + 可选的设备镜像
    └── tables.rs         # include!(OUT_DIR/tables.rs)：生成的表（只有数据）
```

## 目录约定

`<root>/<vendor>/<prim>.<ext>`，其中 `<root>`：

- 仓库内是 [`prim::dir()`](src/lib.rs)，即 `crates/prim`；
- 换 `data.root` 时是 `<data.root>/prim`，由 [`prim::layer_root`](src/lib.rs) 给出。

扩展名由 `emit` 语言决定（`Emit::ext`：`c` / `cu` / `cl` / `cpp` / `metal` / `hip` …），文件名的主干必须是 `prof::atom::Prim` 里的名字。
写错一个名字（`load_async2.c`）会在装载期报 "不在原语闭集内"，而不是静默变成"该设备没有这个能力"。

## 片段与供给签名

档案里同一个原语可以有**多条签名**（`dot` 既有 `fma`（`dt = [f32]`、`lanes = 1`）又有 `dp4a`
（`dt = [i8]`、`lanes = 4`））。片段层每个原语只有**一段**文本，它对应的是 P0 模板真正调用
的那条标量签名；`dp4a` / `mmad` / `cp.async` 这类 tile 落点属于 P3，等模板真的按 tile 写时
再按实现名分文件。

"对应哪一条"不是注释，而是数据：每段片段的**第一行**是签名声明

```
/* cutile-sign: dot/fma */
```

`<prim>/<impl>` 里的 `impl` 就是档案里那条 `provides.impl`。这一行由 [`Fragment::parse`](src/lib.rs)
在装载时解析：缺声明、声明的原语与文件名不一致、`impl` 为空都会直接报错。片段只说"这段文本
写的是哪一条签名"，"这台设备有没有这条签名"仍由档案回答；两边的对齐由 `cutile` 的
`Catalogue::validate` 在**装载期**判定——档案协商会挑中 `fma`、文本却声明写的是 `dp4a`
时不必等到构建期，更不会静默用错算法。

声明行是**元数据**：拼接只取它之后的文本（`Fragment::body`），它既不进生成的源文本，也不进
缓存指纹。因此给已有片段补这一行不会改动已落盘的产物，`verify` 打印的 `片段 N 行` 与缓存键
逐字不变。

因此 `nvidia` / `intel` / `ascend` / `metal` 的 9 段片段与 `cpu` 一样是标量语义的自包含 C：语言外壳不同，
调用约定相同（`@load`、`@quant`、`@dot` … 的位置参数与命名参数一一对上）。`cutile` 的
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

`file` 相对片段层根（即 `crates/prim/cpu/witness.c`）。`grade ≥ L2` 的供给**必须**有见证；
`cutile verify` 先把每一条见证编成可执行文件并执行，值与 `expect` 不符就整台设备失败。
`L0`/`L1` 不要求见证，但**一旦声明就会校验**：文件名可读、入口函数在源码里、期望值非空。
设备执行跑不到的见证还会走**本机自检**：见证的 ABI 是 host C，与片段语言无关，所以自检
跳过档案的 `[compile].preamble` 与 `flags`，把文件编成可执行程序逐条对拍——期望值写错会
当场失败。结论记作 `Scope::Host`：自检只证明源码与期望值自洽，不证明设备能跑，因此不改写
任何等级。

两个范围**不重叠、合起来覆盖每一条声明**：非 C 设备的全部声明进自检，`cpu` 只把它的
`L0`/`L1`（`atomic` 与闭集里 CPU 真正具备的四条）交给自检，`L2` 仍由设备执行给出。`cpu` 的
那几条此前两头落空——设备执行只挑 `L2`，自检又只跑非 C 设备，写下的证据从未被执行。

**一条签名一条见证**：两个 `impl` 不能指向同一个入口函数。见证证明的是"这条签名写的是什么
算法"，共用会让"每条签名都有见证"退化成"每条都指向同一份证据"；`ascend` 的 `quant` /
`load` / `load_async` 因此各有两份函数。`profile.validate` 把这条规则做成装载期判据。

四台加速器的见证覆盖整个原语闭集（16 × 4：`nvidia` 17 条、`intel` 17 条、`ascend` 20 条、
`metal` 17 条）：除了 P0 用到的 9 个，还包括张量核 `mma`、`shuffle` / `broadcast` / `scan` 与
`exp2` / `rcp` 的标量等价实现。这些签名没有片段文本（P0 模板不调用它们），等级停在 `L0`。

`cpu` 的 `witness.c` 同样是 14 条：P0 用到的 9 个（`L2`）、`atomic` 与 `mma` / `scan` / `exp2` /
`rcp`（`L0`）。`shuffle` / `broadcast` **不**声明——`[simt].group = 1` 的单线程没有可交换的
第二条 lane，`prof` 的装载期判据会直接拒绝这种声明。闭集的空白格要么有见证，要么有数据
解释，不留“忘了写”。

`witness.<ext>` 与片段同目录同扩展名，因此**装载片段时按名字跳过它**——否则目录扫描会把它
当成一个名字非法的片段。发布机上没有源码树，所以见证源码由 `build.rs` 编进 `WITNESS`
（与 `BUILTIN` 分开：见证不是片段）。

## 构建期：三张表与两种模式

`BUILTIN`（片段）、`WITNESS`（见证源码）、`DEVICES`（每台设备的原语枚举 + 入口映射表 + 可选镜像）
全部由 `build.rs` 在编译期扫描 `<vendor>/` 生成，**不手写登记**。以前每加一台设备都要补一行
`include_str!`，漏一行的症状是"发布出去的二进制里没有它"，只有测试能发现；现在加设备就是放文件。
生成与校验走同一趟，数据错在构建期就失败：

1. 文件名必须是 `Prim` 闭集里的名字；
2. 片段第一行的签名声明 `/* cutile-sign: <prim>/<impl> */` 必须在档案里有对应 `provides`；
3. 那条供给必须带见证（映射表指向的是镜像里的一个入口）；
4. 目录名必须等于档案的 `meta.vendor`，扩展名必须等于 `meta.emit.ext()`。

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
cargo build -p llm --features device-image                        # 带镜像（feature 逐层透传 cutile → prim）
CUTILE_REQUIRE_IMAGE=metal cargo build -p llm --features device-image   # 缺 metal 镜像直接失败
```

镜像里目前是**见证**编译出的设备侧库（一份完整翻译单元、每条签名一个入口），不是片段：片段是
语句块，`$0..$3` 的绑定由算子模板的 `@prim(...)` 给出，脱离模板编译只能得到与真实调用无关的
代码。这也是为什么只有 `metal` 有配方——MSL 收普通函数，CUDA / OpenCL / AscendC 的镜像只收
`__global__` / `__kernel__` 入口，把 host 见证编进去只会得到空产物。

编译不传档案的 `[compile].flags` / `preamble`：见证是 host C ABI（`int func(float *out)`），
与片段语言无关，`verify` 的本机自检因同一个原因跳过它们。

## 新增设备 / 更换一段实现

1. 建 `<vendor>/` 目录，按 `<prim>.<ext>` 写片段（只写这台设备真正支持的，其余留空），第一行
   必须是 `/* cutile-sign: <prim>/<impl> */`，`impl` 取档案里 `provides.impl` 的那一个；
2. 建 `<vendor>/witness.<ext>`（每条签名一个 `int func(float *out)` 入口），并在
   `crates/prof/devices/<vendor>.toml` 里登记 `provides`——目录名、`meta.vendor`、`meta.emit.ext()`
   三者必须一致，否则构建期报错；
3. `cargo test -p llm-prim` —— `builtin_matches_disk` 对拍内嵌与磁盘，`every_device_directory_has_a_row`
   保证新目录一定进表；
4. `cargo test -p llm-prim --features device-image` 有工具链时验证镜像（`metal` 会断言 `MTLB` 魔数
   与符号数）；
5. `cargo run --bin cutile -- verify --op <op> --devices <vendor>` 有真机时对拍并回填档案的 `grade` 与 `witness`。

**不需要**再改 `src/lib.rs` 或 `Cargo.toml` 里的任何登记表：`BUILTIN` / `WITNESS` / `DEVICES` 都是生成的。

## 什么不放进来

- **设备能力区间**（哪个原语、什么精度、什么形状、什么证据等级）：在 `crates/prof/devices/`。
  片段只说"这条能力怎么写"，不说"这台设备有没有"。
- **算子声明与模板**：在 `crates/op`。
- **机制**：模板词法与拼接（`lower/substitute`）、emit（`lower/emit`）、缓存与编译（`cache`、`backend`）、
  探测与协商（`probe`、`cap`）都在 `cutile`；依赖方向固定为 `cutile → prim → {prof, diagnostic}`，本 crate 不依赖 `cutile`。
- 构建与工具链配置：见根目录的 `Cargo.toml`、`clippy.toml`、`rustfmt.toml`。
