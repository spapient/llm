# prim —— 片段层（原语实现 + 见证源码）

每台设备上每个原语"具体怎么写"的文本集合：`<vendor>/<prim>.<ext>`。
**这一层不含机制**：拼接、emit、编译、执行都在 `cutile`，本 crate 只保证片段能被找到、
文件名落在原语闭集内、内嵌副本与磁盘一致。

```
crates/prim/
├── Cargo.toml
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
└── src/lib.rs            # BUILTIN 内嵌 / WITNESS 内嵌 / builtin 查表 / load 目录装载 / dir 与 layer_root
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
缓存指纹。因此给已有片段补这一行不会改动已落盘的产物，`verify` 的 `source N 行` 与缓存键
逐字不变。

因此 `nvidia` / `intel` / `ascend` 的 9 段片段与 `cpu` 一样是标量语义的自包含 C：语言外壳不同，
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
`nvidia` / `intel` / `ascend` 的 35 条 L0/L1 见证还会走**本机自检**：见证的 ABI 是 host C，
与片段语言无关，所以自检跳过档案的 `[compile].preamble` 与 `flags`，把文件编成可执行程序
逐条对拍——期望值写错会当场失败。结论记作 `Scope::Host`：自检只证明源码与期望值自洽，
不证明设备能跑，因此不改写任何等级。

`witness.<ext>` 与片段同目录同扩展名，因此**装载片段时按名字跳过它**——否则目录扫描会把它
当成一个名字非法的片段。发布机上没有源码树，所以见证源码登记在 `src/lib.rs` 的 `WITNESS`
（与 `BUILTIN` 分开：见证不是片段）。

## 新增设备 / 更换一段实现

1. 建 `<vendor>/` 目录，按 `<prim>.<ext>` 写片段（只写这台设备真正支持的，其余留空），第一行
   必须是 `/* cutile-sign: <prim>/<impl> */`，`impl` 取档案里 `provides.impl` 的那一个；
2. 在 `src/lib.rs` 的 `BUILTIN` 里补一条 `(vendor, ext, fragments)`（否则发布出去的二进制里没有它）；
3. `cargo test -p prim` —— `builtin_matches_disk` 会把"磁盘有、内嵌无"直接判为失败；
4. `cargo run --bin cutile -- verify --op <op> --devices <vendor>` 有真机时对拍并回填档案的 `grade` 与 `witness`。

## 什么不放进来

- **设备能力区间**（哪个原语、什么精度、什么形状、什么证据等级）：在 `crates/prof/devices/`。
  片段只说"这条能力怎么写"，不说"这台设备有没有"。
- **算子声明与模板**：在 `crates/op`。
- **机制**：模板词法与拼接（`lower/substitute`）、emit（`lower/emit`）、缓存与编译（`cache`、`backend`）、
  探测与协商（`probe`、`cap`）都在 `cutile`；依赖方向固定为 `cutile → prim → prof`，本 crate 不依赖 `cutile`。
- 构建与工具链配置：见根目录的 `Cargo.toml`、`clippy.toml`、`rustfmt.toml`。
