# 0019：单消费者吸收（packaging merge）

- 状态：已采纳
- 日期：2026-11-13

## 背景

ADR 0004 的判据是"删掉它会谁编译失败"。按这条判据，一个只被**一个**成员消费、
且自己不含第二份实现的 crate 是**合法的**——它让契约有自己的位置。问题在于这条判据
只管"要不要存在"，不管"存在几个位置"：

- `crates/audit` 拥有 `Toolchain`/`probe`/`compile`/`run` 的**类型**，而 `xcrun`/`cl`/`nvcc`
  的**定位**在 `cutile::backend`；同一件"编译并运行"写了两遍；
- `crates/kv` 的页表只收页数，而"要几页"在 `engine` 里写了四遍；
- `crates/exec` 是纯契约，生产使用者只有 `engine::Sequence` 里的一个 `Stream` 字段；
- `crates/checkpoint` 与 `train::Checkpoint` 各自表述"恢复需要知道什么"（6 个事实 vs 4 个）；
- `crates/regress` 的唯一消费者是 `benches` 的一条 dev 边；
- `crates/einsum` 的存在理由是"让 `cutile`/`nn` 只依赖 `op`"——由 `op::model` 的再导出补丁维持。

这六处有一个共同形状：**边界两边是同一个变化原因**。改约定必须同时在两个 crate 里改，
而没有任何测试会因此变红（两边各自有测试，各自都绿）。

## 决策

除"真重复"之外，追加**第二种**合并理由：**单消费者吸收**。

一个成员可以并入其唯一消费者（或唯一被消费者）的**子模块**，当且仅当三条同时成立：

1. **只有一个生产消费者**（dev 边与工具成员不算）；
2. **被吸收方不含"第二方无法复制的知识"**——没有第二份实现、没有独立版本节奏、
   没有必须物理隔离的机制（例如 `annotation` 是 `proc-macro = true`，Cargo 强制独立 crate）；
3. **吸收后方向仍合法**：跨层向下，或同层向右（ADR 0001 的后果段要求这是显式决策，
   本条即是）。

落点必须是**存活方的子模块**，不是把两个 `lib.rs` 拼在一起：`engine/{exec,kv}.rs`、
`train/snapshot/`、`cutile/report/`、`op/einsum/`、`tool/regress.rs`。

## 后果

- 成员 33 → 26（crate 32 → 25），边 106 → 81（含 3 条改指向的 dev 边），根包依赖 18 → 15。
- **生产代码行数基本不变**（+75，全是注释与一个 `pages_for` helper）。合并换的是成员数与边数，
  不是行数；行数减量必须另找生成化与删除，不能指望合并。
- 测试用例数不变（1 037 → 1 037）：吸收是搬家，不是重写。
- `> 899 行` 的文件数不变（8），但**集中**：`train` 一个 crate 里有 2 个（`store.rs`、
  `snapshot/mod.rs`）。吸收会让"大文件"落到更少的人手里。
- 七条里**只有 `kv → engine` 顺手消掉了真重复**（页数公式 4 → 1，`pages_for`）。
  其余五条如果就此收工，成员少了五个，重复一条没少——这正是本 ADR 要写清的一点：
  **包装合并不自动带来质量收益**。
- 未做的附带动作（`exec` 的 SPI 去留、`snapshot` 拆文件、`Facts` 补两个事实、
  字节级黄金向量、`weights::leb` 收口）逐条记在 `docs/audit.md` §11.4，各自需要独立提交：
  它们改变行为或格式，混进搬家提交会让 revert 不再原子。

## 追加：第七次吸收（`proto → serve`）

`proto`（1 126 行、4 个文件）按同样的三条判据吸收进 `serve` 的 `proto` 子模块：

1. **只有一个生产消费者**：`crates/serve/Cargo.toml` 是唯一引用 `llm-proto` 的清单（根包与其它成员都不依赖它）。
2. **不含第二方无法复制的知识**：帧格式是 `serve` 端点契约的线格式，`Envelope`/`Frame`/`Decoder` 不持有任何独立版本节奏；`serve` 原本就依赖 `diagnostic`，吸收不新增生产依赖（只把 `testkit` 从 `proto` 的 dev 边平移到 `serve`）。
3. **方向仍合法**：`proto` 在 L5、`serve` 在 L6，跨层向下，`tests/architecture.rs` 的方向断言不需要放宽。

收益：成员 27 → 26、边 83 → 81（消失 `proto → diagnostic`、`serve → proto`）、根包依赖不变；`pub` 路径从 `proto::` 变为 `serve::proto::`。
**不带来行数变化**（1 126 行原样搬家），也不合并任何重复实现——与前面六条同性质：这是成员裁决，不是质量修复。

## 重议条件

每一条都写死了拆回去的条件（`docs/audit.md` §11.8）：检查点需要跨语言读写、
`report` 开始出现算子名、出现第二个回退判定消费者、第一个真后端落地、
出现第二种调度策略各自持有页账、`op` 需要再拆。**没有触发条件就不拆**——
吸收是产品裁决，不是不可逆的架构结论。

## 补记（第七次）：`proto → serve`

- 日期：2026-11-13；状态：已执行。
- 触发：`proto` 的生产入边只有 `serve` 一条（`serve → engine, proto, tokenizer`），内容是一份纯线格式模块
  （帧布局、版本规则、载荷上限、CRC 尾），没有第二份实现、没有独立版本节奏，且同层同域（L6 服务）。
- 落点：`crates/serve/src/proto/{mod,frame,decoder,crc}.rs`；`serve/src/lib.rs` 只 `pub mod proto;`，
  **不做根级 `pub use`**——否则 `serve` 的公开面会看起来"也定义了帧格式"，与所有权表冲突。
- 边变化：`serve → proto` 与 `proto → diagnostic` 消失，`proto → +dev: testkit` 平移到 `serve`；
  成员 27 → 26（crate 26 → 25），边 83 → 81，`+dev` 仍为 13。
- 代码变化：`crates/*/src` 53 325 → 53 329（+4：模块声明与两处 `use`）。**没有删任何一行逻辑，也没有消掉任何重复**——
  这是"包装合并不自动带来质量收益"的第二个样本（第一个是 `regress → tool`）。
- 重议条件：出现第二个传输消费者（另一个 crate 要读同一帧格式）时，帧格式必须回到 L5，
  否则第二个消费者只能向上依赖 `serve`。
