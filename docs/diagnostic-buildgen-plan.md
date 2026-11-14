# diagnostic 收敛与构建期内核抽离 — 执行计划

版本：v1.0
状态：已完成（P1–P4 全部落地，逐阶段提交，全 workspace 门禁通过）
适用仓库：`llm`（Rust workspace，根目录 `/Users/fei/code/llm`）
交付方式：分阶段推进，每阶段门禁不过不进下一阶段；每阶段独立 commit，回滚走 `git revert`，不产生 `.bak` 备份文件。

---

## 1. 目标与非目标

### 1.1 目标
- 消除 `op` / `prof` / `tool` 三个 `build.rs` 中重复的目录扫描、TOML 读取、`toml::Value → data::Value` 代码生成逻辑。
- 把 `diagnostic` 顶层 `lib.rs` 内的错误类型按职责拆分为 `code.rs` / `error.rs`，提升内聚度。
- 全过程保持默认 feature 纯 `std`、零运行时依赖、零出边；20 个运行时消费者的 `use diagnostic::...` 路径一行不改。
- 迁移前后生成的 `$OUT_DIR` 产物**字节级一致**（真机对拍，不使用 Mock）。

### 1.2 非目标（明确不做，守住 L1 边界）
- 不引入运行期 TOML 解析器（ADR 0022 红线）。
- 不引入日志 / tracing / `file!` / `line!`（仓库刻意零运行时打印，错误即值）。
- 不收敛人可读渲染（`tool::Report` 的所有权）、不收敛 `doctor` / `verify`（强领域耦合，属 cutile）。
- 不把 `data` 与错误类型拆成两个独立 crate（零依赖、同批消费者，拆只增边不降噪）。
- 不新增任何运行期领域语义。

---

## 2. 命名规范（Unix 共识 / 单单词 / 无自造缩写）

### 2.1 文件名
| 文件 | 职责 |
|---|---|
| `src/lib.rs` | 仅模块声明与统一 re-export，不含实现 |
| `src/code.rs` | `Code` 闭集、`as_str`、`exit_code`、`ALL` |
| `src/error.rs` | `Error`、`Result`、`from_io` |
| `src/data.rs` | `Table` / `Value`（现状不动） |
| `src/buildgen.rs` | 构建期共享内核，`#[cfg(feature = "build")]` |

### 2.2 feature 与依赖
- feature 名：`build`（默认关闭；`default = []`）。
- `Cargo.toml`：`toml = { workspace = true, optional = true }`，`[features] build = ["dep:toml"]`。

### 2.3 buildgen 公开函数（签名即契约）
| 函数 | 签名 | 职责 |
|---|---|---|
| `collect` | `(&Path) -> Vec<PathBuf>` | 列出目录下 `.toml` 文件，按 stem 升序 |
| `read` | `(&Path) -> toml::Table` | 读文件 + 解析 + 顶层必须是表，失败带路径 panic |
| `emit` | `(&toml::Table, &str) -> String` | 递归展开为 `Table::from_pairs(...)` 表达式文本 |
| `relative` | `(&Path, &Path) -> String` | 生成 `rerun-if-changed` 用的相对展示路径 |

命名铁律：单单词、领域词保持全称、`iter/len/buf/max` 等 Unix 共识缩写可用，禁止前后缀与自造缩写。

### 2.4 兼容性
`lib.rs` 用 `pub use code::Code; pub use error::{Error, Result};` 维持现有路径，下游零改动。

---

## 3. 目标结构

```
crates/diagnostic
├── Cargo.toml          # 新增 optional toml + feature build
├── src
│   ├── lib.rs          # re-export + 顶层文档
│   ├── code.rs         # 从 lib.rs 拆出
│   ├── error.rs        # 从 lib.rs 拆出
│   ├── data.rs         # 不变
│   └── buildgen.rs     # 新增，feature 门控
└── tests/
    └── buildgen.rs     # 新增，仅在 --features build 下编译
```

---

## 4. 分阶段执行（每阶段门禁不过不进）

### P1 — 错误类型拆分（零行为变化）
- 内容：`lib.rs` 中 `Code` 迁入 `code.rs`；`Error`/`Result`/`from_io` 迁入 `error.rs`；`lib.rs` re-export。
- 门禁：
  - `cargo build --workspace` 通过；
  - `cargo test -p llm-diagnostic` 全绿（现有测试随类型迁移）；
  - `cargo clippy --workspace --all-targets` 无新增警告；
  - 公开 API 路径不变（`tests/architecture.rs` 边界断言通过）。
- 提交：独立 commit。

### P2 — 引入 buildgen（先不接消费者）
- 内容：新增 optional `toml` 依赖与 `build` feature；实现 `collect/read/emit/relative`；新增单元测试。
- 门禁：
  - 默认 `cargo build -p llm-diagnostic` 仍零依赖（`cargo tree` 不含 toml）；
  - `cargo test -p llm-diagnostic --features build` 全绿；
  - `emit` 覆盖全部 6 类值、嵌套表、数组；非有限浮点与 datetime 必须 panic 且信息含路径。
- 提交：独立 commit。此阶段不影响任何现有构建。

### P3 — 迁移消费者（逐个对拍，先同构后异构）
顺序：`op` → `prof` → `tool`，每个独立 commit、独立门禁。

对拍流程（每个消费者执行一次）：
1. 迁移前在干净工作区构建，留存现生成产物：
   `cp target/debug/build/<crate>/out/<file>.rs /tmp/<crate>.before.rs`；
2. 改写该 crate `build.rs` 调用 `buildgen`，并在其 `[build-dependencies]` 启用
   `diagnostic = { path = "../diagnostic", features = ["build"] }`；
3. 强制重新生成后 `diff -u /tmp/<crate>.before.rs <new>.rs`，**必须无输出**；
4. `cargo test -p <crate>` 通过。

- op/prof（目录型）：保留各自 `OnceLock`/`names()`/`declared()` 外层包装，仅替换扫描与生成调用。
- tool（单文件型）：`read` + `emit`，外层 `vocabulary()` 包装保留。
- 门禁：三 crate 产物字节级一致，且全 workspace 测试通过。

### P4 — 清理与文档
- 删除三个 build.rs 中已被替代的私有函数；确认无残留 `toml::` 直接引用（仅 buildgen 一处）。
- 修订 `docs/ARCHITECTURE.md` 第 74–77 行残留的 `diagnostic::toml` 旧方案，标注 ADR 0022 现行形态。
- 更新 `CHANGELOG.md` 与本计划版本号 / 状态。
- 门禁：`cargo build --workspace --all-targets`、`cargo test --workspace`、`cargo clippy --workspace`、`cargo deny check` 全绿；`.jscpd` 重复率下降。

---

## 5. 测试与质量矩阵（测试专家视角）

| 维度 | 用例 | 期望 |
|---|---|---|
| emit-String | 字符串值 | `Value::text("..")` |
| emit-Integer/Float/Boolean | 标量 | 对应构造式 |
| emit-Float 非有限 | NaN/inf | panic，含路径 |
| emit-Datetime | 日期时间 | panic，提示改字符串 |
| emit-Array/Table | 嵌套与下标路径 | 递归正确，路径串完整 |
| collect | 排序/过滤/空目录 | 仅 `.toml`、按 stem 升序 |
| read | 缺失/非法/顶层非表 | 各自带路径 panic |
| 回归 | 20 crate 依赖方向 | `architecture.rs` 断言通过 |
| 对拍 | 三 crate 生成产物 | 迁移前后 `diff` 为空 |
| 静态 | clippy / deny / 重复率 | 无新增警告，重复下降 |

---

## 6. 风险与回滚

| 风险 | 缓解 |
|---|---|
| feature 泄漏到运行时 | 默认 feature 为空；`cargo tree` 验证默认不含 toml；build-dep 与 runtime dep 分离 |
| 生成产物细微差异 | 逐 crate 字节级 `diff`，不过不合并；保留 panic 文案与路径参数 |
| 下游路径被破坏 | `lib.rs` re-export 保持原路径，P1 即验证全 workspace 编译 |
| 误改用户文件 | 仅动 diagnostic 与三个 build.rs；每阶段独立 commit，回滚 `git revert`，不建备份 |

---

## 7. 验收清单（Definition of Done）
- [ ] `code.rs` / `error.rs` 拆分完成，下游零改动。
- [ ] `buildgen` 在 `build` feature 下提供 `collect/read/emit/relative`，默认零依赖。
- [ ] op/prof/tool 三个 build.rs 改为调用 buildgen，产物字节级一致。
- [ ] 全 workspace `build/test/clippy/deny` 全绿，重复率下降。
- [ ] ARCHITECTURE 残留旧方案修订，CHANGELOG 与本计划更新。
