# 工具链迁移：llm.u `tool` → 本项目 `tool`

> 版本 v3 · 2026-11-12 · 进度：**P1 已完成并按 owner 决定并入 `tool`**（零依赖词法器、CCN 分析、
> 对拍零漂移、阈值配置化），P2–P6 待启动。
>
> **一句话结论**：llm.u 的 `tool`（5765 行单体 CLI，9 子命令，依赖 syn/regex/serde_json 等 8+
> 外部 crate）不整体复制。其能力按本项目分层拆解，**全部并入既有 `tool` crate**（owner 明确决定，
> 不再另建工具 crate）；静态分析以零依赖自研词法器重写，阈值抽到随包 `gate.toml`，由 `tests/`
> 薄门禁接入 `cargo test`（不可绕过）；需独立产出文件的命令在 `tool` 内加 `[[bin]]`。

## 0. 落点决策（owner 拍板）

两个项目的 `tool` 同名异物：

| 事实 | llm.u `tool` | 本项目 `tool`（迁移前） |
|---|---|---|
| 形态 | 单体二进制（`[[bin]]`），19 文件 5765 行 | L1 库，1 文件 115 行 |
| 拥有 | 9 个子命令的全部分析逻辑 | `Report`、`join_names` |
| 依赖 | syn、proc-macro2、regex、serde_json、rand、toml 等 8+ 外部 crate | 零依赖，只接字符串 |

曾考虑新建独立工具 crate，owner 否决：**能力统一进 `tool`，不另起名字**。`tool` 因此从“零依赖
呈现叶”扩展为 L1 工具 crate，依赖同层更靠左的 `config`（复用 workspace 唯一 TOML 解析器）与
`diagnostic`。架构测试已验证这两条同层边方向合规。

## 1. 硬约束（一切设计的前提）

| 约束 | 取值 | 后果 |
|---|---|---|
| 第三方依赖 | 全仓 0 个（含 dev） | 不能用 syn/regex；词法与模式匹配全部自研 |
| unsafe | 全局 `deny` | 分析器纯安全 Rust |
| 错误类型 | 唯一 `diagnostic::{Error,Code}`，禁止 `Result<_,String>` | 旧 String 错误全部重写 |
| 配置化 | 阈值集中在 `gate.toml`，编译期 `include_str!` 嵌入 | 调标准只改数据，分析器无魔数 |
| 命名 | 单单词、无前后缀、禁自造缩写；领域词保持全称 | 见 §4 |

## 2. 能力映射（9 子命令 → 落点）

| llm.u 子命令 | 能力 | 目标落点 | 决策 |
|---|---|---|---|
| `gate ccn` | 圈复杂度/函数长度（lizard 1.25） | `tool` 库 + `tests/` 薄门 | **已完成**，对拍零漂移 |
| `gate arch` | crate DAG/所有权 | 已有 `tests/architecture.rs` | 不迁，cargo metadata 对拍更严 |
| `gate docs` | 文档漂移/锚点/README 树 | `tool` 库 + `tests/` | P2 重写 |
| `gate files` | 行数棘轮/重复块/匿名模块 | `tool` 库 + `tests/` | P2 重写 |
| `gate call` | 命名/禁调用形态/panic 预算 | `tool` 库 + vocabulary 闭集 | P2 重写合并 |
| `measure` | 体积/逐张量分布/吞吐 | `tool` `[[bin]]` | P3，消费 weights/storage |
| `pack` | 配置驱动分发打包 | `tool` `[[bin]]` | P3 |
| `init` | Mamba-3 初始化 | `tool` `[[bin]]` | P4，RNG 分布对拍 |
| `grow` | 1.5B→4B 扩容 | `tool` `[[bin]]` | P4 |
| `delta` | 差量压缩位级验证 | `tool` | P5，双射还原 |
| `fp64` | 53-bit 尾数拆分数学原型 | `tool` 单元测试 | P6，转测试不做命令 |

## 3. 最优论证与诚实边界

| 维度 | llm.u | 本项目 `tool` | 验证 |
|---|---|---|---|
| 外部依赖数 | 8+ 外部 crate | 0（自研词法器，复用内部 config） | cargo metadata 断言无外部源 |
| 不可绕过性 | 外挂 CLI 可跳过 | cargo test 薄门 + 编译期闭集 | 删测试即架构门失败 |
| 假绿防护 | 无扫描下限 | 源文件数下限 + 空结果即失败 | 损坏遍历必红 |
| 配置化 | 阈值散落/部分常量表 | 单一 `gate.toml`，带校验 | 写错配置即失败 |
| 错误契约 | `Result<_,String>` | diagnostic 强类型 | 架构测试禁 String 错误 |
| CCN 口径 | 固定语义 | 同口径，**只承诺不劣于** | 全文件逐函数对拍零漂移 |

**诚实边界**：依赖数、不可绕过性、假绿防护、配置化、错误契约可证明领先；CCN 口径、文件 IO、
打包吞吐有物理下界，只承诺对拍不劣于，触底维度不写“指数级”。

## 4. 目标架构（参考 burn：Config 与行为分离，模块小而专注）

```
crates/tool/
├── Cargo.toml
├── gate.toml          # 质量标准（数据，单一真相）
├── src/
│   ├── lib.rs         # 入口：模块声明 + Report/join_names
│   ├── gate.rs        # Gate 配置加载与校验（burn Config 思想）
│   ├── lex.rs         # 零依赖词法器
│   ├── ccn.rs         # 圈复杂度与函数长度
│   ├── strip.rs       # cfg(test) 条目剔除
│   ├── docs.rs        # 文档漂移（P2）
│   ├── dup.rs         # 重复块与行数（P2）
│   ├── call.rs        # 调用形态与命名（P2）
│   └── bin/tool.rs    # 独立产出命令入口（P3 起）
└── tests/
    └── ccn.rs         # 跨实现对拍
```

复用优先：TOML 解析复用 `config`，错误复用 `diagnostic`，不重写基础设施；命名用单单词 + Unix
共识缩写（len/iter/buf/max/arg），阈值只在 `gate.toml`。

## 5. 阶段路线（门不过不进下一阶段）

| 阶段 | 内容 | 准出门禁 |
|---|---|---|
| P1 | 词法器、ccn、对拍、配置化、并入 tool | 对拍零漂移；test/clippy/doc 全绿（**已达成**） |
| P2 | docs 文档漂移、dup 重复块/行数、call 调用形态 | 注入用例红/绿可验证；与 vocabulary 合并 |
| P3 | measure、pack `[[bin]]` | 真机实测；清单与配置双向对拍 |
| P4 | init、grow | 分布对拍；产物可被 train 加载 |
| P5 | delta 差量 | 位级还原双射 |
| P6 | fp64 转测试、全量门整合、文档更新 | 全仓门禁一次通过，文档无漂移 |

## 6. 同步改动清单（每阶段结束核对）

- ARCHITECTURE.md：tool 的边、所有权表、L1 职责描述保持与实际一致；架构测试自动校验。
- 不新增任何外部依赖；cargo metadata 成员集合与文档一一对应。
