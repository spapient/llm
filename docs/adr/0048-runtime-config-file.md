# 0048：运行时配置文件层——恢复用户级 TOML 配置,修订 D5

- 状态：已采纳（实现随本轮落在 `cli::settings`；回传主树后并入门禁基线）
- 日期：2026-11-19
- 相关：[0022](0022-build-time-corpora.md)（删除第四层配置与运行期 TOML 解析）、[0023](0023-settings-to-cli.md)（设置策略上移 `cli`）、ADR 0006（配置机制与身份分离）

## 背景

ADR 0022 删除"用户级配置文件"这一层时，当时的理由是：配置文件层连同一个 585 行的运行期
TOML 解析器一起被删，语料（`ops/`、`devices/`、`lints/`）全部改为构建期解析。ADR 0023 再把
剩余的三层策略（内置默认、`CUTILE_*`、`--set`）上移到 `cli::settings`，`cutile` 从此只收
`Dirs` 纯数据。

两条 ADR 当时都成立，但它们把两件事捆在了一起：**"语料不该在运行期解析"**（仍然成立）与
**"用户没有任何文件级配置入口"**（本 ADR 推翻）。现状的代价是：

1. 配置键闭集只有 3 个路径键（`cache.dir`/`work.dir`/`data.root`），**值类型只有 `PathBuf`**，
   数值与布尔策略（缓存上限、陈旧锁窗口、HTTP 上限、采样默认温度、policy 常数）全部散落在
   各 crate 的 `const` 里，部署方无法调节；
2. 环境变量承载数值配置的能力受限（全部要按路径文法走），且多键时命令行冗长；
3. 面向发布的二进制没有"一个文件描述一台机器"的部署形态。

## 决策

### 1. 恢复配置文件层,语料解析仍在构建期

配置四层优先级（低 → 高），复用既有 `Origin::rank` 机制：

```text
内置默认(0)  <  配置文件(1)  <  CUTILE_* 环境变量(2)  <  --set 命令行(3)
```

ADR 0022 的"语料构建期解析"不动：`ops/`、`devices/`、`lints/` 依旧由 build.rs 在构建期解析，
配置文件的 schema 只覆盖 `keys!` 宏登记的**运行时策略键**，与语料无关。

ADR 0023 的精神完整保留：`cutile` 仍只收纯数据（`Dirs` 与同构的值对象），不读环境变量、
不解析 TOML、不认识命令行；解析与来源追踪只发生在 L7 `cli::settings`。

### 2. 文件发现顺序（只读第一份命中，不做多文件合并）

1. 显式指定：`--config <path>` 或 `CUTILE_CONFIG`（两者同层，`--config` 优先）；
2. 进程当前目录 `./cutile.toml`（不向上搜索父目录，避免隐式拾取）；
3. 平台用户目录：macOS/Linux `~/.config/cutile/config.toml`、Windows `%APPDATA%\cutile\config.toml`。

文件缺失不是错误（三层回退仍可用）；文件存在但不可读、坏 TOML、未知键、类型不符是
`Code::Invalid` 错误，报错带文件路径与全部登记键清单（与 `--set` 的未知键报错同口径）。

### 3. schema 与值类型闭集

- 根为表；键 = `keys!` 宏里的点号名（单一事实源不变：新增键仍只加一行）。
- 值类型闭集 `enum Value { Path(PathBuf), Bytes(u64), Int(i64), Float(f64) }`（只保留键表实际
  使用的类型，新键需要布尔/文本时再加）；TOML 映射：String→Path、Integer→Bytes(非负)/Int、
  Float→Float；类型不符 = `invalid`。
- 键的类型在 `keys!` 宏行内声明；`describe` 逐键打印"值 ← 来源（配置文件路径）"。

### 4. 依赖路线:toml 成为 L7 唯一外部依赖例外（D5 修订）

**D5 修订为**："L1–L6 生产路径零外部依赖；L7 组合根（`cli`）允许唯一例外 `toml`（含传递
依赖 serde），版本由 `[workspace.dependencies]` 锁定，cargo deny 已过"。

- 被否备选：手写 ~350 行最小 TOML 子集解析器（维持 D5 字面）。代价是新长期维护面
  （转义、错误定位、负例测试），与本轮"减代码"目标相悖；且 `toml` 1.2.1 已在
  workspace.dependencies 并被 build 期使用，锁文件与 deny 结论都不新增。
- 架构门只核对 workspace 内成员边，`cli → toml` 是外部 crate 依赖，不进层表、不违反分层规则。

### 5. 首批键（默认值 = 现值,零行为变化）

| 键 | 类型 | 默认值 | 现值所在 |
|---|---|---|---|
| `cache.limit_bytes` | Bytes | 2 GiB | `cutile::cache::CACHE_LIMIT` |

P1 曾一次注册 9 个数值键（缓存时间窗、HTTP 上限、采样温度、policy 常数）；P5 接线时
实测这些键**没有真实消费路径**——接线需要给 `Cache`/`Httpd` 构造或纯函数签名加参数，
属 API 扩张，且 serve 的 HTTP/采样装配点尚不存在。按"无消费者不预埋"（ADR 0004 同一
判据）裁决：8 个键从键表删除，键表回到"3 路径 + 1 数值"最小集。新增键仍是一行：
真实消费者出现时按本 ADR 的 schema 直接注册即可。

格式上限（`wire::leb`、`weights::writer` 的 MAX_*）不进配置——它们是线格式契约而非部署策略。
消费端模式沿用 `Dirs`：`cli` 把 Settings 投影成纯数据值对象传给内核/装配点，内核不改一行。
