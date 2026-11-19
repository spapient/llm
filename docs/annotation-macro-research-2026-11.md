# annotation 路线联网调研笔记(2026-11-19)

调研目的:验证 annotation crate 的"零依赖手写 token 解析"路线与"注解不持运行期注册表"契约
在 2026 年生态中的位置,为剩余工作(P5-6/5-7)提供决策依据。结论只入档,不推翻既有 ADR。

## 1. 手写解析 vs syn/quote

- 社区共识:通用属性宏默认 syn+quote(按需 feature)保证可维护性与错误 span 质量;
  **仅当宏输入是自定义 DSL/极窄语法时才考虑手写**(编译时间敏感是另一个理由)。
- 本仓 annotation 的输入正是极窄 DSL(属性行 `name = "..", shape = [..], dtype = "..",
  skeleton = ".."` 四种语法片),与手写路线的适用条件吻合;零供应链依赖是 ADR 0005 的
  冻结决策。**结论:维持手写,不引 syn/quote/proc-macro2。**
- 手写路线的弱点(span 质量、负例覆盖)由既有机制缓解:`compile_error!` 指到出错 token、
  `tests/audit.rs` 负例集、`contracts.rs` 的词汇表等式门禁。
- 数据点:Linux kernel pin-init 2026-01 改为引入 syn——它解析的是通用 Rust 语法,与本仓
  的极窄 DSL 不是同一场景。

## 2. 编译期元编程 vs 运行期注册表

- 共识:"known-at-build → derive/编译期;动态插件 → 运行期注册表(inventory/traitreg)"。
- **proc-macro 跨调用累积静态状态是 unsound 且不被 rustc 支持**——这从语言机制层面支持
  ADR 0005"注解不建注册表"(注册表会成为第二个事实源)的裁决。
- 本仓 REGISTRY 唯一事实源 + 注解↔注册表镜像门禁(contracts.rs)与"编译期闭合"共识一致:
  错误在构建期浮现、无隐藏全局状态、生成代码可经 cargo expand 阅读。

## 3. 对剩余工作的指导

- P5-6(#[spec] 动态名字):扩展极窄 DSL 的语法片(名字参数允许运行时表达式),在 syntax.rs
  的 closed 解析与 codegen.rs 的查表框架内做增量,不改变手写路线。
- P5-7(#[backward] 注解):目录表注解化与"编译期闭合"共识一致;实现上注意 backward 的
  公式函数本体仍需手写,注解只省目录行(收益 ~20 行),性价比低——**建议暂缓**,等
  FORMULAS 目录与 REGISTRY 三列互证的维护成本实测上升后再做。
