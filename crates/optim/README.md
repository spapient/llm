# `optim`

`cutile` DSL 的执行半边：接缝、证据与四份报告。

[`cutile`](../cutile) 是无 IR 的编译前端 —— 三份输入（声明层、档案层、片段层）经
① expand → ② negotiate → ③ select → ④ substitute → ⑤ emit 折成源文本与缓存键，全程是可见数据结构、
不碰进程。本 crate 从这里接着往下走：

```text
cutile::{emit, cache}  ──┐
                         ├→  编译（runner / build / device / backend）→ 跑一次（witness）
声明 + 样本 + 签名 ──────┘                                             ↓
                                   语言门（gate）→ 证据（Proof）→ 报告（doctor / verify / bench / ledger）
```

## 为什么这条线画在这里

划线判据是**证据从哪来**，不是代码量：前端只回答「这段源文本长什么样」，跨过 `witness` 之后就
必须有「这台机器上真的跑出过这个结果」。所以 `backend`、`runner`、`build`、`device`、`toolchain`
这些要 fork 子进程的模块，与 `gate`、`witness`、`verify`、`bench`、`doctor`、`ledger` 这些以证据为
前提的模块，必须同处一 crate —— 它们互为前后件，中间任何一刀都会让「没有证据的能力」重新变得
可以表达。

## 模块地图

| 职责 | 模块 |
| --- | --- |
| 统一设备：按 `Emit` 分发的描述符表与编排行 | `device` |
| 后端缝（`Codegen` / `Driver` / `Readback`）与注册表 | `backend` |
| 工具链探测与身份 | `toolchain` |
| 编译 / 执行 / 回读原语 | `runner` |
| ⑤ 指纹、锁、落盘与产物缓存 | `build` |
| 一次编译的上下文与产物（`Run` / `Prepared`） | `pipeline`（私有） |
| 见证执行与语言门 | `witness`、`gate`（私有） |
| 命令策略与结构化报告 | `doctor`、`verify`、`bench`、`execution` |
| 跨档案的后端能力台账 | `ledger` |

`pipeline` 与 `gate` 保持私有：它们是接缝而不是 API —— 外部只该看到「能跑 / 不能跑」的结论，
不该直接拼一次编译。`runner` 的 `driver` / `compile_flags` 按原路径转发（`engine` 取这套原语）。

## 依赖位置

L3，位于 `cutile` 右侧：`… → op → cutile → optim`。层表与边集由
`cargo test -p llm-testkit` 的门核对（`docs/ARCHITECTURE.md`），不靠文档自觉维护。

## 状态

`publish = false`，与工作区其余 crate 一致：工作区内 API 稳定，但不承诺 MSRV 与 semver。
