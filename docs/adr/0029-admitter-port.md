# 0029：准入三态翻译收敛为 `Admitter` 端口

- 状态：已采纳（P1 落地）
- 日期：2026-11-17
- 相关：[0001](../ARCHITECTURE.md)（依赖方向）、[0025](0025-runtime-backend-abstraction.md)（执行/调度归属）；设计取证见 [`../serve-deep-optimization-design.md`](../serve-deep-optimization-design.md) D1

## 背景

`crates/serve` 原有两个准入入口：`admit`（`engine::Scheduler`，按引用）与 `admit_batch`
（`engine::Batch`，按值消费序列）。两者是同一套三态判定的两份拷贝：

- 最坏长度装不下（`fits` 为假）→ `400`；
- 现在页满（准入返回满）→ `503`；
- 序列自身不成立（`invalid`）→ `400`，其余错误类别上抛为服务端故障。

两份拷贝漂移的风险是：将来改 400/503 判定时只改一处，另一条路径行为悄悄偏离。

## 决策

在 serve 内定义私有准入端口，三态到 [`Status`](../../crates/serve/src/lib.rs) 的翻译只保留一份：

- `AdmissionOutcome`：私有结局枚举 `Admitted(u64)` / `Full`；
- `trait Admitter`：`fits(&Sequence) -> bool` 与 `admit(Sequence) -> Result<AdmissionOutcome>`；
- 为 `engine::Scheduler` 与 `engine::Batch` 各写一份薄实现，用全限定固有方法调用避开 trait
  同名方法遮蔽；Scheduler 路径在适配层按值克隆一次序列以抹平按引用/按值差异；
- `admit_response<A: Admitter>` 是唯一的响应翻译函数：`fits` 假 → `BadRequest`（`NEVER_FITS`）、
  `Full` → `Unavailable`（`FULL`）、`Err(invalid)` → `BadRequest`（沿用既有 `refused`），
  其余 `Err` 原样上抛。

公开 `admit` / `admit_batch` 签名保持不变，作为薄委托留在端口之上；`Admitter` 与
`AdmissionOutcome` 均为私有类型，不进所有权表，不新增运行时出边。

## 备选方案

- **不动双路径，只抽一个私有 helper 收文案**：拒绝。判定仍散在两处，漂移风险没有解决。
- **把端口上提到 `engine`**：拒绝。三态到 HTTP 状态码的翻译是服务层语义，engine 不该认识
  `Status`，方向反了（ADR 0001）。

## 后果

### 正面

- 三态判定有单一事实源；未来接入限流、队列等新准入实现，只需实现两个端口方法。
- 公开契约不变：根二进制与既有测试仍调 `admit` / `admit_batch`。

### 负面

- 多一个私有端口类型；`Sequence` 按值/按引用的差异在 Scheduler 适配层用一次克隆抹平。

### 取舍

正确性与可复用性优先于少一个 trait；该 trait 只覆盖准入，不扩展成通用后端接口。

## 不做

- 不在本轮删除 Scheduler 旧路径：根二进制演示与既有测试仍在调用，是否切到 `Batch` 随传输层
  落地时再裁决（设计文档附录 B 待决问题 1）。
- 不把端口公开为 serve 成员。

## 修订

- 2026-11-17：初稿并落地；`Admitter`/`AdmissionOutcome` 私有，公开入口薄委托。
