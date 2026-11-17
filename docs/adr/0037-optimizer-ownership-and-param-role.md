# 0037：优化器归 train、对象安全 trait、矩状态自持；ParamRole 闭集进 ContractSpec（草案）

- 状态：草案（结论以看板为准；实现随 T1 轨道填入，本 ADR 不规定分布式优化器与真实大语料装配）
- 日期：2026-11-17
- 相关：[0002](0002-single-owner-per-type.md)（类型唯一归属）、[0005](0005-annotations-as-contracts.md)（注解契约）；决策台账见 [`../implementation-task-plan.md`](../implementation-task-plan.md) §4 D2 / D11，设计取证见 [`../llmu-parity-optimization-design.md`](../llmu-parity-optimization-design.md) §1.5 P1–P4、§4.3 训练闭环

## 背景

T1 之前 `train` 只有 `Schedule` / `Checkpoint` / `Buffers` / `Pool` / `Store` / `shape` / `kernel`：没有优化器、没有矩状态、没有梯度更新；`autograd` 只出 tape / plan / account / replay / Store，不持梯度存储；`nn::ContractSpec` 只有 `name` / `output` / `ops`，没有参数角色字段。接反传闭环前需要裁决两件事：

1. 优化器（更新策略与矩状态）归哪一层、trait 是什么形状；
2. 参数属于哪个优化器组、是否参与权重衰减，元数据从哪里来。

候选分歧是：把优化器下沉 `autograd`，或按权重条目名字符串猜角色；前者会让"反传记账"与"更新策略"混在同一层，后者让分组规则散落在优化器侧的名字表里。

## 决策

### 1. 优化器归 `train`，不下沉 `autograd`

更新策略留在 `train`：`autograd` 的职责边界维持现状——只产出 tape、反传计划、需求分类与重放序列，不持有梯度存储，也不实现任何参数更新算术。优化器取梯度、裁梯度、写参数的动作发生在 `train::Session` 装配之后；一层不越界到另一层的决策口径沿用 ADR 0002。

### 2. `Optimizer` 是对象安全 trait，矩状态由实现按槽位编号自持

- trait 无泛型方法、无 `Self` 返回，可装成 `&mut dyn Optimizer`：
  `fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()>`。
- `Slot<'_>` 是一次步进的视图：`params: &mut [f32]`（就地写回）、`grads: &[f32]`（只读）、`role: ParamRole`。优化器不拥有参数存储，只持有自己的矩状态。
- 矩状态按槽位编号（`slots` 下标）由优化器实例自持：SGD 无矩状态；Adam 自持一阶矩、二阶矩与全局步数（偏差修正分母）。快照在 T1-4 从优化器导出/回载，槽位数与每槽长度不符是一条 `Code::Invalid`，不做半截恢复。
- 新增优化器只实现本 trait，不另起一套参数组映射；学习率走 `train::Schedule`，优化器只收调度后的当前有效 `lr`。

### 3. 参数角色是 `nn::ParamRole` 闭集，填进 `ContractSpec`

- 闭集四值：`Trainable`（投影/稠密权重，参与衰减）、`Bias`（偏置，豁免衰减）、`Norm`（归一化系数，豁免衰减）、`Embedding`（嵌入表，豁免衰减）。闭集之外的角色在编译期不存在；新增一类参数必须先扩这个闭集。
- 角色由层构造器写进 `ContractSpec::role`：`nn` 是构造期唯一知道自身角色的地方，层构造器填默认值，需要改角色的层走显式构造入口。
- 可训练判定与权重衰减豁免都从角色派生：`ParamRole::trainable()`、`ParamRole::wd_exempt()` 是唯一定义点；不在优化器侧再写第二份"名字 → 组"的表。

### 4. 优化器按角色取组，不猜字段名

- `Group::from_role(ParamRole)` 是角色 → 参数组的唯一映射，与 `ParamRole` 一一对应；`Group` 短名只作诊断与快照标签。
- 分组与衰减豁免不按字段名、层名前缀或条目名字符串猜测：错组、缺角色在接线期是负例断言；组归属对拍走角色映射表，不走名字比对。

### 5. 梯度缓冲归 `train::Buffers`，autograd 不持梯度存储

会话闭环里：tape → 反传可达节点计划 → `Requirement::Grad` 终点分类 → 梯度缓冲（`train::Buffers` 扩展的梯度 lane）→ 优化器步。autograd 侧不出现梯度数组；累积窗口的汇总梯度同样由 train 自持，与矩状态平行入快照。

### 6. CLI 只搬字符串，数值解析在组装点

`cli::Command::Train` 只持有 `Option<String>`（lr / warmup / steps / clip / accumulation / load），不解析数值、不构造会话；会话配置与数值解析在根组装点完成。cli 不依赖 train，命令文法与领域类型分层不回退。

## 备选方案

- **优化器下沉 `autograd`**：否决。autograd 的边界是"记账与重放"，更新策略带状态（矩、步数、裁剪口径），下沉会让两层职责互相渗透；反传结果也会接进本不该存在的更新通路。
- **为每类优化器按参数类型泛型特化**：否决。对象安全分发在前，泛型特化在第二后端/第二优化器族出现后再议；现在预埋多余泛型参数会让 trait 形状提前锁死。
- **角色写进优化器配置，或按权重条目名字符串猜角色**：否决。配置侧的角色表与层声明会漂移；名字前缀规则散在优化器里，重命名条目即静默错组。角色事实源必须在 `nn` 层声明处。
- **优化器直接改写参数槽私有字段**：否决。参数存储归会话/缓冲，优化器只拿 `Slot` 视图就地写回；存储归属与更新职责不混。

## 后果

### 正面

- 优化器可替换：`&mut dyn Optimizer` 之下 SGD 与 Adam 同接口，新增优化器不动会话与 autograd。
- 分组规则单源：角色 → 组只有 `Group::from_role` 一处映射；衰减豁免与可训练判定从角色派生，测试可按角色对拍。
- 状态边界清楚：矩状态归优化器、梯度缓冲归 Buffers、tape/plan/Replay 归 autograd，快照只搬运这几份表，不解释算术。

### 负面（代价）

- 每步步进要把参数/梯度/角色收成 `Slot` 视图表；视图生命周期绑定当步会话，跨步持有矩状态是优化器自己的事。
- 角色闭集扩容（未来新加一类参数）要同时动 `ParamRole`、`Group` 映射与层构造器，不能只在优化器侧加分支。

### 取舍

职责分层与分组单源优先于实现捷径；本 ADR 不规定真实模型装配、大语料通路与分布式并行（见下节）。

### 重议条件

- 出现需要跨参数组共享矩状态的优化器族（如第二矩族、记忆优化器）时，复核 `Slot` 视图与矩状态自持边界是否仍够用。
- 角色闭集出现第五类参数（如 MoE 门控、LoRA 适配器）时，先扩 `ParamRole` 与 `Group::from_role`，再动优化器分支。

## 不做

- 分布式优化器、ZeRO / 张量并行归约：暂缓（§3.8 清单），本 ADR 只定单机单会话形状。
- 参数 offload：只留接口形状，驻留/换入换出的真实通路随后续里程碑；tie 权重回填在会话留 `register_tie` / `backfill_ties` 接缝，源槽整块复制到绑定槽并有对拍，嵌入表与输出头共享的真实装配不在本轮。
- 不引外部优化器 crate，生产路径维持零外部依赖；不在 train 新增 `unsafe`。
- 不改 engine / serve / weights 装载通路与 cutile / op 算术。

## 修订

- 2026-11-17：初稿，落 D2（优化器归属、对象安全 trait、矩状态自持）与 D11（ParamRole 闭集进 ContractSpec）；SGD / Adam、梯度累积、快照矩状态与累积段随 T1-1…T1-5 填入。
