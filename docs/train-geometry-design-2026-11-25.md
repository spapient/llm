# `crates/train` 几何调用链实现设计与验证计划

日期：2026-11-25　范围：`src/geometry_shapes.rs` 与 `src/geometry_kernels.rs`，不修改 `geometry.rs` 的公开枚举。

## 1. 不变核心与可变差异

- 不变核心：`Geometry` 的公开形状、输出公式、窗口边界、归一化与卷积数值顺序、错误类别和诊断站点。
- 可变差异：几何族路由、窗口索引、散射坐标、归一化输出切片。差异继续由枚举字段、表项和闭包注入，
  不新增配置镜像、业务常量或第二套形状体系。

## 2. 正交分层

- 数据定义层：沿用 `Geometry` / `Window` / `Running` 与 `Sampling`，不在内核中读取环境或全局配置。
- 核心内核层：`shapes` 只负责窗口、索引和形状判据；`kernels` 只负责前向/反向数值、散射和归一化。
- 执行入口层：`Geometry::{forward,backward,check_shape}` 保持原有入口，不改变公开调用方式。

依赖方向不变：`geometry → shapes → diagnostic/nn` 与 `geometry → kernels → shapes/nn`，不引入新 crate。

## 3. 复用路径

### `shapes.rs`

1. 用声明式路由表生成几何族分派，保留小路由，避免合并成 CCN 超标的大 match。
2. 非零平面元素积复用一个检查函数，池化与平面检查共享 `ensure_shape`。
3. 一维/二维 argmax 共用“严格更大才替换”的迭代骨架，平手仍取先遇到的位置。
4. 归一化输出使用 `NormOut` 切片视图，移除 `too_many_arguments` 豁免，数值循环与顺序不变。

### `kernels.rs`

1. `backward_grid` 统一输出网格、样本遍历与 `dx` 分配；1D 用单元素第二轴，2D 使用真实输出平面。
2. `scatter_area` 统一平均池化与自适应池化的有效矩形及分母；最近邻和双线性保留各自权重算法。
3. 小型路由表统一池化/空间反向分派，避免重复 match 壳。
4. `NormOut` 同时修改声明与唯一调用点，禁止保留参数豁免。

## 4. 扩展方式

- 新增几何族：在对应路由表增加一行，复用 `ensure_shape` / `backward_grid` / `scatter_area`。
- 新增散射模式：只增加坐标/权重闭包，不复制样本遍历、输出分配或索引公式。
- 新增归一化变体：复用 `NormOut` 和 `norm_group`，不得新增参数豁免或第二份反向骨架。

## 5. 量化目标与证据

| 项目 | 基线 | 验收 |
|---|---|---|
| 文件行数 | shapes 346、kernels 595 | 两个文件均不超过 300，且不新增生产文件 |
| 函数 / 嵌套 | 仓库静态门零命中 | 函数不超过 30 行，CCN 不超过 5，控制流嵌套不超过 3 层 |
| 参数豁免 | `norm_group` 有 1 处 | 声明与调用点全部移除 |
| 重复率 | 0.46% 行、0.59% tokens | 两项均低于 0.5%，不得新增克隆 |
| 数值证据 | 246 训练用例、中心差分与几何前向/反向 | 不减少用例与断言强度，全部通过 |
| 资源证据 | 几何测试与既有缓冲断言 | 不因抽象引入隐藏分配或第二份输出缓冲 |

不宣称未经实测的加速。只有在同一 release 负载、迭代次数和基线对比成立时，才报告性能数字。

## 6. 验证计划

1. `cargo test -p llm-train`：前向、反向、窗口越界、空值、零值、非法形状、归一化和恢复路径。
2. `cargo test -p llm-train --doc` 与全目标 Clippy `-D warnings`。
3. 静态门与逐函数扫描：CCN、函数长度、控制流嵌套、文件行数。
4. `npx --no-install jscpd crates/train/src --min-lines 5 --min-tokens 50 --reporters console`：
   重复行与 tokens 均低于 0.5%。
5. `cargo fmt --all --check`、`git diff --check`。
6. 最终统一执行 `scripts/ci.sh`；任何工具缺失、其它 crate 失败或密度预算失败均逐项披露并标为未完成。

## 7. 当前阻塞修复与最终收口设计

- 梯度范数的不变核心是 `f64` 平方和、有限性预检和全局统一裁剪；`DecayClip::scale` 继续作为唯一实现，
  不让 SGD、Adam、Muon、Hybrid 分别计算范数。此次只修复范数参数重构后的调用契约，不改变更新顺序。
- 几何路由以 `Geometry` 为唯一配置源，前向和反向分别共用路由声明；空间反向统一使用输出平面、
  输入平面和散射闭包。新增同类算子只增加路由项及数值微核，不复制网格遍历。
- 归一化继续共用 `normalized_backward` 和 `norm_group`，卷积继续调用 `nn::Conv2d`，不引入平行数值实现。
- 验证顺序为训练测试编译、裁剪与恢复定向测试、几何与归一化回归、训练全量和文档测试、Clippy、jscpd、
  统一 CI 与 `loc-budget`。任何失败都保留为未完成，不用局部通过替代全仓 60,000 行硬目标。
