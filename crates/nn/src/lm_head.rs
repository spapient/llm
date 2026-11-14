//! `lm_head`：由注解声明的固定形状层。
//!
//! 它曾住在 `src/main.rs`（L7 组装点），而 `docs/ARCHITECTURE.md` 的 L7 明写"不得定义新类型"：
//! 类型属于产出它的层。形状、输入宽度与算子只写在注解里，`impl Contract` 与校验测试由
//! [`crate::spec`] 生成。

/// 固定形状的输出头：`[4096]` 输出、输入宽度 4096，权重编码由 `gemv_i4` 承担。
///
/// 与 [`crate::Linear`] 的分工：宽度在编译期就定下来的层用注解（声明即契约），宽度要按模型
/// 配置计算出来的层用构造函数（`Linear::new` 里做校验）。两者走同一条 `ContractSpec` 校验路径，
/// 也走同一条下界路径：`in` 给了输入宽度，生成的 `bounds` 就把 `(in, out)` 交给算子语义表。
#[crate::spec(name = "lm_head", out = [4096], in = 4096, ops = ["gemv_i4"])]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LmHead;
