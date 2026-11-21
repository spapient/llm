//! 通道级 delta 遗忘（KDA）：第 10 款标杆（Kimi K3）的混合 delta 轴。
//!
//! 一个 K3 块按 3:1 排布 KDA 与 Gated MLA；本模块只落在「一层 KDA 怎么用一次前向更新它的记忆」
//! 这一格，块配方与门控 MLA 的接线在装配侧。
//!
//! # 与指数梯形不是一回事
//!
//! 规则 0.11 把「KDA 冒充 Mamba-3 指数梯形」列为禁止项。两者的差别不在有没有门，而在**写的规则**：
//!
//! - 指数梯形是**累加**：状态按门衰减之后把 `x` 加进去。同一个键写两次，读出来是两次的加权和。
//! - delta 规则是**纠错**：写之前先取记忆已经预测出的值 `S k̂`，只把残差 `v − S k̂` 写进去。同一个键
//!   写两次、值不同，读出来是**后一次**的值——记忆被覆盖，而不是被累加。
//!
//! 这条差别有单测钉着（`the_delta_rule_overwrites_instead_of_accumulating`）：它是本模块存在的理由，
//! 不是实现细节。
//!
//! # 通道级遗忘
//!
//! 门 `α ∈ [0, 1]^{d_k}` 是**逐通道**的，不是逐 token 的标量：记忆 `S ∈ R^{d_v × d_k}` 的第 `j` 列乘
//! `α[j]`。哪些通道该忘、哪些该留由这一层自己学，而不是整块记忆一起衰减——这也是 KDA 与 Mamba-3 的
//! SISO 对角转移在结构上的差别。`α = 0` 是「这一列清空」的合法极端；`α > 1` 被拒，它让记忆每步放大。
//!
//! # 三个入口
//!
//! [`Kda::memory`] 给零记忆（几何由层给出，状态不可能与层错配），[`Kda::step`] 是一步更新，
//! [`Kda::read`] 是 `o = S q`。读用的是**本 token 写完之后**的记忆，于是单个 token 也能读完自己写下的
//! 东西；跨 token 的因果由调用次序保证，不由本模块合成。
//!
//! 键在入口按 L2 归一：delta 规则的精确写入要求 `‖k‖ = 1`，把归一化留在调用方就是让「写进去的强度」
//! 悄悄随键的范数变化。零键被拒而不是归一成 `NaN`。纯算术统一走 [`Scalar`] 泛型，点积与加权累加复用
//! [`crate::csa`] 的那两份。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, dot, finite};

/// 报错出处：几何、门、写强度与键都在入口失败，不留到更新中途。
const SITE: &str = "nn::kda";

/// KDA 的递归记忆：行主序 `[值维, 键维]`，列是门的施加方向。
///
/// 没有公开构造函数：状态只能由 [`Kda::memory`] 给出，于是「状态的维度」与「层的维度」不可能不一致，
/// 快照回载也不会有第二个几何来源。
#[derive(Debug, Clone, PartialEq)]
pub struct Memory<S: Scalar = f32> {
    key_dim: usize,
    value_dim: usize,
    cells: Vec<S>,
}

impl<S: Scalar> Memory<S> {
    /// 键维 `d_k`：门的宽度、列的下标。
    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    /// 值维 `d_v`：每次读出的宽度、行的下标。
    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    /// 扁平记忆：`cells[value * key_dim + key]`。
    pub fn cells(&self) -> &[S] {
        &self.cells
    }

    /// 第 `value` 行：这个值维在各键通道上记住了多少；越界返回 `None`。
    pub fn row(&self, value: usize) -> Option<&[S]> {
        if value >= self.value_dim {
            return None;
        }
        Some(&self.cells[value * self.key_dim..(value + 1) * self.key_dim])
    }
}

/// 一层 KDA 的几何：键维 `d_k` 与值维 `d_v`。
///
/// 不持参数：门、写强度、键与值都来自上层（真实实现里各是一次线性投影），本层只负责「一次更新怎么算」。
/// 方法对元素类型泛型，于是同一份几何既跑 `f32` 也跑 `f64`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kda {
    key_dim: usize,
    value_dim: usize,
}

impl Kda {
    /// 由键维 `d_k`（门的宽度）与值维 `d_v`（读出宽度）构造。
    ///
    /// # Errors
    ///
    /// 任一维为零，或 `d_v × d_k` 溢出——返回 `Code::Invalid`。
    pub fn new(key_dim: usize, value_dim: usize) -> Result<Self> {
        if key_dim == 0 || value_dim == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("键维与值维都必须非零，得到 d_k = {key_dim}、d_v = {value_dim}"),
            )
            .with(SITE));
        }
        if key_dim.checked_mul(value_dim).is_none() {
            return Err(Error::new(
                Code::Invalid,
                format!("记忆尺寸溢出（{value_dim} × {key_dim}）"),
            )
            .with(SITE));
        }
        Ok(Self { key_dim, value_dim })
    }

    /// 键维 `d_k`。
    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    /// 值维 `d_v`。
    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    /// 零记忆：一个 token 都没写过。
    pub fn memory<S: Scalar>(&self) -> Memory<S> {
        Memory {
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            cells: vec![S::ZERO; self.key_dim * self.value_dim],
        }
    }

    /// 一步更新：先按门逐通道遗忘，再把残差 `v − S k̂` 按写强度与 `k̂` 写回。
    ///
    /// # Errors
    ///
    /// 记忆几何与本层不符、门/键/值宽度不符、门不在 `[0, 1]`、写强度不在 `[0, 1]`、键的 L2 范数为零、
    /// 或任一输入非有限——都返回 `Code::Invalid`，且**在写入之前**判定，不会留下半截更新。
    pub fn step<S: Scalar>(
        &self,
        memory: &mut Memory<S>,
        gate: &[S],
        write: S,
        key: &[S],
        value: &[S],
    ) -> Result<()> {
        self.check(memory)?;
        self.check_values("值", value, self.value_dim)?;
        let (write, key) = self.gated_write(gate, write, key)?;
        for (value_index, row) in memory.cells.chunks_exact_mut(self.key_dim).enumerate() {
            write_row(row, gate, write, &key, value[value_index]);
        }
        Ok(())
    }

    /// 门、写强度与单位键：三处都是「写入之前的形状与取值契约」，合成一次调用。
    ///
    /// 门在写强度之前判：门是逐 token 由投影给出的，写强度是标量；先查门才能让报错定位到通道。
    fn gated_write<S: Scalar>(&self, gate: &[S], write: S, key: &[S]) -> Result<(S, Vec<S>)> {
        self.check_gates(gate)?;
        let write = self.write_strength(write)?;
        Ok((write, self.unit_key(key)?))
    }

    /// 读：`o = S q`，即 `o[i] = Σ_j S[i][j] q[j]`。
    ///
    /// 用的是**本 token 写完之后**的记忆：单个 token 写进去、立刻读得出来。
    ///
    /// # Errors
    ///
    /// 记忆几何与本层不符、查询宽度不符或查询含非有限值——返回 `Code::Invalid`。
    pub fn read<S: Scalar>(&self, memory: &Memory<S>, query: &[S]) -> Result<Vec<S>> {
        self.check(memory)?;
        self.check_values("查询", query, self.key_dim)?;
        Ok(memory
            .cells
            .chunks_exact(self.key_dim)
            .map(|row| dot(row, query))
            .collect())
    }

    /// 状态几何必须与本层一致：错配的记忆是别的层的，读出来的数没有意义。
    fn check<S: Scalar>(&self, memory: &Memory<S>) -> Result<()> {
        if memory.key_dim == self.key_dim && memory.value_dim == self.value_dim {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!(
                "记忆几何必须是 [值维 {}, 键维 {}]，得到 [值维 {}, 键维 {}]",
                self.value_dim, self.key_dim, memory.value_dim, memory.key_dim
            ),
        )
        .with(SITE))
    }

    /// 门：长度、有限性与取值域 `[0, 1]` 一起在这个入口判定。
    fn check_gates<S: Scalar>(&self, gate: &[S]) -> Result<()> {
        self.check_len("门", gate.len(), self.key_dim)?;
        for (channel, &alpha) in gate.iter().enumerate() {
            check_channel(channel, alpha)?;
        }
        Ok(())
    }

    /// 写强度 `β ∈ [0, 1]`：1 是「整份残差都写进去」，0 是「只遗忘不写」。
    fn write_strength<S: Scalar>(&self, write: S) -> Result<S> {
        if finite(write) && write >= S::ZERO && write <= S::ONE {
            return Ok(write);
        }
        Err(Error::new(
            Code::Invalid,
            format!("写强度是 {write:?}：必须在 [0, 1] 内（0 只遗忘不写，1 写满残差）"),
        )
        .with(SITE))
    }

    /// 单位键：delta 规则的精确写入要求 `‖k‖ = 1`，归一化因此在入口做一次，零键拒绝。
    fn unit_key<S: Scalar>(&self, key: &[S]) -> Result<Vec<S>> {
        self.check_values("键", key, self.key_dim)?;
        let norm = key
            .iter()
            .copied()
            .fold(S::ZERO, |acc, value| acc.add(value.mul(value)))
            .sqrt();
        if norm <= S::ZERO {
            return Err(Error::new(
                Code::Invalid,
                "键的 L2 范数为零：残差 `v − S k` 会退化成 `v`，写进去的是值而不是纠错",
            )
            .with(SITE));
        }
        Ok(key.iter().map(|&value| value.div(norm)).collect())
    }

    /// 宽度 + 有限性。
    fn check_values<S: Scalar>(&self, what: &str, values: &[S], want: usize) -> Result<()> {
        self.check_len(what, values.len(), want)?;
        match values.iter().position(|&value| !finite(value)) {
            Some(index) => Err(Error::new(
                Code::Invalid,
                format!("{what}第 {index} 项非有限：放行会把 NaN 一路带进记忆"),
            )
            .with(SITE)),
            None => Ok(()),
        }
    }

    /// 宽度校验。
    fn check_len(&self, what: &str, got: usize, want: usize) -> Result<()> {
        if got == want {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!("{what}的宽度必须是 {want}，得到 {got}"),
        )
        .with(SITE))
    }
}

/// 一行的更新：先逐通道遗忘，再把残差 `v − S k̂` 按写强度写回。
///
/// 两步的顺序不能换：先写后忘会把刚写进去的纠错一起衰减掉，门就变成了写强度的双计。
fn write_row<S: Scalar>(row: &mut [S], gate: &[S], write: S, key: &[S], value: S) {
    for (cell, &alpha) in row.iter_mut().zip(gate) {
        *cell = cell.mul(alpha);
    }
    let predicted = dot(row, key);
    accumulate(row, key, write.mul(value.sub(predicted)));
}

/// 单个通道的门：有限且落在 `[0, 1]`（`0` 清空该通道，`1` 一点不忘）。
fn check_channel<S: Scalar>(channel: usize, alpha: S) -> Result<()> {
    if !finite(alpha) {
        return Err(Error::new(
            Code::Invalid,
            format!("门第 {channel} 个通道非有限：遗忘系数必须是 [0, 1] 里的实数"),
        )
        .with(SITE));
    }
    if alpha < S::ZERO || alpha > S::ONE {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "门第 {channel} 个通道是 {alpha:?}：遗忘系数必须在 [0, 1] 内\
                 （0 清空该通道，1 一点不忘）"
            ),
        )
        .with(SITE));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/kda_tests.rs"]
mod tests;
