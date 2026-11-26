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

use crate::invalid;
use diagnostic::Result;
use nn::conv::scan::Scalar;

use crate::csa::{accumulate, dot, finite};

/// 报错出处：几何、门、写强度与键都在入口失败，不留到更新中途。
const SITE: &str = "algorithm::kda";

/// KDA 的递归记忆：行主序 `[值维, 键维]`，列是门的施加方向。
#[derive(Debug, Clone, PartialEq)]
pub struct Memory<S: Scalar = f32> {
    key_dim: usize,
    value_dim: usize,
    cells: Vec<S>,
}

impl<S: Scalar> Memory<S> {
    accessors! {
        /// 键维 `d_k`：门的宽度、列的下标。
        key_dim -> usize = key_dim,
        /// 值维 `d_v`：每次读出的宽度、行的下标。
        value_dim -> usize = value_dim,
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
            return err!("键维与值维都必须非零，得到 d_k = {key_dim}、d_v = {value_dim}");
        }
        if key_dim.checked_mul(value_dim).is_none() {
            return err!("记忆尺寸溢出（{value_dim} × {key_dim}）");
        }
        Ok(Self { key_dim, value_dim })
    }

    accessors! {
        /// 键维 `d_k`。
        key_dim -> usize = key_dim,
        /// 值维 `d_v`。
        value_dim -> usize = value_dim,
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
        self.check_memory(memory)?;
        self.check_values("值", value, self.value_dim)?;
        let (write, key) = self.gated_write(gate, write, key)?;
        for (value_idx, row) in memory.cells.chunks_exact_mut(self.key_dim).enumerate() {
            write_row(row, gate, write, &key, value[value_idx]);
        }
        Ok(())
    }

    /// 门、写强度与单位键：三处都是「写入之前的形状与取值契约」，合成一次调用。
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
        self.check_memory(memory)?;
        self.check_values("查询", query, self.key_dim)?;
        Ok(memory
            .cells
            .chunks_exact(self.key_dim)
            .map(|row| dot(row, query))
            .collect())
    }

    /// 状态几何必须与本层一致：错配的记忆是别的层的，读出来的数没有意义。
    fn check_memory<S: Scalar>(&self, memory: &Memory<S>) -> Result<()> {
        if memory.key_dim == self.key_dim && memory.value_dim == self.value_dim {
            return Ok(());
        }
        err!(
            "记忆几何必须是 [值维 {}, 键维 {}]，得到 [值维 {}, 键维 {}]",
            self.value_dim,
            self.key_dim,
            memory.value_dim,
            memory.key_dim
        )
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
        err!("写强度是 {write:?}：必须在 [0, 1] 内（0 只遗忘不写，1 写满残差）")
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
            return err!("键的 L2 范数为零：残差 `v − S k` 会退化成 `v`，写进去的是值而不是纠错");
        }
        Ok(key.iter().map(|&value| value.div(norm)).collect())
    }

    /// 宽度 + 有限性。
    fn check_values<S: Scalar>(&self, what: &str, values: &[S], want: usize) -> Result<()> {
        self.check_len(what, values.len(), want)?;
        match values.iter().position(|&value| !finite(value)) {
            Some(idx) => err!("{what}第 {idx} 项非有限：放行会把 NaN 一路带进记忆"),
            None => Ok(()),
        }
    }

    /// 宽度校验。
    fn check_len(&self, what: &str, got: usize, want: usize) -> Result<()> {
        if got == want {
            return Ok(());
        }
        err!("{what}的宽度必须是 {want}，得到 {got}")
    }
}

/// 一行的更新：先逐通道遗忘，再把残差 `v − S k̂` 按写强度写回。
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
        return err!("门第 {channel} 个通道非有限：遗忘系数必须是 [0, 1] 里的实数");
    }
    if alpha < S::ZERO || alpha > S::ONE {
        return err!(
            "门第 {channel} 个通道是 {alpha:?}：遗忘系数必须在 [0, 1] 内\
                 （0 清空该通道，1 一点不忘）"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `kda` 的用例。
    //!
    //! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

    use super::*;
    use diagnostic::Code;

    /// 3 通道键、2 维值的层：手算跟得上。
    fn kda() -> Kda {
        Kda::new(3, 2).expect("几何合法")
    }

    /// 单位门：什么都不忘。
    fn keep() -> [f64; 3] {
        [1.0, 1.0, 1.0]
    }

    /// 零记忆。
    fn zeroed() -> Memory<f64> {
        kda().memory()
    }

    /// 单位键 `k / ‖k‖`：写入与读取都用它，精确性才可比。
    fn unit(key: [f64; 3]) -> [f64; 3] {
        let norm = key.iter().map(|value| value * value).sum::<f64>().sqrt();
        [key[0] / norm, key[1] / norm, key[2] / norm]
    }

    /// 两个正交的单位键：写一个不该影响另一个。
    fn axes() -> ([f64; 3], [f64; 3]) {
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    }

    /// 两个正交键各记一条关联：键 `first` 记 `[1, 0]`、键 `second` 记 `[0, 4]`；
    /// 返回层、写完的记忆与两个键。
    fn two_records() -> (Kda, Memory<f64>, [f64; 3], [f64; 3]) {
        let layer = kda();
        let mut memory = zeroed();
        let (first, second) = axes();
        layer
            .step(&mut memory, &keep(), 1.0, &first, &[1.0, 0.0])
            .expect("几何合法");
        layer
            .step(&mut memory, &keep(), 1.0, &second, &[0.0, 4.0])
            .expect("几何合法");
        (layer, memory, first, second)
    }

    testkit::test_cases! {
    /// 写一次、读一次：读出就是写进去的值（`q = k̂` 时的精确性）。
    fn a_write_is_exactly_recalled() {
        let layer = kda();
        let mut memory = zeroed();
        let key = unit([0.3, 0.4, 0.5]);
        let value = [1.0, -2.0];
        layer
            .step(&mut memory, &keep(), 1.0, &key, &value)
            .expect("几何合法");
        let got = layer.read(&memory, &key).expect("几何合法");
        for (idx, (read, wrote)) in got.iter().zip(&value).enumerate() {
            assert!(
                (read - wrote).abs() < 1e-12,
                "第 {idx} 维读出 {read}，写进去的是 {wrote}"
            );
        }
    }

    /// 同一个键写两次、值不同，读出是**后一次**的值：这是 delta 规则与指数梯形的分界。
    fn the_delta_rule_overwrites_instead_of_accumulating() {
        let layer = kda();
        let mut memory = zeroed();
        let (key, _) = axes();
        layer
            .step(&mut memory, &keep(), 1.0, &key, &[1.0, 2.0])
            .expect("几何合法");
        layer
            .step(&mut memory, &keep(), 1.0, &key, &[-3.0, 0.5])
            .expect("几何合法");
        let got = layer.read(&memory, &key).expect("几何合法");
        assert!(
            (got[0] + 3.0).abs() < 1e-12,
            "第 0 维是 {}，累加的话会是 -2",
            got[0]
        );
        assert!(
            (got[1] - 0.5).abs() < 1e-12,
            "第 1 维是 {}，累加的话会是 2.5",
            got[1]
        );
    }

    /// 正交的两个键各记各的：写第二个不扰动第一个。
    fn two_orthogonal_keys_keep_their_own_values() {
        let (layer, memory, first, second) = two_records();
        let a = layer.read(&memory, &first).expect("几何合法");
        let b = layer.read(&memory, &second).expect("几何合法");
        assert!((a[0] - 1.0).abs() < 1e-12 && a[1].abs() < 1e-12, "{a:?}");
        assert!(b[0].abs() < 1e-12 && (b[1] - 4.0).abs() < 1e-12, "{b:?}");
    }

    /// 门是逐通道的：关掉通道 0 只让键 0 的关联失效，通道 1 的照旧。
    fn the_gate_forgets_one_channel_at_a_time() {
        let (layer, mut memory, first, second) = two_records();
        // 只在通道 0 上遗忘，这一次不写（β = 0）。
        layer
            .step(&mut memory, &[0.0, 1.0, 1.0], 0.0, &first, &[0.0, 0.0])
            .expect("几何合法");
        let gone = layer.read(&memory, &first).expect("几何合法");
        assert!(
            gone[0].abs() < 1e-12,
            "通道 0 被清空，键 0 的关联应当读不出来，得到 {}",
            gone[0]
        );
        let kept = layer.read(&memory, &second).expect("几何合法");
        assert!(
            (kept[1] - 4.0).abs() < 1e-12,
            "通道 1 没忘，键 1 的关联还该在，得到 {}",
            kept[1]
        );
    }

    /// 写强度按比例缩放纠错量：`β = 0.25` 读到的就是满写的四分之一。
    fn the_write_strength_scales_the_correction() {
        let layer = kda();
        let (key, _) = axes();
        let value = [3.0, -1.0];
        let mut full = zeroed();
        layer
            .step(&mut full, &keep(), 1.0, &key, &value)
            .expect("几何合法");
        let mut quarter = zeroed();
        layer
            .step(&mut quarter, &keep(), 0.25, &key, &value)
            .expect("几何合法");
        let want = layer.read(&full, &key).expect("几何合法");
        let got = layer.read(&quarter, &key).expect("几何合法");
        for (idx, (read, full)) in got.iter().zip(&want).enumerate() {
            assert!(
                (read - full * 0.25).abs() < 1e-12,
                "第 {idx} 维 {read} 不是满写的四分之一（{full}）"
            );
        }
    }

    /// 写强度为零时只留下遗忘：`α = 0.5` 且不写，读出减半。
    fn a_zero_write_strength_only_forgets() {
        let layer = kda();
        let mut memory = zeroed();
        let (key, _) = axes();
        layer
            .step(&mut memory, &keep(), 1.0, &key, &[2.0, -2.0])
            .expect("几何合法");
        layer
            .step(&mut memory, &[0.5; 3], 0.0, &key, &[0.0, 0.0])
            .expect("几何合法");
        let got = layer.read(&memory, &key).expect("几何合法");
        assert!(
            (got[0] - 1.0).abs() < 1e-12 && (got[1] + 1.0).abs() < 1e-12,
            "{got:?}"
        );
    }

    /// 键的范数不影响写入强度：归一化在入口做，缩放键不会改变记忆。
    fn the_key_is_normalized_at_the_door() {
        let layer = kda();
        let mut short = zeroed();
        let mut long = zeroed();
        let key = [0.2, -0.3, 0.6];
        let stretched = [1.2, -1.8, 3.6];
        layer
            .step(&mut short, &keep(), 1.0, &key, &[1.5, -0.5])
            .expect("几何合法");
        layer
            .step(&mut long, &keep(), 1.0, &stretched, &[1.5, -0.5])
            .expect("几何合法");
        for (cell, (a, b)) in short.cells().iter().zip(long.cells()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "第 {cell} 格 {a} 与 {b} 不同：键的范数不该改变写入强度"
            );
        }
    }

    /// 零键被拒：归一化它只会得到 `NaN`，而残差会退化成 `v`。
    fn a_zero_key_is_refused() {
        let mut memory = zeroed();
        let refused = kda()
            .step(&mut memory, &keep(), 1.0, &[0.0; 3], &[1.0, 2.0])
            .expect_err("零键不能归一");
        assert_eq!(refused.code(), Code::Invalid);
        assert_eq!(memory.cells(), &[0.0; 6], "被拒的一次不该改动记忆");
    }

    /// 非有限输入在入口被拒，且记忆保持原样。
    fn non_finite_inputs_are_refused() {
        let layer = kda();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut memory = zeroed();
            let key = layer
                .step(&mut memory, &keep(), 1.0, &[1.0, bad, 0.0], &[1.0, 2.0])
                .expect_err("键非有限");
            assert_eq!(key.code(), Code::Invalid);
            let value = layer
                .step(&mut memory, &keep(), 1.0, &[1.0, 0.0, 0.0], &[bad, 2.0])
                .expect_err("值非有限");
            assert_eq!(value.code(), Code::Invalid);
            let query = layer
                .read(&memory, &[bad, 0.0, 0.0])
                .expect_err("查询非有限");
            assert_eq!(query.code(), Code::Invalid);
            assert_eq!(memory.cells(), &[0.0; 6]);
        }
    }

    /// 门是遗忘系数：越界拒绝，两个端点合法。
    fn the_gate_is_a_forget_gate() {
        let layer = kda();
        let key = [1.0, 0.0, 0.0];
        let value = [1.0, 1.0];
        for bad in [-0.5, 1.5, f64::NAN] {
            let mut memory = zeroed();
            let refused = layer
                .step(&mut memory, &[bad, 1.0, 1.0], 1.0, &key, &value)
                .expect_err("门越界");
            assert_eq!(refused.code(), Code::Invalid);
        }
        for edge in [0.0, 1.0] {
            let mut memory = zeroed();
            layer
                .step(&mut memory, &[edge; 3], 1.0, &key, &value)
                .expect("两个端点都是合法遗忘系数");
        }
    }

    /// 写强度是比例：越界拒绝，端点合法。
    fn the_write_strength_is_a_fraction() {
        let layer = kda();
        let key = [1.0, 0.0, 0.0];
        let value = [1.0, 1.0];
        for bad in [-0.5, 1.5, f64::NAN] {
            let mut memory = zeroed();
            let refused = layer
                .step(&mut memory, &keep(), bad, &key, &value)
                .expect_err("写强度越界");
            assert_eq!(refused.code(), Code::Invalid);
        }
        for edge in [0.0, 1.0] {
            let mut memory = zeroed();
            layer
                .step(&mut memory, &keep(), edge, &key, &value)
                .expect("两个端点都是合法写强度");
        }
    }

    /// 构造期非零校验，运行期每处宽度与状态归属都在入口判定。
    fn geometry_is_checked_at_the_door() {
        assert_eq!(Kda::new(0, 2).expect_err("键维为零").code(), Code::Invalid);
        assert_eq!(Kda::new(2, 0).expect_err("值维为零").code(), Code::Invalid);

        let layer = kda();
        let key = [1.0, 0.0, 0.0];
        let value = [1.0, 1.0];
        let mut memory = zeroed();
        let narrow_gate = layer
            .step(&mut memory, &[1.0, 1.0], 1.0, &key, &value)
            .expect_err("门窄了一格");
        assert_eq!(narrow_gate.code(), Code::Invalid);
        let narrow_key = layer
            .step(&mut memory, &keep(), 1.0, &[1.0, 0.0], &value)
            .expect_err("键窄了一格");
        assert_eq!(narrow_key.code(), Code::Invalid);
        let narrow_value = layer
            .step(&mut memory, &keep(), 1.0, &key, &[1.0])
            .expect_err("值窄了一格");
        assert_eq!(narrow_value.code(), Code::Invalid);
        let narrow_query = layer.read(&memory, &[1.0, 0.0]).expect_err("查询窄了一格");
        assert_eq!(narrow_query.code(), Code::Invalid);

        let mut foreign = Kda::new(2, 3).expect("几何合法").memory::<f64>();
        let mismatched = layer
            .step(&mut foreign, &keep(), 1.0, &key, &value)
            .expect_err("记忆属于别的层");
        assert_eq!(mismatched.code(), Code::Invalid);
    }

    /// 记忆是行主序 `[值维, 键维]`：一个通道上的写只落在对应列。
    fn the_memory_is_row_major_value_by_key() {
        let mut memory = zeroed();
        kda()
            .step(&mut memory, &keep(), 1.0, &[1.0, 0.0, 0.0], &[1.0, -2.0])
            .expect("几何合法");
        assert_eq!(memory.cells(), &[1.0, 0.0, 0.0, -2.0, 0.0, 0.0]);
        assert_eq!(memory.row(0), Some(&[1.0, 0.0, 0.0][..]));
        assert_eq!(memory.row(1), Some(&[-2.0, 0.0, 0.0][..]));
        assert_eq!(memory.row(2), None);
        assert_eq!(memory.key_dim(), 3);
        assert_eq!(memory.value_dim(), 2);
    }

    /// 没写过的记忆读出零，而不是残缺的记忆。
    fn an_untouched_memory_reads_zero() {
        let got = kda().read(&zeroed(), &[1.0, 2.0, 3.0]).expect("几何合法");
        assert_eq!(got, vec![0.0, 0.0]);
    }

    /// `f32` 与 `f64` 同一套算术：宽窄差别只在精度。
    fn f32_and_f64_agree() {
        let layer = kda();
        let key64: [f64; 3] = unit([0.3, -0.4, 0.5]);
        let key32: [f32; 3] = [key64[0] as f32, key64[1] as f32, key64[2] as f32];
        let mut wide = zeroed();
        layer
            .step(&mut wide, &[0.75; 3], 0.75, &key64, &[1.5, -0.25])
            .expect("几何合法");
        layer
            .step(&mut wide, &[0.9, 0.5, 0.25], 0.5, &key64, &[-1.0, 2.0])
            .expect("几何合法");
        let mut narrow: Memory<f32> = layer.memory();
        layer
            .step(&mut narrow, &[0.75; 3], 0.75, &key32, &[1.5, -0.25])
            .expect("几何合法");
        layer
            .step(&mut narrow, &[0.9, 0.5, 0.25], 0.5, &key32, &[-1.0, 2.0])
            .expect("几何合法");
        for (cell, (a, b)) in wide.cells().iter().zip(narrow.cells()).enumerate() {
            assert!(
                (a - f64::from(*b)).abs() < 1e-6,
                "第 {cell} 格 f64 是 {a}、f32 是 {b}"
            );
        }
    }
    }
}
