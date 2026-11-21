//! `T_θ` 三通道合成律（§1.5，数学验证 31）。
//!
//! `y_cont = π·y_α + (1−π)·y_β`；`y_γ = Encode_event(y_cont)`；`y_t = (1−γ)·y_cont + γ·y_γ`。
//! 系数 `((1−γ)π, (1−γ)(1−π), γ)` 构成单位剖分（和为 1、各项 ≥ 0），故 `y_t` 落在三通道输出的
//! 凸包内。事件编码 `y_γ` 由调用方（[`crate::mix::pulse`]）算出，本模块只做纯合成，无状态。

use op::scan::Scalar;

/// α/β 凸组合：`y_cont = π·y_α + (1−π)·y_β`。
pub fn cont<S: Scalar>(ya: &[S], yb: &[S], pi: S) -> Vec<S> {
    debug_assert_eq!(ya.len(), yb.len());
    let om = S::ONE.sub(pi);
    ya.iter()
        .zip(yb)
        .map(|(&a, &b)| a.mul(pi).add(b.mul(om)))
        .collect()
}

/// 连续/事件门控：`y_t = (1−γ)·y_cont + γ·y_γ`。
pub fn gate<S: Scalar>(yc: &[S], yg: &[S], gamma: S) -> Vec<S> {
    debug_assert_eq!(yc.len(), yg.len());
    let om = S::ONE.sub(gamma);
    yc.iter()
        .zip(yg)
        .map(|(&c, &g)| c.mul(om).add(g.mul(gamma)))
        .collect()
}

/// 全合成：`y_t = (1−γ)(π·y_α + (1−π)·y_β) + γ·y_γ`。
pub fn synthesize<S: Scalar>(ya: &[S], yb: &[S], yg: &[S], pi: S, gamma: S) -> Vec<S> {
    gate(&cont(ya, yb, pi), yg, gamma)
}
mod pulse {
    //! γ 脉冲通道：事件编码 `Encode_event`（§1.5）与 INT8-Spiking 事件臂（E14 / E15）。
    //!
    //! IEEE 上 `Encode_event ≡ Id`（[`encode_id`]，Lipschitz 常数 `L_e = 1`）；FR-HW 走 [`Encoder`]——
    //! 对称 int8 量化 + 整数阈值发放，还原所需的步长随 [`Events`] 一起走，只对发放位置做整数累加
    //! （[`accumulate`]）。γ 通道不是可选装饰，而是跨平台统一契约的必要组成（§1.5 第 6 点）：它让
    //! 同一模型可跨 IEEE/FR-HW 迁移，并允许运行时按输入稀疏度切换事件臂。
    //!
    //! **三项实测在本模块，判定不在这里**：E14 的冻结稀疏度 `S = 1 − N_active/N_total` 由
    //! [`Events::sparsity`] 给出，重构损失由 [`Audit`] 给出，`L_e` 的样本估计由 [`steepest`] 给出。
    //! PASS/FAIL 属 `tools::card` 的协议卡——一列样本只能出一份判决，本模块不另开判定语汇。

    use diagnostic::{Code, Error, Result};
    use op::scan::Scalar;

    use crate::csa::finite;

    /// 报错出处：参数、几何与非有限值都在入口失败。
    const SITE: &str = "nn::mix::pulse";

    /// 对称 int8 的正码上限：码位是 `−127..=127`。
    ///
    /// `−128` 不是码位——把它算进去，`code` 与 `−code` 就不再对称，`code·scale` 还原出来的偏差
    /// 会整体偏一格。
    const MAX_CODE: i32 = 127;

    /// LIF 神经元（FR-HW 事件臂）：膜电位积分 + 泄漏 + 阈值发放 + 重置。
    ///
    /// 状态是膜电位 [`Lif::mem`]；每步 [`Lif::step`] 返回本步发放（`ONE`/`ZERO`），事件驱动的整数
    /// 累加由调用方把返回值累加（§1.4 第 5 类）。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Lif<S> {
        threshold: S,
        leak: S,
        reset: S,
        mem: S,
    }

    impl<S: Scalar> Lif<S> {
        /// 构造：发放阈值 `threshold`、每步泄漏比例 `leak ∈ [0,1]`（`0` = 纯 IF）、发放后重置 `reset`。
        pub fn new(threshold: S, leak: S, reset: S) -> Self {
            Self {
                threshold,
                leak,
                reset,
                mem: S::ZERO,
            }
        }

        /// 推进后的膜电位 `mem·(1−leak) + u`：只算、不提交、不判发放。
        ///
        /// 拆出来的唯一理由是「发放前那一刻的电位」本身要被观测：反传的代理梯度按它判窗，
        /// 而 [`Lif::step`] 一提交就把发放过的通道改写成 `reset`，那个值当场没了。让前向顺手
        /// 记下 `peek` 的返回值，比事后按 `reset` 猜一遍要准。
        pub fn peek(&self, u: S) -> S {
            self.mem.mul(S::ONE.sub(self.leak)).add(u)
        }

        /// 按已算出的膜电位提交：达到阈值则发放并把膜电位重置，返回本步发放（`ONE`/`ZERO`）。
        pub(crate) fn commit(&mut self, potential: S) -> S {
            let spike = if potential >= self.threshold {
                S::ONE
            } else {
                S::ZERO
            };
            self.mem = if spike == S::ONE {
                self.reset
            } else {
                potential
            };
            spike
        }

        /// 推进一步：`mem = mem·(1−leak) + u`，超阈值发放并重置。
        ///
        /// 返回本步发放（`ONE` = 发放，`ZERO` = 未发放）。
        pub fn step(&mut self, u: S) -> S {
            let potential = self.peek(u);
            self.commit(potential)
        }

        /// 当前膜电位（观测用）。
        pub fn mem(&self) -> S {
            self.mem
        }
    }

    /// IEEE 事件编码恒等：`Encode_event(y) = y`（`L_e = 1`，§1.5）。
    ///
    /// 逐元素恒等，因此 IEEE 六平台上 γ 混合与 `γ≡0/1` 输出一致（F0-2 门控同构）。
    pub fn encode_id<S: Scalar>(y: &[S]) -> Vec<S> {
        y.to_vec()
    }

    /// 稀疏事件：只记发放位置与整数码，未发放的位置不占记录。
    ///
    /// 还原所需的步长也在里面（[`Events::decode`] 因此不需要编码器）：事件与「按什么步长还原」
    /// 分开存放时，拿另一层的 `Events` 去解会静默解出另一组数——内积照算，结果却不是这一层的。
    #[derive(Debug, Clone, PartialEq)]
    pub struct Events<S: Scalar> {
        width: usize,
        scale: S,
        indices: Vec<usize>,
        codes: Vec<i8>,
    }

    impl<S: Scalar> Events<S> {
        /// 事件覆盖的宽度（本层激活的元素数 `N_total`）。
        pub fn width(&self) -> usize {
            self.width
        }

        /// 发放的位置数 `N_active`。
        pub fn active(&self) -> usize {
            self.indices.len()
        }

        /// E14 的冻结定义：`S = 1 − N_active/N_total`。
        ///
        /// 宽度为零记 `1`，不是 `0` 也不是 `NaN`：一个位置都没有，就没有一个位置需要算，这与
        /// 「全不发放」是同一个稀疏度；`NaN` 还会在下游被当成最大稀疏度。
        pub fn sparsity(&self) -> f64 {
            if self.width == 0 {
                return 1.0;
            }
            1.0 - self.indices.len() as f64 / self.width as f64
        }

        /// 发放位置，严格升序。
        pub fn indices(&self) -> &[usize] {
            &self.indices
        }

        /// 与 [`Events::indices`] 一一对应的整数码。
        pub fn codes(&self) -> &[i8] {
            &self.codes
        }

        /// 每个码的步长。
        pub fn scale(&self) -> S {
            self.scale
        }

        /// 还原成稠密张量：发放位置取 `code·scale`，其余位置为零。
        ///
        /// 未发放的位置还原成零而不是「保留原值」：事件臂能给出的信息只有发放的那些码，
        /// 补上原值等于偷偷退回连续臂。
        pub fn decode(&self) -> Vec<S> {
            let mut y = vec![S::ZERO; self.width];
            for (&at, &code) in self.indices.iter().zip(&self.codes) {
                y[at] = self.scale.mul(S::from_f64(f64::from(code)));
            }
            y
        }
    }

    /// FR-HW 事件臂的编码器（`Encode_event` 的非恒等形态）：对称 int8 量化 + 整数阈值发放。
    ///
    /// 只持两个参数：`scale` 是每个码的步长，`threshold` 是发放门槛。门槛按**码位**给而不是按
    /// 浮点值给，于是「发不发放」与 `scale` 解耦——换一个步长不会顺手把稀疏度换成另一个数。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Encoder<S> {
        scale: S,
        threshold: i32,
    }

    impl<S: Scalar> Encoder<S> {
        /// 构造：步长有限且为正，门槛落在 `1..=127`。
        ///
        /// 门槛为零会让**每一个**位置都发放（含恰好为零的那些），事件臂于是退化成「带量化误差的
        /// 稠密臂」——稀疏度是 0，能效论证随之不成立，所以在入口就拒绝。
        ///
        /// # Errors
        ///
        /// 步长非有限或非正、门槛不在 `1..=127`——返回 `Code::Invalid`。
        pub fn new(scale: S, threshold: i32) -> Result<Self> {
            if !finite(scale) || scale <= S::ZERO {
                return Err(fail(format!("步长必须是有限正数，得到 {scale:?}")));
            }
            if !(1..=MAX_CODE).contains(&threshold) {
                return Err(fail(format!(
                    "发放门槛必须落在 1..={MAX_CODE}，得到 {threshold}"
                )));
            }
            Ok(Self { scale, threshold })
        }

        /// 每个码的步长。
        pub fn scale(&self) -> S {
            self.scale
        }

        /// 发放门槛（码位口径）。
        pub fn threshold(&self) -> i32 {
            self.threshold
        }

        /// 编码：`code = round(y/scale)` 夹取到 `±127`，`|code| ≥ threshold` 的位置才发放。
        ///
        /// # Errors
        ///
        /// 输入含非有限值返回 `Code::Invalid`：比较对 `NaN` 全为假，放行会把它拾成码位 `0`，
        /// 于是一个坏值静默变成「不发放」。
        pub fn encode(&self, y: &[S]) -> Result<Events<S>> {
            check_finite(y)?;
            let mut indices = Vec::new();
            let mut codes = Vec::new();
            for (at, &value) in y.iter().enumerate() {
                let code = self.quantize(value);
                if i32::from(code).abs() >= self.threshold {
                    indices.push(at);
                    codes.push(code);
                }
            }
            Ok(Events {
                width: y.len(),
                scale: self.scale,
                indices,
                codes,
            })
        }

        /// 一次完整的事件臂前向：`y → Encode_event → (1−γ)·y + γ·y_γ`（§1.5 的 γ 合成）。
        ///
        /// 顺序固定在这一处：先编码再合成，因此 `γ = 0` 时结果与输入逐位相同，`γ = 1` 时结果恰是
        /// [`Events::decode`] 的输出。
        ///
        /// # Errors
        ///
        /// 与 [`Encoder::encode`] 同一判据。
        pub fn blend(&self, y: &[S], gamma: S) -> Result<Vec<S>> {
            let events = self.encode(y)?;
            Ok(super::gate(y, &events.decode(), gamma))
        }

        /// 一个浮点值 → 码位：`round(v/scale)` 夹取到 `±MAX_CODE`。
        ///
        /// 夹取而不是回绕：回绕把超量程的值折成另一个符号的合法码，看起来仍像一个数。
        fn quantize(&self, value: S) -> i8 {
            let raw = value.div(self.scale).round().to_f64();
            let limit = f64::from(MAX_CODE);
            raw.clamp(-limit, limit) as i8
        }
    }

    /// 一段激活的事件臂实测两项：E14 稀疏度与相对重构损失 `‖ŷ − y‖₂ / ‖y‖₂`。
    ///
    /// 只报数、不判门：SpB 2.0 的「64.31% 稀疏 / 0.69% 损失」是同口径的**外部锚点**，本仓的数要
    /// 自己测；判决由 `tools::card` 的协议卡出。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Audit {
        sparsity: f64,
        loss: f64,
    }

    impl Audit {
        /// 对一段激活取两项实测。
        ///
        /// `‖y‖ = 0` 时分母为零，改用**绝对**误差 `‖ŷ‖`：相对误差在零输入上是 `0/0`，记成 `0`
        /// 会把「没测到」与「完美重构」混为一谈。
        ///
        /// # Errors
        ///
        /// 输入含非有限值返回 `Code::Invalid`（与 [`Encoder::encode`] 同一判据）。
        pub fn measure<S: Scalar>(encoder: &Encoder<S>, y: &[S]) -> Result<Self> {
            let events = encoder.encode(y)?;
            let restored = events.decode();
            Ok(Self {
                sparsity: events.sparsity(),
                loss: relative_error(y, &restored),
            })
        }

        /// E14 的稀疏度 `S = 1 − N_active/N_total`。
        pub fn sparsity(&self) -> f64 {
            self.sparsity
        }

        /// 相对重构损失（`‖y‖ = 0` 时退化为绝对误差）。
        pub fn loss(&self) -> f64 {
            self.loss
        }
    }

    /// 样本对上的最大放大比 `max ‖Ê(u) − Ê(v)‖ / ‖u − v‖`（§1.5 的 `L_e`）。
    ///
    /// **量化器不是真 Lipschitz**：两点距离趋于零时，只要它们落在相邻两个码位上，比值就能任意大
    /// （码位之间有台阶）。所以这里报的只是**给定样本对上**的实测比值——`L_e` 要有意义，样本集
    /// （连同点的间距分布）必须与数字一起冻结，否则同一份实现能给出任意大的 `L_e`。
    ///
    /// 没有一对可比的输入时返回 `Ok(None)`：`0` 会被读成「完全不放大」，而事实是没有任何证据。
    ///
    /// # Errors
    ///
    /// 某一对两段长度不等、或任一输入含非有限值——返回 `Code::Invalid`。
    pub fn steepest<S: Scalar>(
        encoder: &Encoder<S>,
        pairs: &[(&[S], &[S])],
    ) -> Result<Option<f64>> {
        let mut worst: Option<f64> = None;
        for &(u, v) in pairs {
            if let Some(ratio) = pair_gain(encoder, u, v)? {
                worst = Some(worst.map_or(ratio, |w| w.max(ratio)));
            }
        }
        Ok(worst)
    }

    /// 一对输入的放大比；两点重合时 `None`（没有距离，比值无定义）。
    fn pair_gain<S: Scalar>(encoder: &Encoder<S>, u: &[S], v: &[S]) -> Result<Option<f64>> {
        check_pair(u, v)?;
        let (ha, hb) = (encoder.encode(u)?.decode(), encoder.encode(v)?.decode());
        let span = squared_distance(u, v);
        if span == 0.0 {
            return Ok(None);
        }
        Ok(Some((squared_distance(&ha, &hb) / span).sqrt()))
    }

    /// 一对样本的入口契约：长度相等且各自都有限。
    fn check_pair<S: Scalar>(u: &[S], v: &[S]) -> Result<()> {
        if u.len() != v.len() {
            return Err(fail(format!(
                "样本对的两段长度必须相等，得到 {} 与 {}",
                u.len(),
                v.len()
            )));
        }
        check_finite(u)?;
        check_finite(v)
    }

    /// 事件驱动的整数累加：`y_j = Σ_{i ∈ 发放} w[j·width + i]·code_i`（§1.4 第 5 类）。
    ///
    /// 只遍历发放位置，因此代价随 `N_active` 而不是 `width` 走——这是事件臂相对于稠密臂的唯一
    /// 实际好处；数值上它必须与「先铺成稠密码表再整行累加」**逐位相同**（累加次序仍是 `i` 升序，
    /// 被跳过位置不贡献任何一项）。
    ///
    /// 累加在 `i32` 里做，容量在入口一次判清（`cols · 127²` 装得下才开工），于是内层不必逐项查
    /// 溢出：回绕后的数看起来仍像一个数，比一个显式的失败危险得多。
    ///
    /// # Errors
    ///
    /// `w.len() != rows × cols`、`events.width() != cols`、尺寸或累加容量溢出——返回 `Code::Invalid`。
    pub fn accumulate<S: Scalar>(
        w: &[i8],
        rows: usize,
        cols: usize,
        events: &Events<S>,
    ) -> Result<Vec<i32>> {
        check_geometry(w.len(), rows, cols, events.width())?;
        let mut out = vec![0i32; rows];
        for (row, slot) in out.iter_mut().enumerate() {
            let line = &w[row * cols..(row + 1) * cols];
            *slot = events
                .indices()
                .iter()
                .zip(events.codes())
                .fold(0i32, |acc, (&at, &code)| {
                    acc + i32::from(line[at]) * i32::from(code)
                });
        }
        Ok(out)
    }

    /// `‖ŷ − y‖₂ / ‖y‖₂`；`‖y‖ = 0` 时退化成 `‖ŷ‖₂`。
    fn relative_error<S: Scalar>(y: &[S], restored: &[S]) -> f64 {
        let residual = squared_distance(y, restored).sqrt();
        let norm = squared_norm(y).sqrt();
        if norm > 0.0 {
            residual / norm
        } else {
            residual
        }
    }

    /// 一段向量的欧氏长度平方。
    fn squared_norm<S: Scalar>(values: &[S]) -> f64 {
        values.iter().map(|&v| v.to_f64().powi(2)).sum()
    }

    /// 两段等长向量的欧氏距离平方。
    fn squared_distance<S: Scalar>(a: &[S], b: &[S]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = x.sub(y).to_f64();
                d * d
            })
            .sum()
    }

    /// 几何与容量一次判清：形状对了才谈累加，容量装得下才允许用普通 `i32` 加法。
    fn check_geometry(got: usize, rows: usize, cols: usize, width: usize) -> Result<()> {
        if rows.checked_mul(cols) != Some(got) {
            return Err(fail(format!(
                "权重长度必须是 rows × cols = {rows} × {cols}，得到 {got}"
            )));
        }
        if width != cols {
            return Err(fail(format!(
                "事件宽度必须等于权重列数 {cols}，得到 {width}"
            )));
        }
        let cell = (MAX_CODE as usize) * (MAX_CODE as usize);
        if cols.saturating_mul(cell) > i32::MAX as usize {
            return Err(fail(format!(
                "列数 {cols} × 单格上限 {cell} 超出 i32 累加容量"
            )));
        }
        Ok(())
    }

    /// 整段激活都有限：比较对 `NaN` 全为假，放行会把它静默拾成码位 `0`。
    fn check_finite<S: Scalar>(values: &[S]) -> Result<()> {
        if values.iter().all(|&v| finite(v)) {
            return Ok(());
        }
        Err(fail("激活含非有限值".to_string()))
    }

    /// 统一的失败构造：出处固定、码位固定，报错文案各自不同。
    fn fail(message: String) -> Error {
        Error::new(Code::Invalid, message).with(SITE)
    }

    #[cfg(test)]
    #[path = "../../../tests/mix_pulse_tests.rs"]
    mod tests;
}

pub use pulse::{Audit, Encoder, Events, Lif, accumulate, encode_id, steepest};

#[cfg(test)]
#[path = "../tests/mix_tests.rs"]
mod tests;
