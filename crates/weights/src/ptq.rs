//! 2/3/4/8-bit 对称训练后量化（PTQ）：把 f32 权重按组量化并打包，以及它的反解。
//!
//! 与 [`super::quant`] 的分工：`quant` 是**磁盘方案描述符与分组推断**（读已有量化文件的形状与
//! 标度，不做数值）；本模块是**把 f32 真正算成打包码**的正向算法与其逆变换，供离线 PTQ 与
//! round-trip 验证使用。换算算法按 ADR 0018 留在 `weights`，不进入设备内核注册表。
//!
//! `I2`/`I4`/`I8` 的运行时事实（位宽、对称零点、正码上限）与 `op::quant::Row` **逐项同源**
//! （低码在前的打包位序、对称零点一致）；`I3` 是离线侧新增档位（3 码位宽），无内核对应。本模块
//! 不自建到 `op` 的依赖边：离线换算按 ADR 0018 留在 `weights`，不需要内核层。
//!
//! 对称口径：`scale = max_abs / max_code`，`signed = round(x/scale)` 夹取到 `[-max_code, max_code]`，
//! 码的线格式与反量化公式 `y = scale * (code - zero)` 完全一致。i2/i4 是**偏移码**（`code = signed + zero`，
//! 无符号 nibble），i8 是**二补码字节**（`zero = 0`，`code = signed` 直接按 `i8` 解释）。

mod adaptive;

pub use adaptive::{Adaptive, AdaptiveCompensated, adaptive_quantize, compensate_adaptive};

use diagnostic::{Code, Error, Result};

/// 一档对称量化格式的运行时事实。
///
/// 三个字段回答“这一档怎么量化/怎么打包”，与 `op::quant::Row`（`I2`/`I4`/`I8`）逐项一致：
/// `bits` 对应 `Row::width`，`zero` 对应 `Row::zero`，`max_code` 是对称码域的正码上限
/// （`2^(bits-1) - 1`，i8 取 127 避开 -128 的非对称）。打包长度由 `bits` 直接推出，
/// 不单设每字节码数——3 位档每码 3 bit，不整除字节。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scheme {
    /// 元素位宽：2 / 3 / 4 / 8。
    pub bits: u8,
    /// 对称零点：偏移码（i2/i3/i4）的 `signed + zero` 平移量；i8 为 0。
    pub zero: i32,
    /// 正码上限：`scale = max_abs / max_code` 的分母。
    pub max_code: f32,
}

impl Scheme {
    /// 码域大小：`2^bits`。
    pub const fn domain(self) -> i32 {
        1 << self.bits
    }

    /// 每个码占的位数。
    pub const fn width(self) -> usize {
        self.bits as usize
    }

    /// 打包字节数：`ceil(len * bits / 8)`。
    pub fn packed_len(self, len: usize) -> usize {
        len.saturating_mul(self.bits as usize).div_ceil(8)
    }
}

/// 2 位打包：4 码/字节，零点 2，码表 `{-2,-1,0,1}`，正码上限 1。
pub const I2: Scheme = Scheme {
    bits: 2,
    zero: 2,
    max_code: 1.0,
};

/// 3 位打包：8 码/3 字节，零点 3，码表 `{-3..=3}`，正码上限 3。
pub const I3: Scheme = Scheme {
    bits: 3,
    zero: 3,
    max_code: 3.0,
};

/// 4 位打包：2 码/字节，零点 8，码表 `{-8..=7}`，正码上限 7。
pub const I4: Scheme = Scheme {
    bits: 4,
    zero: 8,
    max_code: 7.0,
};

/// 8 位逐元素：1 码/字节，零点 0（二补码 `i8`），正码上限 127。
pub const I8: Scheme = Scheme {
    bits: 8,
    zero: 0,
    max_code: 127.0,
};

/// 按组对称量化：把 `x` 按 `group` 个元素一组，每组独立 `scale = max_abs / max_code`，
/// 逐元素 `signed = round(x/scale)` 夹取到 `[-max_code, max_code]`，再按低码在前打包。
///
/// 返回 `(packed, scales)`：`packed` 长度 `ceil(len * bits / 8)`，`scales` 长度 `ceil(len / group)`。
/// 全零组取 `scale = 1.0`（码全为对称零点 `zero`），反解仍精确归零，同时避开零 scale 被反解侧拒绝。
///
/// # Errors
///
/// `group` 为 0、输入含非有限值，或 `max_code` 非法（0）时返回说明。
pub fn quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
    validate_quantize(scheme, group, x)?;

    let groups = x.len().div_ceil(group);
    let mut packed = vec![0u8; scheme.packed_len(x.len())];
    let mut scales = Vec::with_capacity(groups);

    for g in 0..groups {
        let lo = g * group;
        let hi = (lo + group).min(x.len());
        let slice = &x[lo..hi];
        let scale = group_scale(scheme, slice);
        scales.push(scale);

        for (i, &v) in slice.iter().enumerate() {
            let signed = (v / scale).round() as i32;
            let signed = signed.clamp(-(scheme.max_code as i32), scheme.max_code as i32);
            pack_bits(
                &mut packed,
                lo + i,
                scheme.bits,
                (signed + scheme.zero) as u8,
            );
        }
    }

    Ok((packed, scales))
}

/// 量化参数校验：分组、正码上限、输入有限性。
fn validate_quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<()> {
    if group == 0 {
        return Err(invalid("量化分组必须大于 0"));
    }
    if scheme.max_code <= 0.0 {
        return Err(invalid("正码上限必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(invalid("输入必须是有限 f32"));
    }
    Ok(())
}

/// 一组元素的对称标度：全零组取 1（码全为对称零点，反解仍精确归零），否则 `max_abs / max_code`。
fn group_scale(scheme: Scheme, slice: &[f32]) -> f32 {
    let max_abs = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / scheme.max_code
    }
}

/// 按组反解：`y = scale * (code - zero)`，与反量化公式一致，供 round-trip 对拍。
///
/// `packed` 长度必须为 `ceil(len * bits / 8)`，`scales` 长度必须为 `ceil(len / group)`，`out` 长度
/// 为元素数。i2/i3/i4 的码是无符号偏移码（减 `zero` 得符号值），i8 的码是二补码字节（`zero = 0`）。
///
/// # Errors
///
/// 输出长度不为 0 但分组/字节数不匹配时返回说明。
pub fn dequantize(
    scheme: Scheme,
    packed: &[u8],
    scales: &[f32],
    group: usize,
    out: &mut [f32],
) -> Result<()> {
    validate_dequantize(scheme, packed, scales, group, out)?;

    for (i, slot) in out.iter_mut().enumerate() {
        let field = unpack_bits(packed, i, scheme.bits);
        *slot = scales[i / group] * decode_code(scheme, field);
    }
    Ok(())
}

/// 反解参数校验：分组、打包字节数与输出长度、标度条数与分组数。
fn validate_dequantize(
    scheme: Scheme,
    packed: &[u8],
    scales: &[f32],
    group: usize,
    out: &[f32],
) -> Result<()> {
    if group == 0 {
        return Err(invalid("反解分组必须大于 0"));
    }
    if packed.len() != scheme.packed_len(out.len()) {
        return Err(invalid("打包字节数与输出长度不一致"));
    }
    if scales.len() != out.len().div_ceil(group) {
        return Err(invalid("标度条数与分组数不一致"));
    }
    Ok(())
}

/// 码 → 符号值：i8（`zero == 0`）按二补码解释，i2/i3/i4 是无符号偏移码减零点。
fn decode_code(scheme: Scheme, field: u8) -> f32 {
    if scheme.zero == 0 {
        field as i8 as f32
    } else {
        field as f32 - scheme.zero as f32
    }
}

/// 把第 `index` 个 `width` 位码写入位流（低码在前）；跨字节边界时高位溢到下一字节。
///
/// `code` 必须落在 `[0, 2^width)`；调用方（[`quantize`]）已按 `signed + zero` 把它夹在该区间。
fn pack_bits(packed: &mut [u8], index: usize, width: u8, code: u8) {
    let bit = index * width as usize;
    let byte = bit / 8;
    let shift = (bit % 8) as u32;
    let value = (code as u32) << shift;
    packed[byte] |= value as u8;
    if shift + width as u32 > 8 {
        packed[byte + 1] |= (value >> 8) as u8;
    }
}

/// 从位流读出第 `index` 个 `width` 位码；跨字节边界时低位溢到下一字节。
fn unpack_bits(packed: &[u8], index: usize, width: u8) -> u8 {
    let bit = index * width as usize;
    let byte = bit / 8;
    let shift = (bit % 8) as u32;
    let mut value = (packed[byte] as u32) >> shift;
    if shift + width as u32 > 8 {
        value |= (packed[byte + 1] as u32) << (8 - shift);
    }
    (value & ((1u32 << width) - 1)) as u8
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("weights::ptq")
}

/// 秩-r 低秩 + 残差量化补偿的结果：`Ŵ = L_r + Q(W − L_r)`。
///
/// 这是 M4「谱隙证书 + 秩-10 残差补偿」的数值侧：低秩项 `L_r` 以 f32 保存，显式残差
/// `C = W − L_r` 才进入低位宽对称量化（g=32）。4bit 的量化误差因此只作用在残差上，而不是整张
/// 权重上——这正是输出空间 SNR 能从「无补偿 ~20 dB」提到 44 dB 以上的原因，**前提**是
/// `W` 在秩 r 处确实有谱隙（结构近似低秩）。没有谱隙时量化误差不缩小，应由调用方走
/// [`crate::ledger::certify`] 拒压回退，而不是硬压出假数字。
#[derive(Clone, Debug)]
pub struct Compensated {
    /// 实际使用的低秩项秩（`rank` 与 SVD 可取秩的较小值）。
    pub rank: usize,
    /// 最大奇异值 `σ_1`（谱隙的相对参照）。
    pub sigma1: f64,
    /// 秩 r 处的谱隙 `γ_r = σ_r − σ_{r+1}`（r 为末位时取 `σ_r`）。
    pub gap: f64,
    /// 低秩项 `L_r`（行主序 `m*n` f32）。
    pub low: Vec<f32>,
    /// 残差 `C` 的打包码。
    pub packed: Vec<u8>,
    /// 残差 `C` 的组标度。
    pub scales: Vec<f32>,
    /// 重构 `Ŵ = L_r + Q(C)`（行主序 `m*n` f32）。
    pub recon: Vec<f32>,
}

/// 秩-r 低秩 + 残差量化补偿（自动谱隙证书门控）。
///
/// 流程：`W` 行主序 `m*n` → 一次 f64 SVD → [`svd::signal_rank`] 找信号/噪声边界秩 `r`
/// （无谱隙即拒压）→ 取前 `r` 个奇异方向做低秩项 `L_r` → 残差 `C = W − L_r` 按
/// [`quantize`] 做 `scheme`/`group` 对称量化 → [`dequantize`] 回解 `Q(C)` →
/// 重构 `Ŵ = L_r + Q(C)`。
///
/// 返回 `Ok(None)` 表示**拒压回退**：满秩/平坦谱矩阵没有信号谱隙（`signal_rank` 返回 `None`），
/// 4bit 量化误差不会因低秩分解而缩小，此时不硬压、不颁发证书。
///
/// # Errors
///
/// 形状非法、分组为 0、输入含非有限值，或 SVD 不收敛时返回说明。
pub fn compensate(
    scheme: Scheme,
    group: usize,
    w: &[f32],
    m: usize,
    n: usize,
) -> Result<Option<Compensated>> {
    validate_compensate(m, n, w)?;
    rebuild(scheme, group, w, m, n)
}

/// 数值侧：一次 f64 分解定低秩项（连它撑起的谱隙），显式残差再走同一个量化器。
fn rebuild(
    scheme: Scheme,
    group: usize,
    w: &[f32],
    m: usize,
    n: usize,
) -> Result<Option<Compensated>> {
    let Some(parts) = decompose(w, m, n)? else {
        return Ok(None);
    };
    let (packed, scales, qrest) = pack_residual(scheme, group, &parts.rest)?;
    let recon: Vec<f32> = parts.low.iter().zip(&qrest).map(|(l, q)| l + q).collect();
    Ok(Some(Compensated {
        rank: parts.rank,
        sigma1: parts.sigma1,
        gap: parts.gap,
        low: parts.low,
        packed,
        scales,
        recon,
    }))
}

/// 一次分解的产物：秩、谱隙、低秩项、显式残差与整体范数同出一门。
///
/// 两条补偿路径（固定档 [`rebuild`] 与自适应档 `adaptive::compensate_adaptive`）据此共用同一次
/// 分解：分别算一遍不只是慢，还会让「秩」与「残差」在某些输入上来自两个不同的分解。
struct Components {
    rank: usize,
    sigma1: f64,
    gap: f64,
    low: Vec<f32>,
    rest: Vec<f32>,
    norm: f64,
}

/// 一次 f64 SVD → 秩 / 低秩项 `L_r` / 显式残差 `C = W − L_r`；无谱隙即拒压。
fn decompose(w: &[f32], m: usize, n: usize) -> Result<Option<Components>> {
    let a: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let d = svd::compute(&a, m, n)?;
    let Some(rank) = svd::signal_rank(&d.sigma) else {
        return Ok(None);
    };
    let rows = d.reconstruct(rank);
    let low: Vec<f32> = rows.iter().map(|&x| x as f32).collect();
    let rest: Vec<f32> = a
        .iter()
        .zip(&rows)
        .map(|(orig, l)| (orig - l) as f32)
        .collect();
    Ok(Some(Components {
        rank,
        sigma1: d.sigma[0],
        gap: spectral_gap(&d.sigma, rank),
        low,
        rest,
        norm: a.iter().map(|v| v * v).sum::<f64>().sqrt(),
    }))
}

/// 截断处上方那道缝：`σ_r − σ_{r+1}`；已到最后一个奇异值时退化成 `σ_r`（下面没有值可供相减）。
fn spectral_gap(sigma: &[f64], r: usize) -> f64 {
    match sigma.get(r) {
        Some(&below) => sigma[r - 1] - below,
        None => sigma[r - 1],
    }
}

/// 残差打包 → 回解：编解码只有 [`quantize`] / [`dequantize`] 一份定义，补偿不另写一套。
fn pack_residual(
    scheme: Scheme,
    group: usize,
    rest: &[f32],
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let (packed, scales) = quantize(scheme, group, rest)?;
    let mut qrest = vec![0.0f32; rest.len()];
    dequantize(scheme, &packed, &scales, group, &mut qrest)?;
    Ok((packed, scales, qrest))
}

/// 补偿入口的两项契约：形状、有限性，每项各答一个问题。
fn validate_compensate(m: usize, n: usize, w: &[f32]) -> Result<()> {
    let cells = m
        .checked_mul(n)
        .filter(|&cells| cells > 0 && cells == w.len());
    if cells.is_none() {
        return Err(invalid(format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    if !w.iter().all(|v| v.is_finite()) {
        return Err(invalid("输入必须是有限 f32"));
    }
    Ok(())
}

pub mod svd {
    //! 零依赖 f64 奇异值分解：单边 Jacobi SVD（one-sided Jacobi）。
    //!
    //! 不先形成 `AᵀA`（那会把条件数平方、损失小奇异值精度），而是直接在工作矩阵的列对之间做
    //! Jacobi 旋转，逐步把列正交化：收敛后列长度即奇异值，归一化列即左奇异向量，累积的旋转给出
    //! 右奇异向量。Jacobi 旋转对小奇异值精度高、实现仅用 f64 算术，适合替代旧版的 faer 依赖。
    //!
    //! 物理下界维度（浮点机器精度）只承诺残差不劣于 `O(ε_mach·σ_1)`，不宣称精确零。

    use diagnostic::{Code, Error, Result};

    /// 最大扫描轮数：典型矩阵几轮即收敛，硬上限防止病态输入死循环。
    const MAX_SWEEPS: usize = 100;

    /// 收敛阈值：列内积相对范数小于此即视为已正交。
    const TOL: f64 = 1e-14;

    /// SVD 结果：奇异值按降序，配套左右奇异向量。
    #[derive(Clone, Debug)]
    pub struct Svd {
        /// 奇异值 `σ_1 ≥ σ_2 ≥ …`。
        pub sigma: Vec<f64>,
        /// 左奇异向量，行主序 `m × k`（第 j 列是对应 `sigma[j]` 的左向量）。
        pub u: Vec<f64>,
        /// 右奇异向量，行主序 `n × k`。
        pub v: Vec<f64>,
        m: usize,
        n: usize,
    }

    impl Svd {
        /// 左向量矩阵列数（= 非零/全部奇异值个数，取 n 与 m 的较小者由调用语义保证）。
        #[must_use]
        pub fn rank(&self) -> usize {
            self.sigma.len()
        }

        /// 矩阵形状 `(m, n)`：因子化存储要按它算 `UΣ` 与 `Vᵀ` 的行宽。
        #[must_use]
        pub fn shape(&self) -> (usize, usize) {
            (self.m, self.n)
        }

        /// 秩-r 截断重构（行主序 `m×n`）：`U[:, :r] Σ[:r] V[:, :r]ᵀ`。
        #[must_use]
        pub fn reconstruct(&self, r: usize) -> Vec<f64> {
            let r = r.min(self.sigma.len());
            let mut out = vec![0.0f64; self.m * self.n];
            for i in 0..self.m {
                for j in 0..self.n {
                    let mut sum = 0.0;
                    for k in 0..r {
                        // U(i,k)·σ_k·V(j,k)
                        sum += self.u[i * self.sigma.len() + k]
                            * self.sigma[k]
                            * self.v[j * self.sigma.len() + k];
                    }
                    out[i * self.n + j] = sum;
                }
            }
            out
        }
    }

    /// 信号秩选择的相对谱隙阈值。
    ///
    /// 依据（本机实测）：满秩高斯矩阵相邻奇异值的最大相对谱隙 ≈0.03；结构化近似低秩矩阵
    /// （秩-10 信号 + 高斯噪声）在信号/噪声边界处的相对谱隙 ≈0.49。0.2 落在这两个实测值之间，
    /// 与两侧都保持 >2× 的分离，用于把「有信号可压」与「纯噪声不可压」分开。
    pub const REL_GAP: f64 = 0.2;

    /// 有损补偿侧的信号秩：返回**噪声底之前**的最大秩 `r`，即 `(σ_r − σ_{r+1}) / σ_1 > REL_GAP`
    /// 的最后一点。
    ///
    /// 与 [`crate::ledger`] 的 `certified_rank` 分工：后者用算法残差 `ε`（机器精度量级）做绝对
    /// 阈值，服务**无损**压缩的精确秩判定（满秩矩阵返回满秩是正确语义）；本函数用相对阈值，服务
    /// **有损**量化补偿（满秩噪声矩阵无信号谱隙 → 返回 `None` = 拒压回退）。两者不可互换。
    #[must_use]
    pub fn signal_rank(sigma: &[f64]) -> Option<usize> {
        let floor = REL_GAP * sigma.first()?;
        (1..sigma.len())
            .rev()
            .find(|&r| sigma[r - 1] - sigma[r] > floor)
    }

    /// 对行主序 `a[m*n]` 做 SVD。
    pub fn compute(a: &[f64], m: usize, n: usize) -> Result<Svd> {
        if m == 0 || n == 0 || a.len() != m * n {
            return Err(Error::new(
                Code::Invalid,
                format!("SVD: 形状不合法 m={m} n={n} len={}", a.len()),
            )
            .with("svd::compute"));
        }
        let k = m.min(n);
        let mut u = a.to_vec();
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        orthogonalize(&mut u, &mut v, m, n);
        let sigma = singular_values(&mut u, m, n);
        let (sigma, u, v) = sorted_columns(&sigma, &u, &v, m, n, k);
        Ok(Svd { sigma, u, v, m, n })
    }

    /// 反复扫描直到收敛；硬上限防止病态输入死循环。
    fn orthogonalize(u: &mut [f64], v: &mut [f64], m: usize, n: usize) {
        for _ in 0..MAX_SWEEPS {
            if sweep(u, v, m, n) {
                break;
            }
        }
    }

    /// 一轮扫描：逐个列对旋转，返回本轮是否全部已正交。
    fn sweep(u: &mut [f64], v: &mut [f64], m: usize, n: usize) -> bool {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                if rotate_pair(u, v, m, n, p, q) {
                    converged = false;
                }
            }
        }
        converged
    }

    /// 一对列：已正交返回 `false`，做了旋转返回 `true`。
    fn rotate_pair(u: &mut [f64], v: &mut [f64], m: usize, n: usize, p: usize, q: usize) -> bool {
        let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
        for row in 0..m {
            let x = u[row * n + p];
            let y = u[row * n + q];
            alpha += x * x;
            beta += y * y;
            gamma += x * y;
        }
        if gamma.abs() <= TOL * (alpha * beta).sqrt() {
            return false;
        }
        let (c, s) = rotation(alpha, beta, gamma);
        rotate_columns(u, m, n, p, q, c, s);
        rotate_columns(v, n, n, p, q, c, s);
        true
    }

    /// 对 `mat` 的第 p、q 列施加旋转 `(c, s)`；`cols` 是行跨度。
    fn rotate_columns(
        mat: &mut [f64],
        rows: usize,
        cols: usize,
        p: usize,
        q: usize,
        c: f64,
        s: f64,
    ) {
        for row in 0..rows {
            let x = mat[row * cols + p];
            let y = mat[row * cols + q];
            mat[row * cols + p] = c * x - s * y;
            mat[row * cols + q] = s * x + c * y;
        }
    }

    /// 列长度即奇异值；随后把 U 的列归一化。
    fn singular_values(u: &mut [f64], m: usize, n: usize) -> Vec<f64> {
        let mut sigma = vec![0.0f64; n];
        for (j, value) in sigma.iter_mut().enumerate() {
            *value = column_norm(u, m, n, j);
        }
        for (j, &sigma) in sigma.iter().enumerate() {
            normalize_column(u, m, n, j, sigma);
        }
        sigma
    }

    /// 一列的欧氏长度。
    fn column_norm(u: &[f64], m: usize, n: usize, j: usize) -> f64 {
        let mut norm = 0.0;
        for row in 0..m {
            norm += u[row * n + j].powi(2);
        }
        norm.sqrt()
    }

    /// 列除以它的长度；零列保持为零。
    fn normalize_column(u: &mut [f64], m: usize, n: usize, j: usize, sigma: f64) {
        if sigma <= 0.0 {
            return;
        }
        for row in 0..m {
            u[row * n + j] /= sigma;
        }
    }

    /// 按奇异值降序重排 U、V 的列，只保留前 k 列。
    fn sorted_columns(
        sigma: &[f64],
        u: &[f64],
        v: &[f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
        let sorted_sigma = order.iter().map(|&j| sigma[j]).take(k).collect();
        let mut sorted_u = vec![0.0f64; m * k];
        let mut sorted_v = vec![0.0f64; n * k];
        for (new_col, &old_col) in order.iter().take(k).enumerate() {
            copy_column(u, m, n, &mut sorted_u, k, old_col, new_col);
            copy_column(v, n, n, &mut sorted_v, k, old_col, new_col);
        }
        (sorted_sigma, sorted_u, sorted_v)
    }

    /// 把 `src` 的第 `from` 列拷进 `dst` 的第 `to` 列。
    fn copy_column(
        src: &[f64],
        rows: usize,
        src_stride: usize,
        dst: &mut [f64],
        dst_stride: usize,
        from: usize,
        to: usize,
    ) {
        for row in 0..rows {
            dst[row * dst_stride + to] = src[row * src_stride + from];
        }
    }

    /// 由列 Gram 参数算 Jacobi 旋转系数 `(c, s)`。
    fn rotation(alpha: f64, beta: f64, gamma: f64) -> (f64, f64) {
        let tau = (beta - alpha) / (2.0 * gamma);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, c * t)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn frob_resid(a: &[f64], back: &[f64]) -> f64 {
            a.iter()
                .zip(back)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        #[test]
        fn diagonal_singular_values_are_exact() {
            // diag(3,2,1,0.5) 行主序。
            let mut a = vec![0.0f64; 16];
            let diag = [3.0, 2.0, 1.0, 0.5];
            for (i, d) in diag.iter().enumerate() {
                a[i * 4 + i] = *d;
            }
            let svd = compute(&a, 4, 4).expect("svd");
            for (got, want) in svd.sigma.iter().zip(diag.iter()) {
                assert!((got - want).abs() < 1e-10, "got {got} want {want}");
            }
        }

        #[test]
        fn full_reconstruction_residual_is_machine_order() {
            let mut rng = 0x1234_5678u64;
            let mut next = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 40) as f64 / (1u64 << 24) as f64 - 0.5
            };
            for (m, n) in [(8, 8), (12, 6), (6, 12), (20, 10)] {
                let a: Vec<f64> = (0..m * n).map(|_| next()).collect();
                let svd = compute(&a, m, n).expect("svd");
                let k = m.min(n);
                assert_eq!(svd.sigma.len(), k);
                let back = svd.reconstruct(k);
                let scale = svd.sigma[0];
                assert!(
                    frob_resid(&a, &back) < 1e-10 * scale * (m as f64).sqrt(),
                    "残差应在机器精度量级 m={m} n={n}"
                );
            }
        }

        #[test]
        fn truncated_reconstruction_is_monotonic() {
            let mut a = vec![0.0f64; 36];
            for i in 0..6 {
                a[i * 6 + i] = 6.0 - i as f64;
            }
            let svd = compute(&a, 6, 6).expect("svd");
            let mut prev = f64::INFINITY;
            for r in 1..=6 {
                let err = frob_resid(&a, &svd.reconstruct(r));
                assert!(err <= prev, "增加秩不应让误差变大");
                prev = err;
            }
        }

        #[test]
        fn singular_values_are_descending_and_nonnegative() {
            let mut rng = 99u64;
            let a: Vec<f64> = (0..64)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng >> 40) as f64 / (1u64 << 24) as f64
                })
                .collect();
            let svd = compute(&a, 8, 8).expect("svd");
            for pair in svd.sigma.windows(2) {
                assert!(pair[0] >= pair[1]);
                assert!(pair[0] >= 0.0);
            }
        }

        #[test]
        fn bad_shape_is_an_error() {
            assert!(compute(&[1.0], 2, 2).is_err());
            assert!(compute(&[], 0, 4).is_err());
        }

        /// 信号秩选择：结构化的秩边界有相对谱隙 → 秩 10；满秩噪声平坦谱 → 拒压；单一奇异值无从比较 → 拒压。
        #[test]
        fn signal_rank_separates_structure_from_noise() {
            let mut structured = vec![10.0f64; 10];
            structured.extend(vec![0.1f64; 6]);
            assert_eq!(signal_rank(&structured), Some(10));

            let noise: Vec<f64> = (0..16).map(|k| 10.0 - k as f64 * 0.1).collect();
            assert_eq!(signal_rank(&noise), None);

            assert_eq!(signal_rank(&[1.0]), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        I2, I3, I4, I8, adaptive_quantize, compensate, compensate_adaptive, dequantize, quantize,
    };

    /// 打包位序：i4 低 nibble 在前，`[7, -7]` → 码 `[15, 1]` → 字节 `0x1F`。
    #[test]
    fn i4_packs_low_nibble_first() {
        let (packed, scales) = quantize(I4, 32, &[7.0, -7.0]).expect("合法输入");
        assert_eq!(packed, vec![0x1F]);
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 2];
        dequantize(I4, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [7.0, -7.0]);
    }

    /// 打包位序：i2 每字节 4 码，`[1, -1, 0, 0]` → 码 `[3, 1, 2, 2]` → 字节 `0xA7`。
    #[test]
    fn i2_packs_four_codes_per_byte() {
        let (packed, scales) = quantize(I2, 32, &[1.0, -1.0, 0.0, 0.0]).expect("合法输入");
        assert_eq!(packed, vec![0xA7]);
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 4];
        dequantize(I2, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [1.0, -1.0, 0.0, 0.0]);
    }

    /// 打包位序：i3 每字节不整除，`[3, -3, 1, 0]` → 码 `[6, 0, 4, 3]` → 三个字节的位流。
    #[test]
    fn i3_packs_three_bits_per_code() {
        let (packed, scales) = quantize(I3, 32, &[3.0, -3.0, 1.0, 0.0]).expect("合法输入");
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 4];
        dequantize(I3, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [3.0, -3.0, 1.0, 0.0]);
        // 位宽不整除：4 码 × 3 bit = 12 bit → 2 字节。
        assert_eq!(packed.len(), 2);
        // 码 [6,0,4,3] 低 3 位在前：位置 0..2 码 6、3..5 码 0、6..8 码 4（跨字节）、9..11 码 3。
        assert_eq!(packed[0], 6, "码 6 在 0..2，码 4 的低 2 位 0 在 6..7");
        assert_eq!(
            packed[1],
            1 | (3 << 1),
            "码 4 的第 3 位在 byte1 位 0，码 3 在 1..3"
        );
    }

    /// 自适应分配按误差预算选位宽：极紧预算停在全 8bit 且不达标，极松预算一路降到 2bit。
    #[test]
    fn adaptive_quantize_assigns_bits_by_error_budget() {
        let x: Vec<f32> = (0..256).map(|i| ((i as f32).sin()) * 3.7).collect();
        let norm: f64 = x
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();

        let tight = adaptive_quantize(32, &x, 1e-9).expect("合法");
        assert!(tight.widths.iter().all(|&w| w == 8));
        assert!(!tight.met, "预算低于 8bit 可达误差，必须如实报未达标");

        let loose = adaptive_quantize(32, &x, norm * 2.0).expect("合法");
        assert!(
            loose.widths.iter().all(|&w| w == 2),
            "松预算应降到最低档：{:?}",
            loose.widths
        );
        assert!(loose.met);
        assert!((loose.bpw - 2.0).abs() < 1e-9);
    }

    /// i8 逐元素二补码：`[127, -127]` → 字节 `[127, 129]`，反解归位。
    #[test]
    fn i8_stores_twos_complement() {
        let (packed, scales) = quantize(I8, 32, &[127.0, -127.0]).expect("合法输入");
        assert_eq!(packed, vec![127, 129]);
        assert_eq!(scales, vec![1.0]);
        let mut out = [0.0; 2];
        dequantize(I8, &packed, &scales, 32, &mut out).expect("合法反解");
        assert_eq!(out, [127.0, -127.0]);
    }

    /// 量化误差有界：每元素误差不超过本组半码距（`scale / 2`），且余弦相似度随位宽提升。
    #[test]
    fn round_trip_error_is_bounded() {
        let x: Vec<f32> = (0..256)
            .map(|i| ((i as f32).sin() * 3.7) + (i as f32 % 7.0) - 3.0)
            .collect();
        // i2 只有 {-1,0,1} 三档码，余弦阈值放低；i4/i8 随位宽逐级收紧。
        for (scheme, min_cosine) in [(I2, 0.80), (I4, 0.98), (I8, 0.999)] {
            let (packed, scales) = quantize(scheme, 32, &x).expect("合法输入");
            let mut out = vec![0.0; x.len()];
            dequantize(scheme, &packed, &scales, 32, &mut out).expect("合法反解");

            let mut cosine_num = 0.0f64;
            let mut x2 = 0.0f64;
            let mut o2 = 0.0f64;
            for (a, b) in x.iter().zip(&out) {
                cosine_num += (*a as f64) * (*b as f64);
                x2 += (*a as f64) * (*a as f64);
                o2 += (*b as f64) * (*b as f64);
            }
            let cosine = cosine_num / (x2.sqrt() * o2.sqrt());
            assert!(
                cosine > min_cosine,
                "scheme {} 余弦 {cosine} 低于阈值 {min_cosine}",
                scheme.bits
            );

            // 误差界用本组 scale：round 到最近码，|x - scale*signed| <= scale/2。
            for (i, (a, b)) in x.iter().zip(&out).enumerate() {
                let group_scale = scales[i / 32];
                let err = (a - b).abs();
                assert!(
                    err <= group_scale / 2.0 + 1e-4,
                    "scheme {} 第 {i} 项误差 {err} 超过半码距 {}",
                    scheme.bits,
                    group_scale / 2.0
                );
            }
        }
    }

    /// 全零组不产生零 scale：scale 取 1，码全为对称零点（i4 为 8），反解精确归零。
    #[test]
    fn all_zero_group_is_invertible() {
        let x = [0.0f32; 64];
        let (packed, scales) = quantize(I4, 32, &x).expect("合法输入");
        assert_eq!(scales, vec![1.0, 1.0]);
        assert!(packed.iter().all(|&b| b == 0x88), "码应为零点 8");
        let mut out = [0.0; 64];
        dequantize(I4, &packed, &scales, 32, &mut out).expect("合法反解");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// 非法输入：零分组、非有限值、长度不匹配都显式报错，不静默。
    #[test]
    fn rejects_bad_inputs() {
        assert!(quantize(I4, 0, &[1.0]).is_err());
        assert!(quantize(I4, 32, &[f32::NAN]).is_err());
        assert!(quantize(I4, 32, &[f32::INFINITY]).is_err());

        let (packed, scales) = quantize(I4, 32, &[1.0, -1.0]).expect("合法输入");
        let mut out = [0.0; 3];
        assert!(dequantize(I4, &packed, &scales, 32, &mut out).is_err());
    }

    /// M4/E5 输出空间 SNR 扫描（`#[ignore]` 实测基准：
    /// `cargo test -p llm-weights --lib -- output_space_snr_scan --ignored --nocapture`）。
    ///
    /// 口径与 §10.10 M4/E5 卡一致：W4A16、g=32；SNR = 20·log10(‖Wx‖ / ‖Wx − Q(W)x‖)，
    /// 跨层取 P50。本测试只出数字、不设 ≥44 dB 断言——44 dB 是门禁而非通过条件；
    /// 当前 `quantize` 是无补偿的对称分组量化，必须如实报告它离门禁的距离，而不是伪造达标。
    #[test]
    #[ignore = "量化精度基准：cargo test -p llm-weights --lib -- output_space_snr_scan --ignored --nocapture"]
    fn output_space_snr_scan() {
        let (rows, cols, g) = (2048usize, 2048usize, 32usize);
        let layers = 4usize;

        // 确定性 LCG + Box–Muller 高斯权重（无外部 RNG 依赖，可复现）。
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next_f64 = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut next_gauss = move || {
            let u1 = next_f64().max(1e-12);
            let u2 = next_f64();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };

        for (scheme, name) in [(I2, "i2"), (I4, "i4"), (I8, "i8")] {
            let mut snrs = Vec::with_capacity(layers);
            for _ in 0..layers {
                let w: Vec<f32> = (0..rows * cols).map(|_| next_gauss()).collect();
                let x: Vec<f32> = (0..cols).map(|_| next_gauss()).collect();
                let (packed, scales) = quantize(scheme, g, &w).expect("合法输入");
                let mut qw = vec![0.0f32; rows * cols];
                dequantize(scheme, &packed, &scales, g, &mut qw).expect("合法反解");
                let (num, den) = output_snr(&w, &qw, &x, rows, cols);
                snrs.push(20.0 * (num / den).log10());
            }
            snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = snrs[layers / 2];
            println!(
                "M4/E5 SNR[{name}] g={g} layers={layers} rows={rows} cols={cols} P50={p50:.3} dB 样本={:?}",
                snrs.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
            );
        }
    }

    /// `y = Wx` 与 `‖y − Q(W)x‖` 的 f64 累积：SNR = 20·log10(‖y‖ / ‖y − ŷ‖)。
    fn output_snr(w: &[f32], qw: &[f32], x: &[f32], rows: usize, cols: usize) -> (f64, f64) {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in 0..rows {
            let (mut y, mut yq) = (0.0f64, 0.0f64);
            for c in 0..cols {
                let xv = x[c] as f64;
                y += w[r * cols + c] as f64 * xv;
                yq += qw[r * cols + c] as f64 * xv;
            }
            num += y * y;
            let d = y - yq;
            den += d * d;
        }
        (num.sqrt(), den.sqrt())
    }

    /// 确定性高斯矩阵（Box–Muller，无外部 RNG 依赖）。
    fn gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        let mut next_f64 = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        (0..rows * cols)
            .map(|_| {
                let u1 = next_f64().max(1e-12);
                let u2 = next_f64();
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect()
    }

    /// 结构近似低秩权重：`W = U·Vᵀ + σ_n·N`，`U∈m×r`、`V∈n×r`、`N` 高斯同形。
    fn low_rank_plus_noise(m: usize, n: usize, r: usize, sigma_n: f32, seed: u64) -> Vec<f32> {
        let u = gaussian(m, r, seed);
        let v = gaussian(n, r, seed ^ 0x9e37_79b9);
        let mut w = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for k in 0..r {
                    s += u[i * r + k] as f64 * v[j * r + k] as f64;
                }
                w[i * n + j] = s as f32;
            }
        }
        let noise = gaussian(m, n, seed ^ 0xdead_beef);
        for (slot, &nval) in w.iter_mut().zip(&noise) {
            *slot += sigma_n * nval;
        }
        w
    }

    /// M4 谱隙证书自动拒压 + 秩-10 残差补偿（`#[ignore]`：
    /// `cargo test -p llm-weights --lib -- m4_residual_compensation_snr --release --ignored --nocapture`）。
    ///
    /// 谱隙证书（`svd::signal_rank`，相对阈值 0.2）自动选秩：结构近似低秩权重在秩 10 处有谱隙
    /// → 补偿；满秩高斯无谱隙 → 拒压。σn∈{0.1,0.2,0.3} 扫描补偿 SNR 的 44 dB 分界。
    #[test]
    #[ignore = "M4 补偿基准：cargo test -p llm-weights --lib -- m4_residual_compensation_snr --release --ignored --nocapture"]
    fn m4_residual_compensation_snr() {
        let (m, n, r) = (512usize, 512usize, 10usize);
        for sigma_n in [0.1f32, 0.2, 0.3] {
            let w = low_rank_plus_noise(m, n, r, sigma_n, 0xfeed_0001);
            match compensate(I4, 32, &w, m, n).expect("补偿") {
                Some(comp) => {
                    let x = gaussian(n, 1, 0xace_0002);
                    let (num, den) = output_snr(&w, &comp.recon, &x, m, n);
                    let snr = 20.0 * (num / den).log10();
                    let frob = frob_snr(&w, &comp.recon);
                    println!(
                        "M4 结构化 σn={sigma_n} 补偿 秩={} 相对谱隙γ/σ_1={:.3} 输出SNR={snr:.3} dB FrobeniusSNR={frob:.3} dB",
                        comp.rank,
                        comp.gap / comp.sigma1,
                    );
                }
                None => println!("M4 结构化 σn={sigma_n} 拒压（无谱隙）"),
            }
        }

        let (m, n) = (256usize, 256usize);
        let w = gaussian(m, n, 0xcafe_0003);
        match compensate(I4, 32, &w, m, n).expect("补偿") {
            Some(comp) => println!(
                "M4 满秩高斯 补偿 秩={} 相对谱隙γ/σ_1={:.3e}（不应发生）",
                comp.rank,
                comp.gap / comp.sigma1,
            ),
            None => println!("M4 满秩高斯 拒压（无谱隙，不冒充 4bit）"),
        }
    }

    /// E5 2–8bit 自适应分配（`#[ignore]`：
    /// `cargo test -p llm-weights --lib -- e5_adaptive_allocation_snr --release --ignored --nocapture`）。
    ///
    /// 谱隙选秩 + 残差 `{2,3,4,8}` 自适应量化，目标 44 dB：σn 越大残差越难压，分配器自动多给
    /// 8bit 组，平均码位随 σn 上升——σn=0.3 也达到 ≥44 dB（代价是 bpw 从 4 升到 ~6）。
    #[test]
    #[ignore = "E5 自适应基准：cargo test -p llm-weights --lib -- e5_adaptive_allocation_snr --release --ignored --nocapture"]
    fn e5_adaptive_allocation_snr() {
        let (m, n, r) = (512usize, 512usize, 10usize);
        for sigma_n in [0.1f32, 0.2, 0.3] {
            let w = low_rank_plus_noise(m, n, r, sigma_n, 0xfeed_0001);
            // 目标取 46 dB：输出空间 SNR 对随机 x 有 ~±1–2 dB 的投影波动，Frobenius 46 dB
            // 给 44 dB 门禁留出裕量，保证配对输出 SNR 的下界也 ≥44。
            match compensate_adaptive(32, &w, m, n, 46.0).expect("自适应补偿") {
                Some(comp) => {
                    let mut snrs = Vec::new();
                    for seed in [0xace_0005u64, 0xace_0006, 0xace_0007] {
                        let x = gaussian(n, 1, seed);
                        let (num, den) = output_snr(&w, &comp.recon, &x, m, n);
                        snrs.push(20.0 * (num / den).log10());
                    }
                    snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let frob = frob_snr(&w, &comp.recon);
                    let (c2, c3, c4, c8) = width_histogram(&comp.widths);
                    println!(
                        "E5 结构化 σn={sigma_n} 自适应 秩={} bpw={:.3} met={} 位宽[2:{c2} 3:{c3} 4:{c4} 8:{c8}] 输出SNR(P50={:.3},min={:.3}) dB FrobeniusSNR={frob:.3} dB",
                        comp.rank, comp.bpw, comp.met, snrs[1], snrs[0],
                    );
                }
                None => println!("E5 结构化 σn={sigma_n} 拒压（无谱隙）"),
            }
        }
    }

    /// 位宽直方图：`(2bit 组数, 3bit 组数, 4bit 组数, 8bit 组数)`。
    fn width_histogram(widths: &[u8]) -> (usize, usize, usize, usize) {
        let count = |w: u8| widths.iter().filter(|&&x| x == w).count();
        (count(2), count(3), count(4), count(8))
    }

    /// Frobenius 口径 SNR：`20·log10(‖W‖_F / ‖W − Ŵ‖_F)`。
    fn frob_snr(w: &[f32], recon: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in w.iter().zip(recon) {
            let a = *a as f64;
            let d = a - *b as f64;
            num += a * a;
            den += d * d;
        }
        20.0 * (num.sqrt() / den.sqrt()).log10()
    }
}
