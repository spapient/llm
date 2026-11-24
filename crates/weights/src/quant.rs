//! 量化：磁盘方案描述符 + 权重/标度成对推断。

use std::ops::Range;

use crate::bad;
use crate::codec;
use crate::svd;
use diagnostic::{Code, Error, Result};
use tensor::DType;

use crate::Wire;
use crate::{Entry, Header};
use storage::leb::Cursor;
use storage::sink::Sink;

// —— 仿射量化方案（磁盘描述符） ——

/// 仿射量化方案。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Quant {
    /// 分组轴；等于张量 rank 时是 per-tensor。
    pub axis: u32,
    /// 组宽（元素数）；per-tensor 时为 0。
    pub block: u32,
    /// 是否带零点。
    pub zero: bool,
    /// 标度的元素类型（浮点）。
    pub scale_ty: Wire,
    /// 零点的元素类型（整数）；无零点时为 `None`。
    pub zp_ty: Option<Wire>,
}

impl Quant {
    /// 构造并校验一个方案；`rank` 是所属张量的轴数，用来判定 per-tensor 与分组口径。
    pub fn new(
        rank: u32,
        axis: u32,
        block: u32,
        zero: bool,
        scale_ty: Wire,
        zp_ty: Option<Wire>,
    ) -> Result<Self> {
        let quant = Self {
            axis,
            block,
            zero,
            scale_ty,
            zp_ty,
        };
        quant.validate_layout(rank)?;
        quant.validate()?;
        Ok(quant)
    }

    /// 布局校验：轴不超 rank，per-tensor 的 block 为 0，沿轴分组的 block 非 0。
    pub fn validate_layout(&self, rank: u32) -> Result<()> {
        if self.axis > rank {
            return Err(bad("量化轴超出张量 rank"));
        }
        match (self.is_per_tensor(rank), self.block == 0) {
            (true, false) => Err(bad("per-tensor 方案的 block 必须为 0")),
            (false, true) => Err(bad("沿轴分组的方案 block 不能为 0")),
            _ => Ok(()),
        }
    }

    /// 结构校验：标度必须浮点；零点声明与类型必须配套，且类型必须是整数。
    pub fn validate(&self) -> Result<()> {
        if !self.scale_ty.is_float() {
            return Err(bad("量化标度必须是浮点类型"));
        }
        match (self.zero, self.zp_ty.map(Wire::is_integer)) {
            (true, None) => Err(bad("声明零点却没给零点类型")),
            (false, Some(_)) => Err(bad("无零点方案不该带零点类型")),
            (true, Some(false)) => Err(bad("零点必须是整数类型")),
            _ => Ok(()),
        }
    }

    /// 是否为 per-tensor（需要结合 rank 判断）。
    pub fn is_per_tensor(&self, rank: u32) -> bool {
        self.axis == rank
    }

    /// 编码进任意 Sink（不含 present 标志，那一位由段描述符写）。
    pub(crate) fn encode(&self, out: &mut impl Sink, rank: u32) -> Result<()> {
        self.validate_layout(rank)?;
        self.validate()?;
        out.put(&[0]); // scheme = affine
        self.encode_fields(out);
        Ok(())
    }

    /// 从游标解码。
    pub(crate) fn decode(cur: &mut Cursor<'_>, rank: u32) -> Result<Self> {
        read_scheme(cur)?;
        let (axis, block, zero, scale_ty, zp_ty) = Self::decode_fields(cur)?;
        Self::new(rank, axis, block, zero, scale_ty, zp_ty)
    }
}

// 线上字段的唯一声明：编码与解码共用字段顺序、条件尾部与 flags 掩码。
wire_fields! {
    Quant {
        axis: leb_u32,
        block: leb_u32,
        zero: bool_flag<0x01>,
        scale_ty: wire_type,
        zp_ty: wire_type if zero,
    }
}

/// 首字节判据：当前只认 affine(0)，其余按 unsupported 报错（`scheme` 写进文案）。
fn read_scheme(cur: &mut Cursor<'_>) -> Result<()> {
    let scheme = cur.read_u8().map_err(|e| e.with("量化方案"))?;
    if scheme == 0 {
        return Ok(());
    }
    let message = format!("未知量化 scheme {scheme}，当前只支持 affine(0)");
    Err(Error::new(Code::Unsupported, message).with("weights::quant"))
}

// —— 权重/标度成对推断 ——

/// 一次量化推断的结果：打包权重、它的标度条目与推出的分组。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantized<'a> {
    weight: &'a Entry,
    scales: &'a Entry,
    in_features: usize,
    group: usize,
}

impl<'a> Quantized<'a> {
    /// 打包后的权重条目。
    pub fn weight(&self) -> &'a Entry {
        self.weight
    }

    /// 标度条目（每个分组一个缩放因子）。
    pub fn scales(&self) -> &'a Entry {
        self.scales
    }

    /// 分组宽度：多少个元素共享一个缩放因子。
    pub fn group(&self) -> usize {
        self.group
    }

    /// 逻辑输入宽度（元素数）。
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 输出宽度（元素数）：打包不改变行数。
    pub fn out_features(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    /// 每个元素的量化位数：打包容器（`u8` 一个字节两个 nibble、`u4` 一格半字节）是 4 位，
    /// 逐元素容器是它自己的位宽（`i8` → 8）。
    ///
    /// 容器位宽不等于量化位数（`u8` 容器装两组 4 位），所以这里不能直接从 `DType::bits` 取。
    /// 定义域收窄到 `U4` / `U8` / `I8`；兜底分支答的是"逐元素位宽"，
    /// 将来多一个逐元素容器时得到的是真值，不会被静默报成 4 位。
    pub fn bits(&self) -> u8 {
        match self.weight.dtype() {
            DType::U8 | DType::U4 => 4,
            other => other.bits() as u8,
        }
    }

    /// 标度条目的名字。
    pub fn scales_name(&self) -> &'a str {
        self.scales.name()
    }
}

impl Header {
    /// 推断一个条目是不是量化权重的打包部分，并给出它的分组。
    ///
    /// 名字规则按"越具体越先"匹配：`{name}_scale`、`{name}_scales`、`{base}.scales`、
    /// `{base}.scale`（`base` = 去掉末尾 `.weight`）。这四种覆盖了目前见得到的导出格式
    /// （自制的 `_scale` 后缀、optimum 与多数转换脚本的 `.scales`）；不在其中的命名不会被误认。
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`]：条目存在标度伙伴，但两个条目的形状自相矛盾（见模块文档的表格）。
    /// 这类文件读下去只会得到"每个数都错"的下界，所以在这里失败。
    pub fn quantization(&self, name: &str) -> Result<Option<Quantized<'_>>> {
        let Some(weight) = self.get(name) else {
            return Ok(None);
        };
        let Some(scales) = scales_candidates(name)
            .into_iter()
            .find_map(|candidate| self.get(&candidate))
        else {
            return Ok(None);
        };
        analyze(weight, scales)
    }

    /// 全部标度条目的名字：`nn` 用它把标度从"未覆盖条目"里排除。
    ///
    /// 每个条目都有自己的标度伙伴，所以这是 `O(n²)` 次名字查找；条目数量级是几千，成本可以忽略，
    /// 换来的是"没有第二份命名规则"——排除用的判断与推断分组用的是同一个 [`Header::quantization`]。
    ///
    /// # Errors
    ///
    /// 与 [`Header::quantization`] 同一口径：任何一对条目形状矛盾都会在这里失败，而不是被
    /// "排除清单"顺手吞掉。
    pub fn scales_names(&self) -> Result<Vec<&str>> {
        let mut names = Vec::new();
        for entry in self.entries() {
            if let Some(quantized) = self.quantization(entry.name())? {
                names.push(quantized.scales_name());
            }
        }
        Ok(names)
    }
}

/// 标度条目名的候选，按"越具体越先"排序，并去掉重复。
fn scales_candidates(name: &str) -> Vec<String> {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let mut candidates = vec![
        format!("{name}_scale"),
        format!("{name}_scales"),
        format!("{base}.scales"),
        format!("{base}.scale"),
    ];
    candidates.dedup();
    candidates
}

/// 由一对条目推出分组：形状自相矛盾就报错，不是 4 位量化就返回 `None`。
fn analyze<'a>(weight: &'a Entry, scales: &'a Entry) -> Result<Option<Quantized<'a>>> {
    let ctx = format!(
        "量化条目 `{}` 与它的标度 `{}`",
        weight.name(),
        scales.name()
    );
    let Some((second, groups)) = dims(weight, scales, &ctx)? else {
        return Ok(None);
    };
    let Some((in_features, group)) = pack(weight, second, groups, &ctx)? else {
        return Ok(None);
    };
    Ok(Some(Quantized {
        weight,
        scales,
        in_features,
        group,
    }))
}

/// 形状与类型口径：两个条目都必须是秩 2、输出维相等，标度必须是浮点。
fn dims(weight: &Entry, scales: &Entry, ctx: &str) -> Result<Option<(usize, usize)>> {
    let [out, second] = rank2(weight, "打包权重必须是秩 2（out × in）", ctx)?;
    let [scaled_out, groups] = rank2(scales, "标度必须是秩 2（out × 分组数）", ctx)?;
    if scaled_out != out {
        return Err(bad(format!(
            "标度的输出维 {scaled_out} 与权重的输出维 {out} 不等：分组无法对齐到每一行"
        ))
        .with(ctx));
    }
    let float = scales.dtype().is_float();
    Ok(float.then_some((second, groups)))
}

/// 一个条目的秩 2 形状；`expected` 说明要的是哪种秩 2（权重还是标度）。
fn rank2(entry: &Entry, expected: &str, ctx: &str) -> Result<[usize; 2]> {
    let dims = entry.shape().dims();
    match dims {
        [first, second] => Ok([*first, *second]),
        _ => Err(bad(format!("{expected}，实际 {dims:?}")).with(ctx)),
    }
}

/// 分组：逻辑输入宽度必须被标度声明的分组数整除。`None` 表示元素类型不可能是 4 位量化。
fn pack(weight: &Entry, second: usize, groups: usize, ctx: &str) -> Result<Option<(usize, usize)>> {
    let Some(in_features) = width(weight, second, ctx)? else {
        return Ok(None);
    };
    if groups == 0 || !in_features.is_multiple_of(groups) {
        return Err(bad(format!(
            "标度声明 {groups} 个分组，但输入宽度 {in_features} 不是它的整数倍"
        ))
        .with(ctx));
    }
    Ok(Some((in_features, in_features / groups)))
}

/// 每行的逻辑输入宽度：解包视图与逐元素 int8 的第二维就是元素数；无符号字节一个字节两个 nibble。
fn width(weight: &Entry, second: usize, ctx: &str) -> Result<Option<usize>> {
    match weight.dtype() {
        DType::U4 | DType::I8 => Ok(Some(second)),
        DType::U8 => second.checked_mul(2).map(Some).ok_or_else(|| {
            bad(format!(
                "形状 {:?} 的第二维 {second} 乘 2 时溢出",
                weight.shape().dims()
            ))
            .with(ctx)
        }),
        _ => Ok(None),
    }
}

// —— 2/3/4/8-bit 对称训练后量化：f32 → 分组打包码与标度，以及它的反解 ——

/// 一档对称量化格式的运行时事实。
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
    let mut codes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(groups);

    for g in 0..groups {
        let slice = &x[span(group, x.len(), g)];
        let scale = group_scale(scheme, slice);
        scales.push(scale);
        for &v in slice {
            let signed = (v / scale).round() as i32;
            let signed = signed.clamp(-(scheme.max_code as i32), scheme.max_code as i32);
            codes.push((signed + scheme.zero) as u8 as u32);
        }
    }

    // 码位单调递增，交给唯一的流式位打包器：一次顺序写，不再逐元素算 `idx*width/8`。
    Ok((codec::pack_bits(&codes, scheme.bits), scales))
}

/// 量化参数校验：分组、正码上限、输入有限性。
fn validate_quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<()> {
    if group == 0 {
        return Err(bad("量化分组必须大于 0"));
    }
    if scheme.max_code <= 0.0 {
        return Err(bad("正码上限必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(bad("输入必须是有限 f32"));
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

    // 一次流式解包出全部码（与写端同一个位编解码器），再按组取 scale：`scales[g]` 在整组内
    // 不变，按组切片后每组只取一次 scale，逐元素不再算 `idx*width/8` 的位偏移除法。
    let codes = codec::unpack_bits(packed, out.len(), scheme.bits)?;
    let len = out.len();
    // 标度条数已校验为 `ceil(len/group)`，每组区间必非空，无需再判空跳出。
    for (g, &scale) in scales.iter().enumerate() {
        let range = span(group, len, g);
        let lo = range.start;
        for (offset, slot) in out[range].iter_mut().enumerate() {
            *slot = scale * decode_code(scheme, codes[lo + offset] as u8);
        }
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
        return Err(bad("反解分组必须大于 0"));
    }
    if packed.len() != scheme.packed_len(out.len()) {
        return Err(bad("打包字节数与输出长度不一致"));
    }
    if scales.len() != out.len().div_ceil(group) {
        return Err(bad("标度条数与分组数不一致"));
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

/// 第 `g` 组在 `len` 个元素里的区间；末组可能短于 `group`。常规量化与自适应档共用一份。
pub(crate) fn span(group: usize, len: usize, g: usize) -> Range<usize> {
    let lo = g * group;
    lo..(lo + group).min(len)
}

/// 秩-r 低秩 + 残差量化补偿的结果：`Ŵ = L_r + Q(W − L_r)`。
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
pub(crate) struct Components {
    pub(crate) rank: usize,
    pub(crate) sigma1: f64,
    pub(crate) gap: f64,
    pub(crate) low: Vec<f32>,
    pub(crate) rest: Vec<f32>,
    pub(crate) norm: f64,
}

/// 一次 f64 SVD → 秩 / 低秩项 `L_r` / 显式残差 `C = W − L_r`；无谱隙即拒压。
pub(crate) fn decompose(w: &[f32], m: usize, n: usize) -> Result<Option<Components>> {
    let s = svd::Spectral::open(w, m, n)?;
    let Some(rank) = svd::signal_rank(&s.svd.sigma) else {
        return Ok(None);
    };
    let rows = s.svd.reconstruct(rank);
    let low: Vec<f32> = rows.iter().map(|&x| x as f32).collect();
    let rest: Vec<f32> =
        s.a.iter()
            .zip(&rows)
            .map(|(orig, l)| (orig - l) as f32)
            .collect();
    Ok(Some(Components {
        rank,
        sigma1: s.svd.sigma[0],
        gap: svd::spectral_gap(&s.svd.sigma, rank),
        low,
        rest,
        norm: s.norm,
    }))
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
pub(crate) fn validate_compensate(m: usize, n: usize, w: &[f32]) -> Result<()> {
    let cells = m
        .checked_mul(n)
        .filter(|&cells| cells > 0 && cells == w.len());
    if cells.is_none() {
        return Err(bad(format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    if !w.iter().all(|v| v.is_finite()) {
        return Err(bad("输入必须是有限 f32"));
    }
    Ok(())
}

// —— 2–8bit 自适应组量化：误差预算驱动贪心降档，以及谱隙选秩后的自适应残差补偿 ——

/// 候选档位，位宽从低到高（`Scheme::bits` 本身就是位宽，不再并行维护一张位宽表）。
const LADDER: [Scheme; 4] = [I2, I3, I4, I8];

/// 起始档位：所有组都从最宽的 `I8` 起，降到哪由预算决定。
const TOP: usize = 3;

/// 2–8bit 自适应分配的结果：每组的位宽、平均码位、是否达标、反解重构。
#[derive(Clone, Debug)]
pub struct Adaptive {
    /// 每组位宽（2 / 3 / 4 / 8，元素位宽而非档位下标）。
    pub widths: Vec<u8>,
    /// 平均码位（只计码本身，不含组标度与位宽表）。
    pub bpw: f64,
    /// 总误差是否落在目标误差预算内。
    pub met: bool,
    /// 反解重构 `Q(x)`。
    pub recon: Vec<f32>,
}

/// 2–8bit 自适应组量化：按误差预算给每组从 `{2,3,4,8}` 里挑最小位宽。
///
/// 判据是**误差驱动的贪心**：所有组先从 8bit 起，逐步把「每省 1 bit 增误最少」的组降一档
/// （8→4→3→2），直到再降就超出 `target_err`（Frobenius）为止。得到的位置在给定误差预算下近似
/// 最少码位，覆盖 2–8bit 候选（E5 的自适应策略口径）。
///
/// 返回 `recon` 是逐组用各自位宽量化再反解的 `Q(x)`；`met` 表示总误差是否落在预算内。
///
/// # Errors
///
/// 分组为 0、输入含非有限值、目标误差为负时返回说明。
pub fn adaptive_quantize(group: usize, x: &[f32], target_err: f64) -> Result<Adaptive> {
    check_budget(group, x, target_err)?;
    let groups = x.len().div_ceil(group);
    let maxes = maxima(group, x, groups);
    let target2 = target_err * target_err;
    // 内部用档位下标（0=I2 .. 3=I8）；对外只暴露真实位宽。
    let mut levels = vec![TOP; groups];
    let mut total = total_err2(&maxes, group, x.len(), &levels);
    descend(&maxes, group, x.len(), &mut levels, &mut total, target2);
    Ok(Adaptive {
        widths: levels.iter().map(|&level| LADDER[level].bits).collect(),
        bpw: average_bits(group, x.len(), &levels),
        met: total <= target2,
        recon: reconstruct(group, x, &levels)?,
    })
}

/// 自适应补偿的结果：谱隙选秩 + 低秩项 + 残差的 2–8bit 自适应量化。
#[derive(Clone, Debug)]
pub struct AdaptiveCompensated {
    /// 谱隙证书选出的秩。
    pub rank: usize,
    /// 最大奇异值。
    pub sigma1: f64,
    /// 秩 r 处的谱隙。
    pub gap: f64,
    /// 残差平均码位（只计码本身）。
    pub bpw: f64,
    /// 总误差是否落在目标误差预算内。
    pub met: bool,
    /// 每组位宽。
    pub widths: Vec<u8>,
    /// 重构 `Ŵ = L_r + Q_adaptive(C)`。
    pub recon: Vec<f32>,
}

/// 谱隙证书 + 2–8bit 自适应残差补偿：把 4bit 固定档换成按误差预算分配 `{2,3,4,8}`。
///
/// 与 [`compensate`] 同一条低秩分解（`signal_rank` 选秩、无谱隙即拒压）；差别只在残差 `C`
/// 用 [`adaptive_quantize`] 量化，目标误差按 `target_snr_db` 从 `‖W‖_F` 推出。输出 SNR 的实测口径
/// 见调用方（Frobenius 口径对随机输入是输出空间 SNR 的下界，实测通常再高约 2 dB）。
///
/// 返回 `Ok(None)` 表示拒压（无谱隙）。
///
/// # Errors
///
/// 形状非法、输入含非有限值、`target_snr_db` 非有限，或 SVD 不收敛时返回说明。
pub fn compensate_adaptive(
    group: usize,
    w: &[f32],
    m: usize,
    n: usize,
    target_snr_db: f64,
) -> Result<Option<AdaptiveCompensated>> {
    validate(m, n, w, target_snr_db)?;
    let Some(parts) = decompose(w, m, n)? else {
        return Ok(None);
    };
    let adaptive = adaptive_for(group, &parts, target_snr_db)?;
    Ok(Some(assemble(parts, adaptive)))
}

/// 入口两项校验：矩阵几何与目标信噪比。
fn validate(m: usize, n: usize, w: &[f32], target_snr_db: f64) -> Result<()> {
    validate_compensate(m, n, w)?;
    check_target(target_snr_db)
}

/// 按目标信噪比推出误差预算并量化残差。
fn adaptive_for(group: usize, parts: &Components, target_snr_db: f64) -> Result<Adaptive> {
    let target_err = parts.norm * 10f64.powf(-target_snr_db / 20.0);
    adaptive_quantize(group, &parts.rest, target_err)
}

/// 把分解结果与量化结果合成一份反馈记录：低秩部分逐元素加回量化残差。
fn assemble(parts: Components, adaptive: Adaptive) -> AdaptiveCompensated {
    AdaptiveCompensated {
        rank: parts.rank,
        sigma1: parts.sigma1,
        gap: parts.gap,
        bpw: adaptive.bpw,
        met: adaptive.met,
        widths: adaptive.widths,
        recon: parts
            .low
            .iter()
            .zip(&adaptive.recon)
            .map(|(l, q)| l + q)
            .collect(),
    }
}

/// 入口三项契约：分组非零、输入有限、预算非负。
fn check_budget(group: usize, x: &[f32], target_err: f64) -> Result<()> {
    if group == 0 {
        return Err(bad("量化分组必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(bad("输入必须是有限 f32"));
    }
    if target_err.is_nan() || target_err < 0.0 {
        return Err(bad("目标误差必须非负"));
    }
    Ok(())
}

/// 目标 SNR 必须有限：`+inf` 会推出零预算、`NaN` 会推出 `NaN` 预算，两者都等于把「没给目标」
/// 当成「零误差」，而真实结果只会在别处以更难查的方式暴露。
fn check_target(target_snr_db: f64) -> Result<()> {
    if target_snr_db.is_finite() {
        return Ok(());
    }
    Err(bad("目标 SNR 必须有限"))
}

/// 每组最大绝对值：量化 `scale` 的分母只由它决定，与位宽无关，因此全档位共用一份。
fn maxima(group: usize, x: &[f32], groups: usize) -> Vec<f32> {
    (0..groups)
        .map(|g| {
            let range = span(group, x.len(), g);
            x[range].iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
        })
        .collect()
}

/// 第 `g` 组在档位 `level` 下的误差平方。
fn group_err2(maxes: &[f32], group: usize, len: usize, g: usize, level: usize) -> f64 {
    let size = span(group, len, g).count() as f64;
    let scale = f64::from(maxes[g]) / f64::from(LADDER[level].max_code);
    size * scale * scale / 12.0
}

/// 给定档位图的总误差平方（Frobenius 口径）。
fn total_err2(maxes: &[f32], group: usize, len: usize, levels: &[usize]) -> f64 {
    levels
        .iter()
        .enumerate()
        .map(|(g, &level)| group_err2(maxes, group, len, g, level))
        .sum()
}

/// 贪心降档：每轮降一档「每省 1 bit 增误最少」且仍不超预算的组，直到没有组还能降。
fn descend(
    maxes: &[f32],
    group: usize,
    len: usize,
    levels: &mut [usize],
    total: &mut f64,
    target2: f64,
) {
    while let Some((g, delta)) = cheapest(maxes, group, len, levels, *total, target2) {
        levels[g] -= 1;
        *total += delta;
    }
}

/// 这一轮最值得降的那一组：`(组号, 增误)`；没有组能在预算内降档时 `None`。
fn cheapest(
    maxes: &[f32],
    group: usize,
    len: usize,
    levels: &[usize],
    total: f64,
    target2: f64,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for (g, &from) in levels.iter().enumerate() {
        let Some((delta, cost)) = rung(maxes, group, len, g, from) else {
            continue;
        };
        if total + delta > target2 {
            continue;
        }
        if best.is_none_or(|(_, seen, _)| cost < seen) {
            best = Some((g, cost, delta));
        }
    }
    best.map(|(g, _, delta)| (g, delta))
}

/// 一组降一档的 `(增误, 每省 1 bit 的增误)`；已经到最窄档时 `None`。
fn rung(maxes: &[f32], group: usize, len: usize, g: usize, from: usize) -> Option<(f64, f64)> {
    let to = from.checked_sub(1)?;
    let delta = group_err2(maxes, group, len, g, to) - group_err2(maxes, group, len, g, from);
    let saved = f64::from(LADDER[from].bits - LADDER[to].bits);
    Some((delta, delta / saved))
}

/// 逐组按选定位宽量化再反解，拼出重构。
fn reconstruct(group: usize, x: &[f32], levels: &[usize]) -> Result<Vec<f32>> {
    let mut recon = vec![0.0f32; x.len()];
    for (g, &level) in levels.iter().enumerate() {
        let scheme = LADDER[level];
        let range = span(group, x.len(), g);
        let (packed, scales) = quantize(scheme, group, &x[range.clone()])?;
        dequantize(scheme, &packed, &scales, group, &mut recon[range])?;
    }
    Ok(recon)
}

/// 平均码位：只计码本身，不含组标度与位宽表（那两项按组摊薄，与逐元素码位不是一个量纲）。
fn average_bits(group: usize, len: usize, levels: &[usize]) -> f64 {
    let total: usize = levels
        .iter()
        .enumerate()
        .map(|(g, &level)| LADDER[level].bits as usize * span(group, len, g).count())
        .sum();
    total as f64 / len as f64
}

#[cfg(test)]
#[path = "../tests/ptq.rs"]
mod ptq_tests;

#[cfg(test)]
#[path = "../tests/adaptive.rs"]
mod adaptive_tests;

#[cfg(test)]
#[path = "../tests/quant.rs"]
mod tests;
