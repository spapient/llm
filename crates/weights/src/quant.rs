//! 量化：磁盘方案描述符 + 权重/标度成对推断。

use crate::bad;
use diagnostic::{Code, Error, Result};
use tensor::DType;

use crate::Wire;
use crate::{Entry, Header};
use storage::leb::{self, Cursor};
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
        leb::write(out, self.axis);
        leb::write(out, self.block);
        out.put(&[if self.zero { 1 } else { 0 }]);
        leb::write(out, self.scale_ty.code());
        if let Some(zp_ty) = self.zp_ty {
            leb::write(out, zp_ty.code());
        }
        Ok(())
    }

    /// 从游标解码。
    pub(crate) fn decode(cur: &mut Cursor<'_>, rank: u32) -> Result<Self> {
        read_scheme(cur)?;
        let (axis, block, zero) = read_head(cur)?;
        let (scale_ty, zp_ty) = read_types(cur, zero)?;
        Self::new(rank, axis, block, zero, scale_ty, zp_ty)
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

/// 轴、组宽与 flags；flags 只允许最低位（零点开关），其余位是预留。
fn read_head(cur: &mut Cursor<'_>) -> Result<(u32, u32, bool)> {
    let axis = cur.read::<u32>()?;
    let block = cur.read::<u32>()?;
    let flags = cur.read_u8()?;
    if flags & !1 != 0 {
        return Err(bad("量化 flags 含非零预留位"));
    }
    Ok((axis, block, flags & 1 == 1))
}

/// 标度类型；有零点时紧随其后读零点类型，无零点则不占字节。
fn read_types(cur: &mut Cursor<'_>, zero: bool) -> Result<(Wire, Option<Wire>)> {
    let scale_ty = crate::read_wire(cur)?;
    let zp_ty = if zero {
        Some(crate::read_wire(cur)?)
    } else {
        None
    };
    Ok((scale_ty, zp_ty))
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

#[cfg(test)]
#[path = "../tests/quant.rs"]
mod tests;
