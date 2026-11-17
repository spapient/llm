//! 量化：磁盘方案描述符 + 权重/标度成对推断。
//!
//! - [`Quant`]：显式仿射方案（`x ≈ scale · q + zero`），分组、零点与标度类型都是文件里的事实；
//!   `axis == rank` 表示 per-tensor（单个标度，不沿任何真实轴），此时 `block` 必须为 0。
//! - [`Quantized`]：打包权重 + 每个分组一个标度因子的成对结构；分组由标度条目形状推出。
//!
//! 真实的 4 位量化 checkout 里，一个投影张量**从不成对出现**：权重按 4 位打包，另有一个标度条目
//! 存"每 `group` 个元素共享的缩放因子"。分组因此不是猜出来的，而是从标度条目的形状推出来的。
//!
//! `group = in / groups`。这条推断有两条硬约束，缺一条就不是"同一个东西"：
//!
//! 1. 打包权重必须是秩 2，标度必须是秩 2，且两者的输出维相等；
//! 2. 标度的第二维必须整除输入维（`in % groups == 0`），否则 `group` 不是整数。
//!
//! 曾经想要的第三条是"字节数必须等于 `out × ceil(in / 2)`"，但它在本层是**结构性自证**的：
//! [`Entry`] 的长度由 `shape × dtype` 推出，safetensors 又要求声明区间等于这个乘积，所以任何
//! 从 [`Header`] 构造出来的条目都自动满足它——在这里再比一次，等于拿一个值与自己比。
//!
//! 8 位条目到底是"逐元素 int8"还是"两个 nibble 一个字节"，由元素类型的**符号性**分流：
//! [`DType::I8`] 是逐元素有符号字节，第二维就是元素数；[`DType::U8`] 是无符号字节，一个字节
//! 两个 nibble，第二维是字节数、宽度翻倍。分组仍由标度条目形状推出，weights 不认识算子分组档。

use diagnostic::{Code, Error, Result};
use tensor::DType;

use crate::Wire;
use crate::leb::{self, Cursor};
use crate::sink::Sink;
use crate::{Entry, Header};

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

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("weights::quant")
}

// —— 权重/标度成对推断 ——

/// 一次量化推断的结果：打包权重、它的标度条目与推出的分组。
///
/// 三个字段都是借用：推断不复制字节，也不缓存结论——缓存会让"改了一份权重再读"变成隐式状态。
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
    ///
    /// **不要从 [`Quantized::weight`] 的形状里读这个数**：`U8` 打包条目的第二维是字节数（一个字节
    /// 两个 nibble），`I8` 逐元素条目的第二维就是元素数。宽度与分组必须由同一处给出，否则会出现
    /// "分组按 128 个元素算、宽度按 64 个元素算"的错配——那种错不会报错，只会把下界算成真值的一半。
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 输出宽度（元素数）：打包不改变行数。
    pub fn out_features(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    /// 每个元素的位数：逐元素 `I8` 为 8，打包的 `U8`/`U4` 为 4。
    pub fn bits(&self) -> u8 {
        match self.weight.dtype() {
            DType::I8 => 8,
            _ => 4,
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
///
/// 输入宽度不是直接读第二维：`U8` 打包条目的第二维是字节数（一个字节两个 nibble），`I8` 逐元素
/// 与 `U4` 解包视图的第二维才是元素数。这条换算不是约定，是"safetensors 要求 `shape × dtype` 的
/// 字节数等于声明区间"的直接后果：想用无符号字节表达 4 位打包，就只能把形状按字节写。
fn analyze<'a>(weight: &'a Entry, scales: &'a Entry) -> Result<Option<Quantized<'a>>> {
    let context = format!(
        "量化条目 `{}` 与它的标度 `{}`",
        weight.name(),
        scales.name()
    );
    let Some((second, groups)) = dims(weight, scales, &context)? else {
        return Ok(None);
    };
    let Some((in_features, group)) = pack(weight, second, groups, &context)? else {
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
///
/// 返回 `(权重第二维, 标度分组数)`；`None` 表示这是**另一种**方案（整数标度），不猜。
fn dims(weight: &Entry, scales: &Entry, context: &str) -> Result<Option<(usize, usize)>> {
    let [out, second] = rank2(weight, "打包权重必须是秩 2（out × in）", context)?;
    let [scaled_out, groups] = rank2(scales, "标度必须是秩 2（out × 分组数）", context)?;
    if scaled_out != out {
        return Err(invalid(format!(
            "标度的输出维 {scaled_out} 与权重的输出维 {out} 不等：分组无法对齐到每一行"
        ))
        .with(context));
    }
    let float = matches!(scales.dtype(), DType::F32 | DType::F16 | DType::BF16);
    Ok(float.then_some((second, groups)))
}

/// 一个条目的秩 2 形状；`expected` 说明要的是哪种秩 2（权重还是标度）。
fn rank2(entry: &Entry, expected: &str, context: &str) -> Result<[usize; 2]> {
    let dims = entry.shape().dims();
    match dims {
        [first, second] => Ok([*first, *second]),
        _ => Err(invalid(format!("{expected}，实际 {dims:?}")).with(context)),
    }
}

/// 分组：逻辑输入宽度必须被标度声明的分组数整除。`None` 表示元素类型不可能是 4 位量化。
fn pack(
    weight: &Entry,
    second: usize,
    groups: usize,
    context: &str,
) -> Result<Option<(usize, usize)>> {
    let Some(in_features) = width(weight, second, context)? else {
        return Ok(None);
    };
    if groups == 0 || !in_features.is_multiple_of(groups) {
        return Err(invalid(format!(
            "标度声明 {groups} 个分组，但输入宽度 {in_features} 不是它的整数倍"
        ))
        .with(context));
    }
    Ok(Some((in_features, in_features / groups)))
}

/// 每行的逻辑输入宽度：解包视图与逐元素 int8 的第二维就是元素数；无符号字节一个字节两个 nibble。
fn width(weight: &Entry, second: usize, context: &str) -> Result<Option<usize>> {
    match weight.dtype() {
        DType::U4 | DType::I8 => Ok(Some(second)),
        DType::U8 => second.checked_mul(2).map(Some).ok_or_else(|| {
            invalid(format!(
                "形状 {:?} 的第二维 {second} 乘 2 时溢出",
                weight.shape().dims()
            ))
            .with(context)
        }),
        _ => Ok(None),
    }
}

fn invalid(message: String) -> Error {
    Error::new(Code::Invalid, message).with("weights::quant")
}

#[cfg(test)]
mod scheme_tests {
    use super::*;

    fn round(quant: &Quant, rank: u32) -> Quant {
        let mut buf = Vec::new();
        quant.encode(&mut buf, rank).expect("编码合法");
        Quant::decode(&mut Cursor::new(&buf), rank).expect("解码合法")
    }

    #[test]
    fn per_block_and_per_tensor_schemes_round_trip() {
        let block = Quant::new(2, 1, 32, false, Wire::F16, None).expect("per-block");
        assert_eq!(round(&block, 2), block);

        let tensor = Quant::new(2, 2, 0, true, Wire::F32, Some(Wire::I8)).expect("per-tensor+zp");
        assert_eq!(round(&tensor, 2), tensor);
    }

    #[test]
    fn inconsistent_schemes_are_rejected() {
        assert!(
            Quant::new(2, 1, 0, false, Wire::F16, None).is_err(),
            "block=0 沿轴"
        );
        assert!(
            Quant::new(2, 2, 4, false, Wire::F16, None).is_err(),
            "per-tensor 带 block"
        );
        assert!(
            Quant::new(2, 1, 32, false, Wire::I8, None).is_err(),
            "整数标度"
        );
        assert!(
            Quant::new(2, 1, 32, true, Wire::F16, None).is_err(),
            "缺零点类型"
        );
        assert!(
            Quant::new(2, 1, 32, true, Wire::F16, Some(Wire::F32)).is_err(),
            "浮点零点"
        );
    }
}

#[cfg(test)]
mod pair_tests {
    use super::*;
    use tensor::Shape;

    fn header(entries: Vec<Entry>) -> Header {
        // 夹具按顺序把区间摆开：`Header::new` 要求区间不重叠，默认全 0 会让每个用例都失败在
        // 夹具而不是被测逻辑上。
        let mut offset = 0;
        let laid_out = entries
            .into_iter()
            .map(|entry| {
                let start = offset;
                offset += entry.len();
                entry.at(start)
            })
            .collect();
        Header::new(laid_out, 0).expect("测试夹具的区间合法")
    }

    /// 一个 4 位打包的条目：`[out, in]`，字节数是 `ceil(out × in / 2)`（位数组，按行不补齐）。
    fn packed(name: &str, out: usize, in_features: usize) -> Entry {
        let entry = Entry::new(name, DType::U4, Shape::new([out, in_features]));
        assert_eq!(
            entry.len(),
            (out * in_features).div_ceil(2),
            "夹具自证打包口径"
        );
        entry
    }

    fn scales(name: &str, out: usize, groups: usize, dtype: DType) -> Entry {
        Entry::new(name, dtype, Shape::new([out, groups]))
    }

    /// 真实文件的打包条目：`U8` 一个字节两个 nibble，第二维是字节数。
    fn nibbles(name: &str, out: usize, in_features: usize) -> Entry {
        Entry::new(name, DType::U8, Shape::new([out, in_features / 2]))
    }

    #[test]
    fn the_group_comes_from_the_scale_shape() {
        let header = header(vec![
            packed("blk.0.q.weight", 8, 128),
            scales("blk.0.q.scales", 8, 4, DType::F16),
        ]);
        let quantized = header
            .quantization("blk.0.q.weight")
            .expect("自洽")
            .expect("是量化");
        assert_eq!(quantized.group(), 32, "128 个输入 / 4 个分组");
        assert_eq!(quantized.bits(), 4);
        assert_eq!(quantized.scales().name(), "blk.0.q.scales");
        assert_eq!(header.scales_names().expect("自洽"), ["blk.0.q.scales"]);
    }

    #[test]
    fn all_four_naming_families_are_recognised() {
        for (weight, scale) in [
            ("a.weight", "a.weight_scale"),
            ("a.weight", "a.weight_scales"),
            ("a.weight", "a.scales"),
            ("a.weight", "a.scale"),
        ] {
            let header = header(vec![packed(weight, 4, 64), scales(scale, 4, 2, DType::F32)]);
            let quantized = header
                .quantization(weight)
                .expect("自洽")
                .unwrap_or_else(|| panic!("`{scale}` 应被认出"));
            assert_eq!(quantized.group(), 32, "`{scale}`");
        }
    }

    /// 字节数是结构性的：它由 `shape × dtype` 推出，本层不需要再校验一遍。钉成断言，免得
    /// 日后有人真去加一条"拿一个值与自己比"的检查，并误以为它在挡 int8 那种歧义。
    ///
    /// 宽度为奇数时口径是 `ceil(out × in / 2)`，不是 `out × ceil(in / 2)`：这里把 4 位当作一整块
    /// 紧贴的位数组，**按行不补齐**。按行补齐的导出格式会与这条不符——那正是另一种打包方案，
    /// 形状上看不出差异，只能由上层用分组与内核声明去拒。
    #[test]
    fn the_packed_byte_count_is_structural_rather_than_checked_here() {
        for (out, in_features) in [(8, 128), (8, 7), (3, 1)] {
            let entry = Entry::new("a.weight", DType::U4, Shape::new([out, in_features]));
            assert_eq!(
                entry.len(),
                (out * in_features).div_ceil(2),
                "{out} × {in_features} 个 4 位元素"
            );
        }
        // 同一个张量用 8 位书写时，字节数恰好是上面那个数（`in` 为偶数时两者一致）。
        assert_eq!(nibbles("a.weight", 8, 128).len(), 8 * 64);
    }

    #[test]
    fn an_entry_without_a_scale_partner_is_not_quantized() {
        let header = header(vec![packed("a.weight", 4, 64)]);
        assert_eq!(header.quantization("a.weight").expect("无伙伴"), None);
        assert!(header.scales_names().expect("自洽").is_empty());
        assert_eq!(
            header.quantization("absent").expect("不是条目"),
            None,
            "名字不在表里不是错误"
        );
    }

    #[test]
    fn a_disagreeing_output_dimension_is_an_error_not_a_guess() {
        let header = header(vec![
            packed("a.weight", 8, 64),
            scales("a.scales", 4, 2, DType::F16),
        ]);
        let error = header.quantization("a.weight").expect_err("输出维不等");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("输出维 4"), "{error}");
    }

    #[test]
    fn a_group_that_does_not_divide_the_input_is_an_error() {
        let header = header(vec![
            packed("a.weight", 8, 100),
            scales("a.scales", 8, 3, DType::F16),
        ]);
        let error = header.quantization("a.weight").expect_err("100 / 3");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("不是它的整数倍"), "{error}");
    }

    #[test]
    fn a_scale_that_is_not_rank_two_is_an_error() {
        let header = header(vec![
            packed("a.weight", 8, 64),
            Entry::new("a.scales", DType::F16, Shape::new([8])),
        ]);
        let error = header.quantization("a.weight").expect_err("秩 1");
        assert!(error.to_string().contains("秩 2"), "{error}");
    }

    #[test]
    fn the_real_file_shape_reports_the_same_group_as_the_unpacked_view() {
        // 同一层权重，两种书写：解包视图（U4，第二维 = 元素数）与真实文件（U8 打包，第二维 = 字节数）。
        let unpacked = header(vec![
            packed("a.weight", 8, 128),
            scales("a.scales", 8, 4, DType::F16),
        ]);
        let packed_bytes = header(vec![
            nibbles("a.weight", 8, 128),
            scales("a.scales", 8, 4, DType::F16),
        ]);
        let from_unpacked = unpacked
            .quantization("a.weight")
            .expect("自洽")
            .expect("量化");
        let from_file = packed_bytes
            .quantization("a.weight")
            .expect("自洽")
            .expect("量化");
        assert_eq!(from_unpacked.group(), 32);
        assert_eq!(from_file.group(), 32, "U8 条目的第二维是字节数");
        assert_eq!(from_file.in_features(), 128, "宽度是元素数，不是字节数");
        assert_eq!(from_file.out_features(), 8);
        assert_eq!(from_unpacked.in_features(), from_file.in_features());
    }

    #[test]
    fn a_signed_byte_entry_is_one_element_per_byte() {
        // 逐元素 int8：第二维就是元素数，宽度不翻倍，分组按元素数推。
        let weight = Entry::new("a.weight", DType::I8, Shape::new([8, 64]));
        let header = header(vec![weight, scales("a.scales", 8, 2, DType::F16)]);
        let quantized = header
            .quantization("a.weight")
            .expect("自洽")
            .expect("逐元素 int8");
        assert_eq!(quantized.bits(), 8);
        assert_eq!(quantized.in_features(), 64);
        assert_eq!(quantized.group(), 32, "64 元素 / 2 组 = 32");
    }

    #[test]
    fn an_unsigned_byte_entry_packs_two_nibbles_per_byte() {
        // 无符号字节打包：一个字节两个 nibble，宽度按字节数翻倍。
        let weight = Entry::new("a.weight", DType::U8, Shape::new([8, 32]));
        let header = header(vec![weight, scales("a.scales", 8, 2, DType::F16)]);
        let quantized = header
            .quantization("a.weight")
            .expect("自洽")
            .expect("打包 nibble");
        assert_eq!(quantized.bits(), 4);
        assert_eq!(quantized.in_features(), 64, "32 字节 ×2 = 64 个 nibble");
        assert_eq!(quantized.group(), 32, "64 元素 / 2 组 = 32");
    }

    #[test]
    fn a_wide_element_type_is_not_four_bit_quantization() {
        let header = header(vec![
            Entry::new("a.weight", DType::F16, Shape::new([8, 128])),
            scales("a.scales", 8, 4, DType::F16),
        ]);
        assert_eq!(header.quantization("a.weight").expect("自洽"), None);
    }

    #[test]
    fn an_integer_scale_is_another_scheme_rather_than_a_packed_four_bit_one() {
        let header = header(vec![
            packed("a.weight", 8, 64),
            scales("a.scales", 8, 2, DType::I8),
        ]);
        assert_eq!(header.quantization("a.weight").expect("自洽"), None);
    }

    #[test]
    fn the_same_entry_is_never_reported_as_two_layers() {
        // `scales_names` 与 `quantization` 用同一条规则，所以标度必然能被精确排除。
        let header = header(vec![
            packed("x.weight", 4, 32),
            scales("x.weight_scale", 4, 1, DType::F16),
            packed("y.weight", 4, 32),
            scales("y.scale", 4, 1, DType::F16),
        ]);
        let mut names = header.scales_names().expect("自洽");
        names.sort_unstable();
        assert_eq!(names, ["x.weight_scale", "y.scale"]);
    }
}
