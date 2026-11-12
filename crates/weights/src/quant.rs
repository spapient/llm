//! 量化权重的成对结构：打包权重 + 每个分组一个标度因子。
//!
//! 真实的 4 位量化 checkout 里，一个投影张量**从不成对出现**：权重按 4 位打包，另有一个标度条目
//! 存"每 `group` 个元素共享的缩放因子"。分组因此不是猜出来的，而是从标度条目的形状推出来的：
//!
//! ```
//! # use llm_weights::Entry;
//! # use tensor::{DType, Shape};
//! // q_proj.weight  [out, in]  U4（4 位打包）   +  q_proj.scales [out, in / group]  F16
//! ```
//!
//! `group = in / groups`。这条推断有两条硬约束，缺一条就不是"同一个东西"：
//!
//! 1. 打包权重必须是秩 2，标度必须是秩 2，且两者的输出维相等；
//! 2. 标度的第二维必须整除输入维（`in % groups == 0`），否则 `group` 不是整数。
//!
//! 曾经想要的第三条是"字节数必须等于 `out × ceil(in / 2)`"，但它在本层是**结构性自证**的：
//! [`Entry`] 的长度由 `shape × dtype` 推出，safetensors 又要求声明区间等于这个乘积，所以任何
//! 从 [`Header`] 构造出来的条目都自动满足它——在这里再比一次，等于拿一个值与自己比。更要紧的
//! 是它挡不住真正含糊的那件事：8 位元素类型的第二维究竟是"逐元素 int8 的 `in` 个字节"还是
//! "两个 nibble 一个字节"，形状上看不出来。本层如实按字节读（宽度翻倍，分组也跟着翻倍），
//! 由上层拿分组与内核声明的一比：不相等就不计价，而不是套一条用错分组的内核（见
//! `WeightFormat::operator`）。
//!
//! # 三种结局，各有各的理由
//!
//! | 情形 | 返回 | 为什么 |
//! |---|---|---|
//! | 没有伴随的标度条目 | `Ok(None)` | 不是量化，按元素类型处理 |
//! | 有标度但尺寸自相矛盾（秩不是 2、输出维不等、`groups == 0`） | `Err(Invalid)` | 文件说它是量化却对不上，猜下去一定是错数 |
//! | 有标度但打包口径不符（不是 4 位打包、标度不是浮点） | `Ok(None)` | 那是**另一种**量化方案（int8 逐元素、整数标度），不猜 |
//!
//! 表里的分界是"能不能解释成另一种合法方案"：能，就不猜；不能，就报错。

use diagnostic::{Code, Error, Result};
use tensor::DType;

use crate::{Entry, Header};

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
    /// **不要从 [`Quantized::weight`] 的形状里读这个数**：8 位元素类型的第二维是打包后的字节数，
    /// 一个字节两个 nibble。宽度与分组必须由同一处给出，否则会出现"分组按 128 个元素算、宽度按
    /// 64 个元素算"的错配——那种错不会报错，只会把下界算成真值的一半。
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 输出宽度（元素数）：打包不改变行数。
    pub fn out_features(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    /// 每个元素的位数。目前只认 4 位打包。
    pub fn bits(&self) -> u8 {
        4
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
/// 输入宽度不是直接读第二维：**8 位元素类型的第二维是打包后的字节数**（一个字节两个 nibble），
/// 而 4 位元素类型（解包视图）的第二维才是元素数。这条换算不是约定，是"safetensors 要求
/// `shape × dtype` 的字节数等于声明区间"的直接后果：想用一个 8 位 dtype 表达 4 位打包，就只能
/// 把形状按字节写。
fn analyze<'a>(weight: &'a Entry, scales: &'a Entry) -> Result<Option<Quantized<'a>>> {
    let context = format!(
        "量化条目 `{}` 与它的标度 `{}`",
        weight.name(),
        scales.name()
    );
    let [out, second] = weight.shape().dims() else {
        return Err(invalid(format!(
            "打包权重必须是秩 2（out × in），实际 {:?}",
            weight.shape().dims()
        ))
        .with(&context));
    };
    let [scaled_out, groups] = scales.shape().dims() else {
        return Err(invalid(format!(
            "标度必须是秩 2（out × 分组数），实际 {:?}",
            scales.shape().dims()
        ))
        .with(&context));
    };
    if scaled_out != out {
        return Err(invalid(format!(
            "标度的输出维 {scaled_out} 与权重的输出维 {out} 不等：分组无法对齐到每一行"
        ))
        .with(&context));
    }
    if !matches!(scales.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        // 整数标度是另一种方案（int4 + int8 scale），不猜。
        return Ok(None);
    }

    // 每行的逻辑输入宽度：8 位元素类型两个 nibble 一个字节，4 位元素类型一个元素就是一个元素。
    let in_features = match weight.dtype().bits() {
        4 => *second,
        8 => match second.checked_mul(2) {
            Some(width) => width,
            None => {
                return Err(invalid(format!(
                    "形状 {:?} 的第二维 {} 乘 2 时溢出",
                    weight.shape().dims(),
                    second
                ))
                .with(&context));
            }
        },
        // 16 / 32 位元素不可能是 4 位量化：交给按元素类型判断的那条路。
        _ => return Ok(None),
    };
    if *groups == 0 || !in_features.is_multiple_of(*groups) {
        return Err(invalid(format!(
            "标度声明 {groups} 个分组，但输入宽度 {in_features} 不是它的整数倍"
        ))
        .with(&context));
    }
    Ok(Some(Quantized {
        weight,
        scales,
        in_features,
        group: in_features / groups,
    }))
}

fn invalid(message: String) -> Error {
    Error::new(Code::Invalid, message).with("weights::quant")
}

#[cfg(test)]
mod tests {
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

    /// 真实文件的条目：8 位元素类型，第二维是**字节数**（一个字节两个 nibble）。
    fn nibbles(name: &str, out: usize, in_features: usize) -> Entry {
        Entry::new(name, DType::I8, Shape::new([out, in_features / 2]))
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
        // 同一层权重，两种书写：解包视图（U4，第二维 = 元素数）与真实文件（8 位，第二维 = 字节数）。
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
        assert_eq!(from_file.group(), 32, "8 位条目的第二维是字节数");
        assert_eq!(from_file.in_features(), 128, "宽度是元素数，不是字节数");
        assert_eq!(from_file.out_features(), 8);
        assert_eq!(from_unpacked.in_features(), from_file.in_features());
    }

    #[test]
    fn an_element_per_byte_entry_doubles_the_width_and_so_the_group() {
        // 逐元素 int8 与"打包 nibble"在形状上无法区分，本层如实把宽度读成两倍：分组因此也翻倍，
        // 上层拿算子声明的分组一比就会拒绝，而不是算出一个用错分组的流量。
        let weight = Entry::new("a.weight", DType::I8, Shape::new([8, 64]));
        let header = header(vec![weight, scales("a.scales", 8, 2, DType::F16)]);
        let quantized = header
            .quantization("a.weight")
            .expect("自洽")
            .expect("是 8 位 + 标度");
        assert_eq!(
            quantized.group(),
            64,
            "64 个字节 = 128 个 nibble，/2 组 = 64"
        );
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
