//! 层栈：一个模型就是一串层，本模块给出这串层以及它们的**总下界**。
//!
//! 层下界（[`Bounds`]）回答"这一层至少要搬多少、算多少"；模型下界回答同一个问题在整串层上的和。
//! 这个和不是新公式：它就是 [`Bounds::plus`]，所以新增一层不会多出第二份流量口径。
//!
//! 层的清单有两个来源：
//!
//! - 声明：[`Stack::new`] 直接给一串 [`Linear`]，[`Stack::of`] 收任意 [`Layer`]（含注解层）；
//! - 权重：[`Stack::from_header`] 按权重文件的条目推出来——秩为 2 的条目按 `[out, in]` 视为一次
//!   投影，名字去掉 `.weight` 后缀，**编码取自条目自己的 `dtype`**；**其余条目进
//!   [`Stack::uncovered`]，绝不静默丢弃**。
//!
//! 「未覆盖」与「无法计价」是两件事，分别记录：
//!
//! | 清单 | 含义 |
//! |---|---|
//! | [`Stack::uncovered`] | 根本不是矩阵（范数、偏置、位置编码……），不构成一次投影 |
//! | [`Floor::unpriced`] | 是投影，但推不出下界：形状没有合法内核，或这种权重编码还没有算子 |
//!
//! 两者都**不会**让整张模型的下界变成错误。一个 LoRA 适配器（`[4096, 16]`，16 不是量化分组的
//! 整数倍）或一张 f32 权重（还没有浮点投影算子）只让自己无法计价，其余层照常相加——这正是
//! "允许某个组件没有下界"的含义。
//!
//! 维度约定与 [`Linear`] 一致：`shape = [out_features, in_features]`，即 safetensors 里的
//! `weight` 按 PyTorch 的行主序形状书写。
//!
//! # 例子
//!
//! ```
//! use llm_nn::{Linear, Stack};
//!
//! let model = Stack::new(vec![Linear::new("blk.0.attn_q", 4096, 4096).expect("合法")]).expect("名字唯一");
//! let floor = model.floor();
//! assert!(floor.total().read() > 8 * 1024 * 1024, "int4 权重本身就有 8 MiB");
//! assert!(floor.unpriced().is_empty());
//! assert!(model.uncovered().is_empty());
//! ```

use std::fmt;

use diagnostic::{Code, Error, Result};
use weights::{Entry, Header};

use crate::{Bounds, Layer, Linear, WeightFormat};

/// 权重条目名里表示"这是一个权重矩阵"的后缀；层名去掉它，才能与其它按名查找的地方对齐。
pub const WEIGHT_SUFFIX: &str = ".weight";

/// 一条"是投影、但推不出下界"的层：名字 + 原因。
///
/// 原因不是装饰：`floor()` 的返回值要和 `uncovered` 一起被打印出来，读者才能判断"模型下界偏低"
/// 是因为多了一张 LoRA 表、一张 f32 权重，还是因为某个形状确实没有合法内核。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unpriced {
    name: String,
    reason: String,
}

impl Unpriced {
    fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// 层名。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 为什么没有下界。
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for Unpriced {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}（{}）", self.name, self.reason)
    }
}

/// 一次模型计价的结果：能加的那部分 + 加不进去的那部分。
///
/// 没有 `Result`：无法计价不是错误，而是**必须被如实报告**的事实。真正的错误（权重文件损坏、
/// 层重名、零宽投影）在构造 [`Stack`] 时就已经报了。
#[derive(Debug, Clone, PartialEq)]
pub struct Floor {
    total: Bounds,
    unpriced: Vec<Unpriced>,
}

impl Floor {
    /// 各层下界之和。无法计价的层不出现在这个数里。
    pub fn total(&self) -> Bounds {
        self.total
    }

    /// 无法计价的层，保持构造顺序。
    pub fn unpriced(&self) -> &[Unpriced] {
        &self.unpriced
    }
}

/// 一串层及其总下界。
pub struct Stack {
    layers: Vec<Box<dyn Layer>>,
    uncovered: Vec<String>,
}

impl fmt::Debug for Stack {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Stack")
            .field(
                "layers",
                &self
                    .layers
                    .iter()
                    .map(|layer| layer.name())
                    .collect::<Vec<_>>(),
            )
            .field("uncovered", &self.uncovered)
            .finish()
    }
}

impl Stack {
    /// 由一串声明好的线性层构造。
    ///
    /// # Errors
    ///
    /// 层名重复时返回 [`Code::Invalid`]：后面的遍历（参数表、切分、下界归因）都按名字对齐，
    /// 重名会让"到底算的是哪一层"变成猜测。
    pub fn new(layers: Vec<Linear>) -> Result<Self> {
        Self::of(
            layers
                .into_iter()
                .map(|layer| Box::new(layer) as Box<dyn Layer>)
                .collect(),
        )
    }

    /// 由任意一串层构造（可以把注解层与手写层混在一起）。
    ///
    /// 收 `Box<dyn Layer>` 而不是泛型 `L: Layer`：一个栈里既要放 [`Linear`] 这种按模型配置
    /// 构造的层，也要放 `#[layer]` 注解出来的零大小类型，两者没有共同的静态类型。
    ///
    /// # Errors
    ///
    /// 层名重复时返回 [`Code::Invalid`]（理由同 [`Stack::new`]）。
    pub fn of(layers: Vec<Box<dyn Layer>>) -> Result<Self> {
        unique_names(layers.iter().map(|layer| layer.name()))?;
        Ok(Self {
            layers,
            uncovered: Vec::new(),
        })
    }

    /// 按权重文件的条目推出层清单。
    ///
    /// 秩为 2 的条目 → 一次 [`Linear`]（`[out, in]`），**权重编码由条目自己的 `dtype` 决定**
    /// （[`WeightFormat::of_dtype`]）；秩不是 2 的条目 → 未覆盖清单。
    ///
    /// dtype 只说明"元素怎么存"，不说明"多少个元素共享一个缩放因子"。因此 4 位打包的条目只会
    /// 得到 [`WeightFormat::Packed4`]：层照样被列出来，但 [`Stack::floor`] 会明确报"分组未声明、
    /// 无法计价"。缺什么就说什么，不拿一个默认分组去凑。
    ///
    /// # Errors
    ///
    /// - 条目形状为 `[0, n]` / `[n, 0]`：宽度为零的投影没有意义（[`Code::Invalid`]）；
    /// - 去掉 `.weight` 后层名重复（[`Code::Invalid`]）。
    pub fn from_header(header: &Header) -> Result<Self> {
        // 标度条目不是层：它的分组已经算进某条权重那一层的流量（算子语义里 `ng` 那条派生轴），
        // 单独列进"未覆盖"清单只会让读者以为漏了东西。
        let scales = header.scales_names()?;
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        let mut uncovered = Vec::new();
        for entry in header.entries() {
            if scales.contains(&entry.name()) {
                continue;
            }
            match projection(header, entry)? {
                Some(layer) => layers.push(Box::new(layer)),
                None => uncovered.push(entry.name().to_string()),
            }
        }
        unique_names(layers.iter().map(|layer| layer.name()))?;
        Ok(Self { layers, uncovered })
    }

    /// 全部层，保持权重文件里的偏移顺序（构造顺序）。
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    /// 无法归因为投影的条目名（范数、偏置、位置编码……）。
    pub fn uncovered(&self) -> &[String] {
        &self.uncovered
    }

    /// 总下界：各层下界之和，加上无法计价的层的清单。
    ///
    /// 单层的 `Err`（形状没有合法内核）或 `None`（没有可计价的形状）只让**那一层**进清单，
    /// 不会让整张模型算不出来：这样一张 LoRA 表不会把其余 300 层的和一起清零。
    pub fn floor(&self) -> Floor {
        let mut total = Bounds::ZERO;
        let mut unpriced = Vec::new();
        for layer in &self.layers {
            match layer.bounds() {
                Ok(Some(bounds)) => total = total.plus(bounds),
                Ok(None) => unpriced.push(Unpriced::new(
                    layer.name(),
                    "没有可计价的形状（缺输入宽度）",
                )),
                Err(error) => unpriced.push(Unpriced::new(layer.name(), error.to_string())),
            }
        }
        Floor { total, unpriced }
    }
}

/// 一个条目能否被视为一次投影：能则给出层，不能则返回 `None`（进未覆盖清单）。
///
/// 编码的判定分两步，顺序不能反：
///
/// 1. 文件里有没有这条权重的**标度伙伴**？有，分组就从标度的形状读出来（[`Header::quantization`]），
///    于是 `WeightFormat::int4 { group }` 是实测值而不是默认值；
/// 2. 没有伙伴，才退回按元素类型判断（f32 / f16 / i8 / 无分组的 4 位打包）。
///
/// 反过来的顺序（先按 dtype 定格式、再尝试修正）会让"默认 32 个一组"混进真实分组不同的模型里，
/// 得到的下界看起来有数、其实算错了流量。
fn projection(header: &Header, entry: &Entry) -> Result<Option<Linear>> {
    let name = layer_name(entry.name());
    // 量化条目先问标度：宽度与分组都从那一处来。8 位元素类型的第二维是**字节数**，直接当宽度
    // 会把下界算成真值的一半，而且不会有任何报错——这是本函数里唯一需要"先问"的理由。
    if let Some(quantized) = header.quantization(entry.name())? {
        return Linear::with_format(
            name,
            WeightFormat::int4(quantized.group()),
            quantized.in_features(),
            quantized.out_features(),
        )
        .map(Some)
        .map_err(|error| error.with(format!("权重条目 `{}`", entry.name())));
    }
    let [out_features, in_features] = entry.shape().dims() else {
        return Ok(None);
    };
    let Some(format) = WeightFormat::of_dtype(entry.dtype()) else {
        return Ok(None);
    };
    Linear::with_format(name, format, *in_features, *out_features)
        .map(Some)
        .map_err(|error| error.with(format!("权重条目 `{}`", entry.name())))
}

/// 层名：去掉 [`WEIGHT_SUFFIX`]，其它名字原样保留。
fn layer_name(entry: &str) -> &str {
    entry
        .strip_suffix(WEIGHT_SUFFIX)
        .filter(|name| !name.is_empty())
        .unwrap_or(entry)
}

fn unique_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(Error::new(
                Code::Invalid,
                format!("层名 `{name}` 重复：下界与参数表都按名字对齐，重名无法归因"),
            )
            .with("nn::Stack"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LmHead;
    use tensor::{DType, Shape};

    fn header(entries: Vec<Entry>) -> Header {
        Header::new(entries, 0).expect("测试夹具的区间合法")
    }

    fn entry(name: &str, dims: [usize; 2], offset: usize) -> Entry {
        Entry::new(name, DType::F32, Shape::new(dims)).at(offset)
    }

    fn packed(name: &str, dims: [usize; 2], offset: usize) -> Entry {
        Entry::new(name, DType::U4, Shape::new(dims)).at(offset)
    }

    #[test]
    fn a_two_dimensional_entry_becomes_a_projection() {
        let model = Stack::from_header(&header(vec![entry("blk.0.attn_q.weight", [64, 32], 0)]))
            .expect("形状合法");
        let layers = model.layers();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name(), "blk.0.attn_q", "去掉 `.weight` 后缀");
        assert!(model.uncovered().is_empty());
    }

    #[test]
    fn the_entry_dtype_decides_the_format_not_a_global_default() {
        let model = Stack::from_header(&header(vec![entry("blk.0.attn_q.weight", [64, 32], 0)]))
            .expect("形状合法");
        // 文件里写的是 f32：它由稠密算子计价，而不是按 int4 报一个 8 倍偏低的数。
        let floor = model.floor();
        assert!(floor.unpriced().is_empty(), "f32 现在有算子");
        let layer = &model.layers()[0];
        assert_eq!(
            layer.spec().ops(),
            [Linear::DENSE],
            "f32 权重选的是稠密那条算子"
        );
        assert_eq!(
            floor.total(),
            layer.bounds().expect("合法").expect("有算子"),
            "栈的总下界就是各层的和"
        );
        assert_ne!(
            floor.total(),
            Bounds::of_op(Linear::OP, 32, 64).expect("int4 语义"),
            "按 int4 计价会把 f32 权重报成八分之一"
        );
    }

    /// 一份全是 f32 的权重文件（真实 safetensors 最常见的形态）现在能全额计价。
    #[test]
    fn a_dense_model_is_priced_end_to_end() {
        let model = Stack::from_header(&header(vec![
            entry("blk.0.attn_q.weight", [64, 4096], 0),
            entry("blk.0.attn_k.weight", [64, 4096], 64 * 4096 * 4),
        ]))
        .expect("形状合法");
        let floor = model.floor();
        assert_eq!(model.layers().len(), 2);
        assert!(floor.unpriced().is_empty(), "{:?}", floor.unpriced());
        // 每层：权重 64×4096 个 f32 + 输入 4096 个 f32；两层相加。
        let per_layer = (64 * 4096 + 4096) as u64 * 4;
        assert_eq!(floor.total().read(), 2 * per_layer);
        assert_eq!(floor.total().flops(), 2.0 * 2.0 * 64.0 * 4096.0);
    }

    #[test]
    fn a_packed_entry_without_a_group_is_reported_not_guessed() {
        let model = Stack::from_header(&header(vec![packed("blk.0.attn_q.weight", [64, 32], 0)]))
            .expect("形状合法");
        let floor = model.floor();
        assert_eq!(floor.total(), Bounds::ZERO);
        assert!(
            floor.unpriced()[0].reason().contains("分组"),
            "分组缺失必须写在原因里：{}",
            floor.unpriced()[0]
        );
    }

    #[test]
    fn entries_that_are_not_matrices_are_reported_not_dropped() {
        let model = Stack::from_header(&header(vec![
            Entry::new("blk.0.norm.weight", DType::F32, Shape::new([64])).at(0),
            Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([64, 64])).at(256),
        ]))
        .expect("形状合法");
        assert_eq!(model.layers().len(), 1);
        assert_eq!(model.uncovered(), ["blk.0.norm.weight"]);
        // 未覆盖与无法计价是两件事：前者没进层清单，后者进了但推不出下界。
        let floor = model.floor();
        assert_eq!(floor.unpriced().len(), 1, "只有那条 16 位投影无法计价");
        assert!(
            floor.unpriced()[0].reason().contains("16 位"),
            "{}",
            floor.unpriced()[0]
        );
    }

    #[test]
    fn the_model_floor_is_the_sum_of_its_layers() {
        let model = Stack::of(vec![
            Box::new(Linear::new("a", 32, 64).expect("合法")),
            Box::new(Linear::new("b", 64, 128).expect("合法")),
        ])
        .expect("名字唯一");
        let sum = model
            .layers()
            .iter()
            .map(|layer| layer.bounds().expect("合法").expect("有算子"))
            .fold(Bounds::ZERO, Bounds::plus);
        assert_eq!(model.floor().total(), sum);
        assert!(sum.read() > 0 && sum.write() > 0 && sum.flops() > 0.0);
    }

    #[test]
    fn an_annotated_layer_joins_the_same_sum() {
        let model = Stack::of(vec![
            Box::new(LmHead),
            Box::new(Linear::new("blk.0.attn_q", 4096, 4096).expect("合法")),
        ])
        .expect("名字唯一");
        let floor = model.floor();
        assert!(floor.unpriced().is_empty(), "两侧都有形状");
        let head = LmHead.bounds().expect("合法").expect("有算子");
        let projection = Linear::new("blk.0.attn_q", 4096, 4096)
            .expect("合法")
            .bounds()
            .expect("合法")
            .expect("有算子");
        assert_eq!(floor.total(), head.plus(projection), "注解层没有第二条公式");
    }

    #[test]
    fn a_layer_without_a_legal_kernel_only_takes_itself_out() {
        // in = 8 不是量化分组的整数倍：这一层没有合法内核，但其余层照常相加。
        let model = Stack::of(vec![
            Box::new(Linear::new("ok", 64, 4).expect("合法")),
            Box::new(Linear::new("odd", 8, 4).expect("层可构造")),
        ])
        .expect("名字唯一");
        let floor = model.floor();
        assert_eq!(floor.unpriced().len(), 1);
        assert_eq!(floor.unpriced()[0].name(), "odd");
        assert!(
            floor.unpriced()[0].reason().contains("整数倍"),
            "{}",
            floor.unpriced()[0]
        );
        assert_eq!(
            floor.total(),
            Bounds::of_op("gemv_i4", 64, 4).expect("合法"),
            "一层的失败没有清零整张模型"
        );
    }

    #[test]
    fn duplicate_names_after_stripping_the_suffix_are_rejected() {
        let error = Stack::from_header(&header(vec![
            entry("a.weight", [64, 32], 0),
            entry("a", [32, 64], 64 * 32 * 4),
        ]))
        .expect_err("去掉后缀后重名");
        assert!(error.to_string().contains("层名 `a` 重复"), "{error}");
    }

    #[test]
    fn a_zero_width_projection_is_rejected() {
        let error = Stack::from_header(&header(vec![entry("empty.weight", [0, 32], 0)]))
            .expect_err("宽度为零");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("empty"), "{error}");
    }

    /// 真实量化 checkout 的形状：4 位打包权重 + 每 `group` 个元素一个 f16 标度。
    fn quantized(name: &str, out: usize, in_features: usize, groups: usize) -> Vec<Entry> {
        vec![
            packed(&format!("{name}.weight"), [out, in_features], 0),
            Entry::new(
                format!("{name}.scales"),
                DType::F16,
                Shape::new([out, groups]),
            )
            .at(1 << 20),
        ]
    }

    #[test]
    fn a_packed_weight_and_its_scales_become_one_priced_layer() {
        // 128 个输入 / 4 个分组 = 每 32 个元素一个标度，正是内核声明的那个分组。
        let model =
            Stack::from_header(&header(quantized("blk.0.attn_q", 64, 128, 4))).expect("文件自洽");
        assert_eq!(model.layers().len(), 1, "标度不是层");
        assert!(model.uncovered().is_empty(), "标度也不该进未覆盖清单");

        let floor = model.floor();
        assert!(floor.unpriced().is_empty(), "{:?}", floor.unpriced());
        // 分组来自标度的形状，不是默认值：下界必须与算子语义逐个数字一致。
        assert_eq!(
            floor.total(),
            Bounds::of_op(Linear::OP, 128, 64).expect("同一份语义")
        );
        assert_eq!(model.layers()[0].spec().ops(), [Linear::OP]);
    }

    #[test]
    fn a_group_the_kernel_does_not_declare_is_named_in_the_reason() {
        // 512 / 4 = 每 128 个元素一个标度：真实权重里很常见，但不是本内核声明的那一档。
        let model =
            Stack::from_header(&header(quantized("blk.0.attn_q", 64, 512, 4))).expect("文件自洽");
        let floor = model.floor();
        assert_eq!(floor.total(), Bounds::ZERO, "分组不同就不给数字");
        assert_eq!(floor.unpriced().len(), 1);
        let reason = floor.unpriced()[0].reason();
        assert!(reason.contains("128"), "要说出文件里的分组：{reason}");
        assert!(reason.contains("32"), "也要说出内核声明的分组：{reason}");
    }

    #[test]
    fn a_packed_weight_without_scales_says_what_is_missing() {
        let model =
            Stack::from_header(&header(vec![packed("w.weight", [8, 32], 0)])).expect("文件合法");
        let floor = model.floor();
        assert_eq!(floor.unpriced().len(), 1);
        assert!(
            floor.unpriced()[0].reason().contains("标度条目"),
            "{}",
            floor.unpriced()[0]
        );
    }

    #[test]
    fn a_scale_that_contradicts_its_weight_stops_the_whole_stack() {
        // 输出维对不上：这不是"另一种量化方案"，是文件自相矛盾，算下去一定是错数。
        let entries = vec![
            packed("blk.0.attn_q.weight", [64, 128], 0),
            Entry::new("blk.0.attn_q.scales", DType::F16, Shape::new([32, 4])).at(1 << 20),
        ];
        let error = Stack::from_header(&header(entries)).expect_err("输出维不等");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("输出维 32"), "{error}");
    }

    #[test]
    fn a_declared_model_rejects_duplicate_names_too() {
        let one = Linear::new("blk.0", 32, 32).expect("合法");
        let two = Linear::new("blk.0", 32, 32).expect("合法");
        let error = Stack::new(vec![one, two]).expect_err("重名");
        assert!(error.to_string().contains("blk.0"), "{error}");
    }

    #[test]
    fn a_quantized_entry_and_its_unpacked_view_declare_the_same_floor() {
        // 同一层权重的两种书写：解包视图（U4，第二维是元素数）与真实文件（8 位，第二维是字节数）。
        let packed_bytes = Entry::new("blk.0.q.weight", DType::I8, Shape::new([64, 64])).at(0);
        let scales = Entry::new("blk.0.q.scales", DType::F16, Shape::new([64, 4])).at(4096);
        let plain = Entry::new("blk.0.q.weight", DType::U4, Shape::new([64, 128])).at(0);
        let plain_scales = Entry::new("blk.0.q.scales", DType::F16, Shape::new([64, 4])).at(4096);

        let from_file = Stack::from_header(&header(vec![packed_bytes, scales])).expect("形状合法");
        let from_view = Stack::from_header(&header(vec![plain, plain_scales])).expect("形状合法");

        assert_eq!(from_file.layers().len(), 1, "标度不是层");
        assert_eq!(from_view.layers().len(), 1);
        assert!(
            from_file.uncovered().is_empty(),
            "标度被当成未覆盖条目：{:?}",
            from_file.uncovered()
        );
        let a = from_file.floor().total();
        let b = from_view.floor().total();
        assert!(a.bytes() > 0, "量化层应当可计价");
        assert_eq!(a, b, "8 位条目的第二维是字节数，宽度必须由标度推出");
    }

    #[test]
    fn a_scale_with_no_matching_group_is_named_not_silently_priced() {
        // group = 128 个元素一组，而内核声明 32：不是同一个东西，必须点名。
        let weight = Entry::new("blk.0.q.weight", DType::I8, Shape::new([64, 64])).at(0);
        let scales = Entry::new("blk.0.q.scales", DType::F16, Shape::new([64, 1])).at(4096);
        let model = Stack::from_header(&header(vec![weight, scales])).expect("形状合法");

        let floor = model.floor();
        assert_eq!(floor.total(), Bounds::ZERO);
        assert_eq!(floor.unpriced().len(), 1);
        assert!(
            floor.unpriced()[0].reason().contains("128"),
            "理由要点名真实分组：{}",
            floor.unpriced()[0]
        );
        assert!(
            floor.unpriced()[0].reason().contains("32"),
            "理由要点名内核要求的分组：{}",
            floor.unpriced()[0]
        );
    }
}
