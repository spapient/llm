//! 逻辑张量：`DType` + `Shape` + `Strides` + 只读字节视图 [`Tensor`]。
//!
//! **拥有**：`Tensor`、`View`、`DType`、`Shape`、`Strides`、`Element`。**不拥有**：kernel 侧的操作数视图
//! （DSL 的宿主 API，两者语义不同，见 `docs/adr/0003-kernel-and-runtime-types.md`）、
//! 缓冲分配（`mem`）、布局搬运（`storage`）。**依赖**：`mem`、`diagnostic`。
//! **被依赖**：`storage`、`weights`、`kv`、`autograd`、`data`、`nn`、`engine`、`llm`。
//!
//! 这个类型只描述"数据长什么样、放在哪几个字节里"，**不做 `&mut` 分区**——分区是 DSL 的所有权
//! 规则（DESIGN §4.4，P0 尚未实现），两套并发规则会互相打架。
//!
//! # 例子
//!
//! ```
//! use llm_tensor::{DType, Shape, Tensor};
//!
//! let shape = Shape::new([2, 3]);
//! let bytes = [0u8; 24]; // 2 * 3 个 f32
//! let tensor = Tensor::new(DType::F32, shape, &bytes).expect("形状与缓冲匹配");
//! assert_eq!(tensor.numel(), 6);
//! assert_eq!(tensor.strides().as_slice(), [3, 1], "行主序连续布局");
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};

/// 元素类型闭集。
///
/// 只列推理与训练真实会用到的宽度；`U4` 是 int4 权重的存储形态（按位打包，`bits()` 为 4）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32 位浮点。
    F32,
    /// 16 位浮点。
    F16,
    /// bfloat16。
    BF16,
    /// 8 位整数（含 int8 / uint8 两种解释，由算子决定）。
    I8,
    /// 4 位整数（按位打包）。
    U4,
}

impl DType {
    /// 全部元素类型，供测试与 CLI 遍历。
    pub const ALL: [Self; 5] = [Self::F32, Self::F16, Self::BF16, Self::I8, Self::U4];

    /// 单个元素占多少位。
    pub fn bits(self) -> u32 {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::I8 => 8,
            Self::U4 => 4,
        }
    }

    /// 是否为浮点类型（整数类型走精确对拍，见 `cutile::reference`）。
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

/// 形状：每个轴的长度。空形状表示标量，`numel()` 为 1。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// 由各轴长度构造。
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// 轴数（rank）。标量为 0。
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// 某一轴的长度；越界返回 `None`。
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// 元素个数：各轴长度之积，空形状为 1。
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// 该元素类型下需要的字节数（按位向上取整，`U4` 会小于 `numel`）。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        let bits = u64::from(dtype.bits());
        let total = (self.numel() as u64).saturating_mul(bits);
        usize::try_from(total.div_ceil(8)).unwrap_or(usize::MAX)
    }
}

/// 步长：沿每个轴前进一个下标要走多少个元素。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<usize>);

impl Strides {
    /// 由各轴步长构造（不做合法性校验：非连续布局也是合法布局）。
    pub fn new(strides: impl Into<Vec<usize>>) -> Self {
        Self(strides.into())
    }

    /// 行主序连续布局：最后一轴步长为 1，向前累乘。
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![0; dims.len()];
        let mut acc = 1usize;
        for axis in (0..dims.len()).rev() {
            strides[axis] = acc;
            acc = acc.saturating_mul(dims[axis]);
        }
        Self(strides)
    }

    /// 各轴步长。
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// 某一轴的步长；越界返回 `None`。
    pub fn get(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }
}

/// 逻辑张量：元数据 + 只读字节视图。
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// 由元数据与连续布局的缓冲构造：`data` 必须够长，否则报 `invalid`。
    pub fn new(dtype: DType, shape: Shape, data: &'a [u8]) -> Result<Self> {
        let required = shape.element_bytes(dtype);
        if data.len() < required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {} 字节，{dtype:?} 的 {:?} 需要 {required} 字节",
                    data.len(),
                    shape.dims()
                ),
            )
            .with("tensor::Tensor::new"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            dtype,
            shape,
            strides,
            data,
        })
    }

    /// 由分配器发出的区间构造：视图借用 `arena`，不复制。
    pub fn from_region(
        arena: &'a mem::Arena,
        region: mem::Region,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self> {
        let data = arena.slice(region).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("区间 {region:?} 超出分配器的已分配范围"),
            )
            .with("tensor::Tensor::from_region")
        })?;
        Self::new(dtype, shape, data)
    }

    /// 元素类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 只读字节视图。
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }
}

/// 元素类型：把 Rust 标量映射到 [`DType`] 闭集，供 [`View`] 做单态化视图。
///
/// 只实现到 `DType` 已有的宽度：`f16` / `bf16` 没有稳定的 Rust 标量类型，
/// `U4` 是按位打包的存储形态（不是一种元素宽度），两者都不适合做视图的元素类型。
pub trait Element: Copy + 'static {
    /// 对应的元素类型。
    const DTYPE: DType;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
}

impl Element for i8 {
    const DTYPE: DType = DType::I8;
}

impl Element for u8 {
    const DTYPE: DType = DType::I8;
}

/// 泛型只读视图：形状 + 步长 + 调用方的 `&[E]`。
///
/// 与 [`Tensor`] 的分工：`Tensor` 是**字节**视图（权重、打包类型、跨语言边界），
/// `View` 是**元素**视图（内核与算子直接计算）。`View` 不分配、不做类型双关、
/// 不装 `unsafe`：它只接受调用方已经持有的 `&[E]`，因此“零拷贝”是类型保证，
/// 不是注释承诺；代价是它不负责把字节转成元素（那是装载方的职责）。
#[derive(Debug, Clone)]
pub struct View<'a, E: Element> {
    shape: Shape,
    strides: Strides,
    data: &'a [E],
}

impl<'a, E: Element> View<'a, E> {
    /// 由形状与连续布局的缓冲构造：元素个数必须与形状一致。
    pub fn new(shape: Shape, data: &'a [E]) -> Result<Self> {
        let required = shape.numel();
        if data.len() != required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {} 个元素，形状 {:?} 需要 {required} 个",
                    data.len(),
                    shape.dims()
                ),
            )
            .with("tensor::View::new"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// 由任意合法步长构造：转置、行/列切片这类非连续布局也是合法视图。
    ///
    /// 校验的三件事：步长轴数必须与形状一致；零元素视图必须借用空缓冲（与 [`View::new`]
    /// 同一口径）；按最大下标算出的最高偏移必须落在 `data` 内。校验用 `checked_mul`，
    /// 形状异常时返回错误而不是溢出。
    pub fn strided(shape: Shape, strides: Strides, data: &'a [E]) -> Result<Self> {
        if strides.as_slice().len() != shape.rank() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "步长有 {} 个轴，形状 {:?} 有 {} 个轴",
                    strides.as_slice().len(),
                    shape.dims(),
                    shape.rank()
                ),
            )
            .with("tensor::View::strided"));
        }
        if shape.numel() == 0 {
            return if data.is_empty() {
                Ok(Self {
                    shape,
                    strides,
                    data,
                })
            } else {
                Err(Error::new(
                    Code::Invalid,
                    format!("零元素视图不该借用 {} 个元素的缓冲", data.len()),
                )
                .with("tensor::View::strided"))
            };
        }

        let mut highest = 0usize;
        for (axis, dimension) in shape.dims().iter().enumerate() {
            let step = strides.get(axis).unwrap_or(0);
            let last = dimension - 1;
            let offset = last
                .checked_mul(step)
                .and_then(|offset| highest.checked_add(offset))
                .ok_or_else(|| {
                    Error::new(
                        Code::Invalid,
                        format!(
                            "形状 {:?} 与步长 {:?} 算出溢出",
                            shape.dims(),
                            strides.as_slice()
                        ),
                    )
                    .with("tensor::View::strided")
                })?;
            highest = offset;
        }
        if highest >= data.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "最高偏移 {highest} 超出缓冲（{} 个元素）：步长与形状对不上",
                    data.len()
                ),
            )
            .with("tensor::View::strided"));
        }
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// 元素类型（来自 [`Element::DTYPE`]，与 `DType` 闭集对齐）。
    pub fn dtype(&self) -> DType {
        E::DTYPE
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 调用方的缓冲本身（同一个指针，不是副本）。
    pub fn as_slice(&self) -> &'a [E] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }

    /// 按下标取值；秩不符或越界返回 `None`（不 panic）。
    pub fn at(&self, index: &[usize]) -> Option<E> {
        if index.len() != self.shape.rank() {
            return None;
        }
        let mut offset = 0usize;
        for (axis, &position) in index.iter().enumerate() {
            if position >= self.shape.dim(axis)? {
                return None;
            }
            offset = offset.checked_add(position.checked_mul(self.strides.get(axis)?)?)?;
        }
        self.data.get(offset).copied()
    }

    /// 二维视图的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
    ///
    /// 只在列步长为 1 时给出切片：`&[E]` 这个类型本身就承诺"这一段是连续的"，
    /// 非连续的行（例如转置视图的列）必须走 [`View::at`]，不能假装它是一片内存。
    pub fn row(&self, row: usize) -> Option<&'a [E]> {
        let [rows, columns] = <[usize; 2]>::try_from(self.shape.dims()).ok()?;
        if row >= rows || self.strides.get(1)? != 1 {
            return None;
        }
        let start = row.checked_mul(self.strides.get(0)?)?;
        self.data.get(start..start.checked_add(columns)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scalar_has_one_element_and_no_strides() {
        let shape = Shape::new([]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1);
        assert_eq!(Strides::contiguous(&shape).as_slice(), [] as [usize; 0]);
        let tensor = Tensor::new(DType::F32, shape, &[0u8; 4]).expect("标量");
        assert_eq!(tensor.numel(), 1);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn sub_byte_dtypes_need_fewer_bytes_than_elements() {
        let shape = Shape::new([3, 2]);
        assert_eq!(shape.element_bytes(DType::F32), 24);
        assert_eq!(shape.element_bytes(DType::U4), 3, "6 个 int4 = 3 字节");
        assert_eq!(shape.element_bytes(DType::I8), 6);
    }

    #[test]
    fn a_short_buffer_is_rejected_with_the_required_length() {
        let error = Tensor::new(DType::F32, Shape::new([2]), &[0u8; 4]).expect_err("应报错");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("8 字节"), "{error}");
    }

    #[test]
    fn a_region_view_borrows_the_arena() {
        let mut arena = mem::Arena::new();
        let region = arena.alloc(8, 8).expect("分配");
        let tensor =
            Tensor::from_region(&arena, region, DType::F32, Shape::new([2])).expect("视图");
        assert_eq!(tensor.data(), &[0u8; 8]);
        let too_large = mem::Region::new(0, 1);
        assert!(Tensor::from_region(&arena, too_large, DType::F32, Shape::new([2])).is_err());
    }

    #[test]
    fn a_transposed_stride_is_not_contiguous() {
        let shape = Shape::new([2, 3]);
        let tensor = Tensor::new(DType::I8, shape.clone(), &[0u8; 6]).expect("视图");
        assert!(tensor.is_contiguous());
        let transposed = Strides::new([1, 2]);
        assert_ne!(transposed, Strides::contiguous(&shape));
    }

    #[test]
    fn a_view_borrows_the_callers_buffer_instead_of_copying_it() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let view = View::new(Shape::new([2, 2]), &values).expect("视图");
        assert_eq!(view.dtype(), DType::F32);
        assert_eq!(view.numel(), 4);
        assert_eq!(view.strides().as_slice(), [2, 1]);
        assert!(
            std::ptr::eq(view.as_slice().as_ptr(), values.as_ptr()),
            "视图必须借用原缓冲，而不是复制一份"
        );
    }

    #[test]
    fn a_view_is_bounds_checked_without_panicking() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let view = View::new(Shape::new([2, 2]), &values).expect("视图");
        assert_eq!(view.at(&[0, 1]), Some(2.0));
        assert_eq!(view.at(&[1, 0]), Some(3.0));
        assert_eq!(view.at(&[1]), None, "秩不符");
        assert_eq!(view.at(&[2, 0]), None, "越界");
        assert_eq!(view.row(1), Some(&[3.0, 4.0][..]));
        assert_eq!(view.row(2), None);
    }

    #[test]
    fn a_view_rejects_a_buffer_of_the_wrong_length() {
        let error = View::new(Shape::new([2, 2]), &[1.0f32, 2.0]).expect_err("长度不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("需要 4 个"), "{error}");
    }

    #[test]
    fn a_row_of_a_higher_rank_view_is_not_guessed() {
        let values = [0u8; 8];
        let view = View::new(Shape::new([2, 2, 2]), &values).expect("视图");
        assert_eq!(view.row(0), None, "只有二维视图才有“行”");
    }

    #[test]
    fn element_types_stay_inside_the_dtype_closed_set() {
        for dtype in [
            <f32 as Element>::DTYPE,
            <i8 as Element>::DTYPE,
            <u8 as Element>::DTYPE,
        ] {
            assert!(DType::ALL.contains(&dtype), "{dtype:?} 不在闭集里");
        }
        assert_eq!(<f32 as Element>::DTYPE, DType::F32);
        assert_eq!(<i8 as Element>::DTYPE, DType::I8);
        assert_eq!(
            <u8 as Element>::DTYPE,
            DType::I8,
            "无符号与有符号共用 int8 宽度"
        );
    }

    const SIX: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    #[test]
    fn a_transposed_view_reads_through_its_strides() {
        // 2×3 行主序缓冲，按 3×2 转置读：`(i, j)` 落在 `i + 3j`。
        let view = View::strided(Shape::new([3, 2]), Strides::new([1, 3]), &SIX).expect("转置视图");
        assert_eq!(view.at(&[0, 0]), Some(1.0));
        assert_eq!(view.at(&[0, 1]), Some(4.0));
        assert_eq!(view.at(&[2, 1]), Some(6.0));
        assert!(!view.is_contiguous(), "转置后的步长不是行主序");
        assert_eq!(view.numel(), 6);
    }

    #[test]
    fn a_strided_view_rejects_a_step_count_that_does_not_match_the_rank() {
        let error =
            View::strided(Shape::new([2, 2]), Strides::new([4]), &SIX).expect_err("轴数不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("2 个轴"), "{error}");
    }

    #[test]
    fn a_strided_view_rejects_steps_that_run_past_the_buffer() {
        let error = View::strided(Shape::new([2, 2]), Strides::new([2, 1]), &[0.0f32; 3])
            .expect_err("最高偏移越界");
        assert!(error.to_string().contains("最高偏移 3"), "{error}");
    }

    #[test]
    fn a_row_of_a_transposed_view_is_not_returned_as_a_slice() {
        let view = View::strided(Shape::new([3, 2]), Strides::new([1, 3]), &SIX).expect("转置视图");
        assert_eq!(
            view.row(0),
            None,
            "列不连续时不能把跳着取的元素当成一片内存"
        );
        assert_eq!(view.at(&[0, 1]), Some(4.0), "逐点读仍然可用");
    }

    #[test]
    fn a_row_strided_view_still_serves_contiguous_rows() {
        // 隔行取样：形状 2×3，行步长 2、列步长 1。
        let view = View::strided(Shape::new([2, 3]), Strides::new([2, 1]), &SIX).expect("行切片");
        assert_eq!(view.row(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(view.row(1), Some(&[3.0, 4.0, 5.0][..]));
        assert!(!view.is_contiguous(), "行之间跳过了元素，整体不连续");
    }

    #[test]
    fn a_zero_element_view_needs_an_empty_buffer() {
        let empty: [f32; 0] = [];
        assert!(
            View::strided(Shape::new([0, 3]), Strides::new([3, 1]), &empty).is_ok(),
            "零元素视图与 `View::new` 同一口径"
        );
        let error = View::strided(Shape::new([0, 3]), Strides::new([3, 1]), &[1.0f32])
            .expect_err("非空缓冲");
        assert!(error.to_string().contains("零元素"), "{error}");
    }
}
