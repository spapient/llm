//! 代价推导：表达式 + 轴长度 + 操作数 → [`Cost`]，以及中间的轴分解 [`Axes`]。

use crate::einsum::{Bits, Cost, Expr, Extents};
use diagnostic::{Code, Error};

/// 操作数的数据流向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// 只读：算子入口的输入。
    Read,
    /// 只写：算子出口的输出。写入同样计费，因为"至少写一遍输出"是逃不掉的流量。
    Write,
}

/// 表达式里一个操作数的**编译期**形态：名字、元素位宽与读/写方向。
///
/// [`Operand`] 装着 `String`，构造不出常量；这里只放编译期就能定下的三件事，装载成 [`Operand`]
/// 时再补上拥有型名字。它属于表达式层而不是某条骨架：四条骨架（GEMV / 逐元素 / 一元 / 归约）
/// 都要声明操作数表，放在其中任何一条里都会让另外三条反向依赖它。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandSpec {
    /// 缓冲名（与 `.k` 的形参名一致，也是诊断里显示的名字）。
    pub name: &'static str,
    /// 元素位宽。
    pub bits: Bits,
    /// 读写方向。
    pub access: Access,
}

impl OperandSpec {
    /// 只读操作数。
    #[must_use]
    pub const fn read(name: &'static str, bits: Bits) -> Self {
        Self {
            name,
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    #[must_use]
    pub const fn write(name: &'static str, bits: Bits) -> Self {
        Self {
            name,
            bits,
            access: Access::Write,
        }
    }
}

/// f32 一读一写的标准操作数：`a` 读、`y` 写（一元与归约共用同一份）。
///
/// 过去 `unary` 与 `reduce` 各写一份同名的 `f32_operands`，而 `elementwise` 同名却返回
/// 三元素——同一个函数名、两种元数，改错一处只会在运行时才暴露。收口到这里之后，操作数表
/// 与 `OperandSpec` 同层（表达式层），四条骨架按元数各取所需。
#[must_use]
pub const fn f32_io_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// f32 两读一写的标准操作数：`a`、`b` 读、`y` 写（二元逐元素）。
#[must_use]
pub const fn f32_binary_operands() -> [OperandSpec; 3] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::read("b", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// 一个操作数：名字、元素位宽与读/写方向。
///
/// 名字只是诊断用的标签（与内核模板的形参名一致），但必须唯一：同名缓冲出现在两个位置会被
/// 重复计费，而这在真实算子里从来不是本意。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    name: String,
    bits: Bits,
    access: Access,
}

impl Operand {
    /// 只读操作数。
    pub fn read(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    pub fn write(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Write,
        }
    }

    /// 按方向构造（方向来自数据时用这个）。
    pub fn new(name: impl Into<String>, bits: Bits, access: Access) -> Self {
        Self {
            name: name.into(),
            bits,
            access,
        }
    }

    /// 名字。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素位宽。
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// 流向。
    pub fn access(&self) -> Access {
        self.access
    }
}

/// 一条轴在这次计算里的角色。
///
/// 这个分类不是装饰：`Batch` 与 `Free` 决定 TileLang 式的批处理能不能合批，`Reduced` 决定
/// roofline 的 `flops` 口径。把它算出来，上层就不必再猜。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 批轴：出现在输出里，且被至少两个**只读**操作数共享。
    ///
    /// 只统计只读操作数：输出缓冲自己也带着同样的字母，把它算进去会让所有输出轴都变成批轴，
    /// 这个分类就失去了信息量。
    Batch,
    /// 自由轴：出现在输出里，且只被一个操作数拥有。
    Free,
    /// 规约轴：不出现在输出里，求和后被消掉。
    Reduced,
    /// 分组轴：由另一条轴按固定因子切出（[`crate::einsum::Extents::group`]）。
    ///
    /// 它只影响**流量**（`scales[n, g]` 确实只有 `n · m/组宽` 个元素），不计入运算次数：
    /// 遍历 `g` 与遍历 `m` 是同一批元素的两层看法，乘起来就是重复计数。
    Grouped,
}

/// 一条轴：字母、长度与角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    label: char,
    extent: usize,
    role: AxisRole,
}

impl Axis {
    /// 轴字母。
    pub fn label(&self) -> char {
        self.label
    }

    /// 轴长度（元素个数）。
    pub fn extent(&self) -> usize {
        self.extent
    }

    /// 轴角色。
    pub fn role(&self) -> AxisRole {
        self.role
    }
}

/// 表达式的轴分解：输出轴在前（按输出顺序），规约轴在后（按首次出现顺序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axes {
    axes: Vec<Axis>,
}

impl Axes {
    /// 全部轴，顺序稳定（输出轴优先）。
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// 输出轴。
    ///
    /// 只含批轴与自由轴：分组轴虽然不由求和消掉，但它是一条已被别的轴覆盖的派生看法，
    /// 算进输出形状就会把 `y[n]` 说成 `y[n, g]`。
    pub fn output(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| matches!(axis.role, AxisRole::Batch | AxisRole::Free))
    }

    /// 规约轴。
    pub fn reduced(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Reduced)
    }

    /// 分组轴（由其他轴按因子切出，只计流量）。
    pub fn grouped(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Grouped)
    }

    /// 是否有真正的规约轴（决定 `flops` 用哪种口径）。
    ///
    /// 分组轴不算：它不增加运算次数。
    pub fn reduces(&self) -> bool {
        self.reduced().next().is_some()
    }

    /// 输出形状（按输出轴的声明顺序）。
    pub fn output_shape(&self) -> Vec<usize> {
        self.output().map(Axis::extent).collect()
    }

    /// 输出元素个数；标量输出为 `1`。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn output_elements(&self) -> Result<u128, Error> {
        self.output()
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 参与运算的轴的乘积，即"展开后的标量运算次数"的分母。
    ///
    /// 分组轴被排除：它是另一条轴的分组看法，不是独立的循环维。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn all_elements(&self) -> Result<u128, Error> {
        self.axes
            .iter()
            .filter(|axis| axis.role != AxisRole::Grouped)
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 运算次数：有规约轴时按"每个规约元素一次乘加（2 flops）"，否则只算输出元素。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn flops(&self) -> Result<f64, Error> {
        if self.reduces() {
            return Ok(2.0 * self.all_elements()? as f64);
        }
        Ok(self.output_elements()? as f64)
    }
}

/// 一份可估算的语义：表达式 + 每个操作数的位宽与方向。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    expr: Expr,
    operands: Vec<Operand>,
}

impl Formula {
    /// 把表达式与操作数配对。
    ///
    /// # Errors
    ///
    /// - 操作数个数与表达式的项数不一致；
    /// - 操作数重名（同一块缓冲会被重复计费）；
    /// - 写入操作数多于一个，或写入的轴不在输出里（写回中间量不是算子语义）。
    pub fn new(expr: Expr, operands: Vec<Operand>) -> Result<Self, Error> {
        if expr.arity() != operands.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表达式 `{expr}` 有 {} 项，但给了 {} 个操作数",
                    expr.arity(),
                    operands.len()
                ),
            )
            .with("einsum::Formula::new"));
        }
        unique(&operands)?;
        single_write(&operands)?;
        writes_only_outputs(&expr, &operands)?;
        Ok(Self { expr, operands })
    }

    /// 表达式。
    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    /// 操作数（顺序与表达式的项一致）。
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// 轴分解：每条轴的长度与角色。
    ///
    /// # Errors
    ///
    /// 表达式里的字母没有给长度、给了长度的字母没出现在表达式里，或乘积超出 `u128`。
    pub fn axes(&self, extents: &Extents) -> Result<Axes, Error> {
        let mut axes = self.output_axes(extents)?;
        self.push_reduced(extents, &mut axes)?;
        self.check_used(extents, &axes)?;
        Ok(Axes { axes })
    }

    /// 输出轴：角色看"有几个读操作数共享它"（多于一个即 batch）。
    ///
    /// 派生轴不能出现在输出里——它的长度是从基准轴推出来的，写回去等于把一个视图当缓冲。
    fn output_axes(&self, extents: &Extents) -> Result<Vec<Axis>, Error> {
        let mut axes: Vec<Axis> = Vec::with_capacity(extents.len());
        for &label in self.expr.output() {
            let Some((base, divisor)) = extents.grouped(label) else {
                let extent = extent(extents, label)?;
                axes.push(Axis {
                    label,
                    extent,
                    role: self.output_role(label),
                });
                continue;
            };
            return Err(Error::new(
                Code::Invalid,
                format!("轴 `{label}` 是 `{base}` 按 {divisor} 分组的派生轴，不能出现在输出里"),
            )
            .with("einsum::Formula::axes"));
        }
        Ok(axes)
    }

    /// 输出轴的角色：被多个读操作数共享的是 batch，否则是 free。
    fn output_role(&self, label: char) -> AxisRole {
        let shared = self
            .operands
            .iter()
            .zip(self.expr.terms())
            .filter(|(operand, term)| operand.access() == Access::Read && term.contains(&label))
            .count();
        if shared > 1 {
            AxisRole::Batch
        } else {
            AxisRole::Free
        }
    }

    /// 输入项里还没被收进来的轴：派生轴是 [`AxisRole::Grouped`]，其余是 [`AxisRole::Reduced`]。
    fn push_reduced(&self, extents: &Extents, axes: &mut Vec<Axis>) -> Result<(), Error> {
        for term in self.expr.terms() {
            for &label in term {
                if axes.iter().any(|axis| axis.label == label) {
                    continue;
                }
                let role = extents
                    .grouped(label)
                    .map_or(AxisRole::Reduced, |_| AxisRole::Grouped);
                axes.push(Axis {
                    label,
                    extent: extent(extents, label)?,
                    role,
                });
            }
        }
        Ok(())
    }

    /// 给了长度却没出现在表达式里的轴：多半是笔误。
    fn check_used(&self, extents: &Extents, axes: &[Axis]) -> Result<(), Error> {
        for label in extents.labels() {
            if axes.iter().any(|axis| axis.label == label) {
                continue;
            }
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "轴 `{label}` 有长度 {} 但没出现在表达式 `{}` 里：多半是笔误",
                    extent(extents, label)?,
                    self.expr
                ),
            )
            .with("einsum::Formula::axes"));
        }
        Ok(())
    }

    /// 计算代价。
    ///
    /// # Errors
    ///
    /// 轴分解失败（见 [`Formula::axes`]）、维度乘积溢出，或流量超出 `u64`。
    pub fn cost(&self, extents: &Extents) -> Result<Cost, Error> {
        let axes = self.axes(extents)?;
        self.estimate(extents, &axes)
    }

    /// 一次算完：flops 与两个方向的流量。
    ///
    /// 两个方向各自累加：合成一个数之后，"读受限还是写受限"就再也推不出来了。
    fn estimate(&self, extents: &Extents, axes: &Axes) -> Result<Cost, Error> {
        let flops = axes.flops()?;
        let (read, write) = self.traffic(extents)?;
        Ok(Cost {
            flops,
            read: bytes(read, "读")?,
            write: bytes(write, "写")?,
        })
    }

    /// 流量：每个操作数按自己的项长 × 位宽计费，读与写分开累加。
    fn traffic(&self, extents: &Extents) -> Result<(u128, u128), Error> {
        let mut read: u128 = 0;
        let mut write: u128 = 0;
        for (operand, term) in self.operands.iter().zip(self.expr.terms()) {
            let elements = term_elements(extents, term)?;
            let bytes = operand.bits().bytes_for(elements);
            let total = match operand.access() {
                Access::Read => &mut read,
                Access::Write => &mut write,
            };
            accumulate(total, bytes)?;
        }
        Ok((read, write))
    }
}

/// 同名缓冲只允许出现一次：否则同一块内存会被重复计费。
fn unique(operands: &[Operand]) -> Result<(), Error> {
    let mut names: Vec<&str> = Vec::with_capacity(operands.len());
    for operand in operands {
        if names.contains(&operand.name()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "操作数 `{}` 出现了两次：同名缓冲会被重复计费",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
        names.push(operand.name());
    }
    Ok(())
}

/// 一次计算只有一个出口。
fn single_write(operands: &[Operand]) -> Result<(), Error> {
    let writes = operands
        .iter()
        .filter(|operand| operand.access() == Access::Write)
        .count();
    if writes > 1 {
        return Err(
            Error::new(Code::Invalid, "有多个写入操作数：一次计算只有一个出口")
                .with("einsum::Formula::new"),
        );
    }
    Ok(())
}

/// 写入操作数只能写输出轴：写回中间量不是算子语义。
fn writes_only_outputs(expr: &Expr, operands: &[Operand]) -> Result<(), Error> {
    for (operand, term) in operands.iter().zip(expr.terms()) {
        if operand.access() != Access::Write {
            continue;
        }
        if let Some(label) = term.iter().find(|label| !expr.output().contains(label)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "写入操作数 `{}` 的轴 `{label}` 不在输出里：写回中间量不是算子语义",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
    }
    Ok(())
}

/// 一项的元素数：该项每条轴长度的乘积。
fn term_elements(extents: &Extents, term: &[char]) -> Result<u128, Error> {
    term.iter().try_fold(1u128, |total, label| {
        multiply(total, extent(extents, *label)?)
    })
}

/// 按方向累加：两个计数器各自溢出都要报。
fn accumulate(total: &mut u128, bytes: u128) -> Result<(), Error> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}

/// 流量收口到 `u64`：`Cost` 的字段是 `u64`，装不下说明形状不可能是真的。
fn bytes(value: u128, kind: &str) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| {
        Error::new(Code::Invalid, format!("{kind}流量 {value} 字节超出 u64"))
            .with("einsum::Formula::cost")
    })
}

fn extent(extents: &Extents, label: char) -> Result<usize, Error> {
    extents.get(label).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("轴 `{label}` 没有长度：表达式里的每个字母都必须给出轴长度"),
        )
        .with("einsum::Formula::axes")
    })
}

fn multiply(total: u128, length: usize) -> Result<u128, Error> {
    total.checked_mul(length as u128).ok_or_else(overflow)
}

fn overflow() -> Error {
    Error::new(Code::Invalid, "维度乘积或流量超出 u128：形状不可能是真的").with("einsum::Formula")
}
