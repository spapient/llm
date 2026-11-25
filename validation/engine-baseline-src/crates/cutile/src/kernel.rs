//! Typed, target-neutral IR for the `#[dsl]` kernel family.
//!
//! 一张 [`Kernel`] 覆盖三种写法：逐点 `output[idx] = value`、归约 `output[at] += value`、以及由
//! 前两者组合出的融合核（[`Kernel::fuse`]）。这一层做校验、规范化、常量折叠、死形参消除，并发射
//! 两种目标文本：可移植 C（[`Kernel::emit_c`]）与设备 lowering 的 `.k` 正文（[`Kernel::emit_k`]，
//! 经 `shard::ir` 桥接入 `Lower`/设备 dispatch）。`Image` 尚未接上，所以「性能提升」只能由桥接后
//! 的设备路径宣称，不由这一层单独宣称。

use core::fmt;

use crate::cap::{Atom, AtomId, AtomTable};
use crate::pass::Pipeline;
use crate::validate::verify_core;

/// Scalar element types shared by the DSL and template program IR.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// IEEE-754 single precision.
    F32,
    /// IEEE-754 half precision.
    F16,
    /// Brain floating point.
    Bf16,
    /// IEEE-754 double precision.
    F64,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// Rust `usize` / C `size_t`.
    Usize,
    /// Boolean predicate.
    Bool,
    /// `void`，仅用于显式丢弃表达式值的转换。
    Void,
    /// 四车道 IEEE-754 单精度向量。
    F32x4,
}

impl ScalarType {
    /// C spelling used in a declaration or cast.
    pub(crate) fn c_name(self) -> &'static str {
        const NAMES: [&str; 14] = [
            "float",
            "half",
            "bf16",
            "double",
            "signed char",
            "unsigned char",
            "int",
            "unsigned int",
            "long",
            "unsigned long",
            "size_t",
            "_Bool",
            "void",
            "float4",
        ];
        NAMES[self as usize]
    }

    /// Whether this type may be held in a buffer parameter.
    pub(crate) fn buf_ok(self) -> bool {
        !matches!(self, Self::Usize | Self::Bool | Self::Void)
    }
}

/// Buf access required by a kernel parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read-only buffer.
    Read,
    /// Mutable buffer, which may be read and written.
    ReadWrite,
}

impl Access {
    /// C pointer declaration for a buffer of the given element spelling, held with this access.
    fn c_pointer(self, element: &str) -> String {
        match self {
            Access::Read => format!("const {element} *"),
            Access::ReadWrite => format!("{element} *"),
        }
    }
}

/// A checked value type in the kernel signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// Scalar value.
    Scalar(ScalarType),
    /// Contiguous buffer and its element/access contract.
    Buf {
        /// Buf element type.
        element: ScalarType,
        /// Required access.
        access: Access,
    },
}

impl ValueType {
    /// C declaration for this parameter, or `None` when the element type has no buffer spelling.
    pub(crate) fn c_type(self) -> Option<String> {
        match self {
            ValueType::Scalar(element) => Some(element.c_name().to_string()),
            ValueType::Buf { element, access } => {
                element.buf_ok().then(|| access.c_pointer(element.c_name()))
            }
        }
    }

    pub(crate) fn is_buf(self) -> bool {
        matches!(self, ValueType::Buf { .. })
    }

    /// 标量总能发射；缓冲元素必须在 [`ScalarType::buf_ok`] 闭集内，否则校验阶段就打回。
    pub(crate) fn element_ok(self) -> bool {
        match self {
            ValueType::Scalar(_) => true,
            ValueType::Buf { element, .. } => element.buf_ok(),
        }
    }
}

/// A named kernel parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    /// Rust parameter name.
    pub name: String,
    /// Scalar or buffer type.
    pub ty: ValueType,
}

/// Arithmetic operations supported by the pointwise expression IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arith {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Remainder.
    Rem,
    /// Left shift.
    Shl,
    /// Right shift.
    Shr,
    /// Bitwise and.
    BitAnd,
    /// Bitwise or.
    BitOr,
    /// Bitwise xor.
    BitXor,
}

impl Arith {
    /// Infix symbol used when emitting C.
    pub(crate) fn c_symbol(self) -> &'static str {
        const SYMBOLS: [&str; 10] = ["+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"];
        SYMBOLS[self as usize]
    }

    /// Apply a floating-point operation; integer-only operators return `None`.
    pub(crate) fn apply(self, left: f32, right: f32) -> Option<f32> {
        const RULES: [fn(f32, f32) -> Option<f32>; 10] = [
            |left, right| Some(left + right),
            |left, right| Some(left - right),
            |left, right| Some(left * right),
            |left, right| Some(left / right),
            |left, right| Some(left % right),
            no_float_binary,
            no_float_binary,
            no_float_binary,
            no_float_binary,
            no_float_binary,
        ];
        RULES[self as usize](left, right)
    }
}

fn no_float_binary(_: f32, _: f32) -> Option<f32> {
    None
}

/// Comparison operators shared by scalar and template expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareKind {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

impl CompareKind {
    /// C operator spelling.
    pub(crate) fn c_symbol(self) -> &'static str {
        const SYMBOLS: [&str; 6] = ["==", "!=", "<", "<=", ">", ">="];
        SYMBOLS[self as usize]
    }
}

/// Short-circuit logical operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalKind {
    /// Logical and.
    And,
    /// Logical or.
    Or,
}

impl LogicalKind {
    /// C operator spelling.
    pub(crate) fn c_symbol(self) -> &'static str {
        const SYMBOLS: [&str; 2] = ["&&", "||"];
        SYMBOLS[self as usize]
    }
}

/// 一元 f32 数学 / 取整组合子闭集。
///
/// 每个算子只在这一处登记 C 发射名与 Rust 主机求值；校验、遍历、折叠、解释执行都从这张表取，
/// 不为单个变体各写一份 match 臂。结果仍是 `f32`——比较 / 选择这类出布尔值的组合子不在本闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFn {
    /// `|x|`
    Abs,
    /// `√x`
    Sqrt,
    /// `e^x`
    Exp,
    /// `ln x`
    Ln,
    /// `sin x`
    Sin,
    /// `cos x`
    Cos,
    /// 向正无穷取整。
    Ceil,
    /// 向负无穷取整。
    Floor,
    /// 四舍五入到最近整数，半入远离零。
    Round,
    /// 截断到整数。
    Trunc,
}

impl UnaryFn {
    /// C `<math.h>` 函数名，下标与枚举变体声明顺序一一对应。
    const NAMES: [&'static str; 10] = [
        "fabsf", "sqrtf", "expf", "logf", "sinf", "cosf", "ceilf", "floorf", "roundf", "truncf",
    ];

    /// 主机侧 f32 求值表：解释执行、字面量折叠与参考对拍共用同一口径。
    const APPLY: [fn(f32) -> f32; 10] = [
        f32::abs,
        f32::sqrt,
        f32::exp,
        f32::ln,
        f32::sin,
        f32::cos,
        f32::ceil,
        f32::floor,
        f32::round,
        f32::trunc,
    ];

    /// C `<math.h>` 函数名。
    pub(crate) const fn c_name(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    /// 主机侧 f32 求值：解释执行、字面量折叠与参考对拍共用同一口径，避免两边各写一套数学定义。
    pub fn apply(self, value: f32) -> f32 {
        (Self::APPLY[self as usize])(value)
    }
}

/// Typed expression nodes. Buf loads use the kernel's implicit loop index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// An f32 literal stored by its exact IEEE-754 bit pattern.
    F32(u32),
    /// An f32 scalar parameter.
    Scalar(String),
    /// Load one f32 value from a buffer at the current loop index.
    Load(String),
    /// Integer literal.
    Int(i64),
    /// Explicit buffer or local-array indexing.
    Index {
        /// Base expression.
        base: Box<Expr>,
        /// Index expression.
        idx: Box<Expr>,
    },
    /// A declared host / device helper call.
    Call {
        /// Function identifier.
        function: String,
        /// Ordered arguments.
        args: Vec<Expr>,
    },
    /// Scalar cast.
    Cast {
        /// Target scalar type.
        ty: ScalarType,
        /// Input expression.
        value: Box<Expr>,
    },
    /// Comparison producing a boolean predicate.
    Compare {
        /// Comparison operator.
        op: CompareKind,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Short-circuit logical operation.
    Logical {
        /// Logical operator.
        op: LogicalKind,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Conditional selection.
    Select {
        /// Predicate.
        condition: Box<Expr>,
        /// Value when true.
        when_true: Box<Expr>,
        /// Value when false.
        when_false: Box<Expr>,
    },
    /// Assignment, optionally fused with an arithmetic operator.
    Assign {
        /// Target lvalue or output address.
        target: Box<Expr>,
        /// `+=` 等复合运算；`None` 表示普通赋值。
        op: Option<Arith>,
        /// Assigned value.
        value: Box<Expr>,
    },
    /// Prefix or postfix increment / decrement.
    Update {
        /// Updated scalar lvalue.
        target: Box<Expr>,
        /// `1` 为递增，`-1` 为递减。
        delta: i8,
        /// 是否前置。
        prefix: bool,
    },
    /// Logical not.
    Not(Box<Expr>),
    /// Bitwise not.
    BitNot(Box<Expr>),
    /// Address-of.
    Address(Box<Expr>),
    /// Pointer dereference.
    Deref(Box<Expr>),
    /// Negate an f32 expression.
    Neg(Box<Expr>),
    /// A unary math / rounding combinator over one f32 expression.
    Unary {
        /// Registered unary operator.
        op: UnaryFn,
        /// Operand.
        child: Box<Expr>,
    },
    /// Binary f32 arithmetic.
    Binary {
        /// Operation.
        op: Arith,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Fused multiply-add `a * b + c`, evaluated with a single rounding.
    Fma {
        /// Multiplicand.
        a: Box<Expr>,
        /// Multiplier.
        b: Box<Expr>,
        /// Addend.
        c: Box<Expr>,
    },
    /// Element-wise minimum of two f32 expressions.
    Min {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Element-wise maximum of two f32 expressions.
    Max {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
}

/// Verification failure for malformed or unsupported typed IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationError {
    message: String,
}

impl VerificationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VerificationError {}

/// What a kernel stores on every iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Write {
    /// Pointwise store: `output[idx] = value`.
    Map(Expr),
    /// Reduction: `output[at] += value`.
    Reduce {
        /// Constant slot accumulated into.
        at: u32,
        /// Value added to the accumulator on each iteration.
        value: Expr,
    },
}

impl Write {
    /// Expression evaluated once per iteration.
    pub(crate) fn expr(&self) -> &Expr {
        match self {
            Write::Map(value) => value,
            Write::Reduce { value, .. } => value,
        }
    }

    /// Mutable access to the expression evaluated once per iteration.
    pub(crate) fn expr_mut(&mut self) -> &mut Expr {
        match self {
            Write::Map(value) => value,
            Write::Reduce { value, .. } => value,
        }
    }

    /// Fixed slot a reduction accumulates into; `None` for pointwise stores.
    pub(crate) fn slot(&self) -> Option<u32> {
        match self {
            Write::Reduce { at, .. } => Some(*at),
            Write::Map(_) => None,
        }
    }
}

/// 局部值类型：元素、指针层数、可选定长数组与只读约束。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalType {
    /// 元素类型。
    pub element: ScalarType,
    /// 指针层数；0 为普通标量 / 数组。
    pub pointers: u8,
    /// 是否声明为 `const`。
    pub constant: bool,
    /// 定长数组元素数。
    pub array: Option<usize>,
}

/// 一条局部声明。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    /// 局部名。
    pub name: String,
    /// 局部类型。
    pub ty: LocalType,
    /// 可选初值。
    pub init: Option<Expr>,
}

/// 原子的类型化实参。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomArg {
    /// 位置表达式。
    Expr(Expr),
    /// 命名能力或后端参数。
    Named {
        /// 参数名。
        key: String,
        /// 参数值。
        value: String,
    },
}

/// IR 上的原子引用：只存 arena 身份与参数，不存拼好的后端文本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomCall {
    /// 原子身份。
    pub atom: AtomId,
    /// 调用参数。
    pub args: Vec<AtomArg>,
    /// 模板行号。
    pub line: usize,
}

/// 完整模板程序的语句 IR。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    /// 局部声明。
    Decl(LocalDecl),
    /// 原子调用。
    Atom(AtomCall),
    /// 普通表达式语句。
    Expr(Expr),
    /// 嵌套块。
    Block(Vec<Statement>),
    /// `for` 循环。
    For {
        /// 可选初始化语句。
        init: Option<Box<Statement>>,
        /// 可选继续条件。
        condition: Option<Expr>,
        /// 可选迭代表达式。
        step: Option<Expr>,
        /// 循环体。
        body: Vec<Statement>,
    },
    /// 条件分支。
    If {
        /// 条件。
        condition: Expr,
        /// 真分支。
        then_branch: Vec<Statement>,
        /// 假分支；空表示没有 `else`。
        else_branch: Vec<Statement>,
    },
    /// 受校验的编译器指令（当前闭集为 `unroll`）。
    Pragma {
        /// 指令名。
        name: String,
        /// 可选正整数参数。
        value: Option<u32>,
    },
    /// `continue`。
    Continue,
    /// `break`。
    Break,
    /// 可选带值的 `return`。
    Return(Option<Expr>),
}

/// 模板级类型别名。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    /// 别名。
    pub name: String,
    /// 目标标量类型。
    pub ty: ScalarType,
}

/// 模板中的辅助函数或主内核函数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    /// 函数名。
    pub name: String,
    /// 返回类型。
    pub result: ScalarType,
    /// 有序形参。
    pub params: Vec<Parameter>,
    /// 函数体。
    pub body: Vec<Statement>,
}

/// 完整 `.k` 模板的强类型程序：共享 `Parameter` / `Expr`，并持有自己的原子 arena。
#[derive(Debug, Clone, PartialEq)]
pub struct Program<'a> {
    name: String,
    params: Vec<Parameter>,
    atoms: AtomTable<'a>,
    aliases: Vec<TypeAlias>,
    helpers: Vec<Function>,
    body: Vec<Statement>,
}

impl<'a> Program<'a> {
    /// 构造一份程序；实际发射前仍须通过 [`Self::verify`]。
    pub fn new(
        name: impl Into<String>,
        params: Vec<Parameter>,
        atoms: AtomTable<'a>,
        body: Vec<Statement>,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            atoms,
            aliases: Vec::new(),
            helpers: Vec::new(),
            body,
        }
    }

    /// 挂上同一模板的类型别名。
    pub fn with_aliases(mut self, aliases: Vec<TypeAlias>) -> Self {
        self.aliases = aliases;
        self
    }

    /// 挂上同一模板的辅助函数。
    pub fn with_helpers(mut self, helpers: Vec<Function>) -> Self {
        self.helpers = helpers;
        self
    }

    /// 类型别名。
    pub fn aliases(&self) -> &[TypeAlias] {
        &self.aliases
    }

    /// 辅助函数。
    pub fn helpers(&self) -> &[Function] {
        &self.helpers
    }

    /// 程序名。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 有序形参。
    pub fn params(&self) -> &[Parameter] {
        &self.params
    }

    /// 顶层语句。
    pub fn body(&self) -> &[Statement] {
        &self.body
    }

    pub(crate) fn params_mut(&mut self) -> &mut Vec<Parameter> {
        &mut self.params
    }

    pub(crate) fn body_mut(&mut self) -> &mut Vec<Statement> {
        &mut self.body
    }

    pub(crate) fn helpers_mut(&mut self) -> &mut Vec<Function> {
        &mut self.helpers
    }

    /// 原子 arena。
    pub fn atoms(&self) -> &AtomTable<'a> {
        &self.atoms
    }

    /// 读取一个原子调用引用的记录。
    pub fn atom(&self, call: &AtomCall) -> Option<&Atom<'a>> {
        self.atoms.get(call.atom)
    }

    /// 校验签名、局部类型、表达式与原子端口。
    pub fn verify(&self) -> Result<(), VerificationError> {
        crate::validate::verify_program(self)
    }

    /// 常量折叠、代数化简与死形参消除。
    pub fn optimize(&self) -> Result<Self, VerificationError> {
        crate::pass::optimize_program(self.clone())
    }

    /// 后端 `.k` 发射；printer 只消费 IR。
    pub fn emit_k(&self) -> Result<String, VerificationError> {
        crate::printer::render_program(self)
    }

    /// 按输出语言发射，向量类型只落到已登记的真实后端能力。
    pub fn emit(&self, emit: crate::prof::Emit) -> Result<String, VerificationError> {
        crate::printer::render_program_target(self, emit)
    }

    /// 稳定、无指针地址的结构编码，供缓存指纹使用。
    ///
    /// # Errors
    ///
    /// 类型校验、优化或原子参数展开失败时返回说明。
    pub fn encode(&self) -> Result<String, VerificationError> {
        crate::printer::encode_program(self)
    }
}

/// One verified kernel: a single loop that stores into `output` as described by [`Write`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    /// Kernel name.
    pub name: String,
    /// Ordered scalar and buffer parameters.
    pub params: Vec<Parameter>,
    /// Loop index variable.
    pub idx: String,
    /// Name of the `usize` parameter providing the loop bound.
    pub extent: String,
    /// Name of the mutable output buffer.
    pub output: String,
    /// Store performed on every iteration.
    pub write: Write,
}

impl Kernel {
    /// Construct a kernel from an explicit [`Write`].
    pub fn new(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        write: Write,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            idx: idx.into(),
            extent: extent.into(),
            output: output.into(),
            write,
        }
    }

    /// Construct a pointwise kernel: `output[idx] = value`.
    pub fn map(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        value: Expr,
    ) -> Self {
        Self::new(name, params, idx, extent, output, Write::Map(value))
    }

    /// Construct a reduction kernel: `output[at] += value`.
    pub fn reduce(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        at: u32,
        value: Expr,
    ) -> Self {
        Self::new(
            name,
            params,
            idx,
            extent,
            output,
            Write::Reduce { at, value },
        )
    }

    /// Validate names, parameter types, memory access, and expression types.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_core(
            &self.name,
            &self.idx,
            &self.params,
            &self.extent,
            &self.output,
            self.write.expr(),
        )
    }

    /// Run the standard pipeline to a fixpoint: verify, then simplify (algebraic laws + constant
    /// folding), then drop dead parameters, sweeping until a whole round leaves the kernel
    /// unchanged, then verify again. The result is a valid kernel in normal form.
    ///
    /// 轮次不用魔法数：上限是初始 `cost`（形参数 + 节点数），严格收缩的 pass 必然在此之前收敛，耗尽
    /// 即视为流水线震荡并报错。
    pub fn optimize(&self) -> Result<Self, VerificationError> {
        Pipeline::standard().run(self.clone())
    }

    /// Run the contracted pipeline: standard simplification plus multiply-add contraction
    /// (`a * b + c` → a single fused multiply-add). Contraction changes rounding, so it is opt-in;
    /// use it for inner-product / affine kernels that want the fused instruction.
    pub fn optimize_contracted(&self) -> Result<Self, VerificationError> {
        Pipeline::contracted().run(self.clone())
    }

    /// Emit portable C; return status 1 when a guarded length is violated.
    pub fn emit_c(&self) -> Result<String, VerificationError> {
        crate::printer::render(self, crate::printer::Target::C)
    }

    /// Emit the `.k`-format serial body the device lowering seam consumes (the `shard::ir` bridge
    /// feeds it to the vendor backends' `Codegen::lower`): a `void` entry
    /// `cutile_{name}_kernel` with plain `const float *`/`float *`/`int` parameters and a single
    /// guard-free loop. Unlike [`Self::emit_c`], it pairs no buffer lengths, emits no guards and no
    /// `#include` — the vendor backend wraps this body into its grid / chunk / serial skeleton.
    pub fn emit_k(&self) -> Result<String, VerificationError> {
        crate::printer::render(self, crate::printer::Target::K)
    }
}

#[cfg(test)]
#[path = "../tests/ir_tests.rs"]
mod ir_tests;

#[cfg(test)]
#[path = "../tests/unary_tests.rs"]
mod unary_tests;
