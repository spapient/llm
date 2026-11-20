//! 注解层：把"声明"写成属性，由编译器检查，而不是手写 trait 样板再靠人复核。
//!
//! **拥有**：`#[spec]` / `#[model]` 属性宏。**不拥有**：任何领域类型——宏只
//! 生成调用点所在 crate 的代码，生成的实现指向契约路径（`#[spec]` 指向 `nn::Contract` /
//! `nn::ContractSpec`，`#[model]` 指向 `op::model::Model`），因此本 crate 对它们**没有**依赖：proc-macro 只在编译期运行，产出的
//! token 在调用方的 crate 里展开。
//!
//! 两道注解共用同一趟展开（`expand_with`）与同一套"注解只能标 struct / enum / 别名、不得带
//! 泛型"的类型名判据：注解承载的是**可静态判定**的契约（名字/形状/算子、
//! 注册名/生产形状/输出类型），执行与业务仍由手写代码给出，两者用 trait bound 绑死
//! （`nn::Contract`、`op::gemv::Format`）。
//!
//! # 注解里的名字来自词汇表
//!
//! `dtype = "f32"` 是"闭集里的一个名字"。这份闭集不能是注解自己
//! 的一份手抄表：`DType` 增删一个变体时注解必须同时变，而不是安静地留着一个匹配不上
//! 的名字。因此闭集表写成**等式**（`syntax::DTYPES`）：
//!
//! - `DTYPES` 与「`cutile::reference::tolerance` 给出非零阈值」的 `DType` 集合逐项相等。
//!
//! 两边同时被核对：仓库级门禁 `tests/contracts.rs` 核对全量与映射规则（名字 → 变体名），
//! 每个注解生成的 `#[cfg(test)]` 模块核对**这一个调用点**用的确实是同一个变体
//! （`DType::parse_name(..) == Some(MODEL.dtype)`）。
//! 变体名不再在注解里手写一份映射：`dtype` 名的变体名取 `DType::variant`（缩写词全大写
//! 规则也住在 dtype 里），失败仍指向被标注的那一行——而不是另一张表上。
//!
//! **依赖**：仅 `dtype`（workspace 内部 L1 零依赖基座）。不引入 `syn` / `quote` / `proc-macro2`：
//! 注解文法只有四种键（`name = ".."`、`out = [..]`、`in = ..`、`ops = [".."]`），手写解析
//! 一百多行就能覆盖，换来的是 "workspace 零外部依赖"这条供应链契约持续成立
//! （见 `docs/ARCHITECTURE.md` 的依赖规则）。
//!
//! # 三层结构（可测性从这里来）
//!
//! | 层 | 文件 | 职责 | 能否单元测试 |
//! |---|---|---|---|
//! | 文法 | `syntax` | token 模型、文法解析、全部错误路径 | **能**，不接触 `proc_macro` |
//! | 渲染 | `codegen` | 把声明渲染成 `impl` + 校验测试 | **能**，只做字符串拼接 |
//! | 适配 | 本文件 | `TokenStream` ↔ token 模型、把错误映射回 span | 经由 `Streams` 抽象，**能** |
//!
//! 把文法与 `proc_macro` 解耦是这一层的核心决定：`proc_macro::TokenStream` 只在展开期可用，
//! 把规则直接写在它上面，等于让每条错误路径都无法被测试。适配层同样做了依赖倒置——
//! `Streams` 把"读 token、拼 token"抽成一个 trait，测试用字符串实现它，
//! 于是"出错时保留被标注项"这类容易退化的行为也有断言。
//!
//! # 为什么用注解而不是再写一个构造函数
//!
//! 层声明是三样东西的组合：名字、输出形状、要用的算子。手写时它们散落在 `new`、`impl Contract`
//! 与测试里，改名或改形状要改三处；写成注解后只有一处事实源，且两件事被编译器保证：
//!
//! 1. `out` 只接受十进制整数，`ops` 只接受字符串——写错类型是**编译错误**而不是运行期解析失败；
//! 2. 每个注解都会生成一条测试，调用一次 `Contract::spca()`，于是"算子被删了但注解没改"会在
//!    `cargo test` 里失败，而不是等到 emit 阶段才报未知算子。
//!
//! # 例子
//!
//! ```ignore
//! // `in` 是规约轴宽度：给了它，注解层就能把 `(in, out)` 交给算子语义表，
//! // 于是和 `Linear` 走同一条下界路径；不给，层照样合法，只是没有可计价的形状。
//! #[nn::spec(name = "lm_head", out = [32000], in = 4096, ops = ["gemv_i4"])]
//! pub struct LmHead;
//!
//! let layer: &dyn nn::Contract = &LmHead;
//! assert_eq!(layer.name(), "lm_head");
//! assert_eq!(layer.output().dims(), [32000]);
//! assert!(layer.bounds().expect("合法").is_some());
//! ```
//!
//! 生成的东西是确定的三块：原样保留的被标注项、一份 `impl nn::Contract`、一个
//! `#[cfg(test)]` 校验模块。没有隐藏状态，也没有运行期注册表——注册表会是第二个事实源。
//!
//! # 运行时形态
//!
//! 名字或宽度要由构造参数算出的层，把每个值写成 `$构造参数`：
//!
//! ```ignore
//! #[nn::spec(name = $name, out = [$tokens, $dim], ops = [$op], role = $role)]
//! pub struct Embedding {
//!     // `spec: ::nn::ContractSpec` 由注解注入（字段体第一个字段）
//! }
//! ```
//!
//! 注解注入私有的 `spec` 字段，并生成构造宏 `__layer_spec_Embedding!(name, tokens, dim, op, role)`；
//! 构造函数写 `spec: __layer_spec_Embedding!(...)?`，访问器仍由 `spec_accessors!` 读取 `self.spec`。
//! 构造宏与类型同模块（`macro_rules!` 的文本作用域），构造函数必须写在注解所在的模块里。
//! 运行期值无法在编译期核对，因此这种形态不生成校验测试；`in` 也不属于它——下界由层自己实现。
//!
//! # 错误
//!
//! 未知键、重复键、缺少必填键、非字面量、带转义序列的字符串、零宽轴、空算子集、泛型参数、
//! 非 `struct`/`enum` 的被标注项，都会以 `compile_error!` 报出**注解自己的文法**，
//! 并**指在写错的那个 token 上**（而不是笼统地指在被标注的类型上）。
//!
//! # Panics
//!
//! 生成的实现会在首次调用 `Contract::spca()` 时构造 `nn::ContractSpec`；若注解引用了 `op` 声明层
//! 里不存在的算子，这里会 panic 并说明是哪个层。正常情况下走不到：生成的校验测试会在
//! `cargo test` 里先失败。

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

mod codegen;
mod syntax;

#[cfg(test)]
mod fixture;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};
use syntax::{Bracket, Error as SyntaxError, Token};

/// 把 `#[spec(name = "..", out = [..], ops = [".."])]` 展开成 `nn::Contract` 的实现。
///
/// 被标注的类型必须是无泛型参数的 `struct` 或 `enum`：形状与算子列表在编译期固定，
/// 带泛型的类型无法用一份注解覆盖所有实例化。
///
/// 运行时形态把每个值写成 `$构造参数`（例如 `name = $name, out = [$tokens, $dim]`）：适配层给
/// 命名字段 `struct` 注入一个私有 `spec: ::nn::ContractSpec` 字段，并生成构造宏
/// `__layer_spec_<类型>!(...)`；构造函数用它建规格，`impl Contract` 的访问器仍由
/// `spec_accessors!` 从 `self.spec` 读取。`in` 不属于运行时形态——下界由层自己实现。
#[proc_macro_attribute]
pub fn spec(attribute: TokenStream, item: TokenStream) -> TokenStream {
    if syntax::is_runtime(&convert(&attribute.tokenize())) {
        expand_runtime(attribute, item)
    } else {
        expand_with(
            attribute,
            item,
            "spec",
            syntax::parse,
            codegen::render,
            syntax::target_name,
        )
    }
}

/// 展开所需的四个动作：读 token、由源码产出、拼接、重新定位 span。
///
/// 生产实现是 [`proc_macro::TokenStream`]；测试实现是纯字符串。因此本文件里的每条分支
/// （解析失败、生成失败）都不必在过程宏上下文里才能跑到。
trait Streams: Sized {
    /// 取出顶层 token。
    fn tokenize(&self) -> Vec<TokenTree>;

    /// 由源码片段构造；片段非法（例如括号不配平）时返回 `None`。
    fn from_code(code: &str) -> Option<Self>;

    /// 追加另一段流。
    fn append(self, other: Self) -> Self;

    /// 把整段流的 span 换成 `span`（`None` 表示保持原样）。
    fn respan(self, span: Option<Span>) -> Self;

    /// 在命名字段 `struct` 的字段体最前面插入一段字段（运行时 `#[spec]` 的 `spec` 字段）。
    ///
    /// 没有字段体时原样返回：调用前 [`syntax::runtime_target`] 已把这种情况拒绝掉。
    fn insert_field(self, field: &str) -> Self;

    /// 报错文案拼不成 token 流时的退路：一条固定文案的 `compile_error!`。
    fn fallback() -> Self;
}

impl Streams for TokenStream {
    fn tokenize(&self) -> Vec<TokenTree> {
        self.clone().into_iter().collect()
    }

    fn from_code(code: &str) -> Option<Self> {
        code.parse().ok()
    }

    fn append(self, other: Self) -> Self {
        let mut stream = self;
        stream.extend(other);
        stream
    }

    fn respan(self, span: Option<Span>) -> Self {
        let Some(span) = span else {
            return self;
        };
        self.into_iter()
            .map(|mut token| {
                token.set_span(span);
                token
            })
            .collect()
    }

    fn insert_field(self, field: &str) -> Self {
        let Ok(body) = field.parse::<TokenStream>() else {
            return self;
        };
        self.into_iter()
            .map(|token| match token {
                TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => {
                    let mut stream = body.clone();
                    stream.extend(group.stream());
                    let mut inserted = Group::new(Delimiter::Brace, stream);
                    inserted.set_span(group.span());
                    TokenTree::Group(inserted)
                }
                other => other,
            })
            .collect()
    }

    fn fallback() -> Self {
        Self::from_code("compile_error!(\"注解层无法构造错误消息\");").unwrap_or_default()
    }
}

/// 把 `#[model(name = "..", shape = [m, n], dtype = "..")]` 展开成 `MODEL` 常量、两条编译期
/// 断言与一条校验测试。
///
/// 注解承载的是**编译期数据**：注册表的键、生产形状、输出类型。行为不在注解里——它是被标注的
/// `type` 别名指向的那份格式契约（`Quantized<32>` / `Dense<32>`）：生成的 `MODEL` 把
/// `cost` / `sample` / `reference` 写成 `<名字 as Format>` 的单态化实例，`group` 取
/// `Format::GRAIN`。于是"注解说的格式"与"实际实现的格式"不可能各说各的：别名不实现
/// `Format` 就编译不过，生产形状不是粒度的整数倍也是编译错误。
///
/// 与 `#[spec]` 的分工一样，本注解不建注册表：注册表仍是 `REGISTRY` 一张，
/// 生成的测试只核对"注册表里的那一条就是这份 `MODEL`"，不另存一份事实。
#[proc_macro_attribute]
pub fn model(attribute: TokenStream, item: TokenStream) -> TokenStream {
    expand_with(
        attribute,
        item,
        "model",
        syntax::parse_model,
        codegen::render_model,
        syntax::model_target,
    )
}

/// 把一段声明式方言 spec 展开成 `impl ::llm_engine::dialect::Dialect`。
///
/// 用法（Metal 示例，CUDA 是同一张表的另一行）：
///
/// ```ignore
/// #[dialect(
///     entry = "kernel",
///     address_space = "device",
///     device_fn = "",
///     buffer = "    device {ctype}* d_{name} [[buffer({slot})]]",
///     point_param = "uint n [[thread_position_in_grid]]",
///     point_stmt = "",
///     grid_param = "uint3 tid [[thread_position_in_grid]]",
///     grid_stmt = "int g = (int)tid.x;",
/// )]
/// struct Metal;
/// ```
///
/// 八键全必填（缺省值是危险品：拼错就是一份错后端），值一律是普通字符串字面量，允许空串表示
/// 「这一维没有 vendor 差异」。见 [`syntax::parse_dialect`] 与 [`codegen::render_dialect`]。
#[proc_macro_attribute]
pub fn dialect(attribute: TokenStream, item: TokenStream) -> TokenStream {
    expand_with(
        attribute,
        item,
        "dialect",
        syntax::parse_dialect,
        codegen::render_dialect,
        syntax::dialect_target,
    )
}

/// 两道注解共用的一趟展开：解析 → 取类型名 → 渲染 → 拼接。
///
/// 文法与渲染以**具体函数**（不是 `dyn`）传进来：每一步都单态化，公共步骤——错误定位、
/// "失败也要留下被标注项"、拼接顺序——只有一份实现，于是两道注解不会各错一处。
fn expand_with<S, D>(
    attribute: S,
    item: S,
    name: &str,
    parse: fn(&[Token]) -> Result<D, SyntaxError>,
    render: fn(&D, &str) -> String,
    target: fn(&[Token], &str) -> Result<String, SyntaxError>,
) -> S
where
    S: Streams,
{
    match resolve(attribute, item, name, parse, target) {
        Err(output) => output,
        Ok((declaration, ty, item)) => attach(item, &render(&declaration, &ty)),
    }
}

/// 两道注解共用的前半趟：解析注解、取被标注类型名；失败时返回「原被标注项 + 编译错误」。
///
/// 成功时把未改动的被标注项交回调用方——静态形态直接拼接，运行时形态先注入 `spec` 字段。
fn resolve<S, D>(
    attribute: S,
    item: S,
    name: &str,
    parse: fn(&[Token]) -> Result<D, SyntaxError>,
    target: fn(&[Token], &str) -> Result<String, SyntaxError>,
) -> Result<(D, String, S), S>
where
    S: Streams,
{
    let arguments = attribute.tokenize();
    let body = item.tokenize();

    match parse(&convert(&arguments)) {
        Err(error) => Err(reject(item, &error, &arguments)),
        Ok(declaration) => match target(&convert(&body), name) {
            Err(error) => Err(reject(item, &error, &body)),
            Ok(ty) => Ok((declaration, ty, item)),
        },
    }
}

/// 运行时形态的一趟展开：注入 `spec` 字段，再生成由构造参数取值的构造宏。
fn expand_runtime<S: Streams>(attribute: S, item: S) -> S {
    match resolve(
        attribute,
        item,
        "spec",
        syntax::parse_runtime,
        syntax::runtime_target,
    ) {
        Err(output) => output,
        Ok((declaration, ty, item)) => {
            let item = item.insert_field("spec: ::nn::ContractSpec,");
            attach(item, &codegen::render_runtime(&declaration, &ty))
        }
    }
}

/// 把文法错误变成一条指向出错 token 的 `compile_error!`。
fn reject<S: Streams>(item: S, error: &SyntaxError, tokens: &[TokenTree]) -> S {
    let span = tokens.get(error.position).map(TokenTree::span);
    compile_error::<S>(&error.message, span).append(item)
}

/// 拼上生成的实现；生成失败时也要报一条自己的错误，而不是静默丢掉实现。
fn attach<S: Streams>(item: S, generated: &str) -> S {
    match S::from_code(generated) {
        Some(code) => item.append(code),
        None => item.append(compile_error::<S>(
            "内部错误：生成的代码不是合法 token 流，请把这条报给注解层",
            None,
        )),
    }
}

/// 生成一条 `compile_error!`；`span` 决定 IDE 把红线画在哪个 token 上。
fn compile_error<S: Streams>(message: &str, span: Option<Span>) -> S {
    S::from_code(&format!("compile_error!({message:?});"))
        .map(|code| code.respan(span))
        .unwrap_or_else(S::fallback)
}

/// `proc_macro` 的 token 流 → 本 crate 的 token 模型；两者一一对应，下标因此可以互换。
fn convert(tokens: &[TokenTree]) -> Vec<Token> {
    tokens.iter().map(convert_one).collect()
}

fn convert_one(input: &TokenTree) -> Token {
    match input {
        TokenTree::Ident(ident) => Token::Ident(ident.to_string()),
        TokenTree::Punct(punct) => Token::Punct(punct.as_char()),
        TokenTree::Literal(literal) => Token::Literal(literal.to_string()),
        TokenTree::Group(group) => Token::Group(
            bracket(group.delimiter()),
            convert(&group.stream().into_iter().collect::<Vec<_>>()),
        ),
    }
}

fn bracket(delimiter: Delimiter) -> Bracket {
    match delimiter {
        Delimiter::Parenthesis => Bracket::Paren,
        Delimiter::Bracket => Bracket::Square,
        Delimiter::Brace => Bracket::Brace,
        Delimiter::None => Bracket::Invisible,
    }
}

#[cfg(test)]
mod tests;
