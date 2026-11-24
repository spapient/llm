//! 注解层：把契约与 kernel 声明写成属性，由编译器检查，而不是手写重复样板。
//!
//! **拥有**：`#[spec]` / `#[model]` 契约宏、`#[parallel]` 并行策略宏与 `#[dsl]` kernel 前端。宏只在调用点的 crate 里展开，
//! 生成代码指向契约或 IR 路径；kernel 语法由 `syn` 解析、`quote` 生成，编译期依赖不进入运行期依赖图。
//!
//! 注解承载的只是**可静态判定**的契约：层是（名字、输出形状、算子），语义表是（注册名、生产形状、
//! 输出类型、骨架）。行为不在注解里，而在被标注类型所实现的那份 trait 上——两侧由 trait bound 绑死，
//! 于是"注解说的"与"实际实现的"不可能各说各的。
//!
//! # 三层，每层都能单独测
//!
//! | 层 | 文件 | 职责 |
//! |---|---|---|
//! | 文法 | `syntax` | token 模型、游标、逐键读取、被标注项判据；不接触 `proc_macro` |
//! | 声明 | `decl` | 三种声明的键表与解析产物；加一个键 = 加一行 |
//! | 渲染 | `render` | 声明 → 源码；只做字符串拼接 |
//!
//! 适配在 [`spec`] / [`model`] 两个入口里，经由 `Streams` 做了依赖倒置：`proc_macro::TokenStream`
//! 只在展开期可用，把规则直接写在它上面等于让每条错误路径都无法被测试。测试用字符串实现
//! `Streams`，于是"出错时仍保留被标注项"这类容易退化的行为也有断言。
//!
//! # 例子
//!
//! ```ignore
//! // `in` 是规约轴宽度：给了它，注解层就能把 `(in, out)` 交给算子语义表；不给，层照样合法，
//! // 只是没有可计价的形状。
//! #[nn::spec(name = "lm_head", out = [32000], in = 4096, ops = ["gemv_i4"])]
//! pub struct LmHead;
//! ```
//!
//! 生成的东西是确定的三块：原样保留的被标注项、一份 `impl ::nn::Contract`、一个 `#[cfg(test)]`
//! 校验模块。没有隐藏状态，也没有运行期注册表——注册表会是第二个事实源。
//!
//! # 运行时形态
//!
//! 值要由构造参数算出的层，把每个值写成 `$构造参数`：
//!
//! ```ignore
//! #[nn::spec(name = $name, out = [$tokens, $dim], ops = [$op], role = $role)]
//! pub struct Embedding {}
//! ```
//!
//! 注解注入私有的 `spec: ::nn::ContractSpec` 字段（字段体第一个），并生成构造宏
//! `__layer_spec_Embedding!(name, tokens, dim, op, role)`。宏与类型同模块（`macro_rules!` 的文本
//! 作用域），构造函数必须写在注解所在的模块里。运行期值无法在编译期核对，因此这种形态不生成校验
//! 测试；`in` 也不属于它——下界由层自己实现。
//!
//! # 错误
//!
//! 未知键、重复键、缺少必填键、非字面量、带转义序列的字符串、零宽轴、空算子集、泛型参数、
//! 非 `struct`/`enum` 的被标注项，都以 `compile_error!` 报出**注解自己的文法**，并**指在写错的
//! 那个 token 上**（而不是笼统地指在被标注的类型上）。
//!
//! # Panics
//!
//! 生成的实现会在首次调用 `Contract::spca()` 时构造 `::nn::ContractSpec`；若注解引用了 `nn`
//! 声明层里不存在的算子，那里会 panic 并说明是哪个层。正常情况下走不到：生成的校验测试会在
//! `cargo test` 里先失败。

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

mod decl;
mod dsl;
mod ops;
mod render;
mod syntax;

#[cfg(test)]
#[path = "../tests/fixture.rs"]
mod fixture;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};
use std::path::Path;
use syntax::{Bracket, Error as SyntaxError, Token};

/// 把 `#[spec(name = "..", out = [..], in = .., ops = [".."])]` 展开成 `::nn::Contract` 的实现。
#[proc_macro_attribute]
pub fn spec(attribute: TokenStream, item: TokenStream) -> TokenStream {
    if decl::is_runtime(&convert(&attribute.tokenize())) {
        expand_runtime(attribute, item)
    } else {
        expand_with(
            attribute,
            item,
            "spec",
            decl::parse,
            render::render,
            syntax::target_name,
        )
    }
}

/// 展开所需的四个动作：读 token、由源码产出、拼接、重新定位 span。
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

/// 把 `#[model(name = "..", shape = [m, n], dtype = "..")]` 展开成 `MODEL` 常量与一组编译期
/// 断言。
#[proc_macro_attribute]
pub fn model(attribute: TokenStream, item: TokenStream) -> TokenStream {
    expand_with(
        attribute,
        item,
        "model",
        decl::parse_model,
        render::render_model,
        syntax::model_target,
    )
}

/// 把 `#[parallel(tp = .., pp = .., dp = .., reduce = "..", average = .., priority = "..")]`
/// 展开成一条返回 `shard::ParallelConfig` 的策略访问器（注解层不依赖 shard，路径字符串推迟到消费方解析）。
///
/// 键全部可选，缺省口径与 `ParallelConfig::single()` 一致；归约种类与优先级在注解层就按闭集校验。
#[proc_macro_attribute]
pub fn parallel(attribute: TokenStream, item: TokenStream) -> TokenStream {
    expand_with(
        attribute,
        item,
        "parallel",
        decl::parse_parallel,
        render::render_parallel,
        syntax::target_name,
    )
}

/// 将受限的 Rust 逐点 kernel 函数转换为 `cutile::kernel::Kernel` 构造器。
///
/// 第一版接受一层 `for i in 0..n` 循环、一个 `out[i] = expr` 写入，以及 f32 标量/缓冲区。
/// 原函数保留为 CPU 参考实现；生成的 `__dsl_ir_<name>()` 返回可验证、可优化、可发射的 typed IR。
#[proc_macro_attribute]
pub fn dsl(attribute: TokenStream, item: TokenStream) -> TokenStream {
    dsl::expand(attribute, item)
}

/// 编译期扫调用 crate 的 `src/ops/*.toml`，展开成 `pub const OPS: [OpView; N]` 静态表。
///
/// 解析与渲染都在宏进程内完成；生成代码只持有强类型 `OpView`，不残留 `Table` / `Value`
/// 动态树。语料读不到 / 闭集越界时退一条指向具体原因的 `compile_error!`。
#[proc_macro]
pub fn prof_ops(_input: TokenStream) -> TokenStream {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let ops_dir = Path::new(&manifest).join("src").join("ops");
    let code = match ops::collect(&ops_dir) {
        Ok(models) => render::render_ops(&models),
        Err(error) => format!("compile_error!({error:?});"),
    };
    code.parse().unwrap_or_else(|_| {
        r#"compile_error!("注解层内部错误：生成代码不是合法 token 流");"#
            .parse()
            .unwrap_or_default()
    })
}

/// 两道注解共用的一趟展开：解析 → 取类型名 → 渲染 → 拼接。
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
        decl::parse_runtime,
        decl::runtime_target,
    ) {
        Err(output) => output,
        Ok((declaration, ty, item)) => {
            let item = item.insert_field("spec: ::nn::ContractSpec,");
            attach(item, &render::render_runtime(&declaration, &ty))
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
        .map_or_else(S::fallback, |code| code.respan(span))
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
#[path = "../tests/decl_tests.rs"]
mod decl_tests;

#[cfg(test)]
#[path = "../tests/render_tests.rs"]
mod render_tests;

#[cfg(test)]
#[path = "../tests/adapter_tests.rs"]
mod tests;
