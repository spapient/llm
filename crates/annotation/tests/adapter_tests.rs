//! 适配层的单元测试：`Streams` 是替换缝，测试替身把"流"退化成字符串，于是适配层不必进入过程宏上下文。
//!
//! 覆盖边界是显式的：`expand_*` 的**成功**路径要 `tokenize` 出真实 `TokenTree`，而过程宏 API 在
//! 上下文之外不可用，所以这里只钉住它调用的每一步及其顺序；生成文本由 `render_tests` 逐字钉，
//! 真实的宏调用由本仓 89 个算子与各层结构体的编译本身覆盖。

use super::*;
use crate::fixture::lex;

/// 测试用"流"：字符串本身即容器，因此适配层不必进入过程宏上下文。
///
/// `from_code` 复刻词法层唯一会拒绝的东西——括号不配平，用它覆盖"生成失败"的分支。
#[derive(Clone, Debug)]
struct Text(String);

impl Streams for Text {
    fn tokenize(&self) -> Vec<TokenTree> {
        Vec::new()
    }

    fn from_code(code: &str) -> Option<Self> {
        let mut depth: i32 = 0;
        for character in code.chars() {
            match character {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return None;
            }
        }
        (depth == 0).then(|| Self(code.to_string()))
    }

    fn append(self, other: Self) -> Self {
        Self(format!("{} {}", self.0, other.0))
    }

    fn respan(self, _span: Option<Span>) -> Self {
        self
    }

    fn insert_field(self, field: &str) -> Self {
        match self.0.find('{') {
            Some(at) => Self(format!("{} {field}{}", &self.0[..=at], &self.0[at + 1..])),
            None => self,
        }
    }

    fn fallback() -> Self {
        Self("compile_error!(\"注解层无法构造错误消息\");".to_string())
    }
}

fn text(code: &str) -> Text {
    Text(code.to_string())
}

#[test]
fn a_rejected_annotation_keeps_the_annotated_item() {
    let expanded = expand_with(
        text(r#"name = "x""#),
        text("pub struct Head;"),
        "spec",
        decl::parse,
        render::render,
        syntax::target_name,
    );
    assert!(expanded.0.contains("compile_error!"), "{}", expanded.0);
    assert!(
        expanded.0.contains("pub struct Head;"),
        "被标注项必须原样保留：{}",
        expanded.0
    );
}

#[test]
fn a_failed_generation_reports_its_own_error_instead_of_dropping_the_impl() {
    let expanded = attach::<Text>(text("pub struct Head;"), "impl Head {");
    assert!(expanded.0.contains("compile_error!"), "{}", expanded.0);
    assert!(
        expanded.0.starts_with("pub struct Head;"),
        "报错也必须保留被标注项：{}",
        expanded.0
    );
}

#[test]
fn a_successful_generation_is_appended_after_the_item() {
    let expanded = attach::<Text>(text("pub struct Head;"), "impl Head {}");
    assert_eq!(expanded.0, "pub struct Head; impl Head {}");
}

#[test]
fn the_error_message_survives_the_round_trip() {
    let code = compile_error::<Text>("注解键 `x` 重复：名字只能声明一次", None);
    assert!(
        code.0.contains("注解键 `x` 重复：名字只能声明一次"),
        "{}",
        code.0
    );
    assert!(code.0.starts_with("compile_error!"), "{}", code.0);
}

#[test]
fn a_transparent_group_is_not_mistaken_for_a_bracket_group() {
    assert_eq!(
        bracket(Delimiter::None),
        Bracket::Invisible,
        "透明组没有可见括号"
    );
}

#[test]
fn a_runtime_annotation_rejection_keeps_the_annotated_item() {
    let expanded = expand_runtime(
        text(r#"name = $name, out = [$tokens], ops = [$op]"#),
        text("pub struct Head { len: usize }"),
    );
    assert!(expanded.0.contains("compile_error!"), "{}", expanded.0);
    assert!(
        expanded.0.contains("pub struct Head { len: usize }"),
        "被标注项必须原样保留：{}",
        expanded.0
    );
}

/// `expand_runtime` 执行的组合顺序：先注入字段，再把构造器宏接在后面。
///
/// 成功路径本身在单测里不可达——`resolve` 要 `tokenize` 出真实 `TokenTree`，而 `Span::call_site()`
/// 在过程宏上下文之外 panic；这条用例钉住的是那两步行之有效的组合与顺序。
#[test]
fn the_runtime_expansion_inserts_the_slot_before_the_constructor() {
    let declaration =
        decl::parse_runtime(&lex(r#"name = $name, out = [$tokens, $dim], ops = [$op]"#))
            .expect("合法运行时注解");
    let item = text("pub struct Head { len: usize }");
    let expanded = attach(
        item.insert_field("spec: ::nn::ContractSpec,"),
        &render::render_runtime(&declaration, "Head"),
    );
    assert!(
        expanded
            .0
            .starts_with("pub struct Head { spec: ::nn::ContractSpec, len: usize }"),
        "参数槽位是第一个字段：{}",
        expanded.0
    );
    assert!(
        expanded.0.contains("macro_rules! __layer_spec_Head"),
        "构造器宏接在被标注项之后：{}",
        expanded.0
    );
}

#[test]
fn a_field_is_inserted_at_the_front_of_the_struct_body() {
    let item = text("pub struct Head { len: usize }").insert_field("spec: ::nn::ContractSpec,");
    assert_eq!(
        item.0,
        "pub struct Head { spec: ::nn::ContractSpec, len: usize }"
    );
}
