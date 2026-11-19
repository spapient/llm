use super::*;

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
        match self.0.rfind('}') {
            Some(at) => Self(format!("{} {field} {}", &self.0[..at], &self.0[at..])),
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
        syntax::parse,
        codegen::render,
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
    assert_eq!(bracket(Delimiter::None), Bracket::Invisible);
    assert_ne!(bracket(Delimiter::None), Bracket::Square);
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

#[test]
fn a_field_is_inserted_at_the_front_of_the_struct_body() {
    let item = text("pub struct Head { len: usize }").insert_field("spec: ::nn::ContractSpec,");
    assert_eq!(
        item.0,
        "pub struct Head { spec: ::nn::ContractSpec, len: usize }"
    );
}
