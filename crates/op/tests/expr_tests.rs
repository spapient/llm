use super::*;

fn error(text: &str) -> String {
    Expr::parse(text).expect_err("应当被拒绝").to_string()
}

#[test]
fn an_explicit_expression_keeps_terms_and_output_order() {
    let expr = Expr::parse("nm,m,ng,n->n").expect("合法");
    assert_eq!(expr.arity(), 4);
    assert_eq!(expr.output(), ['n']);
    let terms: Vec<&[char]> = expr.terms().collect();
    assert_eq!(terms, [['n', 'm'].as_slice(), &['m'], &['n', 'g'], &['n']]);
    assert_eq!(expr.text(), "nm,m,ng,n->n");
    assert_eq!(expr.to_string(), "nm,m,ng,n->n");
}

#[test]
fn whitespace_is_trimmed_but_not_interpreted() {
    let expr = Expr::parse("  i,j->ij \n").expect("合法");
    assert_eq!(expr.text(), "i,j->ij");
    assert_eq!(expr.arity(), 2);
}

#[test]
fn a_scalar_output_and_scalar_terms_are_allowed() {
    let expr = Expr::parse("m,->").expect("标量输出合法");
    assert_eq!(expr.arity(), 2);
    assert!(expr.output().is_empty());
    let terms: Vec<&[char]> = expr.terms().collect();
    assert_eq!(terms, [['m'].as_slice(), &[]]);
}

#[test]
fn an_empty_expression_is_rejected() {
    assert!(error("").contains("为空"));
    assert!(error("   ").contains("为空"));
}

#[test]
fn a_missing_arrow_is_rejected_with_an_example() {
    let message = error("nm,m");
    assert!(message.contains("缺少 `->`"), "{message}");
    assert!(message.contains("nm,m->n"), "{message}");
}

#[test]
fn a_second_arrow_is_rejected() {
    assert!(error("nm,m->n->n").contains("只能出现一次"));
}

#[test]
fn a_non_letter_is_rejected_with_its_position() {
    let message = error("n1,m->n");
    assert!(message.contains("第 0 项"), "{message}");
    assert!(message.contains("`1`"), "{message}");
}

#[test]
fn a_repeated_letter_inside_one_term_is_rejected() {
    let message = error("nn,m->n");
    assert!(message.contains("重复"), "{message}");
    assert!(
        message.contains("对角线") || message.contains("对角"),
        "{message}"
    );
}

#[test]
fn a_repeated_output_letter_is_rejected() {
    assert!(error("nm->nn").contains("输出轴必须唯一"));
}

#[test]
fn an_output_axis_that_no_input_mentions_is_rejected() {
    let message = error("nm->nk");
    assert!(message.contains("输出轴 `k`"), "{message}");
}

#[test]
fn parsing_through_from_str_is_the_same_path() {
    let parsed: Expr = "ij,jk->ik".parse().expect("合法");
    assert_eq!(parsed, Expr::parse("ij,jk->ik").expect("合法"));
}
