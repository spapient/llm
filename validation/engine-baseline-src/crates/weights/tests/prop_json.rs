//! JSON 解析的性质：任意文本不 panic、结构化生成往返恒等、重复键必拒。
//!
//! 解析器吃的是**对端写的字节**：任意输入都不能 panic 是底线；生成→写文本→
//! 解析→比树则把"解析器悄悄改写结构"（丢键、换序、折算数字）钉死在等号上。

use llm_weights::json::{Value, parse};
use proptest::prelude::*;
use testkit::prop;

/// 递归生成值树：叶子是四种标量，容器深度与广度都受限，避免样本爆炸。
/// 对象键用集合去重——重复键是解析期拒绝的非法输入，不属于往返性质。
fn value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::Integer),
        any::<String>().prop_map(Value::String),
    ];
    leaf.prop_recursive(4, 32, 4, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..=4).prop_map(Value::Array),
            proptest::collection::hash_set(any::<String>(), 0..=4).prop_flat_map(move |keys| {
                let keys: Vec<String> = keys.into_iter().collect();
                let n = keys.len();
                (Just(keys), proptest::collection::vec(inner.clone(), n))
                    .prop_map(|(keys, vals)| Value::Object(keys.into_iter().zip(vals).collect()))
            }),
        ]
    })
}

/// 最小 JSON 写手：只负责把生成的树写成合法文本，转义规则与解析器互逆。
fn write(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Integer(n) => out.push_str(&n.to_string()),
        Value::Decimal(text) => out.push_str(text),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(item, out);
            }
            out.push(']');
        }
        Value::Object(fields) => {
            out.push('{');
            for (i, (key, item)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write(item, out);
            }
            out.push('}');
        }
    }
}

/// 字符串转义：引号、反斜杠与控制字符；其余字符（含多字节 UTF-8）原样写出。
fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

proptest! {
    #![proptest_config(prop::config())]

    /// 任意文本都不能让解析器 panic：错误可以，崩溃不行——头部字节来自文件，
    /// 文件可以是任何东西。
    #[test]
    fn parsing_never_panics_on_any_text(text in any::<String>()) {
        let _ = parse(&text);
    }

    /// 生成→写文本→解析是恒等：键序、整数写法、字符串内容都必须原样回来。
    #[test]
    fn every_generated_tree_round_trips(tree in value()) {
        let mut text = String::new();
        write(&tree, &mut text);
        let parsed = parse(&text).expect("自己写出的合法 JSON 必须可读");
        prop_assert_eq!(parsed, tree, "文本：{}", text);
    }

    /// 重复键在解析期即被拒绝：头部里同一个键写两遍是损坏或攻击，
    /// 悄悄取后者等于替对端做主。
    #[test]
    fn duplicate_keys_are_always_rejected(
        key in any::<String>(),
        first in any::<i64>(),
        second in any::<i64>(),
    ) {
        let mut text = String::new();
        text.push('{');
        write_string(&key, &mut text);
        text.push(':');
        text.push_str(&first.to_string());
        text.push(',');
        write_string(&key, &mut text);
        text.push(':');
        text.push_str(&second.to_string());
        text.push('}');
        prop_assert!(parse(&text).is_err(), "重复键被接受：{text}");
    }
}
