//! A minimal Jinja subset for rendering `chat_template` strings.
//!
//! The subset covers the constructs used by mainstream chat templates:
//! interpolation (`{{ expr }}`), a `for` loop over `messages`, and an
//! `if`/`elif`/`else` chain. Expressions resolve string literals, `bos_token`,
//! `eos_token`, the current message's `role`/`content`, and `loop` metadata.
//! Anything outside this subset is rejected as [`tokenizererror::malformedtemplate`].

use crate::chat::chatmessage;
use crate::error::tokenizererror;

/// A lexed template node.
#[derive(Debug)]
enum node {
    /// Literal text.
    text(String),
    /// An interpolation `{{ ... }}`.
    expr(String),
    /// A statement `{% ... %}`.
    stmt(String),
}

/// A value produced by an expression.
#[derive(Debug)]
enum value {
    string(String),
    boolean(bool),
}

impl value {
    fn intostring(self) -> String {
        match self {
            Self::string(text) => text,
            Self::boolean(flag) => flag.to_string(),
        }
    }
}

/// The message currently in scope for a loop iteration.
struct current<'a> {
    message: &'a chatmessage,
    index: usize,
    first: bool,
    last: bool,
}

/// The full evaluation context.
struct context<'a> {
    messages: &'a [chatmessage],
    bos: &'a str,
    eos: &'a str,
    current: Option<current<'a>>,
}

/// Renders `template` with `messages`, `bos`, and `eos`.
pub(crate) fn renderjinja(
    template: &str,
    messages: &[chatmessage],
    bos: &str,
    eos: &str,
) -> Result<String, tokenizererror> {
    let nodes = tokenize(template)?;
    let context = context {
        messages,
        bos,
        eos,
        current: None,
    };
    let mut out = String::new();
    renderrange(&nodes, 0, nodes.len(), &mut out, &context)?;
    Ok(out)
}

/// Renders `nodes[start..end]` into `out`.
fn renderrange(
    nodes: &[node],
    start: usize,
    end: usize,
    out: &mut String,
    context: &context,
) -> Result<(), tokenizererror> {
    let mut index = start;
    while index < end {
        match &nodes[index] {
            node::text(text) => {
                out.push_str(text);
                index += 1;
            }
            node::expr(expr) => {
                out.push_str(&evalexpr(expr, context)?.intostring());
                index += 1;
            }
            node::stmt(statement) => {
                let statement = statement.trim();
                if let Some(remainder) = statement.strip_prefix("for ") {
                    if !remainder.ends_with(" in messages") {
                        return Err(malformed("unsupported for loop"));
                    }
                    let body_end = matching(nodes, index, "for", "endfor")?;
                    let total = context.messages.len();
                    for (position, message) in context.messages.iter().enumerate() {
                        let subcontext = context {
                            messages: context.messages,
                            bos: context.bos,
                            eos: context.eos,
                            current: Some(current {
                                message,
                                index: position + 1,
                                first: position == 0,
                                last: position + 1 == total,
                            }),
                        };
                        renderrange(nodes, index + 1, body_end, out, &subcontext)?;
                    }
                    index = body_end + 1;
                } else if let Some(condition) = statement.strip_prefix("if ") {
                    let (taken, next) = renderif(nodes, index, end, out, context, condition)?;
                    index = next;
                    if !taken {
                        // nothing; index already advanced past the whole chain
                    }
                } else {
                    return Err(malformed("unexpected statement"));
                }
            }
        }
    }
    Ok(())
}

/// Renders one `if`/`elif`/`else` chain starting at the `if` node `start`.
///
/// Returns whether any branch was taken and the index just past the matching
/// `endif`.
fn renderif(
    nodes: &[node],
    start: usize,
    end: usize,
    out: &mut String,
    context: &context,
    mut condition: &str,
) -> Result<(bool, usize), tokenizererror> {
    let mut index = start + 1;
    loop {
        // Find the next branch boundary: elif, else, or endif.
        let boundary = findboundary(nodes, index, end)?;
        let body_end = boundary.0;
        let boundary_stmt = boundary.1;
        let branch = evalexpr(condition, context)?;
        if matches!(branch, value::boolean(true)) {
            renderrange(nodes, index, body_end, out, context)?;
            // Skip the rest of the chain to its endif.
            let endif = if boundary_stmt == "endif" {
                body_end
            } else {
                matching(nodes, start, "if", "endif")?
            };
            return Ok((true, endif + 1));
        }
        match boundary_stmt {
            "endif" => return Ok((false, body_end + 1)),
            "else" => {
                let endif = matching(nodes, start, "if", "endif")?;
                renderrange(nodes, body_end + 1, endif, out, context)?;
                return Ok((true, endif + 1));
            }
            other => {
                // `elif <cond>`
                condition = other
                    .strip_prefix("elif ")
                    .ok_or_else(|| malformed("malformed elif"))?;
                index = body_end + 1;
            }
        }
    }
}

/// Returns `(boundary_index, boundary_statement)` for the next `elif`/`else`/
/// `endif` at the same nesting level as `start`, scanning from `index`.
fn findboundary(
    nodes: &[node],
    index: usize,
    end: usize,
) -> Result<(usize, &'static str), tokenizererror> {
    let mut depth = 0usize;
    let mut cursor = index;
    while cursor < end {
        if let node::stmt(statement) = &nodes[cursor] {
            let statement = statement.trim();
            if statement.starts_with("if ") {
                depth += 1;
            } else if statement == "endif" {
                if depth == 0 {
                    return Ok((cursor, "endif"));
                }
                depth -= 1;
            } else if statement == "else" && depth == 0 {
                return Ok((cursor, "else"));
            } else if statement.starts_with("elif ") && depth == 0 {
                return Ok((cursor, "elif"));
            }
        }
        cursor += 1;
    }
    Err(malformed("unterminated if"))
}

/// Returns the index of the `close` statement matching the `open` statement at
/// `start`, accounting for nesting.
fn matching(
    nodes: &[node],
    start: usize,
    open: &str,
    close: &str,
) -> Result<usize, tokenizererror> {
    let mut depth = 0usize;
    for (cursor, node) in nodes.iter().enumerate().skip(start + 1) {
        if let node::stmt(statement) = node {
            let statement = statement.trim();
            if statement.starts_with(open) {
                depth += 1;
            } else if statement == close {
                if depth == 0 {
                    return Ok(cursor);
                }
                depth -= 1;
            }
        }
    }
    Err(malformed("unterminated statement"))
}

/// Evaluates an expression to a value.
fn evalexpr(expr: &str, context: &context) -> Result<value, tokenizererror> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err(malformed("empty expression"));
    }
    if let Some(inner) = unquote(expr) {
        return Ok(value::string(inner.to_string()));
    }
    if expr == "True" {
        return Ok(value::boolean(true));
    }
    if expr == "False" {
        return Ok(value::boolean(false));
    }
    if let Some(operand) = expr.strip_prefix("not ") {
        let value = evalexpr(operand, context)?;
        return Ok(value::boolean(!istrue(&value)));
    }
    if let Some((left, right)) = splitop(expr, " and ") {
        let l = istrue(&evalexpr(left, context)?);
        let r = istrue(&evalexpr(right, context)?);
        return Ok(value::boolean(l && r));
    }
    if let Some((left, right)) = splitop(expr, " or ") {
        let l = istrue(&evalexpr(left, context)?);
        let r = istrue(&evalexpr(right, context)?);
        return Ok(value::boolean(l || r));
    }
    if let Some((left, right)) = splitop(expr, " == ") {
        let l = evalexpr(left, context)?.intostring();
        let r = evalexpr(right, context)?.intostring();
        return Ok(value::boolean(l == r));
    }
    if let Some((left, right)) = splitop(expr, " != ") {
        let l = evalexpr(left, context)?.intostring();
        let r = evalexpr(right, context)?.intostring();
        return Ok(value::boolean(l != r));
    }
    evalterm(expr, context)
}

/// Splits a `base.field` or `base['field']` term into `(base, field)`.
fn splitfield(term: &str) -> Option<(&str, &str)> {
    if let Some((base, field)) = term.split_once('.') {
        return Some((base, field));
    }
    if let Some((base, rest)) = term.split_once('[') {
        let field = rest
            .strip_suffix(']')?
            .trim_matches(|character| character == '\'' || character == '"');
        return Some((base, field));
    }
    None
}

/// Evaluates a single term: an identifier or a field access.
fn evalterm(term: &str, context: &context) -> Result<value, tokenizererror> {
    let term = term.trim();
    match term {
        "bos_token" => return Ok(value::string(context.bos.to_string())),
        "eos_token" => return Ok(value::string(context.eos.to_string())),
        _ => {}
    }
    if let Some((base, field)) = splitfield(term) {
        return evalfield(base, field, context);
    }
    Err(malformed(&format!("unknown expression {term:?}")))
}

/// Evaluates a `base.field` or `base['field']` access.
fn evalfield(base: &str, field: &str, context: &context) -> Result<value, tokenizererror> {
    let base = base.trim();
    let field = field.trim();
    if base == "message" {
        let current = context
            .current
            .as_ref()
            .ok_or_else(|| malformed("message outside a loop"))?;
        return match field {
            "role" => Ok(value::string(current.message.role.getname().to_string())),
            "content" => Ok(value::string(current.message.content.clone())),
            other => Err(malformed(&format!("unknown message field {other:?}"))),
        };
    }
    if base == "loop" {
        let current = context
            .current
            .as_ref()
            .ok_or_else(|| malformed("loop outside a loop"))?;
        return match field {
            "last" => Ok(value::boolean(current.last)),
            "first" => Ok(value::boolean(current.first)),
            "index" => Ok(value::string(current.index.to_string())),
            "index0" => Ok(value::string((current.index - 1).to_string())),
            other => Err(malformed(&format!("unknown loop field {other:?}"))),
        };
    }
    Err(malformed(&format!("unknown base {base:?}")))
}

/// Splits `expr` on the first occurrence of `op`, if present.
fn splitop<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    expr.split_once(op)
}

/// Returns the contents of a single- or double-quoted string literal.
fn unquote(expr: &str) -> Option<&str> {
    if expr.len() >= 2 {
        let bytes = expr.as_bytes();
        let first = *bytes.first()?;
        let last = *bytes.last()?;
        if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
            return Some(&expr[1..expr.len() - 1]);
        }
    }
    None
}

/// Reports whether a value is truthy.
fn istrue(value: &value) -> bool {
    matches!(value, value::boolean(true))
}

/// Builds a [`tokenizererror::malformedtemplate`].
fn malformed(detail: &str) -> tokenizererror {
    tokenizererror::malformedtemplate {
        detail: detail.to_string(),
    }
}

/// Tokenizes `template` into nodes, taking whichever of `{{` or `{%` comes
/// first at each step.
fn tokenize(template: &str) -> Result<Vec<node>, tokenizererror> {
    let mut nodes = Vec::new();
    let mut rest = template;
    while !rest.is_empty() {
        let expr = rest.find("{{");
        let stmt = rest.find("{%");
        let (pos, is_stmt) = match (expr, stmt) {
            (Some(expr_pos), Some(stmt_pos)) if expr_pos <= stmt_pos => (expr_pos, false),
            (Some(_), Some(stmt_pos)) => (stmt_pos, true),
            (Some(expr_pos), None) => (expr_pos, false),
            (None, Some(stmt_pos)) => (stmt_pos, true),
            (None, None) => {
                nodes.push(node::text(rest.to_string()));
                break;
            }
        };
        if pos > 0 {
            nodes.push(node::text(rest[..pos].to_string()));
        }
        let after_open = &rest[pos + 2..];
        if is_stmt {
            let close = after_open
                .find("%}")
                .ok_or_else(|| malformed("unterminated {%"))?;
            nodes.push(node::stmt(after_open[..close].trim().to_string()));
            rest = &after_open[close + 2..];
        } else {
            let close = after_open
                .find("}}")
                .ok_or_else(|| malformed("unterminated {{"))?;
            nodes.push(node::expr(after_open[..close].trim().to_string()));
            rest = &after_open[close + 2..];
        }
    }
    Ok(nodes)
}
