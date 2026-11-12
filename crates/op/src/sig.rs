//! 内核签名：从 `.k` 的入口函数声明解析形参，供执行侧派生实参列表与输出缓冲。
//!
//! ABI 过去有三份事实源——模板的 C 签名、样本里"按签名顺序"的缓冲、算子手写的实参字符串——
//! 三者只在 `cc` 编译期对齐，错了报的是 C 的 `incompatible pointer type`，不带算子与缓冲名。
//! 这里把模板变成唯一事实源：装载期解析签名，[`Signature::check`] 再把它与样本逐项比对，
//! 因此类型或名字漂移在 emit 之前就失败，诊断带算子名与形参名。
//!
//! 只支持 P0 需要的那一小撮 C 声明：指针缓冲（`float`、`unsigned char`/`u8`）与整数标量。
//! 不追求通用 C 解析器——遇到闭集之外的类型就报错，而不是猜。

use crate::model::{Sample, Values};
use diagnostic::{Code, Error};

/// 形参的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CType {
    /// `float`（样本里是 [`Values::F32`]）。
    F32,
    /// `unsigned char` / `u8`（样本里是 [`Values::U8`]）。
    U8,
    /// 整数标量（`int`、`long` 等），取值来自 [`Sample::scalars`]。
    Int,
}

impl CType {
    /// C 侧的类型名（诊断与文档用）。
    pub fn c_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::U8 => "unsigned char",
            Self::Int => "int",
        }
    }

    /// 是否对应一个样本缓冲（标量不是缓冲）。
    pub fn is_buffer(self) -> bool {
        !matches!(self, Self::Int)
    }
}

/// 一个形参：名字、类型与指针是否可写。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// 形参名（也是样本里的缓冲名）。
    pub name: String,
    /// 类型。
    pub ctype: CType,
    /// 指针形参是否没有 `const`（可写）；标量恒为 `false`。
    pub mutable: bool,
}

/// 签名错误：`.k` 是输入，因此类别恒为 `invalid`；行号作为一层上下文，文件由调用方补。
fn at(line: usize, detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, detail).with(format!("第 {line} 行"))
}

/// 一次解析得到的内核签名。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    op: String,
    params: Vec<Param>,
}

impl Signature {
    /// 从 `<name>.k` 的入口函数声明解析签名。
    ///
    /// 入口名固定为 `cutile_{op}_kernel`（模板里写作 `cutile_{{op}}_kernel`）。
    ///
    /// # Errors
    ///
    /// 找不到入口、括号不配对、形参类型不在闭集内、名字重复，或可写缓冲不是恰好一个时返回
    /// 带行号与算子名的说明。
    pub fn parse(op: &str, template: &str) -> diagnostic::Result<Self> {
        Self::scan(op, template).map_err(|error| error.with(format!("{op}.k")))
    }

    /// 解析主体；错误带上行号，文件帧由 [`Self::parse`] 统一追加。
    fn scan(op: &str, template: &str) -> diagnostic::Result<Self> {
        // 模板保留 `{{op}}` 插值；直接检查已展开的文本也接受（测试与将来的内联模板）。
        let expanded = format!("cutile_{op}_kernel(");
        let entry = [expanded.as_str(), "cutile_{{op}}_kernel("]
            .into_iter()
            .find(|entry| template.contains(entry))
            .ok_or_else(|| {
                at(
                    1,
                    format!("算子 `{op}` 的模板缺少内核入口 `cutile_{{op}}_kernel(`"),
                )
            })?;
        let start = template.find(entry).expect("刚检查过入口存在");
        let open = start + entry.len() - 1;
        let close = matching(template, open).ok_or_else(|| {
            at(
                line_of(template, open),
                format!("算子 `{op}` 的内核签名括号不配对"),
            )
        })?;
        let inner = &template[open + 1..close];

        let mut params = Vec::new();
        if inner.trim() != "void" && !inner.trim().is_empty() {
            for (offset, text) in split(inner) {
                // 参数可能从下一行开始：行号要落在非空白字符上，而不是逗号之后。
                let indent = inner[offset..].len() - inner[offset..].trim_start().len();
                let line = line_of(template, open + 1 + offset + indent);
                params.push(parse_param(op, text, line)?);
            }
        }

        let signature = Self {
            op: op.to_string(),
            params,
        };
        signature.check_shape()?;
        Ok(signature)
    }

    /// 入口函数名（`cutile_{op}_kernel`）。
    pub fn kernel(&self) -> String {
        format!("cutile_{}_kernel", self.op)
    }

    /// 全部形参（按签名顺序）。
    pub fn params(&self) -> &[Param] {
        &self.params
    }

    /// 唯一的可写输出缓冲名。
    pub fn output(&self) -> Option<&str> {
        self.params
            .iter()
            .find(|param| param.ctype.is_buffer() && param.mutable)
            .map(|param| param.name.as_str())
    }

    /// 把签名与样本逐项对齐：缓冲要同名同类型，标量要有取值，样本也不能多出签名之外的名字。
    ///
    /// # Errors
    ///
    /// 任一不匹配都返回带算子名、形参名与两侧类型的说明。
    pub fn check(&self, sample: &Sample) -> diagnostic::Result<()> {
        for param in &self.params {
            if param.ctype.is_buffer() {
                let buffer = sample.buffer(&param.name).ok_or_else(|| {
                    Error::new(
                        Code::Missing,
                        format!(
                            "算子 `{}` 的内核签名有缓冲 `{}`，样本里没有同名缓冲",
                            self.op, param.name
                        ),
                    )
                })?;
                let actual = describe(&buffer.values);
                if !type_matches(param.ctype, &buffer.values) {
                    return Err(Error::new(
                        Code::Invalid,
                        format!(
                            "算子 `{}` 的缓冲 `{}`：`.k` 声明为 `{} *`，样本给的是 `{actual}`",
                            self.op,
                            param.name,
                            param.ctype.c_name()
                        ),
                    ));
                }
            } else if sample.scalar(&param.name).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!(
                        "算子 `{}` 的标量实参 `{}` 没有取值：样本里缺少同名标量",
                        self.op, param.name
                    ),
                ));
            }
        }

        for buffer in &sample.buffers {
            if !self
                .params
                .iter()
                .any(|param| param.ctype.is_buffer() && param.name == buffer.name)
            {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "算子 `{}` 的样本缓冲 `{}` 不在 `.k` 签名里",
                        self.op, buffer.name
                    ),
                ));
            }
        }
        for (name, _) in &sample.scalars {
            if !self
                .params
                .iter()
                .any(|param| !param.ctype.is_buffer() && param.name == *name)
            {
                return Err(Error::new(
                    Code::Invalid,
                    format!("算子 `{}` 的样本标量 `{name}` 不在 `.k` 签名里", self.op),
                ));
            }
        }
        Ok(())
    }

    /// 按签名顺序渲染内核实参：缓冲名原样、标量取样本里的值。
    ///
    /// # Errors
    ///
    /// 签名与样本不一致时先由 [`Self::check`] 返回说明。
    pub fn call_args(&self, sample: &Sample) -> diagnostic::Result<String> {
        self.check(sample)?;
        let mut parts = Vec::with_capacity(self.params.len());
        for param in &self.params {
            if param.ctype.is_buffer() {
                parts.push(param.name.clone());
            } else {
                // `check` 刚过，这里取不到只能是实现与校验不同步——内部不变量，不是调用方的错。
                let value = sample.scalar(&param.name).ok_or_else(|| {
                    Error::new(
                        Code::Internal,
                        format!("算子 `{}` 的标量实参 `{}` 没有取值", self.op, param.name),
                    )
                })?;
                parts.push(value.to_string());
            }
        }
        Ok(parts.join(", "))
    }

    /// 装载期结构校验：名字唯一、可写缓冲恰好一个。
    fn check_shape(&self) -> diagnostic::Result<()> {
        for (index, param) in self.params.iter().enumerate() {
            if let Some(other) = self.params[..index]
                .iter()
                .find(|other| other.name == param.name)
            {
                return Err(at(
                    1,
                    format!(
                        "算子 `{}` 的签名里形参 `{}` 重复（`{}` 与 `{}`）",
                        self.op, param.name, other.name, param.name
                    ),
                ));
            }
        }
        let outputs = self
            .params
            .iter()
            .filter(|param| param.ctype.is_buffer() && param.mutable)
            .count();
        match outputs {
            1 => Ok(()),
            0 => Err(at(
                1,
                format!(
                    "算子 `{}` 的签名没有可写缓冲：需要一个非 `const` 指针作为输出",
                    self.op
                ),
            )),
            count => Err(at(
                1,
                format!(
                    "算子 `{}` 的签名有 {count} 个可写缓冲；P0 的执行侧只支持一个输出",
                    self.op
                ),
            )),
        }
    }
}

fn describe(values: &Values) -> &'static str {
    match values {
        Values::F32(_) => "float",
        Values::U8(_) => "unsigned char",
    }
}

fn type_matches(ctype: CType, values: &Values) -> bool {
    matches!(
        (ctype, values),
        (CType::F32, Values::F32(_)) | (CType::U8, Values::U8(_))
    )
}

fn line_of(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())]
        .bytes()
        .filter(|b| *b == b'\n')
        .count()
        + 1
}

fn matching(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in text[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + index);
                }
            }
            _ => {}
        }
    }
    None
}

/// 按顶层逗号切分形参，返回 `(相对起点, 文本)`。
fn split(inner: &str) -> Vec<(usize, &str)> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in inner.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push((start, inner[start..index].trim()));
                start = index + 1;
            }
            _ => {}
        }
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push((start, tail));
    }
    parts
}

fn parse_param(op: &str, text: &str, line: usize) -> diagnostic::Result<Param> {
    let pointer = text.contains('*');
    let qualifiers: Vec<&str> = text
        .split(|ch: char| ch.is_whitespace() || ch == '*')
        .filter(|token| !token.is_empty())
        .collect();
    let readonly = qualifiers.contains(&"const");
    let mut tokens = qualifiers;
    let name = tokens.pop().ok_or_else(|| {
        at(
            line,
            format!("算子 `{op}` 的签名里有一个空的形参声明 `{text}`"),
        )
    })?;
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Err(at(
            line,
            format!("算子 `{op}` 的形参名 `{name}` 不是 C 标识符"),
        ));
    }
    let ty: Vec<&str> = tokens
        .into_iter()
        .filter(|token| !matches!(*token, "const" | "volatile" | "restrict" | "__restrict"))
        .collect();
    let ty = ty.join(" ");

    let ctype = if pointer {
        match ty.as_str() {
            "float" => CType::F32,
            "u8" | "unsigned char" => CType::U8,
            _ => {
                return Err(at(
                    line,
                    format!(
                        "算子 `{op}` 的缓冲形参 `{name}` 用类型 `{ty} *`；支持的缓冲类型只有 `float`（f32）与 `unsigned char`/`u8`"
                    ),
                ));
            }
        }
    } else {
        match ty.as_str() {
            "int" | "long" | "long long" | "unsigned" | "unsigned int" | "unsigned long"
            | "size_t" => CType::Int,
            _ => {
                return Err(at(
                    line,
                    format!(
                        "算子 `{op}` 的标量形参 `{name}` 用类型 `{ty}`；只支持整数标量（int、long、unsigned、size_t）"
                    ),
                ));
            }
        }
    };

    Ok(Param {
        name: name.to_string(),
        ctype,
        mutable: pointer && !readonly,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Buffer;

    const TEMPLATE: &str = "typedef unsigned char u8;\n\
        void cutile_probe_kernel(const float *x, const u8 *w, const float *scales,\n\
                                 float *y, int M, int N, int gsz) {\n\
            @dot(y, x, w, k);\n\
        }\n";

    fn sample() -> Sample {
        Sample {
            m: 4,
            n: 2,
            scalars: vec![("M", 4), ("N", 2), ("gsz", 2)],
            buffers: vec![
                Buffer {
                    name: "x",
                    values: Values::F32(vec![1.0; 4]),
                },
                Buffer {
                    name: "w",
                    values: Values::U8(vec![0x98, 0x76]),
                },
                Buffer {
                    name: "scales",
                    values: Values::F32(vec![0.5; 2]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; 2]),
                },
            ],
        }
    }

    #[test]
    fn signature_derives_the_argument_list_from_the_template() {
        let signature = Signature::parse("probe", TEMPLATE).expect("签名");
        assert_eq!(signature.kernel(), "cutile_probe_kernel");
        assert_eq!(signature.output(), Some("y"));
        assert_eq!(
            signature.call_args(&sample()).expect("实参"),
            "x, w, scales, y, 4, 2, 2"
        );
    }

    #[test]
    fn an_unsupported_buffer_type_is_rejected_with_the_parameter_name() {
        let broken = TEMPLATE.replace("const float *scales", "double *scales");
        let error = Signature::parse("probe", &broken).expect_err("double 不是缓冲类型");
        let text = error.to_string();
        assert!(text.contains("probe"), "{text}");
        assert!(text.contains("scales"), "{text}");
        assert!(text.contains("double"), "{text}");
        assert!(text.contains("第 2 行"), "{text}");

        // 形参从下一行开始时，行号要指向该形参，而不是上一行的逗号。
        let broken = TEMPLATE.replace("float *y,", "double *y,");
        let error = Signature::parse("probe", &broken).expect_err("double 不是缓冲类型");
        assert!(error.to_string().contains("第 3 行"), "{error}");
    }

    #[test]
    fn the_output_must_be_the_single_writable_pointer() {
        let broken = TEMPLATE.replace("float *y", "const float *y");
        let error = Signature::parse("probe", &broken).expect_err("没有可写输出");
        assert!(error.to_string().contains("可写缓冲"), "{error}");
    }

    #[test]
    fn sample_names_and_types_must_match_the_signature() {
        let signature = Signature::parse("probe", TEMPLATE).expect("签名");

        let mut mistyped = sample();
        mistyped.buffers[2].values = Values::U8(vec![0; 2]);
        let error = signature.check(&mistyped).expect_err("类型不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("scales"), "{error}");
        assert!(error.to_string().contains("float"), "{error}");

        let mut stray = sample();
        stray.buffers.push(Buffer {
            name: "bias",
            values: Values::F32(vec![0.0; 2]),
        });
        let error = signature.check(&stray).expect_err("多出的缓冲");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("bias"), "{error}");

        let mut missing = sample();
        missing.scalars.retain(|(name, _)| *name != "gsz");
        let error = signature.check(&missing).expect_err("缺标量");
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("gsz"), "{error}");
    }

    #[test]
    fn a_void_signature_has_no_output_and_is_rejected() {
        let error =
            Signature::parse("probe", "void cutile_probe_kernel(void) {}\n").expect_err("没有输出");
        assert!(error.to_string().contains("可写缓冲"), "{error}");
    }
}
