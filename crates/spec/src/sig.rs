//! 内核签名：从 `.k` 的入口函数声明解析形参，供执行侧派生实参列表与输出缓冲。

use crate::model::{DType, Sample};
use crate::parse::{line_error as at, matching_paren, split_top_level};
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

    /// 样本侧对应的 [`DType`]：签名与语义表必须用同一个词描述同一块缓冲。
    pub fn dtype(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::U8 => Some(DType::U8),
            Self::Int => None,
        }
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
        let (open, close) = entry(op, template)?;
        let signature = Self {
            op: op.to_string(),
            params: params(op, template, open, close)?,
        };
        signature.check_shape()?;
        Ok(signature)
    }

    /// 算子名（诊断里显示的名字）。
    pub fn op(&self) -> &str {
        &self.op
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

    /// 唯一可写输出缓冲的类型（与 [`Self::output`] 同源）。
    pub fn output_type(&self) -> Option<CType> {
        self.params
            .iter()
            .find(|param| param.ctype.is_buffer() && param.mutable)
            .map(|param| param.ctype)
    }

    /// 缓冲形参名，按方向筛：`mutable` 为真取可写输出，为假取只读输入；标量形参不在此列。
    pub fn buffer_names(&self, mutable: bool) -> Vec<String> {
        self.params
            .iter()
            .filter(|param| param.ctype.is_buffer() && param.mutable == mutable)
            .map(|param| param.name.clone())
            .collect()
    }

    /// 把签名与样本逐项对齐：缓冲要同名同类型，标量要有取值，样本也不能多出签名之外的名字。
    ///
    /// # Errors
    ///
    /// 任一不匹配都返回带算子名、形参名与两侧类型的说明。
    pub fn check(&self, sample: &Sample) -> diagnostic::Result<()> {
        self.check_params(sample)?;
        self.check_extra_buffers(sample)?;
        self.check_extra_scalars(sample)
    }

    /// 逐个形参对齐样本：缓冲同名同类型，标量有取值。
    fn check_params(&self, sample: &Sample) -> diagnostic::Result<()> {
        for param in &self.params {
            if param.ctype.is_buffer() {
                self.check_buffer(param, sample)?;
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
        Ok(())
    }

    /// 一个缓冲形参：同名缓冲必须存在，元素类型必须与 `.k` 的声明一致。
    fn check_buffer(&self, param: &Param, sample: &Sample) -> diagnostic::Result<()> {
        let buffer = sample.buffer(&param.name).ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!(
                    "算子 `{}` 的内核签名有缓冲 `{}`，样本里没有同名缓冲",
                    self.op, param.name
                ),
            )
        })?;
        if type_matches(param.ctype, &buffer.values) {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!(
                "算子 `{}` 的缓冲 `{}`：`.k` 声明为 `{} *`，样本给的是 `{}`",
                self.op,
                param.name,
                param.ctype.c_name(),
                describe(&buffer.values)
            ),
        ))
    }

    /// 样本里多出来的缓冲名：签名没有它，emit 时谁也接不住。
    fn check_extra_buffers(&self, sample: &Sample) -> diagnostic::Result<()> {
        for buffer in &sample.buffers {
            if !self.has_buffer(buffer.name) {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "算子 `{}` 的样本缓冲 `{}` 不在 `.k` 签名里",
                        self.op, buffer.name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// 样本里多出来的标量名：同上，签名之外的实参无法传给内核。
    fn check_extra_scalars(&self, sample: &Sample) -> diagnostic::Result<()> {
        for (name, _) in &sample.scalars {
            if !self.has_scalar(name) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("算子 `{}` 的样本标量 `{name}` 不在 `.k` 签名里", self.op),
                ));
            }
        }
        Ok(())
    }

    /// 签名里有没有这个缓冲形参。
    fn has_buffer(&self, name: &str) -> bool {
        self.params
            .iter()
            .any(|param| param.ctype.is_buffer() && param.name == name)
    }

    /// 签名里有没有这个标量形参。
    fn has_scalar(&self, name: &str) -> bool {
        self.params
            .iter()
            .any(|param| !param.ctype.is_buffer() && param.name == name)
    }

    /// 按签名顺序渲染内核实参：缓冲名原样、标量取样本里的值。
    ///
    /// # Errors
    ///
    /// 签名与样本不一致时先由 [`Self::check`] 返回说明。
    pub fn call_args(&self, sample: &Sample) -> diagnostic::Result<String> {
        self.check(sample)?;
        let parts: Vec<String> = self
            .params
            .iter()
            .map(|param| self.argument(param, sample))
            .collect::<diagnostic::Result<_>>()?;
        Ok(parts.join(", "))
    }

    /// 一个形参在调用点的写法：缓冲给名字，标量给样本里的取值。
    fn argument(&self, param: &Param, sample: &Sample) -> diagnostic::Result<String> {
        if param.ctype.is_buffer() {
            return Ok(param.name.clone());
        }
        // `check` 刚过，这里取不到只能是实现与校验不同步——内部不变量，不是调用方的错。
        sample
            .scalar(&param.name)
            .map(|value| value.to_string())
            .ok_or_else(|| {
                Error::new(
                    Code::Internal,
                    format!("算子 `{}` 的标量实参 `{}` 没有取值", self.op, param.name),
                )
            })
    }

    /// 装载期结构校验：名字唯一、可写缓冲恰好一个。
    fn check_shape(&self) -> diagnostic::Result<()> {
        self.check_unique()?;
        self.check_single_output()
    }

    /// 形参名只能出现一次：重名会让「按名字取实参」变成看运气。
    fn check_unique(&self) -> diagnostic::Result<()> {
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
        Ok(())
    }

    /// 可写缓冲恰好一个：P0 的执行侧只支持一个输出。
    fn check_single_output(&self) -> diagnostic::Result<()> {
        match self.outputs() {
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

    /// 可写缓冲的个数（`output`/`output_type` 是它的两种投影）。
    fn outputs(&self) -> usize {
        self.params
            .iter()
            .filter(|param| param.ctype.is_buffer() && param.mutable)
            .count()
    }
}

mod check {
    use super::*;
    use crate::model::Values;

    pub(super) fn describe(values: &Values) -> &'static str {
        match values {
            Values::F32(_) => "float",
            Values::U8(_) => "unsigned char",
        }
    }

    pub(super) fn type_matches(ctype: CType, values: &Values) -> bool {
        matches!(
            (ctype, values),
            (CType::F32, Values::F32(_)) | (CType::U8, Values::U8(_))
        )
    }

    pub(super) fn line_of(text: &str, offset: usize) -> usize {
        crate::parse::count_newlines(&text[..offset.min(text.len())]) + 1
    }
}

mod parse {
    use super::*;

    /// 入口函数的括号区间：`cutile_{op}_kernel(` 的 `(` 与配对的 `)`。
    pub(super) fn entry(op: &str, template: &str) -> diagnostic::Result<(usize, usize)> {
        let marker = format!("cutile_{op}_kernel(");
        // 一次定位：`find` 已经给出起点，再 `contains` 一遍只是把同一段文本扫两次。
        let (start, width) = [marker.as_str(), "cutile_{{op}}_kernel("]
            .into_iter()
            .find_map(|entry| template.find(entry).map(|start| (start, entry.len())))
            .ok_or_else(|| {
                at(
                    1,
                    format!("算子 `{op}` 的模板缺少内核入口 `cutile_{{op}}_kernel(`"),
                )
            })?;
        let open = start + width - 1;
        let close = matching_paren(template, open).ok_or_else(|| {
            at(
                line_of(template, open),
                format!("算子 `{op}` 的内核签名括号不配对"),
            )
        })?;
        Ok((open, close))
    }

    /// 括号里的形参表：`void` 与空表都是「没有形参」，其余按顶层逗号切开逐个解析。
    pub(super) fn params(
        op: &str,
        template: &str,
        open: usize,
        close: usize,
    ) -> diagnostic::Result<Vec<Param>> {
        let inner = &template[open + 1..close];
        if inner.trim() == "void" || inner.trim().is_empty() {
            return Ok(Vec::new());
        }
        split(inner)
            .into_iter()
            .map(|(offset, text)| {
                // 参数可能从下一行开始：行号要落在非空白字符上，而不是逗号之后。
                let indent = inner[offset..].len() - inner[offset..].trim_start().len();
                parse_param(op, text, line_of(template, open + 1 + offset + indent))
            })
            .collect()
    }

    /// 按顶层逗号切分形参：切分原语是 [`split_top_level`]，这里只加「裁剪 + 丢空尾」的口径。
    pub(super) fn split(inner: &str) -> Vec<(usize, &str)> {
        let mut parts: Vec<(usize, &str)> = split_top_level(inner)
            .into_iter()
            .map(|(offset, text)| (offset, text.trim()))
            .collect();
        if parts.last().is_some_and(|(_, tail)| tail.is_empty()) {
            parts.pop();
        }
        parts
    }

    /// 一个形参：先认名字（最后一个词），再按「指针 / 标量」两条闭集解析类型。
    pub(super) fn parse_param(op: &str, text: &str, line: usize) -> diagnostic::Result<Param> {
        let pointer = text.contains('*');
        let words = words(text);
        let name = param_name(op, &words, text, line)?;
        let ty = type_text(&words);
        let ctype = ctype(op, name, &ty, line, pointer)?;
        Ok(Param {
            name: name.to_string(),
            ctype,
            mutable: pointer && !words.contains(&"const"),
        })
    }

    /// C 类型：指针读声明表里的缓冲类型，其余是标量类型。
    pub(super) fn ctype(
        op: &str,
        name: &str,
        ty: &str,
        line: usize,
        pointer: bool,
    ) -> diagnostic::Result<CType> {
        if pointer {
            buffer_ctype(op, name, ty, line)
        } else {
            scalar_ctype(op, name, ty, line)
        }
    }

    /// 声明里的词：按空白与 `*` 切开，`const` 这类限定词与类型名都在里面。
    pub(super) fn words(text: &str) -> Vec<&str> {
        text.split(|ch: char| ch.is_whitespace() || ch == '*')
            .filter(|token| !token.is_empty())
            .collect()
    }

    /// 形参名：最后一个词，必须是 C 标识符。
    pub(super) fn param_name<'a>(
        op: &str,
        words: &'a [&'a str],
        text: &str,
        line: usize,
    ) -> diagnostic::Result<&'a str> {
        let Some(name) = words.last() else {
            return Err(at(
                line,
                format!("算子 `{op}` 的签名里有一个空的形参声明 `{text}`"),
            ));
        };
        if name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return Ok(name);
        }
        Err(at(
            line,
            format!("算子 `{op}` 的形参名 `{name}` 不是 C 标识符"),
        ))
    }

    /// 类型文本：去掉形参名与 `const`/`volatile`/`restrict` 这些限定词。
    pub(super) fn type_text(words: &[&str]) -> String {
        words
            .iter()
            .take(words.len().saturating_sub(1))
            .filter(|token| !matches!(**token, "const" | "volatile" | "restrict" | "__restrict"))
            .copied()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 缓冲形参的类型：只有 `float` 与 `unsigned char`/`u8`。
    pub(super) fn buffer_ctype(
        op: &str,
        name: &str,
        ty: &str,
        line: usize,
    ) -> diagnostic::Result<CType> {
        match ty {
            "float" => Ok(CType::F32),
            "u8" | "unsigned char" => Ok(CType::U8),
            _ => Err(at(
                line,
                format!(
                    "算子 `{op}` 的缓冲形参 `{name}` 用类型 `{ty} *`；支持的缓冲类型只有 `float`（f32）与 `unsigned char`/`u8`"
                ),
            )),
        }
    }

    /// 标量形参的类型：只支持整数标量。
    pub(super) fn scalar_ctype(
        op: &str,
        name: &str,
        ty: &str,
        line: usize,
    ) -> diagnostic::Result<CType> {
        match ty {
            "int" | "long" | "long long" | "unsigned" | "unsigned int" | "unsigned long"
            | "size_t" => Ok(CType::Int),
            _ => Err(at(
                line,
                format!(
                    "算子 `{op}` 的标量形参 `{name}` 用类型 `{ty}`；只支持整数标量（int、long、unsigned、size_t）"
                ),
            )),
        }
    }
}
use check::*;
use parse::*;

#[cfg(test)]
#[path = "../tests/sig_tests.rs"]
mod tests;
