//! 请求文法：[`Request`]。
//!
//! 服务层必须有"一个请求长什么样"的唯一定义，否则传输层、端点与测试会各自发明一套字段名。
//! 这里用最小的行文法而不是 JSON：没有解析器依赖（ADR 0022 之后本仓运行期不带 TOML/JSON
//! 解析器），而字段只有三个。文法本身就是契约：
//!
//! ```text
//! id=7
//! prompt=hello world
//! max_new=16
//! ```
//!
//! - 每行 `键=值`，键取第一个 `=` 之前的部分，值取其余全部（提示里可以再有 `=`）；
//! - 空行忽略，行尾的 `\r` 忽略（容忍 CRLF）；
//! - 未知键、重复键、缺字段、数字解析失败都是 `invalid`——服务层据此报 `400`，而不是
//!   把请求静默当成缺省值跑一遍再返回看不懂的结果；
//! - 停止符不在这里：采到哪个 token 算"停"是采样策略的事（[`crate::Sampling`]）。

use diagnostic::{Code, Error, Result};

/// 一条补全请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    id: u64,
    prompt: String,
    max_new: usize,
}

impl Request {
    /// 解析请求体。
    ///
    /// # Errors
    ///
    /// 非 UTF-8、非 `键=值`、未知键、重复键、缺字段，或数字字段解析失败。
    pub fn parse(body: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(body)
            .map_err(|_| Error::new(Code::Invalid, "请求体不是 UTF-8").with(Self::SITE))?;
        let mut builder = Builder::default();
        for (index, line) in text.lines().enumerate() {
            builder.line(index + 1, line)?;
        }
        builder.finish()
    }

    /// 请求编号：由客户端给，重复编号会被准入拒绝成 `400`。
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 提示文本。
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// 最多生成多少个 token。
    pub fn max_new(&self) -> usize {
        self.max_new
    }

    const SITE: &'static str = "serve::Request::parse";
}

/// 逐行累积的请求构造器；字段全是 `Option`，好让"缺了哪个"在 `finish` 时一次说清。
#[derive(Debug, Default)]
struct Builder {
    id: Option<u64>,
    prompt: Option<String>,
    max_new: Option<usize>,
}

impl Builder {
    fn line(&mut self, number: usize, line: &str) -> Result<()> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() {
            return Ok(());
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(Self::line_error(number, "每行必须是 `键=值`"));
        };
        self.set(number, key.trim(), value.trim())
    }

    fn set(&mut self, number: usize, key: &str, value: &str) -> Result<()> {
        match key {
            "id" => self.set_id(number, value),
            "prompt" => self.set_prompt(number, value),
            "max_new" => self.set_max_new(number, value),
            other => Err(Self::line_error(
                number,
                &format!("未知字段 `{other}`：只认 id / prompt / max_new"),
            )),
        }
    }

    fn set_id(&mut self, number: usize, value: &str) -> Result<()> {
        if self.id.is_some() {
            return Err(Self::line_error(number, "字段 `id` 出现了两次"));
        }
        let id = value
            .parse()
            .map_err(|_| Self::line_error(number, &format!("`id` 不是 u64：`{value}`")))?;
        self.id = Some(id);
        Ok(())
    }

    fn set_prompt(&mut self, number: usize, value: &str) -> Result<()> {
        if self.prompt.is_some() {
            return Err(Self::line_error(number, "字段 `prompt` 出现了两次"));
        }
        self.prompt = Some(value.to_string());
        Ok(())
    }

    fn set_max_new(&mut self, number: usize, value: &str) -> Result<()> {
        if self.max_new.is_some() {
            return Err(Self::line_error(number, "字段 `max_new` 出现了两次"));
        }
        let max_new = value
            .parse()
            .map_err(|_| Self::line_error(number, &format!("`max_new` 不是 usize：`{value}`")))?;
        self.max_new = Some(max_new);
        Ok(())
    }

    /// 三个必填字段逐个取出：缺哪个就报哪个，而不是笼统说"请求不完整"。
    fn finish(self) -> Result<Request> {
        let Some(id) = self.id else {
            return Err(Self::missing("id"));
        };
        let Some(prompt) = self.prompt else {
            return Err(Self::missing("prompt"));
        };
        let Some(max_new) = self.max_new else {
            return Err(Self::missing("max_new"));
        };
        Ok(Request {
            id,
            prompt,
            max_new,
        })
    }

    fn line_error(number: usize, message: &str) -> Error {
        Error::new(Code::Invalid, format!("第 {number} 行：{message}")).with(Request::SITE)
    }

    fn missing(field: &str) -> Error {
        Error::new(Code::Invalid, format!("缺少字段 `{field}`")).with(Request::SITE)
    }
}

#[cfg(test)]
#[path = "../tests/request_tests.rs"]
mod tests;
