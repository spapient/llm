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
use std::str::FromStr;

use crate::{Endpoint, Response, Route, Status};

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
        let mut slots = Slots {
            id: None,
            prompt: None,
            max_new: None,
        };
        for (idx, line) in text.lines().enumerate() {
            slots.fill(idx + 1, line)?;
        }
        slots.build()
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

/// 解析途中的三个待填槽位；行级分派与必填收口各占一处，避免 `parse` 堆条件。
struct Slots {
    id: Option<u64>,
    prompt: Option<String>,
    max_new: Option<usize>,
}

impl Slots {
    /// 处理一行：吞空行与 CRLF 尾，拆出 `键=值` 后分派到槽位。
    fn fill(&mut self, number: usize, raw: &str) -> Result<()> {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.trim().is_empty() {
            return Ok(());
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(line_error(number, "每行必须是 `键=值`"));
        };
        self.assign(number, key.trim(), value.trim())
    }

    /// 把一行 `键=值` 分派到对应槽位；重复键、未知键、数值非法都在此报错。
    fn assign(&mut self, number: usize, key: &str, value: &str) -> Result<()> {
        match key {
            "id" => return parse_once(&mut self.id, number, key, value, "u64"),
            "max_new" => return parse_once(&mut self.max_new, number, key, value, "usize"),
            "prompt" => {
                if self.prompt.is_some() {
                    return Err(duplicate(number, key));
                }
                self.prompt = Some(value.to_string());
            }
            _ => {
                return Err(line_error(
                    number,
                    &format!("未知字段 `{key}`：只认 id / prompt / max_new"),
                ));
            }
        }
        Ok(())
    }

    /// 三个必填槽位收口为请求；缺字段具名报错。
    fn build(self) -> Result<Request> {
        Ok(Request {
            id: require(self.id, "id")?,
            prompt: require(self.prompt, "prompt")?,
            max_new: require(self.max_new, "max_new")?,
        })
    }
}

/// 必填槽位取值；缺字段具名报错。
fn require<T>(slot: Option<T>, field: &str) -> Result<T> {
    slot.ok_or_else(|| missing(field))
}

/// 数值字段解析 + 重复键拒绝；`type_name` 只用于错误文案。
fn parse_once<T: FromStr>(
    slot: &mut Option<T>,
    number: usize,
    key: &str,
    value: &str,
    type_name: &str,
) -> Result<()> {
    if slot.is_some() {
        return Err(duplicate(number, key));
    }
    *slot = Some(
        value
            .parse::<T>()
            .map_err(|_| line_error(number, &format!("`{key}` 不是 {type_name}：`{value}`")))?,
    );
    Ok(())
}

fn duplicate(number: usize, key: &str) -> Error {
    line_error(number, &format!("字段 `{key}` 出现了两次"))
}

fn line_error(number: usize, message: &str) -> Error {
    Error::new(Code::Invalid, format!("第 {number} 行：{message}")).with(Request::SITE)
}

fn missing(field: &str) -> Error {
    Error::new(Code::Invalid, format!("缺少字段 `{field}`")).with(Request::SITE)
}

/// 路由分派：[`Router`]。
#[derive(Default)]
pub struct Router {
    endpoints: Vec<Box<dyn Endpoint>>,
}

impl Router {
    /// 空路由表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 挂载一条端点；同一 `方法 路径` 已经存在时返回 `invalid`。
    ///
    /// 泛型接收具体端点，只在存进路由表时装箱；调用方不必先 `Box::new`，也能保留具体类型
    /// 供测试与单态化使用。
    ///
    /// # Errors
    ///
    /// 重复路由（方法与路径都相同）。
    pub fn mount<E: Endpoint + 'static>(&mut self, endpoint: E) -> Result<()> {
        let route = endpoint.route().clone();
        if self.find(route.method(), route.path()).is_some() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "路由 `{} {}` 已经挂过一条端点",
                    route.method(),
                    route.path()
                ),
            )
            .with("serve::Router::mount"));
        }
        self.endpoints.push(Box::new(endpoint));
        Ok(())
    }

    /// 处理一次请求：命中就把请求体交给它，未命中返回 `404`。
    ///
    /// # Errors
    ///
    /// 端点自己报的服务端故障（`internal` / `io` 等）原样上抛；请求侧的问题已在端点内
    /// 落成 [`Response`]（见 [`crate::Endpoint::call`]）。
    pub fn call(&self, method: &str, path: &str, body: &[u8]) -> Result<Response> {
        match self.find(method, path) {
            Some(endpoint) => endpoint.call(body),
            None => Ok(Response::error(
                Status::NotFound,
                &format!("没有路由 `{method} {path}`"),
            )),
        }
    }

    /// 已挂载的路由数。
    pub fn len(&self) -> usize {
        self.endpoints.len()
    }

    /// 是否一条端点都没挂。
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }

    /// 已挂载的路由，按挂载顺序。
    pub fn routes(&self) -> impl Iterator<Item = &Route> {
        self.endpoints.iter().map(super::endpoint::Endpoint::route)
    }

    /// 匹配一条端点：方法大小写不敏感，路径逐字节比较。
    fn find(&self, method: &str, path: &str) -> Option<&dyn Endpoint> {
        self.endpoints.iter().map(Box::as_ref).find(|endpoint| {
            endpoint.route().method().eq_ignore_ascii_case(method)
                && endpoint.route().path() == path
        })
    }
}

#[cfg(test)]
#[path = "../tests/request_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/router_tests.rs"]
mod router_tests;
