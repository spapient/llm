//! 端点契约：路由 [`Route`]、状态 [`Status`]、响应 [`Response`] 与一次调用 [`Endpoint`]。

use crate::macros::code_set;
use crate::protocol::{Envelope, Kind, Version};
use diagnostic::{Code, Error, Result};

/// 一条路由：方法与路径。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    method: String,
    path: String,
}

impl Route {
    /// 解析 `"POST /v1/completions"`；方法与路径都必须非空，路径必须以 `/` 开头。
    pub fn parse(spec: &str) -> Result<Self> {
        let mut parts = spec.split_whitespace();
        let (Some(method), Some(path), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(Error::new(
                Code::Invalid,
                format!("路由必须是 `方法 路径` 两段：`{spec}`"),
            )
            .with("serve::Route::parse"));
        };
        if !path.starts_with('/') {
            return Err(
                Error::new(Code::Invalid, format!("路径必须以 `/` 开头：`{path}`"))
                    .with("serve::Route::parse"),
            );
        }
        Ok(Self {
            method: method.to_ascii_uppercase(),
            path: path.to_string(),
        })
    }

    /// 方法（大写）。
    pub fn method(&self) -> &str {
        &self.method
    }

    /// 路径。
    pub fn path(&self) -> &str {
        &self.path
    }
}

code_set! {
    /// 响应状态：闭集，映射到 HTTP 码。
    pub enum Status {
        /// 正常返回。
        Ok => 200,
        /// 请求不合法。
        BadRequest => 400,
        /// 没有这条路由。
        NotFound => 404,
        /// 容量不足或设备不可用。
        Unavailable => 503,
    }
}

/// 一个响应：状态 + 一帧载荷。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    status: Status,
    envelope: Envelope,
}

impl Response {
    /// 由状态与帧构造。
    pub fn new(status: Status, envelope: Envelope) -> Self {
        Self { status, envelope }
    }

    /// 正常响应：载荷是流式数据块。
    pub fn ok(payload: Vec<u8>) -> Self {
        Self {
            status: Status::Ok,
            envelope: Envelope::new(Version::current(), Kind::Chunk, payload),
        }
    }

    /// 错误响应：载荷是错误消息的 UTF-8 字节。
    pub fn error(status: Status, message: &str) -> Self {
        Self {
            status,
            envelope: Envelope::new(Version::current(), Kind::Error, message.as_bytes().to_vec()),
        }
    }

    /// 状态。
    pub fn status(&self) -> Status {
        self.status
    }

    /// 帧。
    pub fn envelope(&self) -> &Envelope {
        &self.envelope
    }

    /// 编码成写回连接的字节串。
    pub fn frame(&self) -> Result<Vec<u8>> {
        self.envelope.frame()
    }
}

/// 端点契约：一个路由 + 一次调用。
pub trait Endpoint {
    /// 它服务的路由。
    fn route(&self) -> &Route;

    /// 处理一次请求体。
    fn call(&self, body: &[u8]) -> Result<Response>;
}

impl<E: Endpoint + ?Sized> Endpoint for Box<E> {
    fn route(&self) -> &Route {
        self.as_ref().route()
    }

    fn call(&self, body: &[u8]) -> Result<Response> {
        self.as_ref().call(body)
    }
}

/// 把提示文本编码成 token（唯一的分词入口，服务层不得自己切字符串）。
pub fn encode_prompt(
    tokenizer: &dyn tokenizer::Tokenizer,
    text: &str,
) -> Result<Vec<tokenizer::Token>> {
    tokenizer.encode(text)
}
