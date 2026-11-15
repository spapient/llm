//! 端点契约：[`Route`]、[`Status`]、[`Response`]、[`Endpoint`]。
//!
//! **拥有**：`Route`、`Endpoint`、`Response`、`Status`，以及线格式（`proto` 子模块：
//! `Envelope`/`Frame`/`Decoder`/`Kind`/`Version`）。**不拥有**：序列状态与准入（`engine`）、
//! 分词（`tokenizer`）、HTTP/gRPC 服务器（P1：它是 `Endpoint` 的实现）。
//! **依赖**：`engine`、`tokenizer`、`diagnostic`。**被依赖**：`llm`。
//!
//! 这里没有 socket：先把"一个请求进来会发生什么"定死——解析路由、编码提示、尝试准入、
//! 把结果包成帧。等真服务器落地时，它只需要调用 [`Endpoint::call`]。
//!
//! # 例子
//!
//! ```
//! use llm_serve::{Response, Route, Status};
//!
//! let route = Route::parse("POST /v1/completions").expect("路由");
//! assert_eq!(route.method(), "POST");
//! let response = Response::error(Status::BadRequest, "缺 prompt");
//! assert_eq!(response.status().code(), 400);
//! assert!(!response.frame().expect("可编码").is_empty());
//! ```

#![deny(unsafe_code)]

pub mod proto;

use crate::proto::{Envelope, Kind, Version};
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

/// 响应状态：闭集，映射到 HTTP 码。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// 正常返回。
    Ok,
    /// 请求不合法。
    BadRequest,
    /// 容量不足或设备不可用。
    Unavailable,
}

impl Status {
    /// 对应的 HTTP 状态码。
    pub fn code(self) -> u16 {
        match self {
            Self::Ok => 200,
            Self::BadRequest => 400,
            Self::Unavailable => 503,
        }
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
///
/// 实现者可以是内存里的测试端点，也可以是 HTTP 服务器；服务器只负责把字节交给 [`Endpoint::call`]。
pub trait Endpoint {
    /// 它服务的路由。
    fn route(&self) -> &Route;

    /// 处理一次请求体。
    fn call(&self, body: &[u8]) -> Result<Response>;
}

/// 把提示文本编码成 token（唯一的分词入口，服务层不得自己切字符串）。
pub fn encode_prompt(
    tokenizer: &dyn tokenizer::Tokenizer,
    text: &str,
) -> Result<Vec<tokenizer::Token>> {
    tokenizer.encode(text)
}

/// 尝试把一个序列准入调度器，并把结果翻译成响应。
///
/// 三种结局必须是三个不同的东西，混在一起就会让客户端看不到"是我写错了"：
///
/// - **进去**：200 + 序列号；
/// - **现在满**（页被别的序列占着）：可重试的 503，客户端等一会儿再来即可；
/// - **永远装不下**（[`engine::Scheduler::fits()`] 为假，即最坏情况的页数超过整个页池）或
///   **请求自身不成立**（[`engine::Scheduler::admit`] 报 `invalid`：id 已在批里、序列已结束、
///   页宽与调度器不一致）：重试多少次都一样，只有改请求才有用，因此是 400。
///
/// 后两类都必须落成 [`Response`] 而不是 `Err`：端点契约里 `Err` 意味着"服务端自己出错了"，
/// 会被传输层变成 5xx 或直接断连接——客户端于是永远读不到"你的 id 重了"这类信息。只有真正的
/// 服务端故障（`internal` / `io` 等）才继续上抛，状态码与诊断类别因此不会互相打架。
pub fn admit(scheduler: &mut engine::Scheduler, sequence: &engine::Sequence) -> Result<Response> {
    if !scheduler.fits(sequence) {
        return Ok(Response::error(
            Status::BadRequest,
            "这条请求的最坏长度超过整个 KV 页池：调小 max_new 或加大页池",
        ));
    }
    match scheduler.admit(sequence) {
        Ok(true) => Ok(Response::ok(sequence.id().to_le_bytes().to_vec())),
        Ok(false) => Ok(Response::error(
            Status::Unavailable,
            "KV 页不足，请稍后重试",
        )),
        Err(error) => refused(error),
    }
}

/// 准入错误落成响应：`invalid` 全是"这次请求自己不成立"，改什么由客户端决定；
/// `internal` 之类才是本侧的 bug，继续上抛。
fn refused(error: Error) -> Result<Response> {
    if error.code() != Code::Invalid {
        return Err(error);
    }
    Ok(Response::error(
        Status::BadRequest,
        &format!("请求不合法：{error}"),
    ))
}

#[cfg(test)]
mod tests;
