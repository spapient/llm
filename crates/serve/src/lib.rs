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
        // `invalid` 全是"这次请求自己不成立"，改什么由客户端决定；`internal` 之类才是本侧的 bug。
        Err(error) if error.code() == Code::Invalid => Ok(Response::error(
            Status::BadRequest,
            &format!("请求不合法：{error}"),
        )),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::kv::Cache;
    use engine::{Scheduler, Sequence};

    #[test]
    fn a_route_needs_exactly_one_method_and_an_absolute_path() {
        let route = Route::parse("post /v1/completions").expect("路由");
        assert_eq!(route.method(), "POST");
        assert_eq!(route.path(), "/v1/completions");
        assert_eq!(Route::parse("POST").unwrap_err().code(), Code::Invalid);
        assert_eq!(
            Route::parse("POST v1/completions").unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Route::parse("POST /a /b").unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn statuses_map_to_http_codes() {
        assert_eq!(Status::Ok.code(), 200);
        assert_eq!(Status::BadRequest.code(), 400);
        assert_eq!(Status::Unavailable.code(), 503);
    }

    #[test]
    fn a_response_round_trips_through_a_frame() {
        let response = Response::ok(vec![1, 2, 3]);
        let frame = response.frame().expect("编码");
        let parsed = Envelope::parse(&frame).expect("解析");
        assert_eq!(parsed.kind(), Kind::Chunk);
        assert_eq!(parsed.payload(), [1, 2, 3]);
        let error = Response::error(Status::BadRequest, "缺 prompt");
        assert_eq!(error.envelope().kind(), Kind::Error);
        assert_eq!(error.envelope().payload(), "缺 prompt".as_bytes());
    }

    #[test]
    fn a_prompt_is_encoded_through_the_tokenizer_contract() {
        let tokens = encode_prompt(&tokenizer::Bytes, "hi").expect("编码");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn admission_turns_into_an_ok_or_a_retryable_status() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let small = Sequence::new(1, 4, 2, 16, engine::exec::Stream::new(0)).expect("序列");
        let response = admit(&mut scheduler, &small).expect("准入");
        assert_eq!(response.status(), Status::Ok);

        // 提示就占满剩下的一页，但跑到最后要第二页：现在不行，等别人结束再试。
        let full = Sequence::new(2, 20, 4, 16, engine::exec::Stream::new(0)).expect("序列");
        assert_eq!(scheduler.cache().free_pages(), 1);
        let refused = admit(&mut scheduler, &full).expect("拒绝不是错误");
        assert_eq!(refused.status(), Status::Unavailable);
        assert_eq!(refused.status().code(), 503, "现在满 = 可重试");
        assert_eq!(scheduler.running(), 1);
    }

    /// 最坏情况装不下与"现在满"不是一回事：重试没有用，必须改请求，所以不给 503。
    #[test]
    fn an_impossible_request_is_a_bad_request_not_a_retry() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let endless = Sequence::new(9, 4, 1024, 16, engine::exec::Stream::new(0)).expect("序列");
        let response = admit(&mut scheduler, &endless).expect("拒绝不是错误");
        assert_eq!(response.status().code(), 400);
        assert_eq!(
            scheduler.running(),
            0,
            "装不下的请求连页都不该占（否则每次重试都白占一页）"
        );
        assert_eq!(scheduler.cache().in_use(), 0);
    }

    /// 请求自己不成立不能变成传输层错误：客户端看不到 5xx 之外的任何理由。
    #[test]
    fn a_request_that_cannot_enter_at_all_is_a_bad_request_not_an_error() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let small = Sequence::new(1, 4, 2, 16, engine::exec::Stream::new(0)).expect("序列");
        assert_eq!(
            admit(&mut scheduler, &small).expect("准入").status(),
            Status::Ok
        );

        // 同一个 id 再来一次：`admit` 报 `invalid`，而那是客户端的事（换个 id 就行）。
        let duplicate = admit(&mut scheduler, &small).expect("拒绝不是错误");
        assert_eq!(duplicate.status().code(), 400);
        assert_eq!(scheduler.running(), 1, "重复的那次不得多占一个名额");
        assert!(
            String::from_utf8_lossy(duplicate.envelope().payload()).contains("已经在批里"),
            "400 的载荷要说清哪里不合法"
        );

        // 页宽与调度器不一致：同样只能改请求，不能靠重试。
        let wrong_page = Sequence::new(2, 4, 2, 8, engine::exec::Stream::new(0)).expect("序列");
        let mismatched = admit(&mut scheduler, &wrong_page).expect("拒绝不是错误");
        assert_eq!(mismatched.status().code(), 400);
        assert_eq!(scheduler.running(), 1);
    }

    /// 已结束的序列不该再进批（占一页白等）：这是请求状态错了，不是服务端坏了。
    #[test]
    fn a_finished_sequence_is_a_bad_request_rather_than_a_server_failure() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let mut sequence = Sequence::new(5, 0, 1, 16, engine::exec::Stream::new(0)).expect("序列");
        assert_eq!(
            admit(&mut scheduler, &sequence).expect("准入").status(),
            Status::Ok
        );
        scheduler.advance(&mut sequence).expect("推进");
        assert!(sequence.is_done());
        assert!(scheduler.finish(5).expect("结束"));

        let response = admit(&mut scheduler, &sequence).expect("拒绝不是错误");
        assert_eq!(response.status().code(), 400);
        assert!(
            String::from_utf8_lossy(response.envelope().payload()).contains("已结束"),
            "要说清是序列已经结束"
        );
        assert_eq!(scheduler.cache().in_use(), 0, "失败的准入不占页");
    }

    #[test]
    fn an_endpoint_only_needs_a_route_and_a_call() {
        struct Echo(Route);
        impl Endpoint for Echo {
            fn route(&self) -> &Route {
                &self.0
            }

            fn call(&self, body: &[u8]) -> Result<Response> {
                Ok(Response::ok(body.to_vec()))
            }
        }
        let endpoint = Echo(Route::parse("POST /echo").expect("路由"));
        let response = endpoint.call(b"ping").expect("调用");
        assert_eq!(response.envelope().payload(), b"ping");
    }
}
