//! 端点契约与它的第一个真实实现：[`Route`]、[`Status`]、[`Response`]、[`Endpoint`]、
//! 请求文法 [`Request`]、生成协作者 [`Predict`] / [`Sampling`] / [`Greedy`]、
//! 补全端点 [`Completion`] 与路由分派 [`Router`]。
//!
//! **拥有**：`Route`、`Endpoint`、`Response`、`Status`、`Request`、std-only HTTP/1.1
//! 薄适配器（`http`），以及线格式
//! （`protocol`：`Envelope`/`Frame`/`Decoder`/`Kind`/`Version`；`proto` 是兼容重导出路径）；
//! 注入 [`Predict`] 的 CPU 模型提供者收在 `kernels` / `assembly`（`model` 是兼容重导出路径）。
//! **不拥有**：序列状态与页账（`engine`）、分词（`tokenizer`）、算子算术
//! （`op`）、层定义（`nn`）、权重条目字节与读取契约（`weights`）、gRPC 服务器；
//! Metal/CUDA 真设备后端收在独立边界 crate，不回流本层。**依赖**：`engine`、`nn`、`op`、`tokenizer`、`diagnostic`、`weights`。**被依赖**：`llm`。
//!
//! 传输层收在 `http`：单线程把 socket 字节解成 `方法 + 路径 + 请求体` 交给 [`Router::call`]，
//! 后者分派到 [`Completion`]，后者完成"解析 → 编词 → 准入 → 逐步前向 + 采样 → 归还页 → 包成帧"。
//! 因此"一条请求能不能跑完"既可以在进程内验证，也可以经真实 HTTP/1.1 socket 验证。
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

use crate::macros::code_set;

mod completion;
mod generate;
mod http;
mod request;
mod speculate;

// CPU 内核桥与装配宿主（平铺；`model` 兼容路径见下）。
mod assembly;
mod block;
mod kernels;
mod model_host;
mod params;
mod plan;

pub(crate) use params::{descriptors, vocab};

// 线协议：常量/闭集与宏在 `protocol`，帧与增量解码在 `frame`（`proto` 兼容路径见下）。
// CRC-32 的唯一实现在 L1 `wire`：帧尾校验直接复用它，协议层不再自带第二张表。
mod frame;
mod protocol;

pub(crate) use frame::decoder;
pub(crate) use protocol::macros;
pub(crate) use request::router;

/// CRC-32：与 `weights`/`train` 共用 L1 `wire` 的唯一实现。
pub(crate) use wire::crc::crc32;

pub use completion::Completion;
pub use generate::{
    Classical, Ending, Generation, Greedy, Multinomial, Predict, Sampling, SamplingParams,
};
pub use http::{Httpd, fuzz_http_head};
pub use request::Request;
pub use router::Router;
pub use speculate::{Drafter, Verdict, accept};
pub use vocab::tokenizer_from_segment;

/// 兼容旧路径：CPU 模型内核桥与装配宿主。
pub mod model {
    pub use crate::kernels::register_cpu;
    pub use crate::model_host::CpuModel;
    pub use crate::params::Params;
}

/// 兼容旧路径：端点契约的线格式。
pub mod proto {
    pub use crate::protocol::{
        Decoder, Envelope, Frame, HEADER_LEN, Kind, MAGIC, MAX_PAYLOAD, TRAILER_LEN, Version,
    };
}

/// 最坏长度超过整个页池时的说明：两个准入入口共用同一句话。
const NEVER_FITS: &str = "这条请求的最坏长度超过整个 KV 页池：调小 max_new 或加大页池";

/// 页都借出去时的说明：可重试，所以是 `503`。
const FULL: &str = "KV 页不足，请稍后重试";

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
    ///
    /// 三个失败态各对应一类不同的客户端动作：`400` 改请求、`404` 改路径、`503` 等一会儿重试。
    /// 把它们合并成“请求失败”会让客户端只能盲重试。
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
///
/// 实现者可以是内存里的测试端点，也可以是 HTTP 服务器；服务器只负责把字节交给 [`Endpoint::call`]。
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

/// 准入的服务侧结局：已进批（带序列号）或现在页满。
///
/// 是 [`engine::Admission`] 的服务侧投影；序列自身不成立不归这里，继续走 `Err` 由
/// [`refused`] 归类。
enum AdmissionOutcome {
    /// 已进批：之后用序列号找它。
    Admitted(u64),
    /// 页不够：序列原样退回，可重试。
    Full,
}

/// 准入端口：把序列交进调度器或批，并回答"进去了 / 现在满"。
///
/// 三态到 [`Response`] 的翻译只有 [`admit_response`] 一份；实现只把自己的准入结局收成
/// [`AdmissionOutcome`]，不认识 HTTP 状态码。
trait Admitter {
    /// 最坏情况是否装得下：装不下是终局 400，不是可重试的 503。
    fn fits(&self, sequence: &engine::Sequence) -> bool;

    /// 准入；`Ok(Some(id))` 已进，`Ok(None)` 现在满；序列自身不成立走 `Err`。
    fn admit(&mut self, sequence: engine::Sequence) -> Result<AdmissionOutcome>;
}

impl Admitter for engine::Scheduler {
    fn fits(&self, sequence: &engine::Sequence) -> bool {
        engine::Scheduler::fits(self, sequence)
    }

    fn admit(&mut self, sequence: engine::Sequence) -> Result<AdmissionOutcome> {
        // 调度器回的是「已算好的前缀 token 数」：`None` 才是装不下。服务层不需要这个数——
        // 页与位置都由批层自己记账，这里只关心进没进。
        Ok(match engine::Scheduler::admit(self, &sequence)? {
            Some(_) => AdmissionOutcome::Admitted(sequence.id()),
            None => AdmissionOutcome::Full,
        })
    }
}

impl Admitter for engine::Batch {
    fn fits(&self, sequence: &engine::Sequence) -> bool {
        engine::Batch::fits(self, sequence)
    }

    fn admit(&mut self, sequence: engine::Sequence) -> Result<AdmissionOutcome> {
        let id = sequence.id();
        match engine::Batch::admit(self, sequence)? {
            engine::Admission::Admitted => Ok(AdmissionOutcome::Admitted(id)),
            engine::Admission::Full(_) => Ok(AdmissionOutcome::Full),
        }
    }
}

/// 把一次准入的结局翻译成响应。
///
/// 三种结局必须是三个不同的东西，混在一起就会让客户端看不到"是我写错了"：
///
/// - **进去**：200 + 序列号；
/// - **现在满**（页被别的序列占着）：可重试的 503，客户端等一会儿再来即可；
/// - **永远装不下**（[`Admitter::fits`] 为假，即最坏情况的页数超过整个页池）或
///   **请求自身不成立**（准入报 `invalid`：id 已在批里、序列已结束、页宽不一致）：
///   重试多少次都一样，只有改请求才有用，因此是 400。
///
/// 后两类都落成 [`Response`] 而不是 `Err`：端点契约里 `Err` 意味着"服务端自己出错了"，
/// 会被传输层变成 5xx 或断连接——客户端于是永远读不到"你的 id 重了"这类信息。只有真正的
/// 服务端故障（`internal` / `io` 等）才继续上抛。
fn admit_response<A: Admitter>(admitter: &mut A, sequence: engine::Sequence) -> Result<Response> {
    if !admitter.fits(&sequence) {
        return Ok(Response::error(Status::BadRequest, NEVER_FITS));
    }
    match admitter.admit(sequence) {
        Ok(AdmissionOutcome::Admitted(id)) => Ok(Response::ok(id.to_le_bytes().to_vec())),
        Ok(AdmissionOutcome::Full) => Ok(Response::error(Status::Unavailable, FULL)),
        Err(error) => refused(error),
    }
}

/// 尝试把一个序列准入调度器，并把结果翻译成响应。
///
/// # Errors
///
/// 序列自身不成立且错误类别不是 `invalid` 时上抛；`invalid` 一律落成 `400`。
pub fn admit(scheduler: &mut engine::Scheduler, sequence: &engine::Sequence) -> Result<Response> {
    admit_response(scheduler, sequence.clone())
}

/// 把一个序列交进 [`engine::Batch`]：与 [`admit`] 同一套判定，而批会同时持有序列与页账。
///
/// 端点走这条入口：它需要"页进去了"与"序列在批里"同时成立，否则调用方还得自己维护
/// 第二份列表——那正是两份状态开始漂移的地方。
///
/// # Errors
///
/// 序列自身不成立（重名、已结束、页宽与调度器不一致）且错误类别不是 `invalid` 时上抛；
/// `invalid` 一律落成 `400`。
pub fn admit_batch(batch: &mut engine::Batch, sequence: engine::Sequence) -> Result<Response> {
    admit_response(batch, sequence)
}

/// 请求侧错误一律落成 400（`invalid` 全是"这次请求自己不成立"，改什么由客户端决定）；
/// 其余类别（`internal` / `io`）才是本侧的 bug，继续上抛给传输层。
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
mod tests {
    use super::*;
    use engine::kv::{Cache, Pager};
    use engine::{Scheduler, Sequence};

    /// 页池：2 页、每页 16 个 token、每行 2 个元素。
    fn scheduler() -> Scheduler {
        Scheduler::new(Pager::new(Cache::new(2, 16).expect("页池"), 2).expect("行宽"))
    }

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
        let mut scheduler = scheduler();
        let small = Sequence::new(1, &[0; 4], 2, 16, engine::exec::Stream::new(0)).expect("序列");
        let response = admit(&mut scheduler, &small).expect("准入");
        assert_eq!(response.status(), Status::Ok);

        // 提示就占满剩下的一页，但跑到最后要第二页：现在不行，等别人结束再试。
        let full = Sequence::new(2, &[0; 20], 4, 16, engine::exec::Stream::new(0)).expect("序列");
        assert_eq!(scheduler.cache().free_pages(), 1);
        let refused = admit(&mut scheduler, &full).expect("拒绝不是错误");
        assert_eq!(refused.status(), Status::Unavailable);
        assert_eq!(refused.status().code(), 503, "现在满 = 可重试");
        assert_eq!(scheduler.running(), 1);
    }

    /// 最坏情况装不下与"现在满"不是一回事：重试没有用，必须改请求，所以不给 503。
    #[test]
    fn an_impossible_request_is_a_bad_request_not_a_retry() {
        let mut scheduler = scheduler();
        let endless =
            Sequence::new(9, &[0; 4], 1024, 16, engine::exec::Stream::new(0)).expect("序列");
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
        let mut scheduler = scheduler();
        let small = Sequence::new(1, &[0; 4], 2, 16, engine::exec::Stream::new(0)).expect("序列");
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
        let wrong_page =
            Sequence::new(2, &[0; 4], 2, 8, engine::exec::Stream::new(0)).expect("序列");
        let mismatched = admit(&mut scheduler, &wrong_page).expect("拒绝不是错误");
        assert_eq!(mismatched.status().code(), 400);
        assert_eq!(scheduler.running(), 1);
    }

    /// 已结束的序列不该再进批（占一页白等）：这是请求状态错了，不是服务端坏了。
    #[test]
    fn a_finished_sequence_is_a_bad_request_rather_than_a_server_failure() {
        let mut scheduler = scheduler();
        let mut sequence =
            Sequence::new(5, &[0; 0], 1, 16, engine::exec::Stream::new(0)).expect("序列");
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
