//! 端点契约与它的第一个真实实现：[`Route`]、[`Status`]、[`Response`]、[`Endpoint`]、
//! 请求文法 [`Request`]、生成协作者 [`Predict`] / [`Sampling`] / [`Greedy`]、
//! 补全端点 [`Completion`] 与路由分派 [`Router`]。
//!
//! **拥有**：`Route`、`Endpoint`、`Response`、`Status`、`Request`、std-only HTTP/1.1
//! 薄适配器（`http`），以及线格式
//! （`protocol`：`Envelope`/`Frame`/`Decoder`/`Kind`/`Version`；`proto` 是兼容重导出路径）；
//! 注入 [`Predict`] 的 CPU 模型提供者收在 `kernels` / `assembly`（`model` 是兼容重导出路径）。
//! **不拥有**：序列状态与页账（`engine`）、分词（`tokenizer`）、算子参考算术与层定义（`nn`）、
//! 权重条目字节与读取契约（`weights`）、gRPC 服务器；
//! Metal/CUDA 真设备后端收在独立边界 crate，不回流本层。**依赖**：`engine`、`nn`、`train`、`tensor`、`tokenizer`、`diagnostic`、`weights`、`wire`。**被依赖**：`llm`。
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

mod admit;
mod completion;
mod endpoint;
mod generate;
mod http;
mod request;
mod speculate;

// CPU 内核桥与装配宿主（平铺；`model` 兼容路径见下）。
mod assembly;
mod block;
mod kernels;
pub mod model;
mod params;
mod plan;

pub(crate) use admit::refused;
pub(crate) use params::descriptors;

// 线协议：常量/闭集与宏在 `protocol`，帧与增量解码在 `frame`（`proto` 兼容路径见下）。
// CRC-32 的唯一实现在 L1 `storage`：帧尾校验直接复用它，协议层不再自带第二张表。
mod frame;
mod protocol;

pub(crate) use protocol::macros;

/// CRC-32：与 `weights`/`train` 共用 L1 `storage` 的唯一实现。
pub(crate) use storage::crc::crc32;

pub use admit::{admit, admit_batch};
pub use completion::Completion;
pub use endpoint::{Endpoint, Response, Route, Status, encode_prompt};
pub use generate::{
    Classical, Ending, Generation, Greedy, Multinomial, Predict, Sampling, SamplingParams,
};
pub use http::{Httpd, fuzz_http_head};
pub use params::tokenizer_from_segment;
pub use request::{Request, Router};
pub use speculate::{Drafter, Verdict, accept};

/// 兼容旧路径：端点契约的线格式。
pub mod proto {
    pub use crate::protocol::{
        Decoder, Envelope, Frame, HEADER_LEN, Kind, MAGIC, MAX_PAYLOAD, TRAILER_LEN, Version,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::{Code, Result};
    use engine::kv::{Cache, Pager};
    use engine::{Scheduler, Sequence};

    use crate::protocol::{Envelope, Kind};

    /// 页池：2 页、每页 16 个 token、每行 2 个元素。
    fn scheduler() -> Scheduler {
        Scheduler::new(Pager::new(Cache::new(2, 16).expect("页池"), 2).expect("行宽"))
    }

    testkit::test_cases! {
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

    fn statuses_map_to_http_codes() {
        assert_eq!(Status::Ok.code(), 200);
        assert_eq!(Status::BadRequest.code(), 400);
        assert_eq!(Status::Unavailable.code(), 503);
    }

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

    fn a_prompt_is_encoded_through_the_tokenizer_contract() {
        let tokens = encode_prompt(&tokenizer::Bytes, "hi").expect("编码");
        assert_eq!(tokens.len(), 2);
    }

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
    fn an_impossible_request_is_a_bad_request_not_a_retry() {
        let mut scheduler = scheduler();
        let endless = Sequence::new(9, &[0; 4], 1024, 16, engine::exec::Stream::new(0)).expect("序列");
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
        let wrong_page = Sequence::new(2, &[0; 4], 2, 8, engine::exec::Stream::new(0)).expect("序列");
        let mismatched = admit(&mut scheduler, &wrong_page).expect("拒绝不是错误");
        assert_eq!(mismatched.status().code(), 400);
        assert_eq!(scheduler.running(), 1);
    }

    /// 已结束的序列不该再进批（占一页白等）：这是请求状态错了，不是服务端坏了。
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
}
