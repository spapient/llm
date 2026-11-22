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
// CRC-32 的唯一实现在 L1 `wire`：帧尾校验直接复用它，协议层不再自带第二张表。
mod frame;
mod protocol;

pub(crate) use protocol::macros;

/// CRC-32：与 `weights`/`train` 共用 L1 `wire` 的唯一实现。
pub(crate) use wire::crc::crc32;

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
#[path = "../tests/serve_tests.rs"]
mod tests;
pub mod host;
