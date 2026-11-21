use super::*;
use crate::{Endpoint, Response, Route};
use std::net::TcpStream;

/// 固定响应的测试端点：命中即 200，证明分派确实经过 [`Router::call`]。
struct Echo {
    route: Route,
}

impl Endpoint for Echo {
    fn route(&self) -> &Route {
        &self.route
    }

    fn call(&self, _body: &[u8]) -> Result<Response> {
        Ok(Response::ok(vec![0]))
    }
}

fn router() -> Router {
    let mut router = Router::new();
    router
        .mount(Echo {
            route: Route::parse("POST /v1/echo").expect("路由"),
        })
        .expect("挂载");
    router
}

/// 单线程往返：先连好客户端并写完请求，再 `accept` 处理已缓冲的连接，最后读全响应。
///
/// `connect` 在用户态 `accept` 之前就已由内核完成握手，数据也会先进接收缓冲，因此
/// 无需另起线程即可测完整 `socket → Router → socket` 链路。
///
/// 三个用例只差 `timeout`（短超时验证 408）与 `half_close`（请求结束不靠长度），其余
/// 步骤逐字相同：绑定方式随 `timeout` 分岔，半关闭发生在写完请求之后。
fn exchange_with(timeout: Option<Duration>, half_close: bool, request: &[u8]) -> Vec<u8> {
    let httpd = match timeout {
        Some(limit) => Httpd::bind_with_timeout("127.0.0.1:0", limit).expect("绑定短超时测试服务"),
        None => Httpd::bind("127.0.0.1:0").expect("绑定"),
    };
    let addr = httpd.local_addr().expect("地址");
    let mut client = TcpStream::connect(addr).expect("连接");
    client.write_all(request).expect("写");
    if half_close {
        client.shutdown(std::net::Shutdown::Write).expect("半关闭");
    }
    httpd.serve_once(&router()).expect("处理");
    let mut raw = Vec::new();
    client.read_to_end(&mut raw).expect("读");
    raw
}

fn exchange(request: &[u8]) -> Vec<u8> {
    exchange_with(None, false, request)
}

fn exchange_with_timeout(timeout: Duration, request: &[u8]) -> Vec<u8> {
    exchange_with(Some(timeout), false, request)
}

fn exchange_closed_write(request: &[u8]) -> Vec<u8> {
    exchange_with(None, true, request)
}

#[test]
fn a_post_round_trips_through_the_router() {
    let raw = exchange(b"POST /v1/echo HTTP/1.1\r\nContent-Length: 1\r\n\r\n\x07");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
}

#[test]
fn an_unknown_route_is_a_404() {
    let raw = exchange(b"GET /nope HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 404 Not Found\r\n"), "{text}");
}

#[test]
fn an_oversized_declared_body_is_refused() {
    let raw = exchange(b"POST /v1/echo HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 413 "), "{text}");
}

#[test]
fn conflicting_content_lengths_are_refused() {
    let raw =
        exchange(b"POST /v1/echo HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n\x07");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 400 "), "{text}");
}

#[test]
fn repeated_identical_content_lengths_are_accepted() {
    let raw =
        exchange(b"POST /v1/echo HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n\x07");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
}

#[test]
fn transfer_encoding_is_refused() {
    let raw = exchange(b"POST /v1/echo HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 400 "), "{text}");
}

#[test]
fn read_timeout_is_request_timeout() {
    let raw = exchange_with_timeout(Duration::from_millis(50), b"POST /v1/echo HTTP/1.1\r\n");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 408 "), "{text}");
}

#[test]
fn head_limit_has_a_specific_status() {
    const { assert!(MAX_HEAD >= 8 * 1024) };
    assert_eq!(reason(431), "Request Header Fields Too Large");
}

#[test]
fn truncated_body_is_a_bad_request() {
    let raw = exchange_closed_write(b"POST /v1/echo HTTP/1.1\r\nContent-Length: 4\r\n\r\n\x07");
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 400 "), "{text}");
}

#[test]
fn a_body_is_bounded_by_content_length() {
    // 主体里夹着看似请求的字节，也要按 Content-Length 精确截断，不越读。
    let raw = exchange(
        b"POST /v1/echo HTTP/1.1\r\nContent-Length: 1\r\n\r\n\x01\r\nGET /extra HTTP/1.1\r\n",
    );
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
}

#[test]
fn http_head_mutation_sweep_is_stable_and_bounded() {
    let seed = b"POST /v1/echo HTTP/1.1\r\nContent-Length: 2\r\nX-Test: 1\r\n\r\n";
    for length in 0..=seed.len() {
        let prefix = &seed[..length];
        for replacement in [0u8, b'X', b'\r', b'\n', 0xff] {
            let mut candidate = prefix.to_vec();
            candidate.push(replacement);
            candidate.extend_from_slice(&seed[length..]);
            let first = fuzz_http_head(&candidate).map(|_| 1u8);
            let second = fuzz_http_head(&candidate).map(|_| 1u8);
            assert_eq!(first.is_ok(), second.is_ok(), "单输入结论必须稳定");
            assert!(candidate.len() <= MAX_HEAD);
        }
    }
}
