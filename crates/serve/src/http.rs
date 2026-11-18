//! std-only HTTP/1.1 薄适配器：把 socket 字节交给 [`Router::call`]，再把
//! [`Response::frame`] 包成最小响应写回。
//!
//! 这里**不**重写路由、请求文法、帧格式、CRC 或上限——它只是
//! `socket 字节 → Router::call → Response::frame → socket 字节` 的薄胶水。首刀只认
//! 请求行 + `Content-Length` + 主体，响应固定 `Connection: close`；排除 TLS / keep-alive /
//! chunked / 多线程。gRPC 未来在同一条 [`Endpoint`] / [`Router`] 上再挂一个适配器即可，不动端点模型。

use crate::Router;
use diagnostic::{Code, Error, Result};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

const SITE: &str = "serve::http";

/// 请求头字节上限：防御无 `\r\n\r\n` 的恶意连接撑爆内存。
const MAX_HEAD: usize = 8 * 1024;
/// 请求体字节上限：请求文法自身还会再校验，这里只兜住未声明长度的狂读。
const MAX_BODY: usize = 64 * 1024;

/// 一条已解析的 HTTP/1.1 请求。
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// 绑定好的 HTTP 服务端：单线程 accept → 解析 → 分派 → 回写 → 关闭。
pub struct Httpd {
    listener: TcpListener,
}

impl Httpd {
    /// 绑定本地地址；`addr` 形如 `127.0.0.1:8080`。
    pub fn bind(addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(addr).map_err(io)?;
        Ok(Self { listener })
    }

    /// 实际绑定的地址（`bind` 传 `127.0.0.1:0` 时由内核分配，测试据此拿端口）。
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// 处理下一条到达的连接；连接级错误回写 500 后返回，不向上抛。
    pub fn serve_once(&self, router: &Router) -> Result<()> {
        let (mut stream, _) = self.listener.accept().map_err(io)?;
        if let Err(error) = handle(&mut stream, router) {
            let _ = write_response(&mut stream, 500, error.to_string().as_bytes());
        }
        Ok(())
    }

    /// 阻塞的单线程 accept 循环；一条连接失败不拖垮整个服务。
    pub fn serve(&self, router: &Router) -> Result<()> {
        for stream in self.listener.incoming() {
            let mut stream = stream.map_err(io)?;
            if let Err(error) = handle(&mut stream, router) {
                let _ = write_response(&mut stream, 500, error.to_string().as_bytes());
            }
        }
        Ok(())
    }
}

/// 一条连接的完整生命周期：读请求 → 交给路由 → 回写帧。
fn handle(stream: &mut TcpStream, router: &Router) -> Result<()> {
    let Some(request) = read_request(stream)? else {
        return Ok(()); // 无请求的干净关闭。
    };
    match router.call(&request.method, &request.path, &request.body) {
        Ok(response) => {
            let status = response.status().code();
            let body = response.frame()?;
            write_response(stream, status, &body)
        }
        Err(error) => write_response(stream, 500, error.to_string().as_bytes()),
    }
}

/// 读一条请求：请求行 + 头（只取 `Content-Length`）+ 主体。EOF 且零字节时返回 `Ok(None)`。
fn read_request(stream: &mut TcpStream) -> Result<Option<HttpRequest>> {
    let Some((buf, head_end)) = read_head(stream)? else {
        return Ok(None);
    };
    let (method, path, content_len) = parse_head(&buf[..head_end])?;
    let body = read_body(stream, &buf, head_end, content_len)?;
    Ok(Some(HttpRequest { method, path, body }))
}

/// 读头块直到出现 `\r\n\r\n`，返回（缓冲、头结束下标）；干净 EOF 返回 `Ok(None)`。
fn read_head(stream: &mut TcpStream) -> Result<Option<(Vec<u8>, usize)>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        if let Some(pos) = find_head(&buf) {
            return Ok(Some((buf, pos)));
        }
        let n = stream.read(&mut tmp).map_err(io)?;
        if n == 0 {
            return eof_head(&buf);
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > MAX_HEAD {
            return Err(bad("请求头过长"));
        }
    }
}

/// 头读到 EOF：缓冲为空是干净关闭，否则是截断。
fn eof_head(buf: &[u8]) -> Result<Option<(Vec<u8>, usize)>> {
    if buf.is_empty() {
        Ok(None)
    } else {
        Err(bad("连接在读请求头时截断"))
    }
}

/// 读主体到 `Content-Length` 字节：先吃头块之后已缓冲的尾字节，不足再继续读 socket。
fn read_body(
    stream: &mut TcpStream,
    buf: &[u8],
    head_end: usize,
    content_len: usize,
) -> Result<Vec<u8>> {
    if content_len > MAX_BODY {
        return Err(bad("请求体超过上限"));
    }
    let mut body = Vec::with_capacity(content_len);
    body.extend_from_slice(&buf[head_end + 4..]);
    body.truncate(content_len);
    while body.len() < content_len {
        let mut tmp = [0u8; 1024];
        let n = stream.read(&mut tmp).map_err(io)?;
        if n == 0 {
            return Err(bad("连接在读请求体时截断"));
        }
        body.extend_from_slice(&tmp[..n]);
        body.truncate(content_len);
    }
    Ok(body)
}

/// 在缓冲区里找 `\r\n\r\n`（头结束），返回其起始下标。
fn find_head(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// 解析头块：请求行取方法、路径、版本，头里只取 `Content-Length`。
fn parse_head(head: &[u8]) -> Result<(String, String, usize)> {
    let text = std::str::from_utf8(head).map_err(|_| bad("请求头不是 UTF-8"))?;
    let mut lines = text.split("\r\n");
    let (method, path) = request_line(lines.next())?;
    let content_len = content_length(lines)?;
    Ok((method, path, content_len))
}

/// 请求行：`方法 路径 HTTP/版本`，只支持 HTTP/1.1。
fn request_line(line: Option<&str>) -> Result<(String, String)> {
    let line = line.ok_or_else(|| bad("缺请求行"))?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path), Some(version)) = (parts.next(), parts.next(), parts.next())
    else {
        return Err(bad("请求行必须是 `方法 路径 HTTP/版本`"));
    };
    if version != "HTTP/1.1" {
        return Err(bad("只支持 HTTP/1.1"));
    }
    Ok((method.to_owned(), path.to_owned()))
}

/// 头行里找 `Content-Length`；缺失时按 0 处理（无主体）。
fn content_length<'a>(lines: impl Iterator<Item = &'a str>) -> Result<usize> {
    let mut content_len = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
        {
            content_len = value
                .trim()
                .parse()
                .map_err(|_| bad("Content-Length 不是整数"))?;
        }
    }
    Ok(content_len)
}

/// 写最小响应：状态行 + `Content-Length` + `Connection: close` + 主体字节。
fn write_response(stream: &mut TcpStream, status: u16, body: &[u8]) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(io)?;
    stream.write_all(body).map_err(io)?;
    Ok(())
}

/// 状态码 → 原因短语（闭集查表）。
const REASONS: [(u16, &str); 5] = [
    (200, "OK"),
    (400, "Bad Request"),
    (404, "Not Found"),
    (503, "Service Unavailable"),
    (500, "Internal Server Error"),
];

fn reason(status: u16) -> &'static str {
    REASONS
        .iter()
        .find(|(code, _)| *code == status)
        .map(|(_, why)| *why)
        .unwrap_or("Unknown")
}

fn bad(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(SITE)
}

fn io(error: std::io::Error) -> Error {
    Error::new(Code::Io, error.to_string()).with(SITE)
}

#[cfg(test)]
mod tests {
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
    fn exchange(request: &[u8]) -> Vec<u8> {
        let httpd = Httpd::bind("127.0.0.1:0").expect("绑定");
        let addr = httpd.local_addr().expect("地址");
        let mut client = TcpStream::connect(addr).expect("连接");
        client.write_all(request).expect("写");
        httpd.serve_once(&router()).expect("处理");
        let mut raw = Vec::new();
        client.read_to_end(&mut raw).expect("读");
        raw
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
        assert!(text.starts_with("HTTP/1.1 500 "), "{text}");
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
}
