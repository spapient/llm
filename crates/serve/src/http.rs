//! std-only HTTP/1.1 薄适配器：把 socket 字节交给 [`Router::call`]，再把
//! [`Response::frame`] 包成最小响应写回。

use crate::Router;
use diagnostic::Result;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

const SITE: &str = "serve::http";

/// 请求头字节上限：防御无 `\r\n\r\n` 的恶意连接撑爆内存。
const MAX_HEAD: usize = 8 * 1024;
/// 请求体字节上限：请求文法自身还会再校验，这里只兜住未声明长度的狂读。
const MAX_BODY: usize = 64 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);

/// 一条已解析的 HTTP/1.1 请求。
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// 绑定好的 HTTP 服务端：单线程 accept → 解析 → 分派 → 回写 → 关闭。
pub struct Httpd {
    listener: TcpListener,
    socket_timeout: Duration,
}

#[derive(Debug)]
struct HttpError {
    status: u16,
    message: String,
}

impl HttpError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }

    fn io(error: std::io::Error) -> Self {
        if matches!(
            error.kind(),
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
        ) {
            Self::new(408, "读取或写入连接超时")
        } else {
            Self::new(500, error.to_string())
        }
    }
}

impl Httpd {
    /// 绑定本地地址；`addr` 形如 `127.0.0.1:8080`。
    pub fn bind(addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(addr).map_err(io)?;
        Ok(Self {
            listener,
            socket_timeout: SOCKET_TIMEOUT,
        })
    }

    #[cfg(test)]
    fn bind_with_timeout(addr: &str, socket_timeout: Duration) -> Result<Self> {
        let listener = TcpListener::bind(addr).map_err(io)?;
        Ok(Self {
            listener,
            socket_timeout,
        })
    }

    /// 实际绑定的地址（`bind` 传 `127.0.0.1:0` 时由内核分配，测试据此拿端口）。
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// 处理下一条到达的连接；连接级错误回写 500 后返回，不向上抛。
    pub fn serve_once(&self, router: &Router) -> Result<()> {
        let (mut stream, _) = self.listener.accept().map_err(io)?;
        configure_socket(&mut stream, self.socket_timeout);
        if let Err(error) = handle(&mut stream, router) {
            let _ = write_response(&mut stream, error.status, error.message.as_bytes());
        }
        Ok(())
    }

    /// 阻塞的单线程 accept 循环；一条连接失败不拖垮整个服务。
    pub fn serve(&self, router: &Router) -> Result<()> {
        for stream in self.listener.incoming() {
            let mut stream = stream.map_err(io)?;
            configure_socket(&mut stream, self.socket_timeout);
            if let Err(error) = handle(&mut stream, router) {
                let _ = write_response(&mut stream, error.status, error.message.as_bytes());
            }
        }
        Ok(())
    }
}

fn configure_socket(stream: &mut TcpStream, timeout: Duration) {
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
}

/// 一条连接的完整生命周期：读请求 → 交给路由 → 回写帧。
fn handle(stream: &mut TcpStream, router: &Router) -> std::result::Result<(), HttpError> {
    let Some(request) = read_request(stream)? else {
        return Ok(()); // 无请求的干净关闭。
    };
    match router.call(&request.method, &request.path, &request.body) {
        Ok(response) => {
            let status = response.status().code();
            let body = response
                .frame()
                .map_err(|error| HttpError::new(500, error.to_string()))?;
            write_response(stream, status, &body).map_err(HttpError::io)
        }
        Err(error) => {
            write_response(stream, 500, error.to_string().as_bytes()).map_err(HttpError::io)
        }
    }
}

/// 读一条请求：请求行 + 头（只取 `Content-Length`）+ 主体。EOF 且零字节时返回 `Ok(None)`。
fn read_request(stream: &mut TcpStream) -> std::result::Result<Option<HttpRequest>, HttpError> {
    let Some((buf, head_end)) = read_head(stream)? else {
        return Ok(None);
    };
    let (method, path, content_len) = parse_head(&buf[..head_end])?;
    let body = read_body(stream, &buf, head_end, content_len)?;
    Ok(Some(HttpRequest { method, path, body }))
}

/// 读头块直到出现 `\r\n\r\n`，返回（缓冲、头结束下标）；干净 EOF 返回 `Ok(None)`。
fn read_head(stream: &mut TcpStream) -> std::result::Result<Option<(Vec<u8>, usize)>, HttpError> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        if let Some(pos) = find_head(&buf) {
            return Ok(Some((buf, pos)));
        }
        let n = stream.read(&mut tmp).map_err(HttpError::io)?;
        if n == 0 {
            return eof_head(&buf);
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > MAX_HEAD {
            return Err(HttpError::new(431, "请求头过长"));
        }
    }
}

/// 头读到 EOF：缓冲为空是干净关闭，否则是截断。
fn eof_head(buf: &[u8]) -> std::result::Result<Option<(Vec<u8>, usize)>, HttpError> {
    if buf.is_empty() {
        Ok(None)
    } else {
        Err(HttpError::bad("连接在读请求头时截断"))
    }
}

/// 读主体到 `Content-Length` 字节：先吃头块之后已缓冲的尾字节，不足再继续读 socket。
fn read_body(
    stream: &mut TcpStream,
    buf: &[u8],
    head_end: usize,
    content_len: usize,
) -> std::result::Result<Vec<u8>, HttpError> {
    if content_len > MAX_BODY {
        return Err(HttpError::new(413, "请求体超过上限"));
    }
    let mut body = Vec::with_capacity(content_len);
    body.extend_from_slice(&buf[head_end + 4..]);
    body.truncate(content_len);
    while body.len() < content_len {
        let mut tmp = [0u8; 1024];
        let n = stream.read(&mut tmp).map_err(HttpError::io)?;
        if n == 0 {
            return Err(HttpError::bad("连接在读请求体时截断"));
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
fn parse_head(head: &[u8]) -> std::result::Result<(String, String, usize), HttpError> {
    let text = std::str::from_utf8(head).map_err(|_| HttpError::bad("请求头不是 UTF-8"))?;
    let mut lines = text.split("\r\n");
    let (method, path) = request_line(lines.next())?;
    let content_len = content_length(lines)?;
    Ok((method, path, content_len))
}

/// HTTP 头部解析的有界 fuzz 入口；只暴露错误，不持有 socket 或路由状态。
#[doc(hidden)]
pub fn fuzz_http_head(input: &[u8]) -> diagnostic::Result<()> {
    if input.len() > MAX_HEAD {
        return Err(diagnostic::Error::new(diagnostic::Code::Invalid, "请求头过长").with(SITE));
    }
    let Some(head_end) = find_head(input) else {
        return Err(diagnostic::Error::new(diagnostic::Code::Invalid, "请求头未结束").with(SITE));
    };
    parse_head(&input[..head_end]).map(|_| ()).map_err(|error| {
        diagnostic::Error::new(diagnostic::Code::Invalid, error.message).with(SITE)
    })
}

/// 请求行：`方法 路径 HTTP/版本`，只支持 HTTP/1.1。
fn request_line(line: Option<&str>) -> std::result::Result<(String, String), HttpError> {
    let line = line.ok_or_else(|| HttpError::bad("缺请求行"))?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path), Some(version)) = (parts.next(), parts.next(), parts.next())
    else {
        return Err(HttpError::bad("请求行必须是 `方法 路径 HTTP/版本`"));
    };
    if version != "HTTP/1.1" {
        return Err(HttpError::bad("只支持 HTTP/1.1"));
    }
    Ok((method.to_owned(), path.to_owned()))
}

/// 头行里找 `Content-Length`；缺失时按 0 处理（无主体）。
fn content_length<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> std::result::Result<usize, HttpError> {
    let (length, transfer_encoding) = lines.filter_map(|line| line.split_once(':')).try_fold(
        (None, false),
        |(length, transfer), (name, value)| {
            let (length, declared) = apply_header(length, name.trim(), value)?;
            Ok::<_, HttpError>((length, transfer || declared))
        },
    )?;
    finish(length, transfer_encoding)
}

/// 应用一行头：返回（合并后的长度，这一行是否声明了 `Transfer-Encoding`）。
fn apply_header(
    length: Option<usize>,
    name: &str,
    value: &str,
) -> std::result::Result<(Option<usize>, bool), HttpError> {
    match header(name) {
        Header::TransferEncoding => Ok((length, true)),
        Header::ContentLength => Ok((merged(length, value)?, false)),
        Header::Other => Ok((length, false)),
    }
}

/// 并列值合并后仍是「有长度」而不是「无长度」：`merge_lengths` 只判冲突。
fn merged(current: Option<usize>, value: &str) -> std::result::Result<Option<usize>, HttpError> {
    merge_lengths(current, value).map(Some)
}

/// 本层关心的两类头，其余一律忽略（闭集：新增关注点必须显式加一个变体）。
enum Header {
    ContentLength,
    TransferEncoding,
    Other,
}

/// 头名大小写不敏感：`Content-Length` 与 `content-length` 是同一个头。
fn header(name: &str) -> Header {
    if name.eq_ignore_ascii_case("transfer-encoding") {
        Header::TransferEncoding
    } else if name.eq_ignore_ascii_case("content-length") {
        Header::ContentLength
    } else {
        Header::Other
    }
}

/// 一条头行里可以并列多个 `Content-Length`：合并后必须只剩一个值。
fn merge_lengths(current: Option<usize>, value: &str) -> std::result::Result<usize, HttpError> {
    let mut merged = current;
    for piece in value.split(',') {
        let parsed = piece
            .trim()
            .parse::<usize>()
            .map_err(|_| HttpError::bad("Content-Length 不是整数"))?;
        if merged.is_some_and(|known| known != parsed) {
            return Err(HttpError::bad("冲突的 Content-Length"));
        }
        merged = Some(parsed);
    }
    // `split` 至少产出一段，所以合并后一定有值。
    Ok(merged.unwrap_or(0))
}

/// 收尾：`Transfer-Encoding` 一来就没有定长可言，未声明长度按 0 算。
fn finish(length: Option<usize>, transfer_encoding: bool) -> std::result::Result<usize, HttpError> {
    if transfer_encoding {
        return Err(HttpError::bad("本适配器不支持 Transfer-Encoding"));
    }
    Ok(length.unwrap_or(0))
}

/// 写最小响应：状态行 + `Content-Length` + `Connection: close` + 主体字节。
fn write_response(stream: &mut TcpStream, status: u16, body: &[u8]) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

/// 状态码 → 原因短语（闭集查表）。
const REASONS: [(u16, &str); 8] = [
    (200, "OK"),
    (400, "Bad Request"),
    (404, "Not Found"),
    (408, "Request Timeout"),
    (413, "Payload Too Large"),
    (431, "Request Header Fields Too Large"),
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

fn io(error: std::io::Error) -> diagnostic::Error {
    diagnostic::Error::new(diagnostic::Code::Io, error.to_string()).with(SITE)
}

#[cfg(test)]
#[path = "../tests/http_tests.rs"]
mod tests;
