//! Real-socket tests for the blocking HTTP server on an ephemeral loopback
//! port.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use serve::httpserver;

/// Sends `raw` to a one-connection server and returns the response text.
fn roundtrip(
    server: &httpserver<'_, models::llamamodel, cpu::cpuexecutor>,
    addr: std::net::SocketAddr,
    raw: &[u8],
) -> String {
    std::thread::scope(|scope| {
        scope.spawn(|| server.serveone().unwrap());
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        client.write_all(raw).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        response
    })
}

#[test]
fn serves_health_over_a_real_socket() {
    let model = common::model(21);
    let provider = common::provider();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = httpserver::fromlistener(
        listener,
        &model,
        &provider,
        common::serviceconfig(4),
        Some(Duration::from_secs(5)),
    );

    let response = roundtrip(&server, addr, b"GET /health HTTP/1.1\r\nhost: x\r\n\r\n");
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(response.ends_with("\r\n\r\nok"));
}

#[test]
fn serves_generate_over_a_real_socket() {
    let model = common::model(22);
    let provider = common::provider();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = httpserver::fromlistener(
        listener,
        &model,
        &provider,
        common::serviceconfig(4),
        Some(Duration::from_secs(5)),
    );

    let body = "prompt=1,2,3&maxnewtokens=3";
    let raw = format!(
        "POST /generate HTTP/1.1\r\ncontent-type: {}\r\ncontent-length: {}\r\n\r\n{body}",
        common::URLENCODED,
        body.len()
    );
    let response = roundtrip(&server, addr, raw.as_bytes());
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(response.contains("tokens="));
    assert!(response.contains("reason=maxlength"));
}

#[test]
fn deadline_dispatch_still_answers_within_the_bound() {
    let model = common::model(23);
    let provider = common::provider();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    // A generous deadline exercises the scoped-worker dispatch path without
    // asserting a timing-sensitive outcome.
    let server = httpserver::fromlistener(
        listener,
        &model,
        &provider,
        common::serviceconfigwith(4, Some(Duration::from_secs(30))),
        Some(Duration::from_secs(5)),
    );

    let response = roundtrip(&server, addr, b"GET /health HTTP/1.1\r\n\r\n");
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
}

#[test]
fn an_expired_deadline_returns_a_gateway_timeout() {
    let model = common::model(24);
    let provider = common::provider();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    // A zero deadline elapses before the worker can report a result.
    let server = httpserver::fromlistener(
        listener,
        &model,
        &provider,
        common::serviceconfigwith(4, Some(Duration::ZERO)),
        Some(Duration::from_secs(5)),
    );

    let body = "prompt=1,2,3&maxnewtokens=3";
    let raw = format!(
        "POST /generate HTTP/1.1\r\ncontent-type: {}\r\ncontent-length: {}\r\n\r\n{body}",
        common::URLENCODED,
        body.len()
    );
    let response = roundtrip(&server, addr, raw.as_bytes());
    assert!(
        response.starts_with("HTTP/1.1 504 Gateway Timeout\r\n"),
        "response was {response:?}"
    );
}
