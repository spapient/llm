//! Relocated unit tests for the `serve` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/serve/src` hold no test code. Where a test
//! previously reached a crate-private helper, it now drives the public
//! `httprequest::parse` or `inferenceservice::handle` seam (see
//! `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

mod common;

/// Tests relocated from `crates/serve/src/admit.rs`.
mod admit {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    use ::serve::admissions;
    use ::serve::serveerror;

    #[test]
    fn rejects_a_zero_ceiling() {
        assert_eq!(
            admissions::new(0).map(|_| ()),
            Err(serveerror::invalidconfig {
                detail: "maxconcurrent must be at least 1",
            })
        );
    }

    #[test]
    fn grants_and_releases_on_drop() {
        let control = admissions::new(2).unwrap();
        assert_eq!(control.getlimit(), 2);
        assert_eq!(control.getinflight(), 0);
        let guard = control.tryadmit().unwrap();
        assert_eq!(control.getinflight(), 1);
        drop(guard);
        assert_eq!(control.getinflight(), 0);
    }

    #[test]
    fn rejects_past_the_ceiling() {
        let control = admissions::new(1).unwrap();
        let _guard = control.tryadmit().unwrap();
        assert_eq!(
            control.tryadmit().map(|_| ()),
            Err(serveerror::overloaded { limit: 1 })
        );
        assert_eq!(control.getinflight(), 1);
    }

    #[test]
    fn concurrent_admissions_never_exceed_the_ceiling() {
        let control = admissions::new(2).unwrap();
        let barrier = Barrier::new(4);
        let successes = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let control = &control;
                let barrier = &barrier;
                let successes = &successes;
                scope.spawn(move || {
                    let guard = control.tryadmit();
                    if guard.is_ok() {
                        successes.fetch_add(1, Ordering::AcqRel);
                    }
                    barrier.wait();
                    drop(guard);
                });
            }
        });
        assert_eq!(successes.load(Ordering::Acquire), 2);
        assert_eq!(control.getinflight(), 0);
    }
}

/// Tests relocated from `crates/serve/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::io::{BufRead, Read};

    use ::serve::{httprequest, serveerror, transportlimits};
    use infer::infererror;

    #[test]
    fn transport_and_model_failures_stay_distinct() {
        let transport = serveerror::linetoolong { len: 9, limit: 8 };
        let model = serveerror::inference(infererror::emptylogits);
        assert_ne!(transport, model);
        assert!(matches!(model, serveerror::inference(_)));
    }

    #[test]
    fn renders_structured_transport_context() {
        assert_eq!(
            serveerror::toomanyheaders { limit: 4 }.to_string(),
            "too many headers: limit 4"
        );
        assert_eq!(
            serveerror::incompletebody {
                expected: 5,
                received: 2,
            }
            .to_string(),
            "incomplete request body: expected 5 bytes, received 2"
        );
    }

    /// A reader whose reads fail, driving the parser's I/O error mapping.
    struct failingreader;

    impl Read for failingreader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "secret path",
            ))
        }
    }

    impl BufRead for failingreader {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "secret path",
            ))
        }

        fn consume(&mut self, _amount: usize) {}
    }

    fn limits() -> transportlimits {
        transportlimits::new(1024, 128, 8, 256, 8, 64, 16, 16)
    }

    #[test]
    fn wraps_io_kind_without_the_original_error() {
        let error = httprequest::parse(&mut failingreader, &limits()).unwrap_err();
        assert_eq!(
            error,
            serveerror::io {
                kind: std::io::ErrorKind::TimedOut,
            }
        );
        assert!(!error.to_string().contains("secret"));
    }
}

/// Tests relocated from `crates/serve/src/request.rs`.
mod request {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::io::Cursor;

    use ::serve::{httpmethod, httprequest, serveerror, transportlimits};

    fn limits() -> transportlimits {
        transportlimits::new(1024, 128, 8, 256, 8, 64, 16, 16)
    }

    fn parse(bytes: &[u8]) -> Result<httprequest, serveerror> {
        httprequest::parse(&mut Cursor::new(bytes), &limits())
    }

    #[test]
    fn parses_a_get_request() {
        let request =
            parse(b"GET /health?verbose=1 HTTP/1.1\r\nHost: x\r\nUser-Agent: t\r\n\r\n").unwrap();
        assert_eq!(request.getmethod(), httpmethod::get);
        assert_eq!(request.getpath(), "/health");
        assert_eq!(request.gettarget(), "/health?verbose=1");
        assert_eq!(request.getquery("verbose"), Some("1"));
        assert_eq!(request.getheader("host"), Some("x"));
        assert_eq!(request.getheader("user-agent"), Some("t"));
        assert_eq!(request.getbody(), b"");
    }

    #[test]
    fn parses_a_post_body_up_to_content_length() {
        let bytes = b"POST /generate HTTP/1.1\r\nContent-Length: 7\r\n\r\nprompt=";
        let request = parse(bytes).unwrap();
        assert_eq!(request.getmethod(), httpmethod::post);
        assert_eq!(request.getbody(), b"prompt=");
        assert_eq!(request.getbodytext().unwrap(), "prompt=");
    }

    #[test]
    fn rejects_a_malformed_request_line() {
        assert_eq!(parse(b"\r\n\r\n"), Err(serveerror::malformedrequestline));
        assert_eq!(
            parse(b"GET /health\r\n\r\n"),
            Err(serveerror::malformedrequestline)
        );
    }

    #[test]
    fn rejects_an_unsupported_version_and_method() {
        assert_eq!(
            parse(b"GET / HTTP/2.0\r\n\r\n"),
            Err(serveerror::unsupportedversion)
        );
        assert_eq!(
            parse(b"PUT / HTTP/1.1\r\n\r\n"),
            Err(serveerror::unsupportedmethod {
                method: "PUT".to_string(),
            })
        );
    }

    #[test]
    fn rejects_a_malformed_header() {
        assert_eq!(
            parse(b"GET / HTTP/1.1\r\nnocolon\r\n\r\n"),
            Err(serveerror::malformedheader)
        );
        assert_eq!(
            parse(b"GET / HTTP/1.1\r\n: empty\r\n\r\n"),
            Err(serveerror::malformedheader)
        );
    }

    #[test]
    fn enforces_header_count_and_length() {
        let small = transportlimits::new(1024, 32, 1, 256, 8, 64, 16, 16);
        assert_eq!(
            httprequest::parse(
                &mut Cursor::new(b"GET / HTTP/1.1\r\na: 1\r\nb: 2\r\n\r\n"),
                &small,
            ),
            Err(serveerror::toomanyheaders { limit: 1 })
        );
        let long = transportlimits::new(1024, 8, 8, 256, 8, 64, 16, 16);
        assert!(matches!(
            httprequest::parse(&mut Cursor::new(b"GET / HTTP/1.1\r\n\r\n"), &long),
            Err(serveerror::linetoolong { .. })
        ));
    }

    #[test]
    fn enforces_body_bounds_and_length_errors() {
        assert_eq!(
            parse(b"POST / HTTP/1.1\r\nContent-Length: nope\r\n\r\n"),
            Err(serveerror::invalidcontentlength)
        );
        assert_eq!(
            parse(b"POST / HTTP/1.1\r\nContent-Length: 9999\r\n\r\n"),
            Err(serveerror::bodytoolarge {
                len: 9999,
                limit: 256,
            })
        );
        assert_eq!(
            parse(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nab"),
            Err(serveerror::incompletebody {
                expected: 5,
                received: 2,
            })
        );
    }

    #[test]
    fn rejects_incomplete_headers() {
        assert_eq!(
            parse(b"GET / HTTP/1.1\r\n"),
            Err(serveerror::incompleteheaders)
        );
    }
}

/// Tests relocated from `crates/serve/src/response.rs`.
///
/// The former unit tests inspected `httpresponse::fromerror` directly. The
/// status mapping is exercised end to end through `inferenceservice::handle`,
/// which routes each service error through that same mapping.
mod response {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::serve::{httpresponse, inferenceservice};

    use crate::common;

    fn respond(raw: &[u8]) -> httpresponse {
        let model = common::model(1);
        let provider = common::provider();
        let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));
        service.handle(&common::parse(raw))
    }

    #[test]
    fn maps_client_failures_to_statuses() {
        assert_eq!(
            respond(b"GET /nope HTTP/1.1\r\nhost: x\r\n\r\n").getstatus(),
            404
        );
        assert_eq!(
            respond(b"GET /generate HTTP/1.1\r\nhost: x\r\n\r\n").getstatus(),
            405
        );
        assert_eq!(
            respond(b"POST /generate HTTP/1.1\r\ncontent-type: application/json\r\ncontent-length: 18\r\n\r\nprompt=1&maxnewt=2").getstatus(),
            415
        );
    }

    #[test]
    fn maps_overload_to_too_many_requests() {
        let model = common::model(1);
        let provider = common::provider();
        let service = inferenceservice::new(&model, &provider, common::serviceconfig(1));
        let _permit = service.getconfig().getadmission().tryadmit().unwrap();
        let response = service.handle(&common::generaterequest(&[1], 2));
        assert_eq!(response.getstatus(), 429);
    }

    #[test]
    fn maps_a_model_rejection_to_bad_request() {
        // Token 9 is outside the tiny vocabulary, so the engine rejects it.
        assert_eq!(
            respond(&common::post("prompt=9&maxnewtokens=2", common::URLENCODED)).getstatus(),
            400
        );
    }

    #[test]
    fn writes_a_well_formed_response() {
        let response = httpresponse::ok(b"hello".to_vec());
        let mut out = Vec::new();
        response.write(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("content-length: 5\r\n"));
        assert!(text.ends_with("\r\n\r\nhello"));
        assert_eq!(response.getbodytext(), Some("hello"));
    }
}

/// Tests relocated from `crates/serve/src/routes.rs`.
///
/// The former unit tests called the crate-private `mediasupported`,
/// `parsetokens`, and `inputfromrequest` helpers. Their behavior is the request
/// contract of `inferenceservice::handle`, so it is verified there.
mod routes {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::serve::{httpresponse, inferenceservice};

    use crate::common;

    fn respond(raw: &[u8]) -> httpresponse {
        let model = common::model(1);
        let provider = common::provider();
        let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));
        service.handle(&common::parse(raw))
    }

    #[test]
    fn media_type_matching_ignores_parameters_and_case() {
        let accepted = common::post(
            "prompt=1&maxnewtokens=2",
            "Application/X-WWW-Form-Urlencoded; charset=utf-8",
        );
        assert_eq!(respond(&accepted).getstatus(), 200);
        let rejected = common::post("prompt=1&maxnewtokens=2", "application/json");
        assert_eq!(respond(&rejected).getstatus(), 415);
    }

    #[test]
    fn token_lists_are_parsed_and_bounded() {
        let good = respond(&common::post(
            "prompt=1,2,3&maxnewtokens=2",
            common::URLENCODED,
        ));
        assert_eq!(good.getstatus(), 200);
        // A non-numeric token and an oversized list are both rejected.
        assert_eq!(
            respond(&common::post(
                "prompt=1,x&maxnewtokens=2",
                common::URLENCODED
            ))
            .getstatus(),
            400
        );
        let many = std::iter::repeat_n("1", 128).collect::<Vec<_>>().join(",");
        let body = format!("prompt={many}&maxnewtokens=2");
        assert_eq!(
            respond(&common::post(&body, common::URLENCODED)).getstatus(),
            400
        );
    }

    #[test]
    fn input_requires_prompt_and_a_bounded_maxnewtokens() {
        // Missing prompt.
        assert_eq!(
            respond(&common::post("maxnewtokens=2", common::URLENCODED)).getstatus(),
            400
        );
        // Zero and oversized maxnewtokens.
        assert_eq!(
            respond(&common::post("prompt=1&maxnewtokens=0", common::URLENCODED)).getstatus(),
            400
        );
        assert_eq!(
            respond(&common::post(
                "prompt=1&maxnewtokens=9999",
                common::URLENCODED
            ))
            .getstatus(),
            400
        );
    }

    #[test]
    fn form_field_count_is_bounded() {
        // One more field than the transport allows (32) is rejected.
        let extra = (0..33)
            .map(|i| format!("f{i}=1"))
            .collect::<Vec<_>>()
            .join("&");
        let body = format!("prompt=1&maxnewtokens=2&{extra}");
        assert_eq!(
            respond(&common::post(&body, common::URLENCODED)).getstatus(),
            400
        );
    }
}
