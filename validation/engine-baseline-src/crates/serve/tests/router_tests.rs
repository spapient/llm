use super::*;
use crate::Endpoint;

/// 回显端点：把请求体原样作为成功载荷返回，用来验证分派有没有找对端点。
struct Echo {
    route: Route,
}

impl Endpoint for Echo {
    fn route(&self) -> &Route {
        &self.route
    }

    fn call(&self, body: &[u8]) -> Result<Response> {
        Ok(Response::ok(body.to_vec()))
    }
}

fn echo(spec: &str) -> Box<dyn Endpoint> {
    Box::new(Echo {
        route: Route::parse(spec).expect("路由"),
    })
}

#[test]
fn a_mounted_endpoint_receives_its_request() {
    let mut router = Router::new();
    assert!(router.is_empty());
    router.mount(echo("POST /v1/completions")).expect("挂载");
    router.mount(echo("GET /health")).expect("挂载");
    assert_eq!(router.len(), 2);

    let response = router.call("POST", "/v1/completions", b"hi").expect("命中");
    assert_eq!(response.status(), Status::Ok);
    assert_eq!(response.envelope().payload(), b"hi");

    let routes: Vec<String> = router
        .routes()
        .map(|route| format!("{} {}", route.method(), route.path()))
        .collect();
    assert_eq!(routes, ["POST /v1/completions", "GET /health"]);
}

/// 方法大小写不该决定命中与否：传输层传 `post` 还是 `POST` 是它的写法，不是协议差异。
#[test]
fn the_method_is_matched_case_insensitively() {
    let mut router = Router::new();
    router.mount(echo("POST /v1/completions")).expect("挂载");
    assert!(router.call("post", "/v1/completions", b"").is_ok());
    assert_eq!(
        router
            .call("POST", "/v1/completions", b"")
            .expect("命中")
            .status(),
        Status::Ok
    );
}

/// 未命中是 404 而不是 `Err`：打错路径是客户端的问题，服务端没有坏。
#[test]
fn an_unknown_route_is_a_404() {
    let mut router = Router::new();
    router.mount(echo("POST /v1/completions")).expect("挂载");
    for (method, path) in [("GET", "/v1/completions"), ("POST", "/v1/other")] {
        let response = router.call(method, path, b"").expect("分派");
        assert_eq!(response.status(), Status::NotFound);
        assert_eq!(response.status().code(), 404);
    }
}

/// 两条端点抢同一个路由时拒绝第二条：否则"哪一条在跑"会变成挂载顺序的偶然结果。
#[test]
fn a_duplicate_route_is_refused() {
    let mut router = Router::new();
    router.mount(echo("POST /v1/completions")).expect("挂载");
    let error = router.mount(echo("POST /v1/completions")).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("/v1/completions"), "{error}");
    assert_eq!(router.len(), 1);
}

/// 端点自己报的服务端故障要原样上抛，不能被分派层改写成一个状态码。
#[test]
fn a_server_fault_from_the_endpoint_is_propagated() {
    struct Faulty {
        route: Route,
    }

    impl Endpoint for Faulty {
        fn route(&self) -> &Route {
            &self.route
        }

        fn call(&self, _body: &[u8]) -> Result<Response> {
            Err(Error::new(Code::Internal, "页账不一致").with("test::Faulty"))
        }
    }

    let mut router = Router::new();
    router
        .mount(Box::new(Faulty {
            route: Route::parse("GET /metrics").expect("路由"),
        }))
        .expect("挂载");
    let error = router.call("GET", "/metrics", b"").unwrap_err();
    assert_eq!(error.code(), Code::Internal);
}

#[test]
fn every_status_has_its_own_code() {
    let codes: Vec<u16> = Status::ALL.iter().map(|status| status.code()).collect();
    assert_eq!(codes, [200, 400, 404, 503]);
}
