//! 路由分派：[`Router`]。
//!
//! 一个进程里通常不止一条端点（补全、健康检查、指标），而"哪个请求归谁"必须先被答出来，
//! 否则每条端点都要自己解析方法与路径。`Router` 只做这一次匹配：
//!
//! - 方法与路径都要对上（方法大小写不敏感，路径逐字节比较）；
//! - 没命中返回 `404`，而不是 `Err`——客户端打错路径是请求侧的问题，不是服务端坏了；
//! - 挂载时重复路由直接拒绝：两条端点抢同一个路由时，"哪一条在跑"会变成注册顺序的偶然结果。
//!
//! 这里仍然没有 socket：传输层（HTTP / gRPC）是 P1，它落地时只需要把方法与路径交给 [`Router::call`]。

use crate::{Endpoint, Response, Route, Status};
use diagnostic::{Code, Error, Result};

/// 一组端点，按路由分派。
#[derive(Default)]
pub struct Router {
    endpoints: Vec<Box<dyn Endpoint>>,
}

impl Router {
    /// 空路由表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 挂载一条端点；同一 `方法 路径` 已经存在时返回 `invalid`。
    ///
    /// 泛型接收具体端点，只在存进路由表时装箱；调用方不必先 `Box::new`，也能保留具体类型
    /// 供测试与单态化使用。
    ///
    /// # Errors
    ///
    /// 重复路由（方法与路径都相同）。
    pub fn mount<E: Endpoint + 'static>(&mut self, endpoint: E) -> Result<()> {
        let route = endpoint.route().clone();
        if self.find(route.method(), route.path()).is_some() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "路由 `{} {}` 已经挂过一条端点",
                    route.method(),
                    route.path()
                ),
            )
            .with("serve::Router::mount"));
        }
        self.endpoints.push(Box::new(endpoint));
        Ok(())
    }

    /// 处理一次请求：命中就把请求体交给它，未命中返回 `404`。
    ///
    /// # Errors
    ///
    /// 端点自己报的服务端故障（`internal` / `io` 等）原样上抛；请求侧的问题已在端点内
    /// 落成 [`Response`]（见 [`crate::Endpoint::call`]）。
    pub fn call(&self, method: &str, path: &str, body: &[u8]) -> Result<Response> {
        match self.find(method, path) {
            Some(endpoint) => endpoint.call(body),
            None => Ok(Response::error(
                Status::NotFound,
                &format!("没有路由 `{method} {path}`"),
            )),
        }
    }

    /// 已挂载的路由数。
    pub fn len(&self) -> usize {
        self.endpoints.len()
    }

    /// 是否一条端点都没挂。
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }

    /// 已挂载的路由，按挂载顺序。
    pub fn routes(&self) -> impl Iterator<Item = &Route> {
        self.endpoints.iter().map(|endpoint| endpoint.route())
    }

    /// 匹配一条端点：方法大小写不敏感，路径逐字节比较。
    fn find(&self, method: &str, path: &str) -> Option<&dyn Endpoint> {
        let method = method.to_ascii_uppercase();
        self.endpoints
            .iter()
            .map(Box::as_ref)
            .find(|endpoint| endpoint.route().method() == method && endpoint.route().path() == path)
    }
}

#[cfg(test)]
mod tests {
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
}
