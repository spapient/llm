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
    /// # Errors
    ///
    /// 重复路由（方法与路径都相同）。
    pub fn mount(&mut self, endpoint: Box<dyn Endpoint>) -> Result<()> {
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
        self.endpoints.push(endpoint);
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
#[path = "../tests/router_tests.rs"]
mod tests;
