//! 请求文法：[`Request`]。
//!
//! 服务层必须有"一个请求长什么样"的唯一定义，否则传输层、端点与测试会各自发明一套字段名。
//! 这里用最小的行文法而不是 JSON：没有解析器依赖（ADR 0022 之后本仓运行期不带 TOML/JSON
//! 解析器），而字段只有三个。文法本身就是契约：
//!
//! ```text
//! id=7
//! prompt=hello world
//! max_new=16
//! ```
//!
//! - 每行 `键=值`，键取第一个 `=` 之前的部分，值取其余全部（提示里可以再有 `=`）；
//! - 空行忽略，行尾的 `\r` 忽略（容忍 CRLF）；
//! - 未知键、重复键、缺字段、数字解析失败都是 `invalid`——服务层据此报 `400`，而不是
//!   把请求静默当成缺省值跑一遍再返回看不懂的结果；
//! - 停止符不在这里：采到哪个 token 算"停"是采样策略的事（[`crate::Sampling`]）。

use diagnostic::{Code, Error, Result};
use std::str::FromStr;

/// 一条补全请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    id: u64,
    prompt: String,
    max_new: usize,
}

impl Request {
    /// 解析请求体。
    ///
    /// # Errors
    ///
    /// 非 UTF-8、非 `键=值`、未知键、重复键、缺字段，或数字字段解析失败。
    pub fn parse(body: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(body)
            .map_err(|_| Error::new(Code::Invalid, "请求体不是 UTF-8").with(Self::SITE))?;
        let mut builder = Builder::default();
        for (index, line) in text.lines().enumerate() {
            builder.line(index + 1, line)?;
        }
        builder.finish()
    }

    /// 请求编号：由客户端给，重复编号会被准入拒绝成 `400`。
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 提示文本。
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// 最多生成多少个 token。
    pub fn max_new(&self) -> usize {
        self.max_new
    }

    const SITE: &'static str = "serve::Request::parse";
}

/// 请求字段闭集：键名、类型名与取值槽位的唯一事实源；新增字段在这里加一条，
/// 键名清单与未知键文案随之派生，不再散落在匹配臂与错误消息里。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Id,
    Prompt,
    MaxNew,
}

impl Field {
    /// 全部字段，供键名检索与"只认哪些"清单遍历。
    const ALL: [Self; 3] = [Self::Id, Self::Prompt, Self::MaxNew];

    /// 线上键名。
    const fn key(self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Prompt => "prompt",
            Self::MaxNew => "max_new",
        }
    }

    /// 按键名找回字段；未知键返回 `None`。
    fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|field| field.key() == key)
    }

    /// 识别出的键名清单：未知键错误里列出"只认"哪些。
    fn known_names() -> String {
        Self::ALL.map(|field| field.key()).join(" / ")
    }
}

/// 逐行累积的请求构造器；字段全是 `Option`，好让"缺了哪个"在 `finish` 时一次说清。
#[derive(Debug, Default)]
struct Builder {
    id: Option<u64>,
    prompt: Option<String>,
    max_new: Option<usize>,
}

impl Builder {
    fn line(&mut self, number: usize, line: &str) -> Result<()> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() {
            return Ok(());
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(Self::line_error(number, "每行必须是 `键=值`"));
        };
        self.set(number, key.trim(), value.trim())
    }

    fn set(&mut self, number: usize, key: &str, value: &str) -> Result<()> {
        let field = Field::from_key(key).ok_or_else(|| {
            Self::line_error(
                number,
                &format!("未知字段 `{key}`：只认 {}", Field::known_names()),
            )
        })?;
        match field {
            Field::Id => Self::set_parsed(&mut self.id, number, key, value, "u64"),
            Field::Prompt => Self::set_text(&mut self.prompt, number, key, value),
            Field::MaxNew => Self::set_parsed(&mut self.max_new, number, key, value, "usize"),
        }
    }

    /// 文本字段：重复键拒绝，原值存入。辅助件只碰传入槽位，不借整个构造器。
    fn set_text(slot: &mut Option<String>, number: usize, key: &str, value: &str) -> Result<()> {
        if slot.is_some() {
            return Err(Self::duplicate(number, key));
        }
        *slot = Some(value.to_string());
        Ok(())
    }

    /// 数值字段：重复键拒绝，按 [`FromStr`] 解析；`type_name` 只用于错误文案。
    fn set_parsed<T: FromStr>(
        slot: &mut Option<T>,
        number: usize,
        key: &str,
        value: &str,
        type_name: &str,
    ) -> Result<()> {
        if slot.is_some() {
            return Err(Self::duplicate(number, key));
        }
        let parsed = value.parse::<T>().map_err(|_| {
            Self::line_error(number, &format!("`{key}` 不是 {type_name}：`{value}`"))
        })?;
        *slot = Some(parsed);
        Ok(())
    }

    /// 三个必填字段逐个取出：缺哪个就报哪个，而不是笼统说"请求不完整"。
    fn finish(self) -> Result<Request> {
        let Some(id) = self.id else {
            return Err(Self::missing(Field::Id.key()));
        };
        let Some(prompt) = self.prompt else {
            return Err(Self::missing(Field::Prompt.key()));
        };
        let Some(max_new) = self.max_new else {
            return Err(Self::missing(Field::MaxNew.key()));
        };
        Ok(Request {
            id,
            prompt,
            max_new,
        })
    }

    fn duplicate(number: usize, key: &str) -> Error {
        Self::line_error(number, &format!("字段 `{key}` 出现了两次"))
    }

    fn line_error(number: usize, message: &str) -> Error {
        Error::new(Code::Invalid, format!("第 {number} 行：{message}")).with(Request::SITE)
    }

    fn missing(field: &str) -> Error {
        Error::new(Code::Invalid, format!("缺少字段 `{field}`")).with(Request::SITE)
    }
}

pub mod router {
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
            self.endpoints.iter().map(Box::as_ref).find(|endpoint| {
                endpoint.route().method() == method && endpoint.route().path() == path
            })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_parses_key_value_lines() {
        let request = Request::parse(b"id=7\nprompt=hello world\nmax_new=16\n").expect("合法");
        assert_eq!(request.id(), 7);
        assert_eq!(request.prompt(), "hello world");
        assert_eq!(request.max_new(), 16);
    }

    /// 值取第一个 `=` 之后的全部：提示里本来就可能有等号（代码片段、`a=b` 这类输入）。
    #[test]
    fn a_value_may_contain_the_separator() {
        let request = Request::parse(b"id=1\nprompt=x=1+2\nmax_new=1").expect("合法");
        assert_eq!(request.prompt(), "x=1+2");
    }

    /// 空行与 CRLF 都不该改变文法：传输层不做换行规范化时这里仍然收得下。
    #[test]
    fn blank_lines_and_carriage_returns_are_tolerated() {
        let request = Request::parse(b"\r\nid=2\r\n\r\nprompt=hi\r\nmax_new=3\r\n").expect("合法");
        assert_eq!(request.id(), 2);
        assert_eq!(request.prompt(), "hi");
    }

    #[test]
    fn a_line_without_a_separator_is_refused() {
        let error = Request::parse(b"id=1\nprompt\nmax_new=2").unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("第 2 行"), "{error}");
    }

    #[test]
    fn an_unknown_field_is_refused_by_name() {
        let error = Request::parse(b"id=1\nprompt=hi\nmax_new=2\ntemperature=0.7").unwrap_err();
        assert!(error.to_string().contains("temperature"), "{error}");
    }

    #[test]
    fn a_duplicate_field_is_refused() {
        let error = Request::parse(b"id=1\nid=2\nprompt=hi\nmax_new=2").unwrap_err();
        assert!(error.to_string().contains("出现了两次"), "{error}");
    }

    #[test]
    fn every_missing_field_is_named() {
        for body in [
            &b"prompt=hi\nmax_new=2"[..],
            &b"id=1\nmax_new=2"[..],
            &b"id=1\nprompt=hi"[..],
        ] {
            let error = Request::parse(body).unwrap_err();
            assert!(error.to_string().contains("缺少字段"), "{error}");
        }
    }

    #[test]
    fn a_number_that_does_not_parse_is_refused_with_its_field() {
        let error = Request::parse(b"id=seven\nprompt=hi\nmax_new=2").unwrap_err();
        assert!(error.to_string().contains("`id`"), "{error}");
        let error = Request::parse(b"id=1\nprompt=hi\nmax_new=-2").unwrap_err();
        assert!(error.to_string().contains("`max_new`"), "{error}");
    }

    #[test]
    fn a_body_that_is_not_utf8_is_refused() {
        let error = Request::parse(&[0xff, 0xfe]).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("UTF-8"), "{error}");
    }

    /// 空提示是合法的（引擎允许空上下文），但字段本身仍要写出：缺 `prompt` 是另一回事。
    #[test]
    fn an_empty_prompt_is_legal_but_the_field_is_still_required() {
        let request = Request::parse(b"id=1\nprompt=\nmax_new=1").expect("合法");
        assert_eq!(request.prompt(), "");
        assert_eq!(
            Request::parse(b"id=1\nmax_new=1").unwrap_err().code(),
            Code::Invalid
        );
    }
}
