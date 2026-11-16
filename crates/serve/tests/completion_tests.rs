use super::*;
use crate::{Endpoint, Greedy};

/// 固定 logits 的模型桩：端点只关心"下一步算什么"，不关心怎么算出来。
struct Fixed(Vec<f32>);

impl Predict for Fixed {
    fn logits(&mut self, _sequence: &Sequence) -> diagnostic::Result<Vec<f32>> {
        Ok(self.0.clone())
    }
}

/// 采样桩：先给 `remaining` 个 token，然后说"到此为止"。
struct StopAfter {
    remaining: usize,
}

impl Sampling for StopAfter {
    fn sample(&mut self, _logits: &[f32]) -> diagnostic::Result<Option<u32>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(7))
    }
}

/// 模型桩：一步都算不出来。
struct Broken;

impl Predict for Broken {
    fn logits(&mut self, _sequence: &Sequence) -> diagnostic::Result<Vec<f32>> {
        Err(diagnostic::Error::new(diagnostic::Code::Internal, "权重缺失").with("test::Broken"))
    }
}

fn route() -> Route {
    Route::parse("POST /v1/completions").expect("路由")
}

fn endpoint<M: Predict, S: Sampling>(model: M, sampling: S, pages: usize) -> Completion<M, S> {
    // 行宽 2 只是桩：端点与页池都不读行里的数，读的是页宽。
    let pager =
        engine::kv::Pager::new(engine::kv::Cache::new(pages, 4).expect("页池"), 2).expect("行宽");
    Completion::new(route(), Box::new(tokenizer::Bytes), model, sampling, pager)
}

fn tokens(payload: &[u8]) -> Vec<u32> {
    payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

#[test]
fn a_request_runs_to_its_limit_and_returns_its_tokens() {
    let endpoint = endpoint(Fixed(vec![0.0, 5.0, 3.0]), Greedy, 8);
    let response = endpoint
        .call(b"id=1\nprompt=hi\nmax_new=3")
        .expect("一次请求");
    assert_eq!(response.status(), Status::Ok);
    assert_eq!(tokens(response.envelope().payload()), [1, 1, 1]);
    assert_eq!(endpoint.in_use_pages(), Some(0), "跑完必须归还页");
}

/// 采到停止符时序列提前结束：页照样归还，载荷只含已经采到的 token。
#[test]
fn a_stop_token_ends_the_generation_early() {
    let endpoint = endpoint(Fixed(vec![0.0, 1.0]), StopAfter { remaining: 2 }, 8);
    let response = endpoint
        .call(b"id=2\nprompt=hi\nmax_new=8")
        .expect("一次请求");
    assert_eq!(response.status(), Status::Ok);
    assert_eq!(tokens(response.envelope().payload()), [7, 7]);
    assert_eq!(endpoint.in_use_pages(), Some(0));
}

#[test]
fn a_malformed_request_never_reaches_the_scheduler() {
    let endpoint = endpoint(Fixed(vec![1.0]), Greedy, 8);
    let response = endpoint
        .call(b"id=1\nprompt=hi\nmax_new=2\nbogus=1")
        .expect("请求侧错误落成响应");
    assert_eq!(response.status(), Status::BadRequest);
    assert_eq!(endpoint.in_use_pages(), Some(0), "解析失败不该占页");
}

/// 引擎侧的"参数不可能"也要报成 400：`max_new = 0` 是客户端能改的错。
#[test]
fn a_request_the_engine_cannot_serve_is_a_bad_request() {
    let endpoint = endpoint(Fixed(vec![1.0]), Greedy, 8);
    let response = endpoint
        .call(b"id=1\nprompt=hi\nmax_new=0")
        .expect("请求侧错误落成响应");
    assert_eq!(response.status(), Status::BadRequest);
    assert!(!response.envelope().payload().is_empty());
}

/// 最坏长度超过整个页池：重试多少次都一样，属于 400 而不是"稍后重试"。
#[test]
fn a_request_that_can_never_fit_is_refused_with_400() {
    let endpoint = endpoint(Fixed(vec![1.0]), Greedy, 1);
    let response = endpoint
        .call(b"id=1\nprompt=hi\nmax_new=4")
        .expect("请求侧错误落成响应");
    assert_eq!(response.status(), Status::BadRequest);
    assert_eq!(endpoint.in_use_pages(), Some(0));
}

/// 模型自己失败是服务端故障：必须上抛成 `Err`，同时页仍然归还。
#[test]
fn a_model_failure_is_a_server_error_that_still_returns_pages() {
    let endpoint = endpoint(Broken, Greedy, 8);
    let error = endpoint
        .call(b"id=1\nprompt=hi\nmax_new=3")
        .expect_err("模型失败不能伪装成 4xx");
    assert_eq!(error.code(), diagnostic::Code::Internal);
    assert_eq!(endpoint.in_use_pages(), Some(0), "失败路径也必须归还页");
}

/// 模型每一步都看得到"到目前为止的生成历史"：历史只存在序列里，端点不再维护平行列表。
#[test]
fn the_model_sees_the_generated_history() {
    use std::cell::RefCell;
    use std::rc::Rc;

    struct Watching(Rc<RefCell<Vec<Vec<u32>>>>);

    impl Predict for Watching {
        fn logits(&mut self, sequence: &Sequence) -> diagnostic::Result<Vec<f32>> {
            self.0.borrow_mut().push(sequence.tokens().to_vec());
            Ok(vec![0.0, 1.0])
        }
    }

    let seen = Rc::new(RefCell::new(Vec::new()));
    let endpoint = endpoint(Watching(Rc::clone(&seen)), Greedy, 8);
    let response = endpoint
        .call(b"id=1\nprompt=hi\nmax_new=3")
        .expect("一次请求");
    assert_eq!(response.status(), Status::Ok);
    assert_eq!(
        *seen.borrow(),
        [Vec::new(), vec![1], vec![1, 1]],
        "第一步看到空历史，之后每一步看到上一步的输出"
    );
}

/// 同一端点连续两次调用不能互相干扰：页账与序列都在每次调用结束时回到空。
#[test]
fn two_requests_in_a_row_leave_no_residue() {
    let endpoint = endpoint(Fixed(vec![0.0, 1.0]), Greedy, 8);
    for id in [1, 2] {
        let body = format!("id={id}\nprompt=hi\nmax_new=2");
        let response = endpoint.call(body.as_bytes()).expect("一次请求");
        assert_eq!(response.status(), Status::Ok);
        assert_eq!(tokens(response.envelope().payload()), [1, 1]);
    }
    assert_eq!(endpoint.in_use_pages(), Some(0));
}
