use super::*;
use crate::{Endpoint, Greedy};

/// 固定 logits 的模型桩：端点只关心"下一步算什么"，不关心怎么算出来。
struct Fixed(Vec<f32>);

impl Predict for Fixed {
    fn logits(&mut self, _sequence: &Sequence, _batch: &mut Batch) -> diagnostic::Result<Vec<f32>> {
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
    fn logits(&mut self, _sequence: &Sequence, _batch: &mut Batch) -> diagnostic::Result<Vec<f32>> {
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
        fn logits(
            &mut self,
            sequence: &Sequence,
            _batch: &mut Batch,
        ) -> diagnostic::Result<Vec<f32>> {
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

/// 一条按 id 失败的模型桩：命中 `fail` 上抛内部错误；批量前向在模型层整体失败，行内不归属。
struct Split {
    fail: u64,
}

impl Predict for Split {
    fn logits(&mut self, sequence: &Sequence, _batch: &mut Batch) -> diagnostic::Result<Vec<f32>> {
        if sequence.id() == self.fail {
            return Err(
                diagnostic::Error::new(diagnostic::Code::Internal, "权重缺失").with("test::Split"),
            );
        }
        Ok(vec![0.0, 1.0])
    }
}

/// 采样桩：第一次正常给 token，第二次报不可用——核对采样失败同样不污染已记账序列。
struct FlakySampler {
    calls: u32,
}

impl Sampling for FlakySampler {
    fn sample(&mut self, _logits: &[f32]) -> diagnostic::Result<Option<u32>> {
        self.calls += 1;
        if self.calls == 2 {
            return Err(
                diagnostic::Error::new(diagnostic::Code::Invalid, "logits 不可用")
                    .with("test::FlakySampler"),
            );
        }
        Ok(Some(1))
    }
}

/// 把当前步 KV 行按序列号写成标记：核对每条序列的页行归属不串行。
struct Recorder;

impl Predict for Recorder {
    fn logits(&mut self, sequence: &Sequence, batch: &mut Batch) -> diagnostic::Result<Vec<f32>> {
        let marker = sequence.id() as f32;
        batch.write_kv(sequence.id(), sequence.len(), &[marker, 0.0])?;
        Ok(vec![0.0, 1.0])
    }
}

fn admit_sequence(
    state: &mut State<impl Predict, impl Sampling>,
    id: u64,
    prompt_len: usize,
    max_new: usize,
) {
    let page_size = state.batch.scheduler().page_size();
    let prompt = vec![0u32; prompt_len];
    let sequence = Sequence::new(
        id,
        &prompt,
        max_new,
        page_size,
        engine::exec::Stream::new(0),
    )
    .expect("序列");
    state.batch.admit(sequence).expect("准入");
}

/// 两条序列同批准入、同批步进：各自记账，跑完一起摘出，页全部归还。
#[test]
fn two_admitted_sequences_step_together_and_keep_their_own_history() {
    let endpoint = endpoint(Fixed(vec![0.0, 5.0]), Greedy, 8);
    let mut finished = Vec::new();
    {
        let mut state = endpoint.lock().expect("锁");
        admit_sequence(&mut state, 1, 2, 2);
        admit_sequence(&mut state, 2, 2, 2);
        while !state.batch.is_empty() {
            finished.extend(state.step_batch().expect("一轮步进"));
        }
    }
    finished.sort_by_key(|(id, _)| *id);
    assert_eq!(finished.len(), 2);
    assert_eq!(finished[0].1.tokens(), [1, 1]);
    assert_eq!(finished[1].1.tokens(), [1, 1]);
    assert_eq!(endpoint.in_use_pages(), Some(0), "跑完必须归还页");
}

/// 页不足的序列留在批里、历史不变；其余序列照常推进，批不整体停摆。
#[test]
fn a_blocked_sequence_stays_in_the_batch_with_unchanged_history() {
    let endpoint = endpoint(Fixed(vec![0.0, 1.0]), Greedy, 2);
    let mut state = endpoint.lock().expect("锁");
    admit_sequence(&mut state, 1, 0, 8);
    admit_sequence(&mut state, 2, 4, 4);

    let round = state.step_batch().expect("一轮步进");
    assert!(round.is_empty(), "这一轮谁都没跑完");
    let running = state.batch.sequence(1).expect("在批里");
    let blocked = state.batch.sequence(2).expect("在批里");
    assert_eq!(running.produced(), 1, "有页的那条照常推进");
    assert_eq!(blocked.produced(), 0, "被挡的那条计数不动");
    assert!(blocked.tokens().is_empty(), "被挡不记 token");
    assert_eq!(state.batch.blocked(), [2]);
}

/// 整批前向在模型层失败：失败发生在任何记账之前，批内序列页全部归还、序列摘出，不留已记账
/// token；错误上抛成服务端故障。采样层的逐行隔离由 `a_failed_sampling_step...` 用例覆盖。
#[test]
fn a_batch_forward_fault_releases_all_pages_without_accounting_tokens() {
    let endpoint = endpoint(Split { fail: 1 }, Greedy, 8);
    {
        let mut state = endpoint.lock().expect("锁");
        admit_sequence(&mut state, 2, 1, 2);
        admit_sequence(&mut state, 1, 1, 1);

        let error = state.step_batch().expect_err("坏序列上抛服务端错误");
        assert_eq!(error.code(), diagnostic::Code::Internal);
        assert!(state.batch.is_empty(), "前向失败时两条都还没记账，页全部归还、序列摘出");
    }
    // 锁先放掉再查页账：`in_use_pages` 也要取同一把互斥锁，不能重入。
    assert_eq!(endpoint.in_use_pages(), Some(0), "失败路径也必须归还页");
}

/// 采样失败与模型失败同一条隔离路径：先记账的序列不回滚，坏序列归还页摘出。
#[test]
fn a_failed_sampling_step_does_not_pollute_the_accounted_other() {
    let endpoint = endpoint(Fixed(vec![0.0, 1.0]), FlakySampler { calls: 0 }, 8);
    let mut state = endpoint.lock().expect("锁");
    admit_sequence(&mut state, 2, 1, 2);
    admit_sequence(&mut state, 1, 1, 1);

    let error = state.step_batch().expect_err("采样坏序列上抛");
    assert_eq!(error.code(), diagnostic::Code::Invalid);

    let good = state.batch.sequence(2).expect("好序列仍在批里");
    assert_eq!(good.tokens(), [1], "已记账的 token 不回滚");
    assert!(state.batch.sequence(1).is_none(), "采样失败序列已摘出批");
}

/// 当前步 KV 行按序列号写进各自页行：两条同批互不覆盖。
#[test]
fn kv_rows_are_attributed_to_their_own_sequence() {
    let endpoint = endpoint(Recorder, Greedy, 8);
    let mut state = endpoint.lock().expect("锁");
    admit_sequence(&mut state, 3, 1, 2);
    admit_sequence(&mut state, 4, 1, 2);

    state.step_batch().expect("一轮步进");
    for id in [3u64, 4] {
        let row = state.batch.kv(id, 1).expect("读到本序列的页行");
        assert_eq!(row[0], id as f32, "页行标记归属序列号 {id}");
        assert_eq!(row[1], 0.0);
    }
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
