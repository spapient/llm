//! 补全端点：[`Completion`]。
//!
//! 它把四件事按顺序接起来：解析请求（[`crate::Request`]）、编词（`tokenizer`）、准入与推进
//! （[`engine::Batch`]）、采样（[`Sampling`]）。每一步的错误都已归位：
//!
//! - 请求不成立、词表外文本、`max_new` 为零、编号重复 → `400`；
//! - 页不够 → `503`（现在满）或 `400`（永远装不下），判定在 [`crate::admit`]；
//! - 模型或采样本身失败 → `Err`（服务端故障，不该伪装成客户端错误）。
//!
//! 一条序列跑完一定 `finish`：成功、被挡或出错都归还页，因此"页泄漏"不会只在某一条分支上
//! 发生——那正是这类端点最常见的残留问题。
//!
//! 单条请求端点仍一次答一条；批步进在 [`State::step_batch`] 落地：一轮整批前向
//! （`Predict::logits_batch` 一次提交），再逐行采样 → 记账，结束的归还页摘出，被挡的原样保留。
//! 真正的并发传输入口属于传输层（M4，尚未落地）。

use crate::generate::{Ending, Generation, Predict, Sampling};
use crate::{Request, Response, Route, Status};
use engine::exec::Stream;
use engine::{Batch, Progress, Sequence};
use std::sync::{Mutex, MutexGuard};

/// 一条补全端点：请求进、token 出。
///
/// 三个协作者都是注入的：分词器（词表边界）、[`Predict`]（一步前向）、[`Sampling`]（选哪个
/// token）。因此"模型还没有前向实现"并不阻止服务链路端到端跑通——端点本身不假设模型存在。
pub struct Completion<M, S> {
    route: Route,
    tokenizer: Box<dyn tokenizer::Tokenizer>,
    state: Mutex<State<M, S>>,
}

/// 端点内部的三个有状态件：它们必须一起被锁住，否则"先编词再准入"这一段会与别的请求交错。
struct State<M, S> {
    model: M,
    sampling: S,
    batch: Batch,
}

/// 一步生成的结局。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// 生成了一个 token，还能继续。
    Continue,
    /// 这一步没有页：序列原样不动。
    Blocked,
    /// 采到了停止符。
    Stop,
    /// 跑满了 `max_new`。
    Limit,
}

impl Step {
    /// 是否该结束这次生成（`Blocked` 也算结束，但原因不是"完成"）。
    fn ending(self) -> Option<Ending> {
        match self {
            Self::Continue | Self::Blocked => None,
            Self::Stop => Some(Ending::Stop),
            Self::Limit => Some(Ending::Limit),
        }
    }

    /// 是否是"这一步没页"。
    fn blocks(self) -> bool {
        matches!(self, Self::Blocked)
    }
}

impl<M, S> Completion<M, S>
where
    M: Predict,
    S: Sampling,
{
    /// 由路由、分词器、模型、采样策略与 KV 页池构造端点。
    ///
    /// 页池带行宽（`2 · kv_heads · head_dim` 这类数）：那个数只有装载模型的一侧知道，
    /// 服务层不替它猜，也不在端点里引入第二份"一行多宽"。
    pub fn new(
        route: Route,
        tokenizer: Box<dyn tokenizer::Tokenizer>,
        model: M,
        sampling: S,
        pager: engine::kv::Pager,
    ) -> Self {
        Self {
            route,
            tokenizer,
            state: Mutex::new(State {
                model,
                sampling,
                batch: Batch::new(pager),
            }),
        }
    }

    /// 页账（只读）：服务侧报告"占了多少页"时用。
    ///
    /// 锁中毒时返回 `None`——那只会在一次调用 panic 之后出现，此时任何账目都不可信。
    pub fn in_use_pages(&self) -> Option<usize> {
        self.state
            .lock()
            .ok()
            .map(|state| state.batch.scheduler().cache().in_use())
    }

    /// 解析请求并编词；两步都是"客户端能改的错"。
    ///
    /// 编词结果就地取编号：端口交出的是 [`tokenizer::Token`] 整块，这里一趟收集成 `u32`，
    /// 不在端点里再留一份 token 中间向量。
    fn parse(&self, body: &[u8]) -> diagnostic::Result<(Request, Vec<u32>)> {
        let request = Request::parse(body)?;
        let tokens = self.tokenizer.encode(request.prompt())?;
        let prompt = tokens.iter().map(tokenizer::Token::id).collect();
        Ok((request, prompt))
    }

    /// 构造序列：提示段 token 整段留存进序列；页宽取自页池，因此这里不需要第二个
    /// "每页多少 token"的来源。
    fn sequence(
        &self,
        state: &State<M, S>,
        request: &Request,
        prompt: &[u32],
    ) -> diagnostic::Result<Sequence> {
        Sequence::new(
            request.id(),
            prompt,
            request.max_new(),
            state.batch.scheduler().page_size(),
            Stream::new(0),
        )
    }

    /// 准入并把结局翻译成响应；`Status::Ok` 表示真的进去了。
    ///
    /// 走 [`crate::admit_batch`] 而不是直接调页账：序列与页表必须一起进批，否则生成时
    /// 按 id 找不到它，或者页进批了但序列没有。
    fn admit(
        &self,
        state: &mut State<M, S>,
        request: &Request,
        prompt: &[u32],
    ) -> diagnostic::Result<Response> {
        let sequence = self.sequence(state, request, prompt)?;
        crate::admit_batch(&mut state.batch, sequence)
    }

    /// 推进批内所有准入序列一轮；返回本轮跑完的 `(id, 生成结果)`。
    ///
    /// 一轮整批前向（`logits_batch` 一次提交）取整块 logits，再逐条采样记账；被挡的序列留在
    /// 批里、历史不变，下一轮原样重试；并发传输入口留 M4。
    pub fn advance_batch(&self) -> diagnostic::Result<Vec<(u64, Generation)>> {
        let mut state = self.lock()?;
        state.step_batch()
    }

    /// 跑完一条已经进批的序列：生成 + 归还页。
    ///
    /// `finish` 在生成之后无条件执行（先取结局、再取归还结果）：模型或采样报错时页也必须
    /// 回来，否则一次失败的请求会永久占住页池——失败路径上的泄漏只有在服务跑久了才暴露。
    fn complete(&self, state: &mut State<M, S>, id: u64) -> diagnostic::Result<Response> {
        let generated = state.generate(id);
        let finished = state.batch.finish(id);
        let generated = generated?;
        finished?;
        Ok(match generated {
            Some(generation) => Response::ok(generation.frame()),
            None => Response::error(
                Status::Unavailable,
                "生成到一半页用尽：请重试，或调小 max_new / 加大页池",
            ),
        })
    }

    /// 锁住内部状态；中毒的锁只可能来自上一次调用里的 panic。
    fn lock(&self) -> diagnostic::Result<MutexGuard<'_, State<M, S>>> {
        self.state.lock().map_err(|_| {
            diagnostic::Error::new(
                diagnostic::Code::Internal,
                "端点状态锁已中毒：上一次调用 panic 了",
            )
            .with("serve::Completion::lock")
        })
    }

    /// 一次请求的全部步骤；客户端错误在这里都以 `Err(invalid)` 形式出现，由 `call` 翻译成 400。
    fn answer(&self, body: &[u8]) -> diagnostic::Result<Response> {
        let (request, prompt) = self.parse(body)?;
        let mut state = self.lock()?;
        let admitted = self.admit(&mut state, &request, &prompt)?;
        if admitted.status() != Status::Ok {
            return Ok(admitted);
        }
        self.complete(&mut state, request.id())
    }
}

impl<M, S> crate::Endpoint for Completion<M, S>
where
    M: Predict,
    S: Sampling,
{
    fn route(&self) -> &Route {
        &self.route
    }

    fn call(&self, body: &[u8]) -> diagnostic::Result<Response> {
        match self.answer(body) {
            Ok(response) => Ok(response),
            Err(error) => crate::refused(error),
        }
    }
}

impl<M, S> State<M, S>
where
    M: Predict,
    S: Sampling,
{
    /// 反复走一步，直到结束、被挡或出错。
    fn generate(&mut self, id: u64) -> diagnostic::Result<Option<Generation>> {
        loop {
            let step = self.step(id)?;
            if step.blocks() {
                return Ok(None);
            }
            if let Some(ending) = step.ending() {
                return self.done(id, ending);
            }
        }
    }

    /// 结束在这一步：把历史与结局装成一次生成的结果。
    fn done(&self, id: u64, ending: Ending) -> diagnostic::Result<Option<Generation>> {
        Ok(Some(Generation::new(self.history(id)?, ending)))
    }

    /// 已经生成的历史：序列是唯一存放处，端点不另存一份。
    fn history(&self, id: u64) -> diagnostic::Result<Vec<u32>> {
        Ok(self
            .batch
            .sequence(id)
            .ok_or_else(|| missing(id))?
            .tokens()
            .to_vec())
    }

    /// 一步：算 logits、采样、推进页账。
    ///
    /// 采样在 `advance` 之前：先确认"这一步该是哪个 token"，再由页账为它占位。反过来（先占页
    /// 再采样）在采样报错时会留下一个已经推进的序列。
    fn step(&mut self, id: u64) -> diagnostic::Result<Step> {
        let Some(logits) = self.logits(id)? else {
            return Ok(Step::Limit);
        };
        self.sample(id, logits)
    }

    /// 算这一步的 logits；序列已经跑完时返回 `None`（那是"到上限了"，不是"没页"）。
    ///
    /// 借用必须分开：前向要把序列交给模型，又要持 `&mut Batch` 读写页账。序列是从 `batch`
    /// 借出的，继续持有那个不可变借用会和可变借用冲突——这里先克隆一份只读的序列事实交出，
    /// 不可变借用随即释放，再以 `&mut Batch` 进入模型。两个字段互不重叠，函数内部拆开就成立。
    fn logits(&mut self, id: u64) -> diagnostic::Result<Option<Vec<f32>>> {
        let State { model, batch, .. } = self;
        let Some(sequence) = batch.sequence(id).cloned() else {
            return Err(missing(id));
        };
        if sequence.is_done() {
            return Ok(None);
        }
        model.logits(&sequence, batch).map(Some)
    }

    /// 从一步 logits 里取下一个 token，并把它记进序列与页账；`None` 是停止符。
    fn sample(&mut self, id: u64, logits: Vec<f32>) -> diagnostic::Result<Step> {
        let Some(token) = self.sampling.sample(&logits)? else {
            return self.stop(id);
        };
        // 记账与推进是同一个调用：`Blocked` 时 token 不会留在历史里。
        Ok(progress(self.batch.emit(id, token)?))
    }

    /// 采到停止符：让序列提前结束（页仍由 `finish` 归还）。
    fn stop(&mut self, id: u64) -> diagnostic::Result<Step> {
        self.batch
            .sequence_mut(id)
            .ok_or_else(|| missing(id))?
            .stop()?;
        Ok(Step::Stop)
    }

    /// 结束一条序列：装出生成结果、归还页并摘出批。
    fn finish_one(&mut self, id: u64, ending: Ending) -> diagnostic::Result<Generation> {
        let generation = self.done(id, ending)?;
        self.batch.finish(id)?;
        generation.ok_or_else(|| missing(id))
    }

    /// 批内一轮：一次整批前向取整块 logits，再逐条采样记账；结束的摘出批归还页，被挡原样保留。
    ///
    /// 前向失败发生在任何记账之前：归还批内全部序列页再上抛。采样失败只归还报错那条，
    /// 此前已采样记账的条目不回滚。序列快照在进入可变借用前一次性克隆取走，不持有 `batch`
    /// 的不可变借用。
    fn step_batch(&mut self) -> diagnostic::Result<Vec<(u64, Generation)>> {
        let sequences = self.batch.sequences().to_vec();
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let rows = match self.model.logits_batch(&sequences, &mut self.batch) {
            Ok(rows) => rows,
            Err(mut error) => {
                self.release_all(&sequences, &mut error);
                return Err(error);
            }
        };
        let mut finished = Vec::new();
        for (sequence, logits) in sequences.iter().zip(rows) {
            self.sample_or_release(sequence.id(), logits, &mut finished)?;
        }
        Ok(finished)
    }

    /// 一条采样结果入账；采样失败先归还坏序列页再上抛，已记账条目不回滚。
    fn sample_or_release(
        &mut self,
        id: u64,
        logits: Vec<f32>,
        finished: &mut Vec<(u64, Generation)>,
    ) -> diagnostic::Result<()> {
        match self.sample(id, logits) {
            Ok(step) => self.account(id, step, finished),
            Err(mut error) => {
                // 归还页优先于报错上抛；归还本身失败时作为同级诊断附注保留，
                // 不能用它覆盖原始故障——否则客户端看不到模型/采样那一类服务端错误。
                if let Err(cleanup) = self.batch.finish(id) {
                    error = error.also(&cleanup);
                }
                Err(error)
            }
        }
    }

    /// 一条采样结局入账：被挡留在批里；收尾的装出生成结果、归还页摘出批。
    fn account(
        &mut self,
        id: u64,
        step: Step,
        finished: &mut Vec<(u64, Generation)>,
    ) -> diagnostic::Result<()> {
        if step.blocks() {
            return Ok(());
        }
        if let Some(ending) = step.ending() {
            finished.push((id, self.finish_one(id, ending)?));
        }
        Ok(())
    }

    /// 整批前向失败：批内序列都还没记账，全部归还页；归还失败作为同级诊断附注保留。
    fn release_all(&mut self, sequences: &[Sequence], error: &mut diagnostic::Error) {
        for sequence in sequences {
            if let Err(cleanup) = self.batch.finish(sequence.id()) {
                // `also` 按值取走主错误；先换出清理错误再挂附注，不就地移动借用背后的值。
                let base = std::mem::replace(error, cleanup);
                *error = base.also(error);
            }
        }
    }
}

/// 调度器的结局映射成一步的结局：`Blocked` 与 `Done` 必须分开，否则"没页"会被当成"跑完了"。
fn progress(progress: Progress) -> Step {
    match progress {
        Progress::Blocked => Step::Blocked,
        Progress::Done => Step::Limit,
        Progress::Running => Step::Continue,
    }
}

fn missing(id: u64) -> diagnostic::Error {
    diagnostic::Error::new(
        diagnostic::Code::Internal,
        format!("序列 {id} 在生成过程中离开了批：页账与端点状态不一致"),
    )
    .with("serve::Completion")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Endpoint, Greedy};

    /// 固定 logits 的模型桩：端点只关心"下一步算什么"，不关心怎么算出来。
    struct Fixed(Vec<f32>);

    impl Predict for Fixed {
        fn logits(
            &mut self,
            _sequence: &Sequence,
            _batch: &mut Batch,
        ) -> diagnostic::Result<Vec<f32>> {
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
        fn logits(
            &mut self,
            _sequence: &Sequence,
            _batch: &mut Batch,
        ) -> diagnostic::Result<Vec<f32>> {
            Err(diagnostic::Error::new(diagnostic::Code::Internal, "权重缺失").with("test::Broken"))
        }
    }

    fn route() -> Route {
        Route::parse("POST /v1/completions").expect("路由")
    }

    fn endpoint<M: Predict, S: Sampling>(model: M, sampling: S, pages: usize) -> Completion<M, S> {
        // 行宽 2 只是桩：端点与页池都不读行里的数，读的是页宽。
        let pager = engine::kv::Pager::new(engine::kv::Cache::new(pages, 4).expect("页池"), 2)
            .expect("行宽");
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
        fn logits(
            &mut self,
            sequence: &Sequence,
            _batch: &mut Batch,
        ) -> diagnostic::Result<Vec<f32>> {
            if sequence.id() == self.fail {
                return Err(
                    diagnostic::Error::new(diagnostic::Code::Internal, "权重缺失")
                        .with("test::Split"),
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
        fn logits(
            &mut self,
            sequence: &Sequence,
            batch: &mut Batch,
        ) -> diagnostic::Result<Vec<f32>> {
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
            assert!(
                state.batch.is_empty(),
                "前向失败时两条都还没记账，页全部归还、序列摘出"
            );
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
}
