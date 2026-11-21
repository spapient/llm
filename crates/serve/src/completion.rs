//! 补全端点：[`Completion`]。

use crate::generate::{Ending, Generation, Predict, Sampling};
use crate::{Request, Response, Route, Status};
use engine::exec::Stream;
use engine::ifb::{Candidate, Phase, Verdict};
use engine::{Batch, Progress, Sequence};
use std::sync::{Mutex, MutexGuard};

/// 一块预填充算多少个提示位置：一次图提交的粒度，也是前缀复用的对齐粒度。
const PREFILL_CHUNK: usize = 64;

/// 一条补全端点：请求进、token 出。
pub struct Completion<M, S> {
    route: Route,
    tokenizer: Box<dyn tokenizer::Tokenizer>,
    state: Mutex<State<M, S>>,
}

mod state;

use state::State;

impl<M, S> Completion<M, S>
where
    M: Predict,
    S: Sampling,
{
    /// 由路由、分词器、模型、采样策略与 KV 页池构造端点。
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
                budget: engine::ifb::Budget::unlimited(),
                prefill: true,
            }),
        }
    }

    /// 给端点装上一次迭代的两条预算（`B_max` / `T_max`）。
    #[must_use]
    pub fn with_budget(mut self, budget: engine::ifb::Budget) -> Self {
        // 此刻还没有别的持有者，`get_mut` 既不用加锁也不会中毒。
        if let Ok(state) = self.state.get_mut() {
            state.budget = budget;
        }
        self
    }

    /// 页账（只读）：服务侧报告"占了多少页"时用。
    pub fn in_use_pages(&self) -> Option<usize> {
        self.state
            .lock()
            .ok()
            .map(|state| state.batch.scheduler().cache().in_use())
    }

    /// 前缀树里可复用的块数（只读诊断）：报告"共享缓存有多深"时用。
    pub fn cached_blocks(&self) -> Option<usize> {
        self.state
            .lock()
            .ok()
            .map(|state| state.batch.scheduler().cached_blocks())
    }

    /// 解析请求并编词；两步都是"客户端能改的错"。
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
    fn admit(
        &self,
        state: &mut State<M, S>,
        request: &Request,
        prompt: &[u32],
    ) -> diagnostic::Result<Response> {
        // `B_max` 撞上与"页不够"同口径：现在满、可重试，不是请求不成立。
        //
        // 上限取 `min(B_max, T_max)`：解码一轮每条序列占一行，于是 `T_max < B_max` 时后者才是
        // 真正的容量。只判 `B_max` 会放进"进得来、却永远排不进一轮"的序列（`pack` 报 `OverTokens`）。
        if state.batch.len() >= state.budget.seqs().min(state.budget.tokens()) {
            return Ok(Response::error(Status::Unavailable, crate::admit::FULL));
        }
        let sequence = self.sequence(state, request, prompt)?;
        crate::admit_batch(&mut state.batch, sequence)
    }

    /// 推进准入序列一轮；返回本轮跑完的 `(id, 生成结果)`。
    pub fn advance_batch(&self) -> diagnostic::Result<Vec<(u64, Generation)>> {
        let mut state = self.lock()?;
        state.step_batch()
    }

    /// 跑完一条已经进批的序列：生成 + 归还页。
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
        self.serve(&mut state, &request, &prompt)
    }

    /// 准入通过才往下走：被拒的请求把响应原样带回，它不是错误。
    fn serve(
        &self,
        state: &mut State<M, S>,
        request: &Request,
        prompt: &[u32],
    ) -> diagnostic::Result<Response> {
        let admitted = self.admit(state, request, prompt)?;
        if admitted.status() != Status::Ok {
            return Ok(admitted);
        }
        let id = request.id();
        self.prefill_request(state, id)?;
        self.complete(state, id)
    }

    /// 预填充失败与解码失败同口径：先把页还回去再上抛，否则这条请求的页永远占着。
    fn prefill_request(&self, state: &mut State<M, S>, id: u64) -> diagnostic::Result<()> {
        let Err(mut error) = state.prefill(id, usize::MAX) else {
            return Ok(());
        };
        if let Err(cleanup) = state.batch.finish(id) {
            error = error.also(&cleanup);
        }
        Err(error)
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

#[cfg(test)]
#[path = "../tests/completion_tests.rs"]
mod tests;
