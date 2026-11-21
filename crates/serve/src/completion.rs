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
//! 单条请求端点仍一次答一条；批步进在 [`State::step_batch`] 落地：一轮先问
//! [`Batch::round`] 这一轮**跑哪些行**（`engine::ifb::pack` 在预算下拍板），再按拍板结果推进——
//! 上下文候选各算自己分到的那一块，生成行一次整批前向（`Predict::logits_batch` 提交一次），
//! 最后逐行采样 → 记账，结束的归还页摘出。没排进本的序列原样留批、历史不动，下一轮重新排队。
//! 真正的并发传输入口属于传输层（M4，尚未落地）。
//!
//! 准入与解码之间还隔着**分块预填充**：提示段的 K/V 必须逐位算进页账，否则解码读到的是零填充
//! 历史（长提示与短提示看到的"过去"一样多）。分块是"一次提交算 [`PREFILL_CHUNK`] 个位置"，
//! 块边界同时是前缀复用点——块算完就可以被下一个同前缀请求整块接走。

use crate::generate::{Ending, Generation, Predict, Sampling};
use crate::{Request, Response, Route, Status};
use engine::exec::Stream;
use engine::ifb::{Candidate, Phase, Verdict};
use engine::{Batch, Progress, Sequence};
use std::sync::{Mutex, MutexGuard};

/// 一块预填充算多少个提示位置：一次图提交的粒度，也是前缀复用的对齐粒度。
///
/// 取 64 而不是"整段"：整段会让一个长提示在准入后的那一次提交里独占一次前向，短请求只能等；
/// 取 1 又会把"一个块一次提交"摊成逐位提交，块边界上登记的页引用也随之多一轮。64 与常见页宽
/// 同量级，块算完的那一页下一轮就能被别的请求接走。
const PREFILL_CHUNK: usize = 64;

/// 一条补全端点：请求进、token 出。
///
/// 三个协作者都是注入的：分词器（词表边界）、[`Predict`]（一步前向）、[`Sampling`]（选哪个
/// token）。因此"模型还没有前向实现"并不阻止服务链路端到端跑通——端点本身不假设模型存在。
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
                budget: engine::ifb::Budget::unlimited(),
                prefill: true,
            }),
        }
    }

    /// 给端点装上一次迭代的两条预算（`B_max` / `T_max`）。
    ///
    /// 容器式构造：默认不限，容量只在真的部署时才是一个确定的数，`new` 里猜一个默认值只会
    /// 让"为什么这里卡住"多一个隐藏来源。
    #[must_use]
    pub fn with_budget(mut self, budget: engine::ifb::Budget) -> Self {
        // 此刻还没有别的持有者，`get_mut` 既不用加锁也不会中毒。
        if let Ok(state) = self.state.get_mut() {
            state.budget = budget;
        }
        self
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

    /// 前缀树里可复用的块数（只读诊断）：报告"共享缓存有多深"时用。
    ///
    /// 锁中毒时返回 `None`，口径同 [`Completion::in_use_pages`]。
    pub fn cached_blocks(&self) -> Option<usize> {
        self.state
            .lock()
            .ok()
            .map(|state| state.batch.scheduler().cached_blocks())
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
        // `B_max` 撞上与"页不够"同口径：现在满、可重试，不是请求不成立。
        //
        // 上限取 `min(B_max, T_max)`：解码一轮每条序列占一行，于是 `T_max < B_max` 时后者才是
        // 真正的容量。只判 `B_max` 会放进"进得来、却永远排不进一轮"的序列（`pack` 报 `OverTokens`）。
        if state.batch.len() >= state.budget.seqs().min(state.budget.tokens()) {
            return Ok(Response::error(Status::Unavailable, crate::FULL));
        }
        let sequence = self.sequence(state, request, prompt)?;
        crate::admit_batch(&mut state.batch, sequence)
    }

    /// 推进准入序列一轮；返回本轮跑完的 `(id, 生成结果)`。
    ///
    /// 一轮只跑打包结果里的行（`State::round`）：上下文候选各算自己那一块，生成行一次整批
    /// 前向取整块 logits；没排进本的序列留在批里、历史不变，下一轮原样重试。并发传输入口留 M4。
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
    ///
    /// 单选路径没有批内预算可挤，于是把上限交给块算式本身（[`PREFILL_CHUNK`]）。
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
