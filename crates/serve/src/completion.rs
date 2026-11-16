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
//! 当前实现一次只处理一条序列（`Mutex` 串行化）：连续批需要的 [`engine::Batch`] 已经就位，
//! 但真正的并发入口属于传输层（P1，尚未落地）。端点因此先保证"一条请求能端到端跑完"这件事
//! 是对的，再谈吞吐。

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
    fn parse(&self, body: &[u8]) -> diagnostic::Result<(Request, Vec<tokenizer::Token>)> {
        let request = Request::parse(body)?;
        let prompt = self.tokenizer.encode(request.prompt())?;
        Ok((request, prompt))
    }

    /// 构造序列：页宽取自页池，因此这里不需要第二个"每页多少 token"的来源。
    fn sequence(
        &self,
        state: &State<M, S>,
        request: &Request,
        prompt_len: usize,
    ) -> diagnostic::Result<Sequence> {
        Sequence::new(
            request.id(),
            prompt_len,
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
        prompt_len: usize,
    ) -> diagnostic::Result<Response> {
        let sequence = self.sequence(state, request, prompt_len)?;
        crate::admit_batch(&mut state.batch, sequence)
    }

    /// 跑完一条已经进批的序列：生成 + 归还页。
    ///
    /// `finish` 在生成之后无条件执行（先取结局、再取归还结果）：模型或采样报错时页也必须
    /// 回来，否则一次失败的请求会永久占住页池——失败路径上的泄漏只有在服务跑久了才暴露。
    fn complete(&self, state: &mut State<M, S>, id: u64) -> diagnostic::Result<Response> {
        let generated = generate(state, id);
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
        let admitted = self.admit(&mut state, &request, prompt.len())?;
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

/// 反复走一步，直到结束、被挡或出错。
fn generate<M, S>(state: &mut State<M, S>, id: u64) -> diagnostic::Result<Option<Generation>>
where
    M: Predict,
    S: Sampling,
{
    loop {
        let step = step(state, id)?;
        if step.blocks() {
            return Ok(None);
        }
        if let Some(ending) = step.ending() {
            return done(state, id, ending);
        }
    }
}

/// 结束在这一步：把历史与结局装成一次生成的结果。
fn done<M, S>(
    state: &State<M, S>,
    id: u64,
    ending: Ending,
) -> diagnostic::Result<Option<Generation>> {
    Ok(Some(Generation::new(history(state, id)?, ending)))
}

/// 已经生成的历史：序列是唯一存放处，端点不另存一份。
fn history<M, S>(state: &State<M, S>, id: u64) -> diagnostic::Result<Vec<u32>> {
    Ok(state
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
fn step<M, S>(state: &mut State<M, S>, id: u64) -> diagnostic::Result<Step>
where
    M: Predict,
    S: Sampling,
{
    let Some(logits) = logits(state, id)? else {
        return Ok(Step::Limit);
    };
    sample(state, id, logits)
}

/// 算这一步的 logits；序列已经跑完时返回 `None`（那是"到上限了"，不是"没页"）。
///
/// 单独一支是因为借用必须分开：前向要读序列（借 `batch`），又要改模型（借 `model`），
/// 直接从 `&mut State` 上同时拿这两样会让借用检查器拒绝——两个字段互不重叠，函数内部拆开就成立。
fn logits<M, S>(state: &mut State<M, S>, id: u64) -> diagnostic::Result<Option<Vec<f32>>>
where
    M: Predict,
{
    let State { model, batch, .. } = state;
    let Some(sequence) = batch.sequence(id) else {
        return Err(missing(id));
    };
    if sequence.is_done() {
        return Ok(None);
    }
    model.logits(sequence).map(Some)
}

/// 从一步 logits 里取下一个 token，并把它记进序列与页账；`None` 是停止符。
fn sample<M, S>(state: &mut State<M, S>, id: u64, logits: Vec<f32>) -> diagnostic::Result<Step>
where
    M: Predict,
    S: Sampling,
{
    let Some(token) = state.sampling.sample(&logits)? else {
        return stop(state, id);
    };
    // 记账与推进是同一个调用：`Blocked` 时 token 不会留在历史里。
    Ok(progress(state.batch.emit(id, token)?))
}

/// 采到停止符：让序列提前结束（页仍由 `finish` 归还）。
fn stop<M, S>(state: &mut State<M, S>, id: u64) -> diagnostic::Result<Step> {
    state
        .batch
        .sequence_mut(id)
        .ok_or_else(|| missing(id))?
        .stop()?;
    Ok(Step::Stop)
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
#[path = "../tests/completion_tests.rs"]
mod tests;
