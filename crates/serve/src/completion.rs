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
//! 单条请求端点仍一次答一条；批步进在 [`State::step_batch`] 落地：批内多条准入序列一轮依次
//! 走 logits → 采样 → 记账，结束的归还页摘出，被挡的原样保留。模型一步仍是单序列几何，真正的
//! 并发传输入口属于传输层（M4，尚未落地）。

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
    /// 模型仍是单序列几何：一步一条序列，调度侧把多条排在同一批里依次走；内核级批处理形状
    /// 不在本刀范围。被挡的序列留在批里、历史不变，下一轮原样重试；并发传输入口留 M4。
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
            let id = sequence.id();
            match self.sample(id, logits) {
                Ok(step) => self.account(id, step, &mut finished)?,
                Err(mut error) => {
                    // 归还页优先于报错上抛；归还本身失败时作为同级诊断附注保留，
                    // 不能用它覆盖原始故障——否则客户端看不到模型/采样那一类服务端错误。
                    if let Err(cleanup) = self.batch.finish(id) {
                        error = error.also(&cleanup);
                    }
                    return Err(error);
                }
            }
        }
        Ok(finished)
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
#[path = "../tests/completion_tests.rs"]
mod tests;
