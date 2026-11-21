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

/// 一块提示的推进结果。
///
/// 三种结局必须分开：`Done` 才能把已算位置推进到块尾；`Deferred` 是页紧（下一轮会好），
/// `Unsupported` 是模型没这条路径（已经退化，不再重试）。合成一个 `bool` 会把"等一等"说成"没有"。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    /// 这块算完并已登记。
    Done,
    /// 页不够，这一轮先不算。
    Deferred,
    /// 模型没有分块预填充，整段已记为退化。
    Unsupported,
}

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
    /// 一次迭代的可行域（`B_max` / `T_max`）：它和批一起被锁，准入与步进读到的是同一份数。
    budget: engine::ifb::Budget,
    /// 模型是否会分块预填充。首次遇到 `Unsupported` 就置 `false`：此后不再逐轮重试，
    /// 也不把提示段登记成可复用前缀——零填充的页不是"算好的提示"，不能冒充缓存命中。
    prefill: bool,
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

    /// 把一条已准入序列的提示段按块算进页账：这一轮最多推进 `quota` 行。
    ///
    /// 每块一次前向、每块之后才登记块：登记按"已算位置"而不是"写到哪一行"推进，因为写是提前
    /// 一格的（第 `at` 步写第 `at + 1` 行）。顺序反过来会把还没算的块当成可复用的前缀。
    ///
    /// `quota` 是本轮打包给这条序列的行数（`engine::ifb::pack` 的拍板结果），它只**收紧**块大小、
    /// 不放大：[`State::next_end`] 用同一份预算算式给块尾，两处取小的那个，于是"这一轮算多少行"
    /// 只有一个来源。
    ///
    /// 模型没有分块预填充（[`Predict::prefill`] 的默认实现报 `Unsupported`）时把整段记为已算：
    /// 端点退化回"提示段不进页"的旧行为，而不是拒绝服务——桩模型与只跑解码的通路照旧能跑。
    fn prefill(&mut self, id: u64, quota: usize) -> diagnostic::Result<()> {
        if !self.prefill {
            return Ok(());
        }
        // 能力先问、页后要：不支持的模型不该先花掉一笔页预算才发现自己没有这条路径。
        if !self.model.prefills() {
            self.prefill = false;
            return Ok(());
        }
        let sequence = self
            .batch
            .sequence(id)
            .cloned()
            .ok_or_else(|| missing(id))?;
        self.advance_prompt(sequence, id, quota)
    }

    /// 推进提示段，最多推进 `quota` 行；已算完或这一轮没有整页预算就是一个字都不写。
    ///
    /// 循环留在这一层而不是提到一批：单选路径要一次把整段算完（`quota` 不定上限），批内路径
    /// 则只给它分到的那几行——同一个入口、同一个块算式，差别只在预算。
    fn advance_prompt(
        &mut self,
        sequence: Sequence,
        id: u64,
        quota: usize,
    ) -> diagnostic::Result<()> {
        let mut at = sequence.prefilled();
        let mut left = quota;
        while let Some((from, end)) = self.next_block(&sequence, at, left) {
            // 页紧与模型不支持都不该判这条请求非法：一个下一轮会好，一个已经退化。
            match self.prefill_block(&sequence, id, from, end)? {
                Block::Done => {
                    left -= end - from;
                    at = end;
                }
                Block::Deferred | Block::Unsupported => return Ok(()),
            }
        }
        Ok(())
    }

    /// 这一轮从 `at` 起要算的那一块：配额耗尽或已算完就是 `None`。
    ///
    /// 「块尾」与「配额」在这里只取一次小的：[`State::next_end`] 给预算内的块尾，`left` 是本轮
    /// 打包分给这条序列的行数，两者相等时循环自然收尾，不需要额外的计数判断。
    fn next_block(&self, sequence: &Sequence, at: usize, left: usize) -> Option<(usize, usize)> {
        let end = self
            .next_end(at, sequence.prompt_len())?
            .min(at.saturating_add(left));
        (end > at).then_some((at, end))
    }

    /// 下一块的块尾：`None` 是已算完，或这一轮没有可推进的整页预算。
    ///
    /// 块大小不在这里 `min` 一次就完事：[`engine::ifb::chunk`] 是唯一的算式，调用方只管用
    /// 它给的数——对齐口径只能有一处，否则总有一处会把「整页」漏成「随便切」。
    fn next_end(&self, at: usize, prompt_len: usize) -> Option<usize> {
        let chunk = engine::ifb::chunk(
            prompt_len - at,
            self.batch.scheduler().page_size(),
            self.budget.tokens().min(PREFILL_CHUNK),
        );
        (chunk > 0).then_some(at + chunk)
    }

    /// 算一块提示位置并登记它：一块只做一次前向、一次登记。
    fn prefill_block(
        &mut self,
        sequence: &Sequence,
        id: u64,
        at: usize,
        end: usize,
    ) -> diagnostic::Result<Block> {
        // 一块要写到行 `end`（1-based），先按块尾把页要到：预填充一次推进多行，不走调度器
        // 逐步预留那条路，缺这一步就会写到页表还没盖住的行上。
        if !self.batch.reserve(id, end + 1)? {
            // 页不够：这一轮先不算，块位置一个都不推进。已算的前缀仍然有效，下一轮接着算。
            return Ok(Block::Deferred);
        }
        let block = self.step_span(sequence, at, end)?;
        self.settle_block(id, end, block)
    }

    /// 逐位算一块：任一位报不支持就停在原地，返回 `Unsupported` 而不是半块。
    fn step_span(
        &mut self,
        sequence: &Sequence,
        at: usize,
        end: usize,
    ) -> diagnostic::Result<Block> {
        for position in at..end {
            if !self.step_prompt(sequence, position)? {
                return Ok(Block::Unsupported);
            }
        }
        Ok(Block::Done)
    }

    /// 一块算完之后登记它：只有算完整块才登记，页紧与退化都不留半块账。
    fn settle_block(&mut self, id: u64, end: usize, block: Block) -> diagnostic::Result<Block> {
        if block == Block::Done {
            self.batch.mark_prefilled(id, end)?;
        }
        Ok(block)
    }

    /// 把一个提示位置算进页账：`false` 表示模型没有分块预填充，已把整段记为退化。
    fn step_prompt(&mut self, sequence: &Sequence, position: usize) -> diagnostic::Result<bool> {
        match self.model.prefill(sequence, &mut self.batch, position) {
            Ok(_) => Ok(true),
            Err(error) if error.code() == diagnostic::Code::Unsupported => {
                // 退化：提示段不进页，也不登记（零页不是算好的块）。端点照旧能解码。
                self.prefill = false;
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }

    /// 批内一轮：按 [`Batch::round`] 的打包结果推进，再把生成行一次整批前向 + 逐行记账。
    ///
    /// 一轮只碰打包结果里的行：上下文候选各算自己分到的那一块，生成行一次提交。被挤掉的上下文
    /// 候选这一轮一行不算、也不解码（历史不完整时解码读到的是零填充的"过去"），下一轮重新排队。
    /// 前向失败发生在任何记账之前：归还本轮全部序列页再上抛。采样失败只归还报错那条，此前已
    /// 采样记账的条目不回滚。序列快照在进入可变借用前一次性克隆取走，不持有 `batch` 的不可变借用。
    fn step_batch(&mut self) -> diagnostic::Result<Vec<(u64, Generation)>> {
        let plan = self.round()?;
        self.prefill_context(&plan)?;
        let sequences = self.ready(&plan)?;
        self.decode_rows(&sequences)
    }

    /// 生成行一次整批前向再逐行记账：一条都没有就是这个也没什么可提交的。
    fn decode_rows(
        &mut self,
        sequences: &[Sequence],
    ) -> diagnostic::Result<Vec<(u64, Generation)>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.batch_rows(sequences)?;
        self.account_rows(sequences, rows)
    }

    /// 先把拍板给上下文候选的那几块算掉：一行也没分到时它什么都不做。
    ///
    /// 上下文与生成分两步而不是一次过滤：提示段没算完的序列不进解码，因此两者不是一个集合。
    fn prefill_context(&mut self, plan: &[Candidate]) -> diagnostic::Result<()> {
        plan.iter()
            .filter(|candidate| candidate.phase() == Phase::Context)
            .try_for_each(|candidate| self.prefill(candidate.id(), candidate.rows()))
    }

    /// 本轮的行与序：`Packed` 就是拍板结果，`Idle` 是空转。
    ///
    /// 预算先按本层一次提交的粒度收紧（[`PREFILL_CHUNK`]）：打包说这轮给 128 行，本层也只会
    /// 一次提交 64 行，两边算出的块尾就会打架。收紧后两边算式同源，配额即块尾。
    ///
    /// `OverSeqs`/`OverTokens` 走到这里是内部矛盾而不是"这一轮很挤"：准入按 `min(B_max, T_max)`
    /// 挡人，`pack` 只在"生成行自己就超过 `T_max`"时报这两条。静默空转会把一种配置错误伪装成
    /// "没有请求"，因此上抛。
    fn round(&self) -> diagnostic::Result<Vec<Candidate>> {
        let budget =
            engine::ifb::Budget::new(self.budget.seqs(), self.budget.tokens().min(PREFILL_CHUNK));
        match self.batch.round(budget) {
            Verdict::Packed { rows, .. } => Ok(rows),
            Verdict::Idle => Ok(Vec::new()),
            Verdict::OverSeqs { seqs, max } => {
                Err(misfit(format!("批内 {seqs} 条序列超过 B_max = {max}")))
            }
            Verdict::OverTokens { tokens, max } => {
                Err(misfit(format!("生成行 {tokens} 行超过 T_max = {max}")))
            }
        }
    }

    /// 本轮要解码的序列：生成候选；预填充退化时上下文候选也算（没有块要算，剩下的只有解码）。
    fn ready(&self, plan: &[Candidate]) -> diagnostic::Result<Vec<Sequence>> {
        plan.iter()
            .filter(|candidate| !self.prefill || candidate.phase() == Phase::Generation)
            .map(|candidate| {
                self.batch
                    .sequence(candidate.id())
                    .cloned()
                    .ok_or_else(|| missing(candidate.id()))
            })
            .collect()
    }

    /// 一次整批前向取回整块 logits；失败时把批内全部序列页归还再上抛。
    fn batch_rows(&mut self, sequences: &[Sequence]) -> diagnostic::Result<Vec<Vec<f32>>> {
        match self.model.logits_batch(sequences, &mut self.batch) {
            Ok(rows) => Ok(rows),
            Err(mut error) => {
                self.release_all(sequences, &mut error);
                Err(error)
            }
        }
    }

    /// 逐条采样入账：结束的装出生成结果并摘出批，被挡的原样留在批里。
    fn account_rows(
        &mut self,
        sequences: &[Sequence],
        rows: Vec<Vec<f32>>,
    ) -> diagnostic::Result<Vec<(u64, Generation)>> {
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

/// 打包结论与准入判据打架：不是"这一轮很挤"，而是预算与准入用了两套上限。
fn misfit(why: String) -> diagnostic::Error {
    diagnostic::Error::new(
        diagnostic::Code::Internal,
        format!("IFB 打包拒绝本轮（{why}）：准入按 min(B_max, T_max) 挡人，走到这里说明两者不一致"),
    )
    .with("serve::Completion")
}

#[cfg(test)]
#[path = "../tests/completion_tests.rs"]
mod tests;
