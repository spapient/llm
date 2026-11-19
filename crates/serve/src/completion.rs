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
//!
//! 准入与解码之间还隔着**分块预填充**：提示段的 K/V 必须逐位算进页账，否则解码读到的是零填充
//! 历史（长提示与短提示看到的"过去"一样多）。分块是"一次提交算 [`PREFILL_CHUNK`] 个位置"，
//! 块边界同时是前缀复用点——块算完就可以被下一个同前缀请求整块接走。

use crate::generate::{Ending, Generation, Predict, Sampling};
use crate::{Request, Response, Route, Status};
use engine::exec::Stream;
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
        if state.batch.len() >= state.budget.seqs() {
            return Ok(Response::error(Status::Unavailable, crate::FULL));
        }
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
        let Err(mut error) = state.prefill(id) else {
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

    /// 把一条已准入序列的提示段按块算进页账。
    ///
    /// 每块一次前向、每块之后才登记块：登记按"已算位置"而不是"写到哪一行"推进，因为写是提前
    /// 一格的（第 `at` 步写第 `at + 1` 行）。顺序反过来会把还没算的块当成可复用的前缀。
    ///
    /// 模型没有分块预填充（[`Predict::prefill`] 的默认实现报 `Unsupported`）时把整段记为已算：
    /// 端点退化回"提示段不进页"的旧行为，而不是拒绝服务——桩模型与只跑解码的通路照旧能跑。
    fn prefill(&mut self, id: u64) -> diagnostic::Result<()> {
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
        self.advance_prompt(sequence, id)
    }

    /// 逐块推进提示段，直到整个提示段落定、或这一轮推不动（页不够）。
    fn advance_prompt(&mut self, sequence: Sequence, id: u64) -> diagnostic::Result<()> {
        let prompt_len = sequence.prompt_len();
        let mut at = sequence.prefilled();
        while let Some(end) = self.next_end(at, prompt_len) {
            match self.prefill_block(&sequence, id, at, end)? {
                Block::Done => at = end,
                // 页紧与模型不支持都不该判这条请求非法：一个下一轮会好，一个已经退化。
                Block::Deferred | Block::Unsupported => return Ok(()),
            }
        }
        Ok(())
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
        self.prefill_pending(&sequences)?;
        let rows = self.batch_rows(&sequences)?;
        self.account_rows(&sequences, rows)
    }

    /// 先把批内还没算完的提示段推一轮：块算完才能在下一步读它当历史。
    fn prefill_pending(&mut self, sequences: &[Sequence]) -> diagnostic::Result<()> {
        for sequence in sequences {
            if self.prefill && !sequence.prefilled_ready() {
                self.prefill(sequence.id())?;
            }
        }
        Ok(())
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

    /// 批满时新请求回 503 而不是 400：这是"现在挤不下"，等一条结束就会好。
    ///
    /// 与"页不够"同口径。把满批判成请求非法，会让客户端在容量压力下永久放弃本可成功的请求。
    #[test]
    fn a_full_batch_turns_new_requests_away_until_one_finishes() {
        let endpoint = endpoint(Fixed(vec![1.0]), Greedy, 8).with_budget(engine::ifb::Budget::new(1, 32));
        // 占住唯一的位置：手工准入一条，让批恒满（`call` 自己会跑完并归还，留不下驻留序列）。
        let sequence = Sequence::new(9, &[1, 2], 2, 4, Stream::new(0)).expect("序列");
        endpoint
            .state
            .lock()
            .expect("锁")
            .batch
            .admit(sequence)
            .expect("准入");
        let response = endpoint
            .call(b"id=1\nprompt=hi\nmax_new=2")
            .expect("满批落成响应");
        assert_eq!(response.status(), Status::Unavailable);
        assert_eq!(
            endpoint.state.lock().expect("锁").batch.len(),
            1,
            "被挡的请求不该进批"
        );
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

    /// 记录预填充位置的模型桩：每个位置写一行 KV，回固定 logits。
    struct Prefilling {
        seen: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
    }

    impl Predict for Prefilling {
        fn logits(
            &mut self,
            _sequence: &Sequence,
            _batch: &mut Batch,
        ) -> diagnostic::Result<Vec<f32>> {
            Ok(vec![0.0, 1.0])
        }

        fn prefills(&self) -> bool {
            true
        }

        fn prefill(
            &mut self,
            sequence: &Sequence,
            batch: &mut Batch,
            at: usize,
        ) -> diagnostic::Result<Vec<f32>> {
            self.seen.borrow_mut().push(at);
            // 页行账是 1-based：行 0 是空槽，第 `at` 个 token 落在行 `at + 1`。
            batch.write_kv(sequence.id(), at + 1, &[0.0, 0.0])?;
            Ok(vec![0.0, 1.0])
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

    /// 分块预填充把提示段算进页账，满块随之登记成跨请求可复用前缀；第二个同提示请求接走那一块，
    /// 提示段一个位置都不重算，生成的 token 与首跑逐位相同。
    ///
    /// 这条用例守的是「页账与位置账是同一份」：只共享页不改已算位置，第二个请求会把同一段提示
    /// 重算一遍，而它写的正是它刚打算去读的页。
    #[test]
    fn a_second_request_with_the_same_prompt_skips_the_prefilled_prefix() {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let endpoint = endpoint(
            Prefilling {
                seen: std::rc::Rc::clone(&seen),
            },
            Greedy,
            16,
        );
        let first = endpoint
            .call(b"id=1\nprompt=abcd\nmax_new=2")
            .expect("首跑");
        assert_eq!(
            first.status(),
            Status::Ok,
            "首跑被拒：{}",
            String::from_utf8_lossy(first.envelope().payload())
        );
        assert_eq!(seen.borrow().as_slice(), [0, 1, 2, 3], "提示段逐位算进页账");
        assert_eq!(
            endpoint.in_use_pages(),
            Some(1),
            "只留前缀树持有的那一块，请求自己的页已归还"
        );

        seen.borrow_mut().clear();
        let second = endpoint
            .call(b"id=2\nprompt=abcd\nmax_new=2")
            .expect("复用");
        assert_eq!(second.status(), Status::Ok);
        assert!(
            seen.borrow().is_empty(),
            "整个提示段都在接走的块里，一个位置都不重算"
        );
        assert_eq!(
            second.envelope().payload(),
            first.envelope().payload(),
            "复用前缀不能改变结果"
        );
        assert_eq!(endpoint.in_use_pages(), Some(1), "没有多留页");
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
