//! 准入的服务侧投影：把序列交进调度器或批，并把结局翻成 [`Response`]。

use diagnostic::{Code, Error, Result};

use crate::endpoint::{Response, Status};

/// 最坏长度超过整个页池时的说明：两个准入入口共用同一句话。
const NEVER_FITS: &str = "这条请求的最坏长度超过整个 KV 页池：调小 max_new 或加大页池";

/// 页都借出去时的说明：可重试，所以是 `503`。
pub(crate) const FULL: &str = "KV 页不足，请稍后重试";

/// 准入的服务侧结局：已进批（带序列号）或现在页满。
enum AdmissionOutcome {
    /// 已进批：之后用序列号找它。
    Admitted(u64),
    /// 页不够：序列原样退回，可重试。
    Full,
}

/// 准入端口：把序列交进调度器或批，并回答"进去了 / 现在满"。
trait Admitter {
    /// 最坏情况是否装得下：装不下是终局 400，不是可重试的 503。
    fn fits(&self, sequence: &engine::Sequence) -> bool;

    /// 准入；`Ok(Some(id))` 已进，`Ok(None)` 现在满；序列自身不成立走 `Err`。
    fn admit(&mut self, sequence: engine::Sequence) -> Result<AdmissionOutcome>;
}

impl Admitter for engine::Scheduler {
    fn fits(&self, sequence: &engine::Sequence) -> bool {
        engine::Scheduler::fits(self, sequence)
    }

    fn admit(&mut self, sequence: engine::Sequence) -> Result<AdmissionOutcome> {
        // 调度器回的是「已算好的前缀 token 数」：`None` 才是装不下。服务层不需要这个数——
        // 页与位置都由批层自己记账，这里只关心进没进。
        Ok(match engine::Scheduler::admit(self, &sequence)? {
            Some(_) => AdmissionOutcome::Admitted(sequence.id()),
            None => AdmissionOutcome::Full,
        })
    }
}

impl Admitter for engine::Batch {
    fn fits(&self, sequence: &engine::Sequence) -> bool {
        engine::Batch::fits(self, sequence)
    }

    fn admit(&mut self, sequence: engine::Sequence) -> Result<AdmissionOutcome> {
        let id = sequence.id();
        match engine::Batch::admit(self, sequence)? {
            engine::Admission::Admitted => Ok(AdmissionOutcome::Admitted(id)),
            engine::Admission::Full(_) => Ok(AdmissionOutcome::Full),
        }
    }
}

/// 把一次准入的结局翻译成响应。
fn admit_response<A: Admitter>(admitter: &mut A, sequence: engine::Sequence) -> Result<Response> {
    if !admitter.fits(&sequence) {
        return Ok(Response::error(Status::BadRequest, NEVER_FITS));
    }
    match admitter.admit(sequence) {
        Ok(AdmissionOutcome::Admitted(id)) => Ok(Response::ok(id.to_le_bytes().to_vec())),
        Ok(AdmissionOutcome::Full) => Ok(Response::error(Status::Unavailable, FULL)),
        Err(error) => refused(error),
    }
}

/// 尝试把一个序列准入调度器，并把结果翻译成响应。
///
/// # Errors
///
/// 序列自身不成立且错误类别不是 `invalid` 时上抛；`invalid` 一律落成 `400`。
pub fn admit(scheduler: &mut engine::Scheduler, sequence: &engine::Sequence) -> Result<Response> {
    admit_response(scheduler, sequence.clone())
}

/// 把一个序列交进 [`engine::Batch`]：与 [`admit`] 同一套判定，而批会同时持有序列与页账。
///
/// 端点走这条入口：它需要"页进去了"与"序列在批里"同时成立，否则调用方还得自己维护
/// 第二份列表——那正是两份状态开始漂移的地方。
///
/// # Errors
///
/// 序列自身不成立（重名、已结束、页宽与调度器不一致）且错误类别不是 `invalid` 时上抛；
/// `invalid` 一律落成 `400`。
pub fn admit_batch(batch: &mut engine::Batch, sequence: engine::Sequence) -> Result<Response> {
    admit_response(batch, sequence)
}

/// 请求侧错误一律落成 400（`invalid` 全是"这次请求自己不成立"，改什么由客户端决定）；
/// 其余类别（`internal` / `io`）才是本侧的 bug，继续上抛给传输层。
pub(crate) fn refused(error: Error) -> Result<Response> {
    if error.code() != Code::Invalid {
        return Err(error);
    }
    Ok(Response::error(
        Status::BadRequest,
        &format!("请求不合法：{error}"),
    ))
}
