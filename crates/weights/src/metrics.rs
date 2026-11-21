//! 容器度量：条目数、参数总数、体积对比与无损口径；全部在打开容器后现算。

use diagnostic::Result;

use crate::stored::{Stored, lossless_entry};

/// E8 体积门：`ratio_param ≤ 1/24`（LLD §E8「24× 无损」）。
///
/// 门槛写成常量而不是配置项：它是验收定义本身，可调的门就不叫门。改这个值等于改验收，
/// 必须同时改 [`docs/lld.md`](../../../docs/lld.md) 的 §E8 与历史账。
pub const E8_RATIO: f64 = 1.0 / 24.0;

/// 度量累加器：逐段把 numel、FP16 参考体积、无损判定与 REF 命中/节省字节收拢到一处。
pub(crate) struct MetricsAcc {
    params: u64,
    v_fp16: u64,
    ref_hits: usize,
    ref_saved_bytes: u64,
    lossless: bool,
}

impl Default for MetricsAcc {
    /// 空容器按无损计：无条目即无有损段；逐段再用「与」收口，任一段有损即整体有损。
    fn default() -> Self {
        Self {
            params: 0,
            v_fp16: 0,
            ref_hits: 0,
            ref_saved_bytes: 0,
            lossless: true,
        }
    }
}

impl MetricsAcc {
    /// 累加单段；引用段计入命中数与逻辑字节节省。
    pub(crate) fn add(&mut self, seg: &Stored) -> Result<()> {
        let numel = tensor::product(seg.shape());
        self.params = self.params.saturating_add(numel);
        self.v_fp16 = self.v_fp16.saturating_add(numel.saturating_mul(2));
        self.lossless = self.lossless && lossless_entry(seg);
        if seg.is_ref() {
            self.ref_hits += 1;
            let logical = u64::try_from(seg.logical_len()?).unwrap_or(u64::MAX);
            self.ref_saved_bytes = self.ref_saved_bytes.saturating_add(logical);
        }
        Ok(())
    }

    /// 收口为 [`Metrics`]：条目数与容器体积由调用方提供，体积比在此处现算。
    pub(crate) fn finish(self, entries: usize, v_artifact: usize) -> Metrics {
        let ratio = if self.v_fp16 == 0 {
            0.0
        } else {
            v_artifact as f64 / self.v_fp16 as f64
        };
        Metrics {
            entries,
            params: self.params,
            v_artifact,
            v_fp16: self.v_fp16,
            ratio,
            lossless: self.lossless,
            ref_hits: self.ref_hits,
            ref_saved_bytes: self.ref_saved_bytes,
        }
    }
}

/// 容器度量：条目数、参数总数、体积对比与无损口径；全部在打开容器后现算。
///
/// 不读文件内保存的度量字段，也不替量化条目下"无损"结论：带量化方案的条目按既有
/// [`crate::quant::Quant`] 语义判为有损，未识别的 codec 同样不计入无损。
#[derive(Debug, Clone)]
pub struct Metrics {
    pub(crate) entries: usize,
    pub(crate) params: u64,
    pub(crate) v_artifact: usize,
    pub(crate) v_fp16: u64,
    pub(crate) ratio: f64,
    pub(crate) lossless: bool,
    pub(crate) ref_hits: usize,
    pub(crate) ref_saved_bytes: u64,
}

impl Metrics {
    /// 条目总数（段表行数，含引用段）。
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// 参数总数：各段 `numel` 乘积之和，溢出饱和。
    pub fn params(&self) -> u64 {
        self.params
    }

    /// 容器文件字节数（含头部、对齐填充与两级 CRC）。
    pub fn v_artifact(&self) -> usize {
        self.v_artifact
    }

    /// 全量 FP16 参考体积：Σ `numel × 2`，溢出饱和。
    pub fn v_fp16(&self) -> u64 {
        self.v_fp16
    }

    /// 体积比：`V_artifact / V_FP16`；`V_FP16` 为 0 时记 0。
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// 参数级体积比 `V_artifact / (params × 2)`，即 LLD §E8 的 `ratio_param`。
    ///
    /// 与 [`Self::ratio`] 的分母在今天的实现里数值相同（`v_fp16` 就是 `Σ numel × 2`），
    /// 但口径不同：`v_fp16` 是"参考体积"，会随参考格式演进而变；`params × 2` 只由张量
    /// 元素数决定，跨格式可复算。E8 的验收走这一条——否则参考体积口径一改，历史结论
    /// 就会莫名其妙地失效。`params` 为 0 时记 0（无参容器没有可比的参数级体积）。
    pub fn ratio_param(&self) -> f64 {
        let denom = self.params.saturating_mul(2);
        if denom == 0 {
            return 0.0;
        }
        self.v_artifact as f64 / denom as f64
    }

    /// E8 裁决：24× 体积门与逐位无损**同时**成立才算通过（LLD §E8「必须且不可降级」）。
    ///
    /// 空容器（`params == 0`）一律判否：`ratio_param` 在无分母时记 0，若不加这一条，
    /// 一份 0 字节的容器就会"通过"24× 门——那是假阳性，不是证据。
    ///
    /// 这里判的是**账**：`lossless` 来自条目声明的 codec，`ratio_param` 来自落盘字节。
    /// 真正的"解码后逐位相等"由解码头（`Archive::load`）负责兑现，裁决方必须先跑完
    /// 整表解码再引用本函数，不能只凭本函数的返回值对外宣称"已验证"。
    pub fn e8_pass(&self) -> bool {
        self.params > 0 && self.lossless && self.ratio_param() <= E8_RATIO
    }

    /// 无损标志：所有条目都不带量化方案，且 codec 属于无损的 `Store`/`Rans`/`Axes`。
    pub fn lossless(&self) -> bool {
        self.lossless
    }

    /// REF 命中数：被去重写成引用的段数。
    pub fn ref_hits(&self) -> usize {
        self.ref_hits
    }

    /// REF 节省字节数：引用段逻辑字节之和（未重复落盘的部分），溢出饱和。
    pub fn ref_saved_bytes(&self) -> u64 {
        self.ref_saved_bytes
    }
}
