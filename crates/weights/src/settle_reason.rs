//! 记账的 [`Reason`]：为什么选了这个 codec、有没有真的更小。
//!
//! 从 [`crate::settle`] 拆出：判定与产出的结论留在那边，这里只有一组把「实测字节」换算成
//! 「理由」的纯函数，便于逐条单测而不用造整份归档。

use crate::settle::{GAIN_FLOOR, MAX_ENTROPY, Reason, SMALL_BYTES};
use crate::{Codec, Stored};


/// 每元素实际占的位数：按打包后的字节数四舍五入，不读方案里的"名义位宽"。
pub(crate) fn bits_per(seg: &Stored, volume: usize) -> u32 {
    let numel = tensor::product(seg.shape());
    if numel == 0 {
        return 0;
    }
    // `+ numel / 2` 是四舍五入：1.5 bit 记 2，不把逐组打包的余量说成"没量化"。
    let bits = (volume as u64).saturating_mul(8).saturating_add(numel / 2) / numel;
    u32::try_from(bits).unwrap_or(u32::MAX)
}

/// 原因判定：先看往返，再按容器形态分开答，同一条只落进一档。
pub(crate) fn reason_of(
    seg: &Stored,
    raw_len: usize,
    volume: usize,
    entropy: f64,
    bitwise: bool,
) -> Reason {
    if !bitwise {
        return Reason::Mismatch;
    }
    if seg.is_ref() {
        return Reason::Deduplicated;
    }
    // 小段先答：编码的固定开销与熵无关，先判大小才不会把 8 字节的偏置说成"不可压"。
    if raw_len < SMALL_BYTES {
        return Reason::Small;
    }
    frame_reason(seg.codec(), raw_len, volume, entropy)
}

/// 按帧形态答：轴字典是被选中的无损形态，`store`/`rans` 才有"省没省"的问题。
pub(crate) fn frame_reason(codec: Codec, raw_len: usize, volume: usize, entropy: f64) -> Reason {
    match codec {
        Codec::Axes => Reason::Axes,
        Codec::Store => store_reason(entropy),
        Codec::Rans => rans_reason(raw_len, volume),
    }
}

/// `store` 帧：熵到顶说明内容真的没冗余，否则是"有冗余但没编"。
pub(crate) fn store_reason(entropy: f64) -> Reason {
    if entropy >= MAX_ENTROPY {
        Reason::Incompressible
    } else {
        Reason::Raw
    }
}

/// `rans` 帧：省下的比例够不够抵一次重编码。
pub(crate) fn rans_reason(raw_len: usize, volume: usize) -> Reason {
    if gain(raw_len, volume) < GAIN_FLOOR {
        Reason::BelowGain
    } else {
        Reason::Rans
    }
}

/// 省下的比例：`1 − volume / raw_len`；逻辑长度为 0 时视作没省。
pub(crate) fn gain(raw_len: usize, volume: usize) -> f64 {
    if raw_len == 0 {
        return 0.0;
    }
    1.0 - volume as f64 / raw_len as f64
}

/// 字节流的 Shannon 熵（bit/byte）：256 桶直方图，空流记 0。
pub(crate) fn entropy_of(bytes: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }
    let total = bytes.len() as f64;
    if total == 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

/// 名字里第一个整数分量：`backbone.layers.12.mlp.w` 给 12，`embed` 给 `None`。
pub(crate) fn layer_of(name: &str) -> Option<u32> {
    name.split('.').find_map(|part| part.parse().ok())
}
