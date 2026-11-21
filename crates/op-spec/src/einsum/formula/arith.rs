use super::*;

/// 一项的元素数：该项每条轴长度的乘积。
pub(super) fn term_elements(extents: &Extents, term: &[char]) -> Result<u128, Error> {
    term.iter().try_fold(1u128, |total, label| {
        multiply(total, extent(extents, *label)?)
    })
}

/// 按方向累加：两个计数器各自溢出都要报。
pub(super) fn accumulate(total: &mut u128, bytes: u128) -> Result<(), Error> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}

/// 流量收口到 `u64`：`Cost` 的字段是 `u64`，装不下说明形状不可能是真的。
pub(super) fn bytes(value: u128, kind: &str) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| {
        Error::new(Code::Invalid, format!("{kind}流量 {value} 字节超出 u64"))
            .with("einsum::Formula::cost")
    })
}

pub(super) fn extent(extents: &Extents, label: char) -> Result<usize, Error> {
    extents.get(label).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("轴 `{label}` 没有长度：表达式里的每个字母都必须给出轴长度"),
        )
        .with("einsum::Formula::axes")
    })
}

pub(super) fn multiply(total: u128, length: usize) -> Result<u128, Error> {
    total.checked_mul(length as u128).ok_or_else(overflow)
}

pub(super) fn overflow() -> Error {
    Error::new(Code::Invalid, "维度乘积或流量超出 u128：形状不可能是真的").with("einsum::Formula")
}
