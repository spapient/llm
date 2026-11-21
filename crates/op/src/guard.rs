//! 缓冲长度校验：`(实际, 需要, 名称)` 逐条比对，失败即带名称与两个数字返回。
//!
//! 归一化、循环与转置卷积三类层共用同一套报错口径，此前各自抄了一份 `check` / `check_all`。
//! 三处都只关心"够不够"，不关心多余的尾部（调用方可以带更大的缓冲），所以判据是 `>=` 而不是 `==`。

use diagnostic::{Code, Error, Result};

/// 一条长度约束：缓冲不足时报出名称、实际长度与下限。
fn check(actual: usize, required: usize, what: &str) -> Result<()> {
    if actual < required {
        return Err(Error::new(
            Code::Invalid,
            format!("{what}: 缓冲 {actual}，需要 {required}"),
        ));
    }
    Ok(())
}

/// 一组约束，逐条短路；把多处 `?` 收进一个出口，调用点只写一次。
pub(crate) fn check_all(bounds: &[(usize, usize, &str)]) -> Result<()> {
    for &(actual, required, what) in bounds {
        check(actual, required, what)?;
    }
    Ok(())
}
