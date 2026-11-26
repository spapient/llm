use super::*;

testkit::test_cases! {
/// 三条打包/逐元素配置自洽：每字节码数 = 8 / 位宽；打包表长度等于码域；零点落在码域内。
fn rows_are_self_consistent() {
    for row in [I2, I4, I8] {
        assert_eq!(row.cpb, 8 / row.bits.bits() as usize, "cpb 必须由位宽推出");
        assert!(
            row.zero >= 0 && row.zero < row.domain() as i32,
            "零点必须在码域内"
        );
        if row.packed() {
            assert_eq!(
                row.table.len(),
                row.domain() as usize,
                "打包表必须覆盖整个码域"
            );
            assert_eq!(row.offsets.len(), row.cpb, "样本偏移与每字节码槽同长");
        }
    }
}

/// 对称表逐位回退：`table[code] = code − zero`，与骨架解包口径一致。
fn packed_tables_are_symmetric() {
    for row in [I2, I4] {
        for (code, &value) in row.table.iter().enumerate() {
            assert_eq!(
                value,
                (code as i32 - row.zero) as f32,
                "code {code} 表值不符"
            );
        }
    }
}
}
