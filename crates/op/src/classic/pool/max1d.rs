use super::super::*;

/// 声明档输出位置数。
pub const OL: usize = 8;
/// 声明档输入长度：`(OL − 1)·S + K = 16`。
pub const L: usize = 16;

/// 声明档算术：`<S: Scalar>` 泛型，四种精度共用一条取窗口最大的路径。
pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
    crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, out)
}

// 每输出读两个输入元素、写一个；读两遍字节、写一遍。
window_op!(
    MaxPool1dModel,
    MaxPool1d,
    "max_pool1d_f32",
    64,
    64,
    L,
    OL,
    2.0,
    2
);
