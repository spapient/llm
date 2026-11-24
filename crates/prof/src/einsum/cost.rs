//! 一次计算的标量代价，roofline 的分子（cutile README §12.1）。

/// 一次计算的标量代价，roofline 的分子（cutile README §12.1）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// 浮点运算次数。
    pub flops: f64,
    /// 最少读入字节：算子入口的输入。
    pub read: u64,
    /// 最少写出字节：算子出口的输出。
    pub write: u64,
}

impl Cost {
    /// 零代价：空的表达式或纯标量计算。
    pub const ZERO: Self = Self {
        flops: 0.0,
        read: 0,
        write: 0,
    };

    /// 由三个分量构造（便于测试与适配旧的调用点）。
    pub const fn new(flops: f64, read: u64, write: u64) -> Self {
        Self { flops, read, write }
    }

    /// 最少流量（字节）：读与写合计。
    pub fn bytes(&self) -> u64 {
        self.read.saturating_add(self.write)
    }

    /// 两笔代价相加：复合算子的总代价。
    pub fn plus(self, other: Self) -> Self {
        Self {
            flops: self.flops + other.flops,
            read: self.read.saturating_add(other.read),
            write: self.write.saturating_add(other.write),
        }
    }
}
