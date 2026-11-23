//! 运行时 lane 缓冲与视图：`f32` 与按位打包 `u8` 两条存储形态的公共定义。
//!
//! engine 的执行缝缓冲与 train 的训练槽缓冲共用同一份枚举，不再各写一份 `F32/U8` 副本。
//! 上层只经这些视图取切片；**形状不在这里解释**——几何归 [`tensor::Shape`](llm_tensor::Shape)
//! 与算子实例几何，执行缝按编号绑定缓冲（ADR 0044）。

/// 一条存储 lane 的运行时标签。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// 32 位浮点。
    F32,
    /// 按位打包的整型（`I8` / `U4` 等），字节序与内核一致。
    U8,
}

impl Lane {
    /// 名字：只用于错误说明。
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U8 => "u8",
        }
    }
}

/// 一段自持缓冲：32 位浮点，或按位打包的字节。
#[derive(Debug, Clone, PartialEq)]
pub enum LaneBuf {
    /// 32 位浮点。
    F32(Vec<f32>),
    /// 按位打包的整型：i8 逐元素一字节，i4 每字节两个 nibble；字节序与内核一致。
    U8(Vec<u8>),
}

impl LaneBuf {
    /// 这条缓冲的存储形态。
    pub fn lane(&self) -> Lane {
        match self {
            Self::F32(_) => Lane::F32,
            Self::U8(_) => Lane::U8,
        }
    }

    /// 当成 `f32` 读；不是浮点缓冲时为 `None`。
    pub fn f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }

    /// 当成打包字节读；不是字节缓冲时为 `None`。
    pub fn u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(values) => Some(values),
            Self::F32(_) => None,
        }
    }

    /// 元素个数：f32 元素数或打包字节数。
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U8(values) => values.len(),
        }
    }

    /// 是否没有元素。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 字节数：切片在编译期就知道元素宽度，热路径上没有分支。
    pub fn bytes(&self) -> u64 {
        match self {
            Self::F32(values) => size_of_val(values.as_slice()) as u64,
            Self::U8(values) => size_of_val(values.as_slice()) as u64,
        }
    }

    /// 只读视图。
    pub fn view(&self) -> LaneView<'_> {
        match self {
            Self::F32(values) => LaneView::F32(values),
            Self::U8(values) => LaneView::U8(values),
        }
    }

    /// 可写视图。
    pub fn view_mut(&mut self) -> LaneViewMut<'_> {
        match self {
            Self::F32(values) => LaneViewMut::F32(values),
            Self::U8(values) => LaneViewMut::U8(values),
        }
    }
}

/// 只读视图：一次执行/前向步的输入操作数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaneView<'a> {
    /// 32 位浮点。
    F32(&'a [f32]),
    /// 按位打包的整型。
    U8(&'a [u8]),
}

impl<'a> LaneView<'a> {
    /// 这一格是哪个存储形态。
    pub fn lane(&self) -> Lane {
        match self {
            Self::F32(_) => Lane::F32,
            Self::U8(_) => Lane::U8,
        }
    }

    /// 当成 `f32` 读；不是浮点缓冲时为 `None`。
    pub fn f32(&self) -> Option<&'a [f32]> {
        match self {
            Self::F32(values) => Some(*values),
            Self::U8(_) => None,
        }
    }

    /// 当成按位打包的整型读；不是字节缓冲时为 `None`。
    pub fn u8(&self) -> Option<&'a [u8]> {
        match self {
            Self::U8(values) => Some(*values),
            Self::F32(_) => None,
        }
    }
}

/// 可写视图：一次执行/前向步的输出操作数。
#[derive(Debug, PartialEq)]
pub enum LaneViewMut<'a> {
    /// 32 位浮点。
    F32(&'a mut [f32]),
    /// 按位打包的整型。
    U8(&'a mut [u8]),
}

impl<'a> LaneViewMut<'a> {
    /// 这一格是哪个存储形态。
    pub fn lane(&self) -> Lane {
        match self {
            Self::F32(_) => Lane::F32,
            Self::U8(_) => Lane::U8,
        }
    }

    /// 当成 `f32` 写；不是浮点缓冲时为 `None`。
    pub fn f32_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }

    /// 当成按位打包的整型写；不是字节缓冲时为 `None`。
    pub fn u8_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::U8(values) => Some(values),
            Self::F32(_) => None,
        }
    }
}
