//! 宿主侧：分区即网格，所有权即并发安全（DESIGN §4.4）。
//!
//! P0 只落地"分区比推导 + 整除校验"这一半：`&mut` 分片保证同一片不会被两个 kernel 写，
//! 网格由分片数推出，调用点不写 `grid`。

use core::fmt;

/// 网格：由分片比推出，不是人写的。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// 一维网格大小。
    pub x: u32,
}

impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "grid.x={}", self.x)
    }
}

/// 一维张量：P0 只需要能表达"分区 + 独占"的最小形态。
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    len: usize,
    data: Vec<T>,
}

impl<T> Tensor<T> {
    /// 由数据构造。
    pub fn new(data: Vec<T>) -> Self {
        Self {
            len: data.len(),
            data,
        }
    }

    /// 元素个数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 只读视图。
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// 可变视图。
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// 按 `chunk` 切分：分片数即 `grid.x`；不整除直接报错（DESIGN §4.4）。
    pub fn partition(&mut self, chunk: usize) -> Result<Partitioned<'_, T>, String> {
        if chunk == 0 {
            return Err("分片大小不能为 0".to_string());
        }
        if !self.len.is_multiple_of(chunk) {
            return Err(format!(
                "{} 个元素无法按 {chunk} 整除；网格必须由分片比推导，不能取整",
                self.len
            ));
        }
        let parts = self.len / chunk;
        Ok(Partitioned {
            tensor: self,
            grid: Grid { x: parts as u32 },
            chunk,
        })
    }
}

/// 拿到独占分片的张量视图；`&mut` 保证同一片不会被两个 kernel 同时写。
#[derive(Debug)]
pub struct Partitioned<'a, T> {
    tensor: &'a mut Tensor<T>,
    grid: Grid,
    chunk: usize,
}

impl<'a, T> Partitioned<'a, T> {
    /// 推导出的网格。
    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// 每个分片的元素数。
    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// 分片数。
    pub fn parts(&self) -> usize {
        self.grid.x as usize
    }

    /// 第 `index` 个分片（独占）。
    pub fn part_mut(&mut self, index: usize) -> Option<&mut [T]> {
        if index >= self.parts() {
            return None;
        }
        let start = index * self.chunk;
        self.tensor.data.get_mut(start..start + self.chunk)
    }

    /// 消费视图，取回底层张量。
    pub fn into_inner(self) -> &'a mut Tensor<T> {
        self.tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_derives_the_grid() {
        let mut tensor = Tensor::new((0..4096).collect::<Vec<i32>>());
        let view = tensor.partition(128).expect("可整除");
        assert_eq!(view.grid(), Grid { x: 32 });
        assert_eq!(view.parts(), 32);
        assert_eq!(view.chunk(), 128);
    }

    #[test]
    fn non_divisible_shape_is_rejected() {
        let mut tensor = Tensor::new(vec![0u8; 100]);
        let error = tensor.partition(32).unwrap_err();
        assert!(error.contains("无法按 32 整除"), "{error}");
    }

    #[test]
    fn zero_chunk_is_rejected() {
        let mut tensor = Tensor::new(vec![0u8; 8]);
        assert!(tensor.partition(0).is_err());
    }

    #[test]
    fn parts_are_disjoint_slices() {
        let mut tensor = Tensor::new((0..8).collect::<Vec<i32>>());
        let mut view = tensor.partition(4).expect("可整除");
        view.part_mut(0).expect("第一片")[0] = 42;
        view.part_mut(1).expect("第二片")[0] = 43;
        assert!(view.part_mut(2).is_none(), "越界分片必须是 None");
        let tensor = view.into_inner();
        assert_eq!(tensor.data(), &[42, 1, 2, 3, 43, 5, 6, 7]);
    }
}
