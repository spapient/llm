//! 集合操作的优先级排队：待提交操作按 [`Priority`] 序数稳定排序后依次下发。
//!
//! 队列只决定**同一批次内**的提交顺序，不改变任何归约数值，也不引入线程池；入队顺序与
//! 调度顺序解耦，高优先级（梯度同步等阻塞主步的操作）先于后台预取类操作被排空。

use crate::policy::Priority;

/// 一条待提交的集合操作：调度优先级 + 调用方携带的泛型载荷（组号、缓冲标记等）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job<T> {
    priority: Priority,
    payload: T,
}

impl<T> Job<T> {
    /// 绑定一条操作与其调度优先级。
    pub fn new(priority: Priority, payload: T) -> Self {
        Self { priority, payload }
    }

    /// 调度优先级。
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// 解出载荷。
    pub fn into_payload(self) -> T {
        self.payload
    }
}

/// 按优先级排队的提交队列。
///
/// 入队不排序；[`Self::drain`] 时按优先级序数从小到大稳定排序——同一优先级保持入队先后，
/// 不打乱同档操作的提交次序。
pub struct Queue<T> {
    jobs: Vec<Job<T>>,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Queue<T> {
    /// 空队列。
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// 入队一条操作；容量随入队增长，不预先按组数分配。
    pub fn submit(&mut self, priority: Priority, payload: T) {
        self.jobs.push(Job::new(priority, payload));
    }

    /// 待提交条数。
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// 是否没有待提交操作。
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// 排空队列：按优先级序数升序、同档按入队先后返回全部作业。队列清空。
    pub fn drain(mut self) -> Vec<Job<T>> {
        self.jobs.sort_by_key(|job| job.priority);
        self.jobs
    }
}

#[cfg(test)]
#[path = "../tests/schedule_tests.rs"]
mod tests;
