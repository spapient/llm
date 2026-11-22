//! 逐点融合：把消费者的中间缓冲换成生产者的值表达式，两趟落进一趟循环。
//!
//! 融合只在这一层改写表达式并重新校验，不碰发射器：消掉中间缓冲后的核仍是一张普通 `Kernel`，
//! 因此常量折叠、长度守卫与 C 发射全部原样复用。归约链、缺失的链接、可写链接与 `extent` 不一致
//! 一律拒绝，而不是猜一个意思。

use crate::kernel::{
    Access, Expr, Kernel, Parameter, ScalarType, ValueType, VerificationError, Write,
};

impl Kernel {
    /// Fuse a pointwise consumer into this pointwise producer.
    ///
    /// 消费者把 `self.output` 声明成只读缓冲时，中间缓冲被消掉：消费者里每一处对它的取值都换成
    /// `self` 的值表达式，于是两步落进同一个循环、只写最终输出。归约链、缺失的链接、可写链接与
    /// 长度不一致一律拒绝，而不是猜一个意思。
    ///
    /// # Errors
    ///
    /// 任一侧校验失败、写回不是逐点、消费者没有把 `self.output` 声明成只读 `f32` 缓冲、
    /// 或两侧 `extent` 不是同一个参数名时返回说明。
    pub fn fuse(&self, next: &Kernel) -> Result<Self, VerificationError> {
        verify_link(self, next)?;
        let (Write::Map(value), Write::Map(after)) = (&self.write, &next.write) else {
            return Err(VerificationError::new("fusion needs two pointwise kernels"));
        };
        let fused = Kernel::map(
            format!("{}_{}", self.name, next.name),
            merged_params(self, next),
            self.idx.clone(),
            self.extent.clone(),
            next.output.clone(),
            substitute(after, &self.output, value),
        );
        fused.verify()?;
        Ok(fused)
    }
}

/// 先各自校验，再确认链路成立：把三次提前返回收进一个入口，压低 `fuse` 的分支复杂度。
fn verify_link(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    producer.verify()?;
    consumer.verify()?;
    check_link(producer, consumer)
}

/// 融合前提：两侧都是逐点核、共享同一个 `extent`，且消费者把生产者的输出当只读缓冲读。
fn check_link(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    check_pointwise(producer, consumer)?;
    check_extent(producer, consumer)?;
    check_input(producer, consumer)
}

fn check_pointwise(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    if producer.write.slot().is_some() || consumer.write.slot().is_some() {
        return Err(VerificationError::new("fusion needs two pointwise kernels"));
    }
    Ok(())
}

fn check_extent(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    if producer.extent != consumer.extent {
        return Err(VerificationError::new(format!(
            "fused kernels must share extent `{}`",
            producer.extent
        )));
    }
    Ok(())
}

fn check_input(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    let link = consumer
        .params
        .iter()
        .find(|parameter| parameter.name == producer.output);
    match link {
        Some(Parameter {
            ty:
                ValueType::Buf {
                    element: ScalarType::F32,
                    access: Access::Read,
                },
            ..
        }) => Ok(()),
        Some(_) => Err(VerificationError::new(format!(
            "fused input `{}` must be a read-only f32 buffer",
            producer.output
        ))),
        None => Err(VerificationError::new(format!(
            "consumer does not read `{}`",
            producer.output
        ))),
    }
}

/// 融合核的形参：去掉中间缓冲，再按「同名同型」去重。
///
/// 两侧本来就必须共享同一个 `extent` 形参（`check_extent` 只比较名字），留两份会被「参数不重名」
/// 闸挡下，于是融合永远不成立。同名**异型**不去重：那正是两侧真的冲突，仍由同一道闸报出来。
fn merged_params(producer: &Kernel, consumer: &Kernel) -> Vec<Parameter> {
    let mut merged: Vec<Parameter> = Vec::new();
    for parameter in producer.params.iter().chain(&consumer.params) {
        let linked = parameter.name == producer.output;
        let seen = merged.iter().any(|prior| prior == parameter);
        if !linked && !seen {
            merged.push(parameter.clone());
        }
    }
    merged
}

/// 把消费者里对中间缓冲的每一次取值换成生产者的值表达式。
fn substitute(expression: &Expr, name: &str, replacement: &Expr) -> Expr {
    match expression {
        Expr::Load(loaded) => substitute_load(loaded, name, replacement),
        other => substitute_compound(other, name, replacement),
    }
}

fn substitute_load(loaded: &str, name: &str, replacement: &Expr) -> Expr {
    if loaded == name {
        return replacement.clone();
    }
    Expr::Load(loaded.to_string())
}

fn substitute_compound(expression: &Expr, name: &str, replacement: &Expr) -> Expr {
    match expression {
        Expr::Neg(value) => Expr::Neg(Box::new(substitute(value, name, replacement))),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(substitute(left, name, replacement)),
            right: Box::new(substitute(right, name, replacement)),
        },
        leaf => leaf.clone(),
    }
}
