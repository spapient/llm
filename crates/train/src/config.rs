//! 训练装配配置：只组合强类型纯数据，不读文件、环境变量或命令行。

use diagnostic::Result;

use crate::{LoaderConfig, TrainConfig};

/// 一次训练装配的静态配置；运行状态仍由 `Session`、`Loader` 等对象分别持有。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainSpec {
    /// 会话、学习率调度与梯度累积配置。
    pub session: TrainConfig,
    /// 数据批几何与互斥采样策略。
    pub loader: LoaderConfig,
}

impl TrainSpec {
    /// 在装配边界统一校验各子配置。
    pub fn new(session: TrainConfig, loader: LoaderConfig) -> Result<Self> {
        session.validate()?;
        loader.validate()?;
        Ok(Self { session, loader })
    }

    /// 拆成各运行对象的独立配置。
    pub fn into_parts(self) -> (TrainConfig, LoaderConfig) {
        (self.session, self.loader)
    }
}
