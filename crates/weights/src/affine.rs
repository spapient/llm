//! 显式量化方案：分组、零点与标度类型都是文件里的事实，不再靠文件名后缀推断。
//!
//! P1 只定义一套仿射方案（`x ≈ scale · q + zero`）并做完整往返与结构校验；反量化数学在后续阶段。
//! `axis == rank` 表示 per-tensor（单个标度，不沿任何真实轴），此时 `block` 必须为 0。

use diagnostic::{Code, Error, Result};

use crate::dtype::Wire;
use crate::leb::{self, Cursor};
use crate::sink::Sink;

/// 仿射量化方案。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Quant {
    /// 分组轴；等于张量 rank 时是 per-tensor。
    pub axis: u32,
    /// 组宽（元素数）；per-tensor 时为 0。
    pub block: u32,
    /// 是否带零点。
    pub zero: bool,
    /// 标度的元素类型（浮点）。
    pub scale_ty: Wire,
    /// 零点的元素类型（整数）；无零点时为 `None`。
    pub zp_ty: Option<Wire>,
}

impl Quant {
    /// 构造并校验一个方案；`rank` 是所属张量的轴数，用来判定 per-tensor 与分组口径。
    pub fn new(
        rank: u32,
        axis: u32,
        block: u32,
        zero: bool,
        scale_ty: Wire,
        zp_ty: Option<Wire>,
    ) -> Result<Self> {
        let quant = Self {
            axis,
            block,
            zero,
            scale_ty,
            zp_ty,
        };
        quant.validate_layout(rank)?;
        quant.validate()?;
        Ok(quant)
    }

    /// 布局校验：轴不超 rank，per-tensor 的 block 为 0，沿轴分组的 block 非 0。
    pub fn validate_layout(&self, rank: u32) -> Result<()> {
        if self.axis > rank {
            return Err(bad("量化轴超出张量 rank"));
        }
        if self.is_per_tensor(rank) {
            if self.block != 0 {
                return Err(bad("per-tensor 方案的 block 必须为 0"));
            }
        } else if self.block == 0 {
            return Err(bad("沿轴分组的方案 block 不能为 0"));
        }
        Ok(())
    }

    /// 结构校验：标度必须浮点、零点必须整数。
    pub fn validate(&self) -> Result<()> {
        if !self.scale_ty.is_float() {
            return Err(bad("量化标度必须是浮点类型"));
        }
        if self.zero {
            match self.zp_ty {
                Some(zp_ty) if !zp_ty.is_float() && zp_ty != Wire::Bool => {}
                Some(_) => return Err(bad("零点必须是整数类型")),
                None => return Err(bad("声明零点却没给零点类型")),
            }
        } else if self.zp_ty.is_some() {
            return Err(bad("无零点方案不该带零点类型"));
        }
        Ok(())
    }

    /// 是否为 per-tensor（需要结合 rank 判断）。
    pub fn is_per_tensor(&self, rank: u32) -> bool {
        self.axis == rank
    }

    /// 编码进任意 Sink（不含 present 标志，那一位由段描述符写）。
    pub(crate) fn encode(&self, out: &mut impl Sink, rank: u32) -> Result<()> {
        self.validate_layout(rank)?;
        self.validate()?;
        out.put(&[0]); // scheme = affine
        leb::write_u32(out, self.axis);
        leb::write_u32(out, self.block);
        out.put(&[if self.zero { 1 } else { 0 }]);
        leb::write_u32(out, u32::from(self.scale_ty.code()));
        if let Some(zp_ty) = self.zp_ty {
            leb::write_u32(out, u32::from(zp_ty.code()));
        }
        Ok(())
    }

    /// 从游标解码。
    pub(crate) fn decode(cur: &mut Cursor<'_>, rank: u32) -> Result<Self> {
        let scheme = cur.read_u8().map_err(|e| e.with("量化方案"))?;
        if scheme != 0 {
            return Err(Error::new(
                Code::Unsupported,
                format!("未知量化 scheme {scheme}，当前只支持 affine(0)"),
            )
            .with("weights::quant"));
        }
        let axis = cur.read_u32()?;
        let block = cur.read_u32()?;
        let flags = cur.read_u8()?;
        let zero = flags & 1 == 1;
        if flags & !1 != 0 {
            return Err(bad("量化 flags 含非零预留位"));
        }
        let scale_ty = read_wire(cur)?;
        let zp_ty = if zero { Some(read_wire(cur)?) } else { None };
        let quant = Self {
            axis,
            block,
            zero,
            scale_ty,
            zp_ty,
        };
        quant.validate_layout(rank)?;
        quant.validate()?;
        Ok(quant)
    }
}

fn read_wire(cur: &mut Cursor<'_>) -> Result<Wire> {
    let code = cur.read_u32()?;
    Wire::from_code(u16::try_from(code).map_err(|_| bad("wire code 超出 u16"))?)
        .ok_or_else(|| bad("未知 wire 类型 code"))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("weights::quant")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(quant: &Quant, rank: u32) -> Quant {
        let mut buf = Vec::new();
        quant.encode(&mut buf, rank).expect("编码合法");
        Quant::decode(&mut Cursor::new(&buf), rank).expect("解码合法")
    }

    #[test]
    fn per_block_and_per_tensor_schemes_round_trip() {
        let block = Quant::new(2, 1, 32, false, Wire::F16, None).expect("per-block");
        assert_eq!(round(&block, 2), block);

        let tensor = Quant::new(2, 2, 0, true, Wire::F32, Some(Wire::I8)).expect("per-tensor+zp");
        assert_eq!(round(&tensor, 2), tensor);
    }

    #[test]
    fn inconsistent_schemes_are_rejected() {
        assert!(
            Quant::new(2, 1, 0, false, Wire::F16, None).is_err(),
            "block=0 沿轴"
        );
        assert!(
            Quant::new(2, 2, 4, false, Wire::F16, None).is_err(),
            "per-tensor 带 block"
        );
        assert!(
            Quant::new(2, 1, 32, false, Wire::I8, None).is_err(),
            "整数标度"
        );
        assert!(
            Quant::new(2, 1, 32, true, Wire::F16, None).is_err(),
            "缺零点类型"
        );
        assert!(
            Quant::new(2, 1, 32, true, Wire::F16, Some(Wire::F32)).is_err(),
            "浮点零点"
        );
    }
}
