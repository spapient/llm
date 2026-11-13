//! 后量子签名侧车（detached envelope）：与 `.pk` 容器分开存放，签名绑定容器的逐张量 Merkle 根。
//!
//! 职责分层与 CRC 互补：CRC 防传输误码、不防恶意篡改；ML-DSA 防具备量子计算能力的攻击者
//! 伪造权重。验签任一环失败都返回错误，调用方据此 fail-closed 拒绝加载，不做“放行并告警”。
//!
//! 被签消息遵循 FIPS 204 外部消息格式：`M' = 0x00 || len(ctx) || ctx || root`，再由
//! `mu = SHAKE256(tr || M')` 收束到 64 字节；固定 `ctx` 做域分离，防止其他用途的签名被重放。

use diagnostic::{Code, Error, Result};

use crate::dsa::{ML_DSA_44, ML_DSA_65, ML_DSA_87, Params, SecretKey};
use crate::{keccak, leb};

/// 侧车魔数 `"LLEN"`。
pub const MAGIC: u32 = 0x4E45_4C4C;

/// 侧车格式版本。
pub const VERSION: u8 = 1;

/// 域分离上下文：把密钥绑定到“权重发布”用途，长度远小于 FIPS 204 的 255 上限。
const CONTEXT: &[u8] = b"llm.weights.v1";

/// 签名侧车：参数集标识、验签公钥与签名；字段闭集，解析拒绝多余字节。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Envelope {
    set: u8,
    pk: Vec<u8>,
    sig: Vec<u8>,
}

/// 由参数集参数取标准标识（44/65/87）。
fn set_id(params: &Params) -> Result<u8> {
    match (params.k, params.l) {
        (4, 4) => Ok(44),
        (6, 5) => Ok(65),
        (8, 7) => Ok(87),
        _ => Err(Error::new(Code::Invalid, "未知 ML-DSA 参数集").with("envelope::set_id")),
    }
}

/// 构造外部消息代表 mu：绑定 tr、域分离上下文与 Merkle 根。
fn mu(tr: &[u8; 64], root: &[u8; 32]) -> [u8; 64] {
    let mut msg = vec![0x00, CONTEXT.len() as u8];
    msg.extend(CONTEXT);
    msg.extend(root);
    let mut input = tr.to_vec();
    input.extend(msg);
    let mut out = [0u8; 64];
    out.copy_from_slice(&keccak::shake256(&input, 64));
    out
}

impl Envelope {
    /// 用私钥对容器 Merkle 根签名，连同显式公钥封入侧车；全零 rnd 为确定性签名。
    pub fn seal(params: &Params, pk: &[u8], sk: &SecretKey, root: &[u8; 32]) -> Result<Self> {
        let m = mu(&sk.tr, root);
        let sig = match (params.k, params.l) {
            (4, 4) => params.sign_internal::<4, 4>(sk, &m, &[0u8; 32]),
            (6, 5) => params.sign_internal::<6, 5>(sk, &m, &[0u8; 32]),
            (8, 7) => params.sign_internal::<8, 7>(sk, &m, &[0u8; 32]),
            _ => return Err(Error::new(Code::Invalid, "未知 ML-DSA 参数集").with("Envelope::seal")),
        }
        .ok_or_else(|| Error::new(Code::Internal, "签名达到尝试上限").with("Envelope::seal"))?;
        Ok(Self {
            set: set_id(params)?,
            pk: pk.to_vec(),
            sig,
        })
    }

    /// 编码为侧车字节：魔数、版本、参数集，公钥与签名各带 LEB 长度前缀。
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(MAGIC.to_le_bytes());
        out.push(VERSION);
        out.push(self.set);
        leb::write_u32(&mut out, self.pk.len() as u32);
        out.extend(&self.pk);
        leb::write_u32(&mut out, self.sig.len() as u32);
        out.extend(&self.sig);
        out
    }

    /// 解析侧车字节：魔数、版本、长度前缀逐项核对，末尾多余字节即失败。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cur = leb::Cursor::new(bytes);
        let head = cur.take(4)?;
        let magic = u32::from_le_bytes(head.try_into().expect("4"));
        if magic != MAGIC {
            return Err(bad("侧车魔数不匹配"));
        }
        if cur.read_u8()? != VERSION {
            return Err(bad("侧车版本不支持"));
        }
        let set = cur.read_u8()?;
        if !matches!(set, 44 | 65 | 87) {
            return Err(bad("未知 ML-DSA 参数集"));
        }
        let pk_len = cur.read_u32()? as usize;
        let pk = cur.take(pk_len)?.to_vec();
        let sig_len = cur.read_u32()? as usize;
        let sig = cur.take(sig_len)?.to_vec();
        if !cur.remaining().is_empty() {
            return Err(bad("侧车末尾含多余字节"));
        }
        Ok(Self { set, pk, sig })
    }

    /// fail-closed 校验：重算 tr 与消息代表，按参数集分派验签，任一不符即错误。
    pub fn check(&self, root: &[u8; 32]) -> Result<()> {
        let mut tr = [0u8; 64];
        tr.copy_from_slice(&keccak::shake256(&self.pk, 64));
        let m = mu(&tr, root);
        let ok = match self.set {
            44 => ML_DSA_44.verify_internal::<4, 4>(&self.pk, &m, &self.sig),
            65 => ML_DSA_65.verify_internal::<6, 5>(&self.pk, &m, &self.sig),
            87 => ML_DSA_87.verify_internal::<8, 7>(&self.pk, &m, &self.sig),
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(bad("ML-DSA 验签失败：根被篡改或公钥不符"))
        }
    }
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("Envelope")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsa::ML_DSA_65;

    #[test]
    fn a_tampered_root_is_rejected() {
        let xi: [u8; 32] = core::array::from_fn(|j| (j + 5) as u8);
        let (pk, sk) = ML_DSA_65.key_gen_internal::<6, 5>(&xi);
        let root = [7u8; 32];
        let env = Envelope::seal(&ML_DSA_65, &pk, &sk, &root).expect("封侧车");
        env.check(&root).expect("原始根通过");
        let mut bad_root = root;
        bad_root[0] ^= 1;
        assert!(env.check(&bad_root).is_err());
    }

    #[test]
    fn round_trip_bytes_and_rejects_garbage() {
        let xi: [u8; 32] = core::array::from_fn(|j| (j + 9) as u8);
        let (pk, sk) = ML_DSA_65.key_gen_internal::<6, 5>(&xi);
        let env = Envelope::seal(&ML_DSA_65, &pk, &sk, &[3u8; 32]).expect("封侧车");
        let bytes = env.encode();
        assert_eq!(Envelope::decode(&bytes).expect("解析"), env);
        assert!(Envelope::decode(&bytes[..bytes.len() - 1]).is_err());
    }
}
