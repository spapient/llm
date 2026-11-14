//! 后量子签名侧车：与 `.pk` 容器分开存放，签名绑定容器的逐张量 Merkle 根。
//!
//! 与 CRC 互补：CRC 防传输误码、ML-DSA 防量子攻击者伪造权重，任一环验签失败即 fail-closed。
//! 消息按 FIPS 204 外部格式 `M' = 0x00 || len(ctx) || ctx || root` 取 `mu = SHAKE256(tr || M')`。

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

/// 构造外部消息代表 mu：绑定 tr、域分离上下文与 Merkle 根。
fn mu(tr: &[u8; 64], root: &[u8; 32]) -> [u8; 64] {
    let input = [&tr[..], &[0x00, CONTEXT.len() as u8], CONTEXT, root].concat();
    let mut out = [0u8; 64];
    out.copy_from_slice(&keccak::shake256(&input, 64));
    out
}

/// 按参数集分派确定性签名（rnd 全零），并给出标准标识：两处都只认 `(k, l)` 闭集。
fn sign(params: &Params, sk: &SecretKey, m: &[u8; 64]) -> Result<(u8, Vec<u8>)> {
    let (set, sig) = match (params.k, params.l) {
        (4, 4) => (44, params.sign_internal::<4, 4>(sk, m, &[0u8; 32])),
        (6, 5) => (65, params.sign_internal::<6, 5>(sk, m, &[0u8; 32])),
        (8, 7) => (87, params.sign_internal::<8, 7>(sk, m, &[0u8; 32])),
        _ => return Err(Error::new(Code::Invalid, "未知 ML-DSA 参数集").with("Envelope::seal")),
    };
    sig.map(|sig| (set, sig))
        .ok_or_else(|| Error::new(Code::Internal, "签名达到尝试上限").with("Envelope::seal"))
}

/// 魔数与版本判据；不回答参数集与载荷。
fn header(cur: &mut leb::Cursor<'_>) -> Result<()> {
    if cur.take(4)? != MAGIC.to_le_bytes().as_slice() {
        return Err(bad("侧车魔数不匹配"));
    }
    if cur.read_u8()? != VERSION {
        return Err(bad("侧车版本不支持"));
    }
    Ok(())
}

/// 公钥与签名两段（同构：LEB 长度 + 字节）；之后不得留下多余字节。
fn body(mut cur: leb::Cursor<'_>) -> Result<[Vec<u8>; 2]> {
    let mut parts = [Vec::new(), Vec::new()];
    for part in &mut parts {
        let len = cur.read::<u32>()? as usize;
        *part = cur.take(len)?.to_vec();
    }
    if !cur.remaining().is_empty() {
        return Err(bad("侧车末尾含多余字节"));
    }
    Ok(parts)
}

impl Envelope {
    /// 用私钥对容器 Merkle 根签名，连同显式公钥封入侧车；全零 rnd 为确定性签名。
    pub fn seal(params: &Params, pk: &[u8], sk: &SecretKey, root: &[u8; 32]) -> Result<Self> {
        let (set, sig) = sign(params, sk, &mu(&sk.tr, root))?;
        Ok(Self {
            set,
            pk: pk.to_vec(),
            sig,
        })
    }

    /// 编码为侧车字节：魔数、版本、参数集，公钥与签名各带 LEB 长度前缀。
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = MAGIC.to_le_bytes().to_vec();
        out.push(VERSION);
        out.push(self.set);
        leb::write(&mut out, self.pk.len());
        out.extend(&self.pk);
        leb::write(&mut out, self.sig.len());
        out.extend(&self.sig);
        out
    }

    /// 解析侧车字节：魔数、版本、长度前缀逐项核对，末尾多余字节即失败。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cur = leb::Cursor::new(bytes);
        header(&mut cur)?;
        let set = cur.read_u8()?;
        if !matches!(set, 44 | 65 | 87) {
            return Err(bad("未知 ML-DSA 参数集"));
        }
        let [pk, sig] = body(cur)?;
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
        ok.then_some(())
            .ok_or_else(|| bad("ML-DSA 验签失败：根被篡改或公钥不符"))
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

    #[test]
    fn a_truncated_signature_is_rejected_not_sliced() {
        // 长度前缀来自文件：声明 1 字节的签名能通过 decode，但验签必须当场判无效，
        // 而不是按参数集的偏移去切一个 1 字节的切片。
        let xi: [u8; 32] = core::array::from_fn(|j| (j + 11) as u8);
        let (pk, _sk) = ML_DSA_44.key_gen_internal::<4, 4>(&xi);
        let mut bytes = MAGIC.to_le_bytes().to_vec();
        bytes.push(VERSION);
        bytes.push(44);
        leb::write(&mut bytes, pk.len());
        bytes.extend(&pk);
        leb::write(&mut bytes, 1u32);
        bytes.push(0);
        let env = Envelope::decode(&bytes).expect("长度前缀自洽，decode 通过");
        assert_eq!(env.sig.len(), 1);
        assert!(env.check(&[0u8; 32]).is_err(), "截断签名不得验签成功");
    }
}
