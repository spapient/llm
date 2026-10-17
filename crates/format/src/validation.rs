use lib::{jsonvalue, limits};

use crate::error::formaterror;
use crate::metadata::tensormetadata;

/// Parses and validates one raw safetensors entry into [`tensormetadata`].
///
/// The entry is read from its JSON value; a schema mismatch is reported as
/// [`formaterror::invalidentry`]. The dtype token, dimensions, element budget,
/// and byte range are then validated by [`tensormetadata::new`].
///
/// # Errors
///
/// Returns a [`formaterror`] for any malformed, unsupported, or inconsistent
/// entry.
pub(crate) fn parseentry(
    name: &str,
    value: &jsonvalue,
    datalength: u64,
    lim: &limits,
) -> Result<tensormetadata, formaterror> {
    let object = value.getobject().ok_or_else(|| formaterror::invalidentry {
        name: name.to_string(),
        reason: "tensor entry is not a JSON object".to_string(),
    })?;
    let dtype = object
        .get("dtype")
        .and_then(jsonvalue::getstr)
        .ok_or_else(|| formaterror::invalidentry {
            name: name.to_string(),
            reason: "missing or non-string 'dtype'".to_string(),
        })?;
    let shape = object
        .get("shape")
        .and_then(jsonvalue::getarray)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    item.getunsigned()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| formaterror::invalidentry {
                            name: name.to_string(),
                            reason: "non-integer shape dimension".to_string(),
                        })
                })
                .collect::<Result<Vec<usize>, formaterror>>()
        })
        .unwrap_or_else(|| {
            Err(formaterror::invalidentry {
                name: name.to_string(),
                reason: "missing or non-array 'shape'".to_string(),
            })
        })?;
    let offsets = object
        .get("data_offsets")
        .and_then(jsonvalue::getarray)
        .ok_or_else(|| formaterror::invalidentry {
            name: name.to_string(),
            reason: "missing or non-array 'data_offsets'".to_string(),
        })?;
    if offsets.len() != 2 {
        return Err(formaterror::invalidentry {
            name: name.to_string(),
            reason: "'data_offsets' must have exactly two integers".to_string(),
        });
    }
    let mut beginend = [0u64; 2];
    for (slot, item) in offsets.iter().enumerate() {
        beginend[slot] = item
            .getinteger()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| formaterror::invalidentry {
                name: name.to_string(),
                reason: "non-integer 'data_offsets' value".to_string(),
            })?;
    }
    let [begin, end] = beginend;
    tensormetadata::new(name.to_string(), dtype, shape, begin, end, datalength, lim)
}

/// Verifies that no two tensor byte ranges overlap.
///
/// Zero-length tensors occupy no bytes and are ignored. Ranges are compared by
/// sorting on the start offset and tracking the maximum end seen so far, so a
/// range nested inside another is still detected. The comparison is
/// conservative: it only rejects genuinely overlapping non-empty ranges.
///
/// # Errors
///
/// Returns [`formaterror::overlappingtensors`] naming the first overlapping
/// pair.
pub fn checkoverlaps(tensors: &[tensormetadata]) -> Result<(), formaterror> {
    let mut spans: Vec<(u64, u64, &str)> = tensors
        .iter()
        .filter(|t| t.getend() > t.getbegin())
        .map(|t| (t.getbegin(), t.getend(), t.getname()))
        .collect();
    spans.sort_unstable_by_key(|span| span.0);
    let mut maxend = 0u64;
    let mut maxname: Option<&str> = None;
    for (begin, end, name) in spans {
        if let Some(previous) = maxname
            && begin < maxend
        {
            return Err(formaterror::overlappingtensors {
                first: previous.to_string(),
                second: name.to_string(),
            });
        }
        if end > maxend {
            maxend = end;
            maxname = Some(name);
        }
    }
    Ok(())
}
