use lib::limits;
use serde::Deserialize;
use serde_json::Value;

use crate::error::formaterror;
use crate::metadata::tensormetadata;

/// The raw safetensors schema for one tensor entry.
#[derive(Deserialize)]
struct rawentry {
    dtype: String,
    shape: Vec<usize>,
    #[serde(rename = "data_offsets")]
    dataoffsets: [u64; 2],
}

/// Parses and validates one raw safetensors entry into [`tensormetadata`].
///
/// The entry is deserialized into the expected schema; a schema mismatch is
/// reported as [`formaterror::invalidentry`]. The dtype token, dimensions,
/// element budget, and byte range are then validated by
/// [`tensormetadata::new`].
///
/// # Errors
///
/// Returns a [`formaterror`] for any malformed, unsupported, or inconsistent
/// entry.
pub(crate) fn parseentry(
    name: &str,
    value: &Value,
    datalength: u64,
    lim: &limits,
) -> Result<tensormetadata, formaterror> {
    let raw: rawentry =
        serde_json::from_value(value.clone()).map_err(|e| formaterror::invalidentry {
            name: name.to_string(),
            reason: e.to_string(),
        })?;
    let [begin, end] = raw.dataoffsets;
    tensormetadata::new(
        name.to_string(),
        &raw.dtype,
        raw.shape,
        begin,
        end,
        datalength,
        lim,
    )
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
