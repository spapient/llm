use ::tensor::{shape, tensor, tensorview};
use api::{backenderror, gatherkernel};

use crate::support::finish;

/// The stable name reported by [`cpuembedding`].
const EMBEDDINGOP: &str = "embedding";

/// Gathers rows from a table by index.
///
/// The first axis of `table` is the row index space. Every index MUST be in
/// `[0, vocab)`. The output stacks the selected rows in index order, so it has
/// shape `[indices.len(), table.shape[1..]]`. A rank-one table has a row width
/// of one.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpuembedding;

impl gatherkernel<f32> for cpuembedding {
    fn getname(&self) -> &'static str {
        EMBEDDINGOP
    }

    fn apply(
        &self,
        table: &tensorview<'_, f32>,
        indices: &[usize],
    ) -> Result<tensor<f32>, backenderror> {
        let dims = table.getshape().getdims();
        let vocab = dims[0];
        let row = dims[1..]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(backenderror::sizeoverflow)?;
        let total = indices
            .len()
            .checked_mul(row)
            .ok_or(backenderror::sizeoverflow)?;
        let src = table.getdata();
        let mut data = Vec::with_capacity(total);
        for &index in indices {
            if index >= vocab {
                return Err(backenderror::indexoutofrange { index, len: vocab });
            }
            data.extend_from_slice(&src[index * row..(index + 1) * row]);
        }
        let mut outdims = Vec::with_capacity(dims.len());
        outdims.push(indices.len());
        outdims.extend_from_slice(&dims[1..]);
        let outshape = shape::new(outdims).map_err(|_| backenderror::sizeoverflow)?;
        finish(&outshape, data)
    }
}
