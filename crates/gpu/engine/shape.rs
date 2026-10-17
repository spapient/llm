use ::tensor::{shape, tensorview};
use api::backenderror;

/// Concatenates two flat buffers.
pub(crate) fn concatdata(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut data = Vec::with_capacity(a.len() + b.len());
    data.extend_from_slice(a);
    data.extend_from_slice(b);
    data
}

/// Concatenates two rank-two shapes along their sequence axis.
pub(crate) fn concatshape(a: &shape, b: &shape) -> Result<shape, backenderror> {
    let ad = a.getdims();
    let bd = b.getdims();
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[1] {
        return Err(backenderror::shapemismatch { op: "attention" });
    }
    shape::new(vec![ad[0] + bd[0], ad[1]]).map_err(|_| backenderror::sizeoverflow)
}

/// Validates scan ranks, channel alignment, and state-width alignment,
/// returning `(n, d, statewidth)`.
///
/// The spectral precondition on `a` is a separate predicate ([`scanstable`]):
/// the single-operator path of all three backends applies it; `musa`'s
/// recorded path does not read operand data and therefore skips it.
pub(crate) fn scanshape(
    input: &shape,
    init: &shape,
    a: &shape,
    b: &shape,
    c: &shape,
    skip: &shape,
) -> Result<(usize, usize, usize), backenderror> {
    let id = input.getdims();
    let initd = init.getdims();
    let ad = a.getdims();
    let bd = b.getdims();
    let cd = c.getdims();
    let skipd = skip.getdims();
    if id.len() != 2
        || initd.len() != 2
        || ad.len() != 2
        || bd.len() != 2
        || cd.len() != 2
        || skipd.len() != 1
    {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    let d = id[1];
    if ad[0] != d || bd[0] != d || cd[0] != d || skipd[0] != d || initd[0] != d {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    let statewidth = initd[1];
    if ad[1] != statewidth || bd[1] != statewidth || cd[1] != statewidth {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    Ok((id[0], d, statewidth))
}

/// Enforces the scan's spectral precondition: every `a` entry is finite and
/// inside `[-1, 1]`.
pub(crate) fn scanstable(a: &tensorview<'_, f32>) -> Result<(), backenderror> {
    for &value in a.getdata() {
        if value.abs() > 1.0 || value.is_nan() {
            return Err(backenderror::invalidparameter { op: "scan" });
        }
    }
    Ok(())
}

/// Builds the gather output shape: `rows` replaces the table's first dimension.
pub(crate) fn embeddingoutshape(tableshape: &shape, rows: usize) -> Result<shape, backenderror> {
    let mut outdims = Vec::with_capacity(tableshape.getrank());
    outdims.push(rows);
    outdims.extend_from_slice(&tableshape.getdims()[1..]);
    shape::new(outdims).map_err(|_| backenderror::sizeoverflow)
}
