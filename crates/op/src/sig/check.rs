use super::*;
use crate::model::Values;

pub(super) fn describe(values: &Values) -> &'static str {
    match values {
        Values::F32(_) => "float",
        Values::U8(_) => "unsigned char",
    }
}

pub(super) fn type_matches(ctype: CType, values: &Values) -> bool {
    matches!(
        (ctype, values),
        (CType::F32, Values::F32(_)) | (CType::U8, Values::U8(_))
    )
}

pub(super) fn line_of(text: &str, offset: usize) -> usize {
    crate::parse::count_newlines(&text[..offset.min(text.len())]) + 1
}
