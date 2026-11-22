//! 形参类型 → `ParamKind` 的映射。收窄到四个形状，别的类型在宏展开期就报错。

use syn::{Error, Type, TypePath, TypeReference, spanned::Spanned};

use super::ParamKind;

pub(super) fn parameter_kind(ty: &Type) -> syn::Result<ParamKind> {
    match ty {
        Type::Reference(reference) => reference_kind(reference),
        Type::Path(path) if path.qself.is_none() => scalar_kind(path),
        _ => Err(Error::new(
            ty.span(),
            "kernel parameters must be `f32`, `usize`, `&[f32]`, or `&mut [f32]`",
        )),
    }
}

fn scalar_kind(path: &TypePath) -> syn::Result<ParamKind> {
    match path.path.get_ident().map(ToString::to_string).as_deref() {
        Some("f32") => Ok(ParamKind::F32),
        Some("usize") => Ok(ParamKind::Usize),
        _ => Err(Error::new(
            path.span(),
            "scalar kernel parameters must be `f32` or `usize`",
        )),
    }
}

fn reference_kind(reference: &TypeReference) -> syn::Result<ParamKind> {
    f32_element(slice_element(reference)?)?;
    Ok(if reference.mutability.is_some() {
        ParamKind::BufWrite
    } else {
        ParamKind::BufRead
    })
}

fn slice_element(reference: &TypeReference) -> syn::Result<&Type> {
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return Err(Error::new(
            reference.span(),
            "buffer parameters must be slices (`&[f32]` or `&mut [f32]`)",
        ));
    };
    Ok(slice.elem.as_ref())
}

fn f32_element(ty: &Type) -> syn::Result<()> {
    if is_float(ty) {
        return Ok(());
    }
    Err(Error::new(ty.span(), "buffer element type must be `f32`"))
}

fn is_float(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.qself.is_none() && path.path.is_ident("f32")
}
