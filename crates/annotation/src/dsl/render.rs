//! `#[dsl]` 的 IR 构造器生成：把已校验的形状摊平成 `cutile::ir::Kernel::new(...)`。

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::ItemFn;

use super::{Body, Param, ir_value_type};

pub(super) fn render(function: &ItemFn, params: &[Param], body: &Body) -> TokenStream2 {
    let ir_params = params.iter().map(|parameter| {
        let name = &parameter.name;
        let ty = ir_value_type(parameter.kind);
        quote! {
            ::cutile::ir::Parameter {
                name: #name.to_string(),
                ty: #ty,
            }
        }
    });
    let fn_name = function.sig.ident.to_string();
    let generated_name = format_ident!("__dsl_ir_{}", fn_name);
    let visibility = &function.vis;
    let idx = &body.idx;
    let extent = &body.extent;
    let output = &body.output;
    let value = &body.value;
    quote! {
        #[doc(hidden)]
        #[allow(missing_docs)]
        #visibility fn #generated_name() -> ::cutile::ir::Kernel {
            ::cutile::ir::Kernel::new(
                #fn_name,
                vec![#(#ir_params),*],
                #idx,
                #extent,
                #output,
                #value,
            )
        }
    }
}
