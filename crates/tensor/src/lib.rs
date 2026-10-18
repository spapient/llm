#![forbid(unsafe_code)]
//! Core tensor types and numerical primitives for the LLM runtime.
//!
//! `tensor` owns tensor storage, shape, layout, element typing, and
//! read-only views. It depends on `lib` for resource limits and error
//! helpers and MUST NOT depend on any execution backend.
//!
//! # Model
//!
//! A [`shape`] is a non-empty list of dimensions. A [`layout`] maps a logical
//! tensor onto a linear element buffer; [`layout::contiguous`] builds the
//! canonical row-major layout. A [`tensor`] owns a dense element buffer whose
//! length equals the shape's element count, and a [`tensorview`] borrows a
//! tensor's shape, layout, and data without copying or sharing ownership.
//!
//! All size and offset arithmetic is checked; a computation that would overflow
//! returns a [`tensorerror`] instead of wrapping or panicking.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace at
//! the crate root.
//!
//! # Scope
//!
//! This crate provides metadata and storage only. It defines no operators and
//! no execution logic. The [`element`] trait is implemented for the primitive
//! types with a native Rust representation (`f32`, `f64`, `i8`, `u8`, `i32`,
//! `i64`), plus the hand-written half-precision types [`f16`] and [`bf16`]
//! (`halffloat.rs`), which are storage-only: no backend declares their
//! execution. [`dtype`] is the storage
//! taxonomy and is a strict superset of [`precision`](lib::precision): `dtype`
//! additionally contains integer storage and storage-only `f64`, none of which
//! is a compute precision.

mod bytes;
mod dtypes;
mod elements;
mod error;
mod halffloat;
mod layouts;
mod shapes;
mod tensors;
mod views;

pub use bytes::{byteable, bytesof, frombytes};
pub use dtypes::dtype;
pub use elements::element;
pub use error::tensorerror;
pub use halffloat::{bf16, f16};
pub use layouts::layout;
pub use shapes::shape;
pub use tensors::tensor;
pub use views::tensorview;
