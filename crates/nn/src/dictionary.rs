use std::collections::BTreeMap;

use ::tensor::tensor;

use crate::error::nnerror;
use crate::parameters::{parameter, tensoreq};

/// A named, order-stable collection of parameter tensors.
///
/// A `statedict` maps a fully qualified parameter name to the tensor it holds.
/// It is the persistence and inspection unit: a composed layer collects its
/// parameters under a hierarchical prefix (`block.0.attn.q_proj.weight`), so the
/// dictionary is the single place where names and values meet. Entries are
/// ordered by name, so iteration is deterministic without a separate sort.
///
/// A `statedict` owns its tensors outright. It holds no live reference into a
/// layer, so a snapshot cannot observe a layer mid-mutation.
#[derive(Clone, Debug, Default)]
pub struct statedict {
    entries: BTreeMap<String, tensor<f32>>,
}

impl PartialEq for statedict {
    fn eq(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .all(|(k, v)| other.entries.get(k).is_some_and(|o| tensoreq(v, o)))
    }
}

impl statedict {
    /// Creates an empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Returns `true` when the dictionary holds no parameters.
    #[must_use]
    pub fn isempty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of parameters.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.entries.len()
    }

    /// Inserts a tensor under a fully qualified name.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::duplicateparameter`] when the name already exists, so
    /// a name collision between two layers is reported rather than silently
    /// overwriting a weight.
    pub fn insert(&mut self, name: impl Into<String>, value: tensor<f32>) -> Result<(), nnerror> {
        let name = name.into();
        if self.entries.contains_key(&name) {
            return Err(nnerror::duplicateparameter { name });
        }
        self.entries.insert(name, value);
        Ok(())
    }

    /// Adds a parameter under `prefix`.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::duplicateparameter`] when the composed name exists.
    pub fn add(&mut self, prefix: &str, p: parameter) -> Result<(), nnerror> {
        self.insert(joinname(prefix, p.getname()), p.getvalue().clone())
    }

    /// Returns the tensor named `name`, or `None`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&tensor<f32>> {
        self.entries.get(name)
    }

    /// Returns the tensor named `name` mutably, or `None`.
    #[must_use]
    pub fn getmut(&mut self, name: &str) -> Option<&mut tensor<f32>> {
        self.entries.get_mut(name)
    }

    /// Returns the fully qualified names in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Returns the entries in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &tensor<f32>)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Composes a hierarchical parameter name from a prefix and a leaf name.
#[must_use]
pub fn joinname(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}
