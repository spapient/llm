use crate::dictionary::statedict;
use crate::error::nnerror;

/// A named, composable unit that owns parameters and exports them by name.
///
/// `module` is the contract a layer satisfies to participate in parameter
/// collection. It deliberately does not include a forward method: layers have
/// different operand arities (a projection takes one tensor, an embedding takes
/// indices, multi-head attention takes three operands), so a single signature
/// would force an enum-wrapped input that adds indirection without adding
/// meaning. Each layer instead exposes a forward whose signature matches its
/// own arity, and every layer implements `module` for name and parameter export.
pub trait module {
    /// Returns the stable layer name.
    fn getname(&self) -> &'static str;

    /// Collects this module's parameters into `dict` under `prefix`.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::duplicateparameter`] when a composed name collides
    /// with an existing entry.
    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror>;

    /// Collects this module into a fresh dictionary at the root prefix.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::duplicateparameter`] on a name collision, which cannot
    /// happen for a single well-formed module but is propagated rather than
    /// suppressed.
    fn getstatedict(&self) -> Result<statedict, nnerror> {
        let mut dict = statedict::new();
        self.collect("", &mut dict)?;
        Ok(dict)
    }
}
