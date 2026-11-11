//! Tile-level compute primitives for the `llm` runtime.

/// Adds two numbers.
///
/// # Examples
///
/// ```
/// assert_eq!(cutile::add(2, 2), 4);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
