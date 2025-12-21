//! Adler-32 checksum (RFC 1950) used for zlib wrappers.

/// Calculate Adler-32 checksum of data.
///
/// Follows the reference algorithm with modulus 65521.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    // Process in reasonably sized blocks to limit modulo operations.
    const NMAX: usize = 5552;
    let mut chunks = data.chunks(NMAX);
    while let Some(chunk) = chunks.next() {
        for &b in chunk {
            s1 = (s1 + b as u32) % MOD_ADLER;
            s2 = (s2 + s1) % MOD_ADLER;
        }
    }

    (s2 << 16) | s1
}

#[cfg(test)]
mod tests {
    use super::adler32;

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32(&[]), 1);
    }

    #[test]
    fn test_adler32_known_values() {
        assert_eq!(adler32(b"hello"), 0x062C0215);
        assert_eq!(adler32(b"Adler-32"), 0x0C34027B);
        assert_eq!(adler32(b"123456789"), 0x091E01DE);
    }
}
