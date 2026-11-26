pub(crate) const FINGERPRINT_OFFSET: u64 = 14695981039346656037;
const FINGERPRINT_PRIME: u64 = 1099511628211;

pub(crate) fn update_byte_fingerprint(
    fingerprint: &mut u64,
    bytes: &[u8],
) {
    for &byte in bytes {
        *fingerprint = (*fingerprint ^ byte as u64).wrapping_mul(FINGERPRINT_PRIME);
    }
}

pub(crate) fn compute_byte_fingerprint(bytes: &[u8]) -> u64 {
    let mut fingerprint = FINGERPRINT_OFFSET;
    update_byte_fingerprint(
        &mut fingerprint,
        bytes,
    );

    return fingerprint;
}

#[cfg(test)]
mod tests {
    use super::{
        compute_byte_fingerprint,
        FINGERPRINT_OFFSET,
        update_byte_fingerprint,
    };

    #[test]
    fn test_fingerprint_matches_known_fnv_vectors() {
        for (bytes, expected) in [(&b""[..], 0xcbf29ce484222325u64,), (&b"a"[..], 0xaf63dc4c8601ec8c,), (&b"foobar"[..], 0x85944171f73967e8,),] {
            assert_eq!(
                compute_byte_fingerprint(bytes),
                expected,
            );
        }
    }

    #[test]
    fn test_streaming_fingerprint_matches_single_buffer() {
        let bytes: Vec<u8> = (0..=255).collect();

        for chunk_size in [1, 3, 17, 128, 256,] {
            let mut fingerprint = FINGERPRINT_OFFSET;

            for chunk in bytes.chunks(chunk_size) {
                update_byte_fingerprint(
                    &mut fingerprint,
                    chunk,
                );
            }

            assert_eq!(
                fingerprint,
                compute_byte_fingerprint(&bytes),
            );
        }
    }
}
