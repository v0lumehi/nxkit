//! Blizzard BLTE (Block Table) container parser and decompressor used inside CASC `.data` archives.

use crate::SplitError;

/// BLTE header magic (`"BLTE"`).
pub const MAGIC: [u8; 4] = *b"BLTE";

const MAX_DEPTH: usize = 8;
const MAX_DECOMP: usize = 512 << 20;

/// One chunk descriptor in a multi-chunk BLTE header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlteChunk {
    /// Stored byte length of the chunk (including its 1-byte encoding mode prefix).
    pub comp_size: usize,
    /// Uncompressed byte length of the chunk (0 when unspecified).
    pub decomp_size: usize,
}

/// Parsed BLTE header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlteHeader {
    /// Total header size in bytes (0 for a single-chunk stream without a chunk table).
    pub header_size: usize,
    /// Chunk descriptors (empty when `header_size == 0`).
    pub chunks: Vec<BlteChunk>,
}

impl BlteHeader {
    /// Total BLTE container byte length (header + all chunk payloads), or `None` for single-chunk streams (`header_size == 0`).
    pub fn payload_size(&self) -> Option<usize> {
        if self.header_size == 0 {
            return None;
        }
        let mut total = self.header_size;
        for c in &self.chunks {
            total = total.checked_add(c.comp_size)?;
        }
        Some(total)
    }
}

fn be32(d: &[u8], o: usize) -> Option<usize> {
    let b = d.get(o..o.checked_add(4)?)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
}

fn be64(d: &[u8], o: usize) -> Option<u64> {
    let b = d.get(o..o.checked_add(8)?)?;
    Some(u64::from_be_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn valid_mode(b: u8) -> bool {
    matches!(b, b'N' | b'Z' | b'4' | b'F' | b'E')
}

/// Inspects the first bytes of a candidate BLTE header.
///
/// Returns `Some(0)` for a single-chunk header (`header_size == 0`), `Some(header_size)` for a multi-chunk
/// table (`header_size >= 36`), or `None` if `d` does not start a valid BLTE header.
pub fn header_len(d: &[u8]) -> Option<usize> {
    if d.get(..4) != Some(&MAGIC[..]) || d.len() < 9 {
        return None;
    }
    let hs = be32(d, 4)?;
    if hs == 0 {
        return valid_mode(d[8]).then_some(0);
    }
    if d.len() < 12 || hs < 36 || !(hs - 12).is_multiple_of(24) || d[8] != 0x0f {
        return None;
    }
    let count = ((d[9] as usize) << 16) | ((d[10] as usize) << 8) | (d[11] as usize);
    if count == 0 || 12 + 24 * count != hs {
        return None;
    }
    Some(hs)
}

/// Parses a BLTE header from `d`.
pub fn parse_header(d: &[u8]) -> Option<BlteHeader> {
    let hs = header_len(d)?;
    if hs == 0 {
        return Some(BlteHeader {
            header_size: 0,
            chunks: Vec::new(),
        });
    }
    if d.len() < hs {
        return None;
    }
    let count = (hs - 12) / 24;
    let mut chunks = Vec::with_capacity(count);
    let mut total_comp = hs;
    for i in 0..count {
        let at = 12 + 24 * i;
        let comp_size = be32(d, at)?;
        let decomp_size = be32(d, at + 4)?;
        if comp_size == 0 {
            return None;
        }
        total_comp = total_comp.checked_add(comp_size)?;
        chunks.push(BlteChunk {
            comp_size,
            decomp_size,
        });
    }
    Some(BlteHeader {
        header_size: hs,
        chunks,
    })
}

fn fmt_err(m: String) -> SplitError {
    SplitError::Format(m)
}

/// Decodes a BLTE container (`d`).
///
/// Returns `Ok(None)` when the container uses encryption (`'E'`), which cannot be decrypted without CDN keys.
pub fn decode(d: &[u8], label: &str) -> Result<Option<Vec<u8>>, SplitError> {
    decode_in(d, label, 0)
}

fn decode_in(d: &[u8], label: &str, depth: usize) -> Result<Option<Vec<u8>>, SplitError> {
    if depth > MAX_DEPTH {
        return Err(fmt_err(format!("{label}: BLTE frames nested too deep")));
    }
    let hdr = parse_header(d).ok_or_else(|| fmt_err(format!("{label}: invalid BLTE header")))?;
    if hdr.header_size == 0 {
        return decode_chunk(&d[8..], 0, label, depth);
    }
    let total_decomp = hdr
        .chunks
        .iter()
        .try_fold(0usize, |acc, c| acc.checked_add(c.decomp_size))
        .ok_or_else(|| fmt_err(format!("{label}: BLTE uncompressed size overflow")))?;
    if total_decomp > MAX_DECOMP {
        return Err(SplitError::Unsupported(format!(
            "{label}: BLTE payload over {MAX_DECOMP} bytes"
        )));
    }
    let mut out = Vec::with_capacity(total_decomp);
    let mut at = hdr.header_size;
    for (idx, c) in hdr.chunks.iter().enumerate() {
        let end = at
            .checked_add(c.comp_size)
            .ok_or_else(|| fmt_err(format!("{label}: BLTE chunk {idx} size overflow")))?;
        let chunk = d
            .get(at..end)
            .ok_or_else(|| fmt_err(format!("{label}: truncated BLTE chunk {idx}")))?;
        let Some(dec) = decode_chunk(chunk, c.decomp_size, label, depth)? else {
            return Ok(None);
        };
        if out.len().saturating_add(dec.len()) > MAX_DECOMP {
            return Err(SplitError::Unsupported(format!(
                "{label}: BLTE payload over {MAX_DECOMP} bytes"
            )));
        }
        out.extend_from_slice(&dec);
        at = end;
    }
    Ok(Some(out))
}

fn decode_chunk(
    chunk: &[u8],
    expected: usize,
    label: &str,
    depth: usize,
) -> Result<Option<Vec<u8>>, SplitError> {
    let Some((&mode, body)) = chunk.split_first() else {
        return Err(fmt_err(format!("{label}: empty BLTE chunk")));
    };
    let dec = match mode {
        b'N' => body.to_vec(),
        b'Z' => {
            let limit = if expected > 0 { expected } else { MAX_DECOMP };
            crate::zlibx::zlib_stream(body, limit)
                .ok_or_else(|| fmt_err(format!("{label}: damaged BLTE zlib chunk")))?
        }
        b'4' => decode_lz4(body, expected, label)?,
        b'F' => return decode_in(body, label, depth + 1),
        b'E' => return Ok(None),
        other => {
            return Err(fmt_err(format!(
                "{label}: unknown BLTE chunk encoding {other:#x}"
            )));
        }
    };
    if expected > 0 && dec.len() != expected {
        return Err(fmt_err(format!(
            "{label}: BLTE chunk decoded to {} bytes, expected {expected}",
            dec.len()
        )));
    }
    Ok(Some(dec))
}

fn decode_lz4(body: &[u8], expected: usize, label: &str) -> Result<Vec<u8>, SplitError> {
    let limit = if expected > 0 { expected } else { MAX_DECOMP };
    if body.starts_with(&[0x04, 0x22, 0x4d, 0x18]) {
        return crate::lz4::frame(body, limit);
    }
    // BLTE '4' v1 header: version(1)=1, uncompressed_size(8 BE), block_shift(1)
    if body.len() >= 10 && body[0] == 1 && (8..=24).contains(&body[9]) {
        let size = be64(body, 1).unwrap_or(0) as usize;
        if size <= limit && (expected == 0 || size == expected) {
            let bsize = 1usize << body[9];
            let rest = &body[10..];
            if rest.len() == size {
                return Ok(rest.to_vec());
            }
            if size <= bsize
                && let Ok(v) = crate::lz4::block(rest, size)
            {
                return Ok(v);
            }
            if let Some(v) = decode_lz4_blocks(rest, size, bsize) {
                return Ok(v);
            }
        }
    }
    if expected > 0 {
        crate::lz4::block(body, expected)
    } else {
        crate::lz4::block_upto(body, limit)
    }
    .map_err(|e| fmt_err(format!("{label}: BLTE LZ4 chunk: {e}")))
}

fn decode_lz4_blocks(mut rest: &[u8], total: usize, bsize: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(total);
    while out.len() < total {
        let hdr = rest.get(..4)?;
        let csz = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
        let blk = rest.get(4..4 + csz)?;
        rest = &rest[4 + csz..];
        let want = (total - out.len()).min(bsize);
        if csz == want {
            out.extend_from_slice(blk);
        } else {
            out.extend_from_slice(&crate::lz4::block(blk, want).ok()?);
        }
    }
    (out.len() == total && rest.is_empty()).then_some(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Builds a valid uncompressed zlib stream (stored DEFLATE blocks + Adler-32).
    pub(crate) fn zlib_stored(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let mut pos = 0;
        while pos < data.len() || pos == 0 {
            let end = (pos + 65535).min(data.len());
            let last = u8::from(end == data.len());
            let chunk = &data[pos..end];
            let len = chunk.len() as u16;
            out.push(last);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(chunk);
            pos = end;
            if data.is_empty() {
                break;
            }
        }
        let (mut a, mut b) = (1u32, 0u32);
        for &x in data {
            a = (a + u32::from(x)) % 65521;
            b = (b + a) % 65521;
        }
        out.extend_from_slice(&((b << 16) | a).to_be_bytes());
        out
    }

    pub(crate) fn make_blte(chunks: &[&[u8]], decomp_sizes: &[usize]) -> Vec<u8> {
        let hs = 12 + 24 * chunks.len();
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(hs as u32).to_be_bytes());
        out.push(0x0f);
        let n = chunks.len() as u32;
        out.extend_from_slice(&n.to_be_bytes()[1..]);
        for (c, &ds) in chunks.iter().zip(decomp_sizes) {
            out.extend_from_slice(&(c.len() as u32).to_be_bytes());
            out.extend_from_slice(&(ds as u32).to_be_bytes());
            out.extend_from_slice(&[0u8; 16]);
        }
        for c in chunks {
            out.extend_from_slice(c);
        }
        out
    }

    #[test]
    fn decodes_raw_zlib_lz4_and_nested_chunks() {
        let mut c0 = vec![b'N'];
        c0.extend_from_slice(b"hello ");

        let mut c1 = vec![b'Z'];
        c1.extend_from_slice(&zlib_stored(b"world!"));

        // LZ4 raw block: literal "abc" (len 3) + match len 9 at offset 3 -> "abcabcabcabc"
        let mut c2 = vec![b'4', 1];
        c2.extend_from_slice(&12u64.to_be_bytes());
        c2.push(16);
        c2.extend_from_slice(&[0x35, b'a', b'b', b'c', 3, 0]);

        let inner = make_blte(&[&c0, &c1], &[6, 6]);
        let mut c_nested = vec![b'F'];
        c_nested.extend_from_slice(&inner);

        let outer = make_blte(&[&c_nested, &c2], &[12, 12]);
        let hdr = parse_header(&outer).unwrap();
        assert_eq!(hdr.payload_size(), Some(outer.len()));
        assert_eq!(
            decode(&outer, "test").unwrap(),
            Some(b"hello world!abcabcabcabc".to_vec())
        );
    }

    #[test]
    fn single_chunk_and_encrypted_behave_cleanly() {
        let mut single = Vec::from(MAGIC);
        single.extend_from_slice(&0u32.to_be_bytes());
        single.push(b'Z');
        single.extend_from_slice(&zlib_stored(b"single-chunk"));
        assert_eq!(header_len(&single), Some(0));
        assert_eq!(
            decode(&single, "single").unwrap(),
            Some(b"single-chunk".to_vec())
        );

        let enc = make_blte(&[b"E\x08\0\0\0\0\0\0\0\0\x04\0\0\0\0Sdeadbeef"], &[8]);
        assert_eq!(decode(&enc, "enc").unwrap(), None);
    }
}
