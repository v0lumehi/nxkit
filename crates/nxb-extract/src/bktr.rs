//! BKTR (LibHac's `BucketTree`, `IndirectStorage`, `AesCtrCounterExtendedStorage`): the update's RomFS section is a
//! relocation table over the base section and the patch's own data, which is AES-CTR encrypted with a generation per
//! physical range. Both tables sit at the end of the patch section, encrypted with the section's plain counter.

use std::sync::Arc;

use crate::aes::Aes128;
use crate::cache::BlockCache;
use crate::error::{ExtractError, Result};
use crate::nca::{FsHeader, u32le, u64le};
use crate::storage::{Counter, CtrSection, IndirectStorage, Relocation, Storage};

const NODE: usize = 0x4000;

/// The fixed-size entries of a bucket tree with one offset node (L1) and its entry-set nodes.
pub(crate) fn bucket_entries(
    table: &[u8],
    entry_size: usize,
    expected: u32,
) -> Result<(Vec<&[u8]>, u64)> {
    if expected == 0 && table.is_empty() {
        return Ok((Vec::new(), 0));
    }
    if table.len() < 16 {
        return Err(ExtractError::format("BKTR table shorter than header"));
    }
    let sets = u32le(table, 4) as usize;
    let end = u64le(table, 8);
    if sets > (NODE - 16) / 8 {
        return Err(ExtractError::Unsupported(
            "BKTR tables with a second offset level".into(),
        ));
    }
    let mut out = Vec::with_capacity(expected as usize);
    for s in 0..sets {
        let start = NODE * (1 + s);
        let node = table
            .get(start..(start + NODE).min(table.len()))
            .filter(|n| n.len() >= 16)
            .ok_or_else(|| ExtractError::format("BKTR entry set outside the table"))?;
        let count = u32le(node, 4) as usize;
        if count
            .checked_mul(entry_size)
            .and_then(|n| n.checked_add(16))
            .is_none_or(|n| n > node.len())
        {
            return Err(ExtractError::format("BKTR entry count"));
        }
        for i in 0..count {
            out.push(&node[16 + i * entry_size..16 + (i + 1) * entry_size]);
        }
    }
    if out.len() != expected as usize {
        return Err(ExtractError::format(format!(
            "BKTR: {} entries, header says {expected}",
            out.len()
        )));
    }
    Ok((out, end))
}

/// Builds the patched section of an update: `base` is the base NCA's decrypted data section, `raw` the patch NCA's
/// raw (encrypted) data section, `ctr_base` its offset in the patch NCA.
pub fn patched_section(
    base: Arc<dyn Storage>,
    raw: Arc<dyn Storage>,
    ctr_base: u64,
    fs: &FsHeader,
    key: impl Fn() -> Result<Aes128>,
    cache: Arc<BlockCache>,
) -> Result<(IndirectStorage, Arc<CtrSection>)> {
    let h = &fs.raw;
    let (ind_off, ind_size) = (u64le(h, 0x100), u64le(h, 0x108));
    let (ex_off, ex_size) = (u64le(h, 0x120), u64le(h, 0x128));
    if h.get(0x110..0x114) != Some(b"BKTR") || h.get(0x130..0x134) != Some(b"BKTR") {
        return Err(ExtractError::format("patch section without BKTR headers"));
    }
    let (ind_count, ex_count) = (u32le(h, 0x118), u32le(h, 0x138));
    if ind_size > (256 << 20) || ex_size > (256 << 20) {
        return Err(ExtractError::format("BKTR tables too large"));
    }
    // the tables, decrypted with the section's plain counter
    let plain = CtrSection::new(
        raw.clone(),
        key()?,
        ctr_base,
        Counter::Upper(fs.upper),
        cache.clone(),
    );

    let patch = if ex_count == 0 || ex_size == 0 {
        Arc::new(CtrSection::new(
            raw,
            key()?,
            ctr_base,
            Counter::Upper(fs.upper),
            cache,
        ))
    } else {
        let ex_table = plain.read_vec(ex_off, ex_size as usize)?;
        let (ex, ex_end) = bucket_entries(&ex_table, 0x10, ex_count)?;
        let mut ranges = Vec::with_capacity(ex.len() + 1);
        for e in ex {
            let off = u64le(e, 0);
            let encryption = e[8];
            let generation = u32le(e, 12);
            let mut upper = fs.upper;
            upper[4..8].copy_from_slice(&generation.to_be_bytes());
            ranges.push((
                off,
                if encryption == 1 {
                    Counter::Plain
                } else {
                    Counter::Upper(upper)
                },
            ));
        }
        if ranges.first().map(|r| r.0) != Some(0) || ranges.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(ExtractError::format("BKTR counter table not sorted from 0"));
        }
        // past the counter table's end (the tables themselves): the plain counter
        ranges.push((ex_end, Counter::Upper(fs.upper)));
        Arc::new(CtrSection::with_ranges(
            raw,
            key()?,
            ctr_base,
            ranges,
            cache,
        ))
    };

    let (entries, ind_end) = if ind_count == 0 || ind_size == 0 {
        (
            vec![Relocation {
                virt: 0,
                phys: 0,
                source: 0,
            }],
            base.size(),
        )
    } else {
        let ind_table = plain.read_vec(ind_off, ind_size as usize)?;
        let (ind, ind_end) = bucket_entries(&ind_table, 0x14, ind_count)?;
        let entries = ind
            .iter()
            .map(|e| Relocation {
                virt: u64le(e, 0),
                phys: u64le(e, 8),
                source: u32le(e, 16),
            })
            .collect();
        (entries, ind_end)
    };
    let patch_dyn: Arc<dyn Storage> = patch.clone();
    Ok((
        IndirectStorage::new(entries, ind_end, base, patch_dyn)?,
        patch,
    ))
}

#[cfg(test)]
mod tests {
    use super::{NODE, bucket_entries, patched_section};
    use crate::aes::Aes128;
    use crate::cache::BlockCache;
    use crate::nca::FsHeader;
    use crate::storage::{MemStorage, Storage};
    use std::sync::Arc;

    #[test]
    fn bucket_entries_handles_empty_short_and_unpadded_tables() {
        let (entries, end) = bucket_entries(&[], 0x14, 0).unwrap();
        assert!(entries.is_empty());
        assert_eq!(end, 0);

        assert!(bucket_entries(&[], 0x14, 1).is_err());
        assert!(bucket_entries(&[0u8; 15], 0x14, 0).is_err());

        // Header-only table (16 bytes, < NODE) with 0 sets.
        let mut hdr = [0u8; 16];
        hdr[8..16].copy_from_slice(&0x5000u64.to_le_bytes());
        let (entries, end) = bucket_entries(&hdr, 0x14, 0).unwrap();
        assert!(entries.is_empty());
        assert_eq!(end, 0x5000);

        // Unpadded entry-set node (total length < 2 * NODE).
        let mut unpadded = vec![0u8; NODE + 16 + 0x14];
        unpadded[4..8].copy_from_slice(&1u32.to_le_bytes());
        unpadded[8..16].copy_from_slice(&0x8000u64.to_le_bytes());
        unpadded[NODE + 4..NODE + 8].copy_from_slice(&1u32.to_le_bytes());
        unpadded[NODE + 16..NODE + 16 + 0x14].fill(0xAA);
        let (entries, end) = bucket_entries(&unpadded, 0x14, 1).unwrap();
        assert_eq!(end, 0x8000);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], &[0xAA; 0x14]);

        // Entry count exceeding the unpadded node length is rejected.
        assert!(bucket_entries(&unpadded[..unpadded.len() - 1], 0x14, 1).is_err());
        assert!(bucket_entries(&unpadded[..NODE + 15], 0x14, 1).is_err());
    }

    #[test]
    fn patched_section_handles_zero_and_unpadded_bktr_tables() {
        let key = [0x11u8; 16];
        let upper = [0x22u8; 8];
        let base_bytes: Vec<u8> = (0..0x200u32).map(|i| i as u8).collect();
        let base: Arc<dyn Storage> = Arc::new(MemStorage(base_bytes.clone()));

        // Both ex and ind empty (count == 0, size == 0).
        let mut fs_raw = [0u8; 0x200];
        fs_raw[0x110..0x114].copy_from_slice(b"BKTR");
        fs_raw[0x130..0x134].copy_from_slice(b"BKTR");
        let fs = FsHeader {
            encryption: 4,
            hash_type: 0,
            fs_type: 0,
            upper,
            raw: fs_raw.to_vec(),
        };
        let cache = Arc::new(BlockCache::new(16));
        let (ind, _patch) = patched_section(
            base.clone(),
            Arc::new(MemStorage(vec![0u8; 0x100])),
            0x4000,
            &fs,
            || Ok(Aes128::new(&key)),
            cache.clone(),
        )
        .unwrap();
        assert_eq!(ind.size(), base_bytes.len() as u64);
        let got = ind.read_vec(0, base_bytes.len()).unwrap();
        assert_eq!(got, base_bytes);

        // ex_count == 0 with non-empty 16-byte header, and an unpadded single-entry ind table.
        let mut ind_table = vec![0u8; NODE + 16 + 0x14];
        ind_table[4..8].copy_from_slice(&1u32.to_le_bytes());
        ind_table[8..16].copy_from_slice(&0x100u64.to_le_bytes());
        ind_table[NODE + 4..NODE + 8].copy_from_slice(&1u32.to_le_bytes());
        // Single entry mapping [0, 0x100) to patch (source 1) at phys 0.
        ind_table[NODE + 16..NODE + 24].copy_from_slice(&0u64.to_le_bytes());
        ind_table[NODE + 24..NODE + 32].copy_from_slice(&0u64.to_le_bytes());
        ind_table[NODE + 32..NODE + 36].copy_from_slice(&1u32.to_le_bytes());

        let patch_plain = vec![0x5Au8; 0x100];
        let mut sec = patch_plain.clone();
        let ind_off = sec.len() as u64;
        sec.extend_from_slice(&ind_table);
        let ctr_base = 0x4000u64;
        Aes128::new(&key).ctr(upper, ctr_base, &mut sec);

        let mut fs2_raw = [0u8; 0x200];
        fs2_raw[0x100..0x108].copy_from_slice(&ind_off.to_le_bytes());
        fs2_raw[0x108..0x110].copy_from_slice(&(ind_table.len() as u64).to_le_bytes());
        fs2_raw[0x110..0x114].copy_from_slice(b"BKTR");
        fs2_raw[0x118..0x11C].copy_from_slice(&1u32.to_le_bytes());
        fs2_raw[0x130..0x134].copy_from_slice(b"BKTR");
        let fs2 = FsHeader {
            encryption: 4,
            hash_type: 0,
            fs_type: 0,
            upper,
            raw: fs2_raw.to_vec(),
        };
        let (ind2, _) = patched_section(
            base,
            Arc::new(MemStorage(sec)),
            ctr_base,
            &fs2,
            || Ok(Aes128::new(&key)),
            cache,
        )
        .unwrap();
        assert_eq!(ind2.size(), 0x100);
        assert_eq!(ind2.read_vec(0, 0x100).unwrap(), patch_plain);
    }
}
