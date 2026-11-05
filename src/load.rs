//! Index loading for the standalone CLI.
//!
//! `reference` = the unchanged `ProductIndex::load` (ser::Reader::open + typed decode + validate +
//! key index), timed only as a whole.
//! `fast` = the SAME checks in the SAME precedence, restructured to avoid needless passes and
//! copies, with every subphase timed:
//!   * the file is read once into a buffer sized from its metadata (parallel positional reads;
//!     `read_to_end` grew and recopied its buffer), instead of read + per-array `to_vec` copy +
//!     element-wise decode;
//!   * the whole-file FNV-1a checksum (format-defined, sequential) runs on its own thread while
//!     the container table is parsed and the arrays are decoded straight from the buffer into
//!     their typed vectors in parallel. Nothing parsed is USED before the checksum is known:
//!     a checksum mismatch is reported first, exactly as `Reader::open` does, then table/parse
//!     errors, then the `ProductIndex::load` scalar/name checks, then dtype checks in the original
//!     array order, then the unchanged structural `validate`, then the unchanged key index.
//! No check is skipped, sampled or deferred past the point where the index is returned.
//!
//! 2026-10-02 (opt-in syncmer sampler): both loaders also recognize CANDIDATE indexes (kind 'Y',
//! the 14 product arrays plus `selection`), exactly as the copied syncmer_core
//! `ProductIndex::load` does: the kind byte picks the schema, the selection array is decoded
//! and parsed after the 14 arrays, the unchanged structural `validate` re-derives every anchor
//! key's selection from its canonical code, and then BOTH loaders apply the CLI's sampler
//! admission (`sampler::admit`: approved settings, nominal thinning re-derived from c).

use crate::kernel::threshold;
use crate::product::{KeyIndex, ProductIndex, LoadReport, KIND_SYNCMER, PRODUCT_ARRAYS,
                     SYNCMER_PRODUCT_ARRAYS};
use crate::ser::{fnv1a, Dt, KIND_PRODUCT, MAGIC, VERSION};
use std::time::Instant;

#[derive(Default, Clone)]
pub struct LoadPhases {
    pub loader: String,
    pub open_and_size: f64,
    pub read: f64,
    pub checksum_thread: f64,
    pub parse_and_decode: f64,
    pub checksum_wait: f64,
    pub validate: f64,
    pub lookup_init: f64,
    pub total: f64,
    pub bytes: u64,
}

pub fn load_reference(path: &str) -> Result<(ProductIndex, LoadReport, LoadPhases), String> {
    let t = Instant::now();
    let (idx, rep) = ProductIndex::load(path)?;
    crate::sampler::admit(&idx, path)?;
    let ph = LoadPhases { loader: "reference (unchanged ProductIndex::load)".into(),
                          lookup_init: rep.lookup_init_seconds, total: t.elapsed().as_secs_f64(),
                          bytes: rep.bytes, ..Default::default() };
    Ok((idx, rep, ph))
}

fn read_parallel(path: &str, parts: usize) -> Result<Vec<u8>, String> {
    // positional reads as tasks of the CURRENT rayon pool (no extra OS threads)
    use rayon::prelude::*;
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    let len = f.metadata().map_err(|e| format!("{}: {}", path, e))?.len() as usize;
    let mut raw = vec![0u8; len];
    let chunk = len.div_ceil(parts.max(1)).max(1 << 20);
    let res: Vec<Result<(), String>> = raw.par_chunks_mut(chunk).enumerate()
        .map(|(i, ch)| f.read_exact_at(ch, (i * chunk) as u64)
             .map_err(|e| format!("{}: {}", path, e)))
        .collect();
    for r in res {
        r?;
    }
    // a file that grew while being read would otherwise be silently truncated to its old size
    let now = f.metadata().map_err(|e| format!("{}: {}", path, e))?.len() as usize;
    if now != len {
        return Err(format!("{}: file size changed while loading ({} -> {})", path, len, now));
    }
    Ok(raw)
}

struct Table<'a> {
    scalars: Vec<u64>,
    meta: String,
    strings: Vec<String>,
    arrays: Vec<(String, Dt, &'a [u8], usize)>,
}

/// The container table, with `Reader::open`'s checks and messages, borrowing array bytes.
fn parse<'a>(path: &str, raw: &'a [u8], allowed: &[&str]) -> Result<Table<'a>, String> {
    let hdr = 12 + 1 + 4 + 8;
    let end = raw.len() - 8;
    let mut p = hdr;
    let rd = |p: &mut usize| -> Result<u64, String> {
        if *p + 8 > end {
            return Err(format!("{}: payload ends inside a scalar", path));
        }
        let v = u64::from_le_bytes(raw[*p..*p + 8].try_into().unwrap());
        *p += 8;
        Ok(v)
    };
    let nsc = rd(&mut p)? as usize;
    let mut scalars = Vec::with_capacity(nsc.min(1 << 16));
    for _ in 0..nsc {
        scalars.push(rd(&mut p)?);
    }
    let ml = rd(&mut p)? as usize;
    if p + ml > end {
        return Err(format!("{}: metadata runs past the end", path));
    }
    let meta = String::from_utf8(raw[p..p + ml].to_vec())
        .map_err(|e| format!("{}: metadata is not UTF-8: {}", path, e))?;
    p += ml;
    let nstr = rd(&mut p)? as usize;
    let mut strings = Vec::with_capacity(nstr.min(1 << 20));
    for _ in 0..nstr {
        let l = rd(&mut p)? as usize;
        if p + l > end {
            return Err(format!("{}: a stored name runs past the end", path));
        }
        strings.push(String::from_utf8(raw[p..p + l].to_vec())
                     .map_err(|_| format!("{}: a stored name is not UTF-8", path))?);
        p += l;
    }
    let narr = rd(&mut p)? as usize;
    if narr != allowed.len() {
        return Err(format!("{}: artifact declares {} arrays, this build expects {}",
                           path, narr, allowed.len()));
    }
    let mut arrays: Vec<(String, Dt, &'a [u8], usize)> = Vec::new();
    for _ in 0..narr {
        if p >= end {
            return Err(format!("{}: the array table ends early", path));
        }
        let nl = raw[p] as usize;
        p += 1;
        if p + nl > end {
            return Err(format!("{}: an array name runs past the end", path));
        }
        let name = String::from_utf8(raw[p..p + nl].to_vec())
            .map_err(|_| format!("{}: an array name is not UTF-8", path))?;
        p += nl;
        if p + 9 > end {
            return Err(format!("{}: array {} has no header", path, name));
        }
        let dt = Dt::from_code(raw[p])
            .ok_or_else(|| format!("{}: array {} has unknown dtype {:?}",
                                   path, name, raw[p] as char))?;
        p += 1;
        let ln = rd(&mut p)? as usize;
        if !allowed.contains(&name.as_str()) {
            return Err(format!("{}: unexpected array {:?}; wrong schema", path, name));
        }
        if arrays.iter().any(|a| a.0 == name) {
            return Err(format!("{}: array {:?} appears twice", path, name));
        }
        let nb = ln.checked_mul(dt.width())
            .ok_or_else(|| format!("{}: array {} declares an impossible length", path, name))?;
        if p + nb > end {
            return Err(format!("{}: array {} declares {} elements that do not fit",
                               path, name, ln));
        }
        arrays.push((name, dt, &raw[p..p + nb], ln));
        p += nb;
    }
    if p != end {
        return Err(format!("{}: {} bytes of unread payload remain", path, end - p));
    }
    for a in allowed {
        if !arrays.iter().any(|x| x.0 == *a) {
            return Err(format!("{}: required array {} is missing", path, a));
        }
    }
    Ok(Table { scalars, meta, strings, arrays })
}

/// Typed little-endian decode, parallel over 1 MiB chunks.
fn decode<T: Send + Default + Clone, const W: usize>(b: &[u8], f: fn([u8; W]) -> T) -> Vec<T> {
    use rayon::prelude::*;
    let n = b.len() / W;
    let mut out = vec![T::default(); n];
    const CH: usize = 1 << 17;
    out.par_chunks_mut(CH).enumerate().for_each(|(ci, o)| {
        let base = ci * CH * W;
        for (i, x) in o.iter_mut().enumerate() {
            let j = base + i * W;
            *x = f(b[j..j + W].try_into().unwrap());
        }
    });
    out
}

enum Col {
    U64(Vec<u64>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    Other,
}

pub fn load_fast(path: &str, read_parts: usize)
    -> Result<(ProductIndex, LoadReport, LoadPhases), String> {
    let t0 = Instant::now();
    let mut ph = LoadPhases { loader: "fast (same checks and precedence; single read; checksum \
thread concurrent with speculative parse, typed decode, validate and key index)".into(),
                              ..Default::default() };
    let raw = read_parallel(path, read_parts)?;
    ph.read = t0.elapsed().as_secs_f64();
    ph.bytes = raw.len() as u64;
    // header checks: exactly Reader::open's, before anything else
    let hdr = 12 + 1 + 4 + 8;
    if raw.len() < hdr + 8 {
        return Err(format!("{}: {} bytes is shorter than the header", path, raw.len()));
    }
    if &raw[..12] != MAGIC {
        return Err(format!("{}: wrong magic {:?}; not a subcore artifact",
                           path, String::from_utf8_lossy(&raw[..12])));
    }
    let kind = raw[12];
    // the reference loader picks the schema from the kind byte: 'Y' -> candidate schema, any
    // other kind -> the original schema (so a wrong kind is reported against 'P', as before)
    if kind != KIND_PRODUCT && kind != KIND_SYNCMER {
        return Err(format!("{}: this is a {:?} artifact but a {:?} one was asked for",
                           path, kind as char, KIND_PRODUCT as char));
    }
    let version = u32::from_le_bytes(raw[13..17].try_into().unwrap());
    if version != VERSION {
        return Err(format!("{}: format version {} but this build reads {}",
                           path, version, VERSION));
    }
    let plen = u64::from_le_bytes(raw[17..25].try_into().unwrap()) as usize;
    if raw.len() != hdr + plen + 8 {
        return Err(format!("{}: declares a {}-byte payload but the file is {} bytes -- \
truncated, appended to, or a different schema", path, plen, raw.len()));
    }
    let want = u64::from_le_bytes(raw[raw.len() - 8..].try_into().unwrap());
    // Everything after the checksum is computed SPECULATIVELY while the checksum runs and is
    // released only if the checksum matches. Error precedence is Reader::open's and
    // ProductIndex::load's: checksum, table, load checks, dtypes (array order), validate.
    let mut sp = LoadPhases::default();
    let (got, ck_secs, wait, spec) = std::thread::scope(|s| {
        let ck = s.spawn(|| {
            let t = Instant::now();
            let g = fnv1a(&raw[..raw.len() - 8]);
            (g, t.elapsed().as_secs_f64())
        });
        let spec = speculative(path, &raw, kind == KIND_SYNCMER, &mut sp);
        let tw = Instant::now();
        let (g, cs) = ck.join().expect("checksum thread panicked");
        (g, cs, tw.elapsed().as_secs_f64(), spec)
    });
    ph.checksum_thread = ck_secs;
    ph.checksum_wait = wait;
    ph.parse_and_decode = sp.parse_and_decode;
    ph.validate = sp.validate;
    ph.lookup_init = sp.lookup_init;
    if want != got {
        return Err(format!("{}: checksum {:016x} does not match the content {:016x}; the \
artifact is corrupt", path, want, got));
    }
    drop(raw);
    let idx = spec?;
    crate::sampler::admit(&idx, path)?;
    ph.total = t0.elapsed().as_secs_f64();
    let slots = idx.map.slots();
    Ok((idx, LoadReport { bytes: ph.bytes, read_seconds: ph.total - ph.lookup_init,
                          lookup_init_seconds: ph.lookup_init, checksum: got, map_slots: slots },
        ph))
}

/// Table parse, ProductIndex::load checks, typed decode, validate and key index -- the same
/// steps and messages as the reference loader, on the not-yet-verified buffer.
fn speculative(path: &str, raw: &[u8], candidate: bool, ph: &mut LoadPhases)
    -> Result<ProductIndex, String> {
    use rayon::prelude::*;
    let t = Instant::now();
    let tab = if candidate { parse(path, raw, &SYNCMER_PRODUCT_ARRAYS)? }
              else { parse(path, raw, &PRODUCT_ARRAYS)? };
    let r_scalars = &tab.scalars;
    if r_scalars.len() != 7 {
        return Err(format!("{}: {} scalars, expected 7", path, r_scalars.len()));
    }
    let (k, g, c, thresh, n_genomes, ngn, nrn) =
        (r_scalars[0] as usize, r_scalars[1] as usize, r_scalars[2] as usize, r_scalars[3],
         r_scalars[4] as usize, r_scalars[5] as usize, r_scalars[6] as usize);
    if k == 0 || k > 32 || g == 0 {
        return Err(format!("{}: k={} g={} is not a usable configuration (k must be 1..=32, g positive)", path, k, g));
    }
    if thresh != threshold(c as u64) {
        return Err(format!("{}: stored threshold {} does not match c={}", path, thresh, c));
    }
    if tab.strings.len() != ngn + nrn || ngn != n_genomes {
        return Err(format!("{}: {} names for {} genomes and {} records",
                           path, tab.strings.len(), ngn, nrn));
    }
    // dtype checks in the reference's array order, BEFORE any decode result is used
    let order: [(&str, Dt); 14] = [
        ("sel_keys", Dt::U64), ("sel_csr", Dt::U32), ("aoff", Dt::I64), ("aidx", Dt::U32),
        ("anchor_seg", Dt::I64), ("st", Dt::I64), ("en", Dt::I64), ("seg_rec", Dt::I64),
        ("dom_physical", Dt::I64), ("dom_bases", Dt::I64), ("dom_starts", Dt::I64),
        ("dom_anchors", Dt::I64), ("rec_len", Dt::I64), ("rec_genome", Dt::I64)];
    let mut pick: Vec<&[u8]> = Vec::new();
    for (name, want) in order {
        let a = tab.arrays.iter().find(|a| a.0 == name)
            .ok_or_else(|| format!("array {} is missing", name))?;
        if a.1 != want {
            return Err(format!("array {} is {:?}, expected {:?}", name, a.1, want));
        }
        pick.push(a.2);
    }
    // candidate: the reference decodes `selection` (u64) after the 14 arrays
    let sel_bytes = if candidate {
        let a = tab.arrays.iter().find(|a| a.0 == "selection")
            .ok_or_else(|| "array selection is missing".to_string())?;
        if a.1 != Dt::U64 {
            return Err(format!("array selection is {:?}, expected {:?}", a.1, Dt::U64));
        }
        Some(a.2)
    } else {
        None
    };
    let cols: Vec<Col> = (0..14usize).into_par_iter().map(|i| match order[i].1 {
        Dt::U64 => Col::U64(decode::<u64, 8>(pick[i], u64::from_le_bytes)),
        Dt::U32 => Col::U32(decode::<u32, 4>(pick[i], u32::from_le_bytes)),
        _ => Col::I64(decode::<i64, 8>(pick[i], i64::from_le_bytes)),
    }).collect();
    let mut it = cols.into_iter();
    let mut u64s = || match it.next() { Some(Col::U64(v)) => v, _ => unreachable!() };
    let sel_keys = u64s();
    let mut rest: Vec<Col> = it.collect();
    let mut nx = rest.drain(..);
    macro_rules! nxt {
        (U32) => { match nx.next() { Some(Col::U32(v)) => v, _ => unreachable!() } };
        (I64) => { match nx.next() { Some(Col::I64(v)) => v, _ => unreachable!() } };
    }
    let mut idx = ProductIndex {
        k, g, c, thresh, n_genomes, sel_keys,
        sel_csr: nxt!(U32), aoff: nxt!(I64), aidx: nxt!(U32), anchor_seg: nxt!(I64),
        st: nxt!(I64), en: nxt!(I64), seg_rec: nxt!(I64), dom_physical: nxt!(I64),
        dom_bases: nxt!(I64), dom_starts: nxt!(I64), dom_anchors: nxt!(I64),
        rec_len: nxt!(I64), rec_genome: nxt!(I64),
        genome_names: tab.strings[..ngn].to_vec(), rec_names: tab.strings[ngn..].to_vec(),
        meta_json: tab.meta.clone(), map: KeyIndex::build(&[], &[]),
        sel: crate::selection::Selection::shipped(k),
    };
    drop(nx);
    if let Some(b) = sel_bytes {
        let v = decode::<u64, 8>(b, u64::from_le_bytes);
        let sel = crate::selection::Selection::from_array(&v)
            .map_err(|e| format!("{}: {}", path, e))?;
        if sel.is_shipped() || sel.k != k {
            return Err(format!("{}: candidate product carries an inconsistent selection", path));
        }
        idx.sel = sel;
    }
    ph.parse_and_decode = t.elapsed().as_secs_f64();
    let tv = Instant::now();
    idx.validate(path)?;
    ph.validate = tv.elapsed().as_secs_f64();
    let tm = Instant::now();
    idx.map = KeyIndex::build(&idx.sel_keys, &idx.sel_csr);
    ph.lookup_init = tm.elapsed().as_secs_f64();
    Ok(idx)
}
