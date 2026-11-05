//! Shared container format for both persisted artifacts.
//!
//! One writer and one reader, so the product index and the dense truth/debug export cannot
//! drift apart in how they are framed, validated or checksummed. Layout:
//!
//! ```text
//!   magic(12)  kind(u8)  version(u32)  payload_len(u64)
//!   payload:   n_scalars(u64) [u64]*          -- a BINARY header, never parsed out of JSON
//!              meta_len(u64)  meta(utf8)      -- opaque here; Python owns its content
//!              n_strings(u64) [len(u64) bytes]*
//!              n_arrays(u64)  [namelen name dtype len data]*
//!   fnv1a(u64) over every byte above
//! ```
//!
//! The scalars are binary because a loader that recovers `k` by scanning a metadata string for
//! `"k":` finds whichever `k` appears first -- which, the day someone nests a field called `k`,
//! is a wrong index that still loads.

use std::io::{Read, Write};

pub const MAGIC: &[u8; 12] = b"SUBCOREIDX2\n";
pub const VERSION: u32 = 2;
pub const KIND_PRODUCT: u8 = b'P';
pub const KIND_DENSE: u8 = b'D';

/// FNV-1a over the payload. A checksum, not a security primitive: it is here so a truncated or
/// byte-flipped index is REJECTED instead of decoded into plausible wrong coordinates. No
/// hashing dependency on purpose -- an offline build must not fail for a checksum.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dt {
    U64,
    I64,
    I32,
    U32,
    U8,
}

impl Dt {
    fn code(self) -> u8 {
        match self {
            Dt::U64 => b'Q',
            Dt::I64 => b'q',
            Dt::I32 => b'i',
            Dt::U32 => b'I',
            Dt::U8 => b'B',
        }
    }
    pub(crate) fn from_code(c: u8) -> Option<Dt> {   // method1_rust_cli: pub(crate) for load.rs
        Some(match c {
            b'Q' => Dt::U64,
            b'q' => Dt::I64,
            b'i' => Dt::I32,
            b'I' => Dt::U32,
            b'B' => Dt::U8,
            _ => return None,
        })
    }
    pub fn width(self) -> usize {
        match self {
            Dt::U64 | Dt::I64 => 8,
            Dt::I32 | Dt::U32 => 4,
            Dt::U8 => 1,
        }
    }
}

#[derive(Default)]
pub struct Writer {
    scalars: Vec<u64>,
    meta: String,
    strings: Vec<String>,
    arrays: Vec<(String, Dt, Vec<u8>, usize)>,
}

fn le(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

impl Writer {
    pub fn new() -> Writer {
        Writer::default()
    }
    pub fn scalar(&mut self, v: u64) -> &mut Self {
        self.scalars.push(v);
        self
    }
    pub fn meta(&mut self, s: &str) -> &mut Self {
        self.meta = s.to_string();
        self
    }
    pub fn strings(&mut self, s: &[String]) -> &mut Self {
        self.strings.extend_from_slice(s);
        self
    }
    pub fn u64s(&mut self, n: &str, v: &[u64]) -> &mut Self {
        let mut b = Vec::with_capacity(v.len() * 8);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        self.arrays.push((n.to_string(), Dt::U64, b, v.len()));
        self
    }
    pub fn i64s(&mut self, n: &str, v: &[i64]) -> &mut Self {
        let mut b = Vec::with_capacity(v.len() * 8);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        self.arrays.push((n.to_string(), Dt::I64, b, v.len()));
        self
    }
    pub fn i32s(&mut self, n: &str, v: &[i32]) -> &mut Self {
        let mut b = Vec::with_capacity(v.len() * 4);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        self.arrays.push((n.to_string(), Dt::I32, b, v.len()));
        self
    }
    pub fn u32s(&mut self, n: &str, v: &[u32]) -> &mut Self {
        let mut b = Vec::with_capacity(v.len() * 4);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        self.arrays.push((n.to_string(), Dt::U32, b, v.len()));
        self
    }
    pub fn u8s(&mut self, n: &str, v: &[u8]) -> &mut Self {
        self.arrays.push((n.to_string(), Dt::U8, v.to_vec(), v.len()));
        self
    }

    pub fn write(&self, path: &str, kind: u8) -> std::io::Result<u64> {
        let mut p: Vec<u8> = Vec::new();
        p.extend_from_slice(&le(self.scalars.len() as u64));
        for &s in &self.scalars {
            p.extend_from_slice(&le(s));
        }
        p.extend_from_slice(&le(self.meta.len() as u64));
        p.extend_from_slice(self.meta.as_bytes());
        p.extend_from_slice(&le(self.strings.len() as u64));
        for s in &self.strings {
            p.extend_from_slice(&le(s.len() as u64));
            p.extend_from_slice(s.as_bytes());
        }
        p.extend_from_slice(&le(self.arrays.len() as u64));
        for (n, dt, data, ln) in &self.arrays {
            p.push(n.len() as u8);
            p.extend_from_slice(n.as_bytes());
            p.push(dt.code());
            p.extend_from_slice(&le(*ln as u64));
            p.extend_from_slice(data);
        }
        let mut out: Vec<u8> = Vec::with_capacity(p.len() + 64);
        out.extend_from_slice(MAGIC);
        out.push(kind);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&le(p.len() as u64));
        out.extend_from_slice(&p);
        let ck = fnv1a(&out);
        out.extend_from_slice(&le(ck));
        let n = out.len() as u64;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&out)?;
        f.sync_all()?;
        Ok(n)
    }
}

pub struct Reader {
    pub kind: u8,
    pub version: u32,
    pub scalars: Vec<u64>,
    pub meta: String,
    pub strings: Vec<String>,
    pub arrays: std::collections::HashMap<String, (Dt, Vec<u8>, usize)>,
    pub bytes: u64,
    pub checksum: u64,
}

impl Reader {
    /// Read and FULLY validate the container. Every failure is a rejection; nothing is
    /// repaired, and nothing falls back to rebuilding from a FASTA.
    pub fn open(path: &str, want_kind: u8, allowed: &[&str]) -> Result<Reader, String> {
        let mut raw = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("{}: {}", path, e))?
            .read_to_end(&mut raw)
            .map_err(|e| format!("{}: {}", path, e))?;
        let hdr = 12 + 1 + 4 + 8;
        if raw.len() < hdr + 8 {
            return Err(format!("{}: {} bytes is shorter than the header", path, raw.len()));
        }
        if &raw[..12] != MAGIC {
            return Err(format!("{}: wrong magic {:?}; not a subcore artifact",
                               path, String::from_utf8_lossy(&raw[..12])));
        }
        let kind = raw[12];
        if kind != want_kind {
            return Err(format!("{}: this is a {:?} artifact but a {:?} one was asked for",
                               path, kind as char, want_kind as char));
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
        let got = fnv1a(&raw[..raw.len() - 8]);
        if want != got {
            return Err(format!("{}: checksum {:016x} does not match the content {:016x}; the \
artifact is corrupt", path, want, got));
        }
        let end = raw.len() - 8;
        let mut p = hdr;
        let mut rd = |p: &mut usize| -> Result<u64, String> {
            if *p + 8 > end {
                return Err(format!("{}: payload ends inside a scalar", path));
            }
            let v = u64::from_le_bytes(raw[*p..*p + 8].try_into().unwrap());
            *p += 8;
            Ok(v)
        };
        let nsc = rd(&mut p)? as usize;
        let mut scalars = Vec::with_capacity(nsc);
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
        let mut strings = Vec::with_capacity(nstr);
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
        let mut arrays = std::collections::HashMap::new();
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
            if arrays.contains_key(&name) {
                return Err(format!("{}: array {:?} appears twice", path, name));
            }
            let nb = ln.checked_mul(dt.width())
                .ok_or_else(|| format!("{}: array {} declares an impossible length", path, name))?;
            if p + nb > end {
                return Err(format!("{}: array {} declares {} elements that do not fit",
                                   path, name, ln));
            }
            arrays.insert(name, (dt, raw[p..p + nb].to_vec(), ln));
            p += nb;
        }
        if p != end {
            return Err(format!("{}: {} bytes of unread payload remain", path, end - p));
        }
        for a in allowed {
            if !arrays.contains_key(*a) {
                return Err(format!("{}: required array {} is missing", path, a));
            }
        }
        Ok(Reader { kind, version, scalars, meta, strings, arrays,
                    bytes: raw.len() as u64, checksum: got })
    }

    fn get(&mut self, name: &str, dt: Dt) -> Result<(Vec<u8>, usize), String> {
        let (d, b, n) = self.arrays.remove(name)
            .ok_or_else(|| format!("array {} is missing", name))?;
        if d != dt {
            return Err(format!("array {} is {:?}, expected {:?}", name, d, dt));
        }
        Ok((b, n))
    }
    pub fn u64s(&mut self, name: &str) -> Result<Vec<u64>, String> {
        let (b, n) = self.get(name, Dt::U64)?;
        Ok((0..n).map(|i| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap())).collect())
    }
    pub fn i64s(&mut self, name: &str) -> Result<Vec<i64>, String> {
        let (b, n) = self.get(name, Dt::I64)?;
        Ok((0..n).map(|i| i64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap())).collect())
    }
    pub fn i32s(&mut self, name: &str) -> Result<Vec<i32>, String> {
        let (b, n) = self.get(name, Dt::I32)?;
        Ok((0..n).map(|i| i32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())).collect())
    }
    pub fn u32s(&mut self, name: &str) -> Result<Vec<u32>, String> {
        let (b, n) = self.get(name, Dt::U32)?;
        Ok((0..n).map(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())).collect())
    }
    pub fn u8s(&mut self, name: &str) -> Result<Vec<u8>, String> {
        let (b, _n) = self.get(name, Dt::U8)?;
        Ok(b)
    }
}
