//! Minimal output, matching `minimal_timing.write_output`:
//!   segments.npz  numpy `savez` layout: STORED (uncompressed) zip members `<name>.npy` in the
//!                 same order, each a version-1.0 .npy (numpy's exact header text and 64-byte
//!                 padding) followed by the little-endian array bytes. Member bytes are identical
//!                 to numpy's; zip container metadata (timestamps, zip64 extras) may differ.
//!   record-map.json  copied byte for byte.
//!   report.json   the same keys and values (sorted keys, indent 2), with provenance fields
//!                 describing THIS binary.
use crate::reduce::J;
use std::io::Write;

pub enum Arr<'a> {
    I64(&'a [i64]),
    F64(&'a [f64]),
}

fn npy_header(arr: &Arr) -> Vec<u8> {
    let (descr, n) = match arr {
        Arr::I64(v) => ("<i8", v.len()),
        Arr::F64(v) => ("<f8", v.len()),
    };
    let h = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({},), }}", descr, n);
    let hlen = h.len() + 1;
    let pad = 64 - ((6 + 2 + 2 + hlen) % 64);
    let mut header = h.into_bytes();
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out
}

fn arr_len(arr: &Arr) -> usize {
    match arr { Arr::I64(v) => v.len(), Arr::F64(v) => v.len() }
}

fn arr_chunks<'a>(arr: &'a Arr<'a>) -> impl Iterator<Item = [u8; 8]> + 'a {
    let i: Box<dyn Iterator<Item = [u8; 8]> + 'a> = match arr {
        Arr::I64(v) => Box::new(v.iter().map(|x| x.to_le_bytes())),
        Arr::F64(v) => Box::new(v.iter().map(|x| x.to_le_bytes())),
    };
    i
}

struct Central {
    name: String,
    crc: u32,
    size: u64,
    offset: u64,
    zip64_size: bool,
    zip64_offset: bool,
}

/// A STORED zip like `np.savez`, with ZIP64 for large members/archives. Array payloads are
/// streamed rather than duplicated in a multi-GiB temporary Vec. The archive is fsynced and
/// atomically renamed into place, so a write failure cannot leave the requested path partial.
pub fn write_npz(path: &std::path::Path, members: &[(&str, Arr)]) -> Result<(), String> {
    if path.exists() { return Err(format!("{}: exists", path.display())); }
    let tmp = path.with_extension(format!("npz.tmp-{}", std::process::id()));
    let f = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp)
        .map_err(|e| format!("{}: {}", tmp.display(), e))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    let mut central: Vec<Central> = Vec::new();
    let mut off: u64 = 0;
    let (time, date) = (0u16, 33u16); // 1980-01-01 00:00 (container metadata only)
    let err = |e: std::io::Error| format!("{}: {}", tmp.display(), e);
    let result = (|| -> Result<(), String> {
      for (name, arr) in members {
        let header = npy_header(arr);
        let size = header.len() as u64 + arr_len(arr) as u64 * 8;
        let fname = format!("{}.npy", name);
        let zip64_size = size >= u32::MAX as u64;
        let zip64_offset = off >= u32::MAX as u64;
        let mut extra = Vec::new();
        if zip64_size {
            extra.extend_from_slice(&1u16.to_le_bytes());
            extra.extend_from_slice(&16u16.to_le_bytes());
            extra.extend_from_slice(&size.to_le_bytes());
            extra.extend_from_slice(&size.to_le_bytes());
        }
        // Bit 3 permits one-pass streaming: CRC and sizes follow in a data descriptor.
        let version = if zip64_size { 45u16 } else { 20u16 };
        let mut crc = flate2::Crc::new();
        let mut lh = Vec::new();
        lh.extend_from_slice(&0x04034b50u32.to_le_bytes());
        lh.extend_from_slice(&version.to_le_bytes());
        lh.extend_from_slice(&8u16.to_le_bytes());
        lh.extend_from_slice(&0u16.to_le_bytes());
        lh.extend_from_slice(&time.to_le_bytes());
        lh.extend_from_slice(&date.to_le_bytes());
        lh.extend_from_slice(&0u32.to_le_bytes());
        lh.extend_from_slice(&(if zip64_size { u32::MAX } else { 0 }).to_le_bytes());
        lh.extend_from_slice(&(if zip64_size { u32::MAX } else { 0 }).to_le_bytes());
        lh.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        lh.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        lh.extend_from_slice(fname.as_bytes());
        lh.extend_from_slice(&extra);
        w.write_all(&lh).map_err(err)?;
        w.write_all(&header).map_err(err)?;
        crc.update(&header);
        let mut buf = Vec::with_capacity(1 << 20);
        for bytes in arr_chunks(arr) {
            buf.extend_from_slice(&bytes);
            if buf.len() == buf.capacity() {
                w.write_all(&buf).map_err(err)?;
                crc.update(&buf);
                buf.clear();
            }
        }
        if !buf.is_empty() { w.write_all(&buf).map_err(err)?; crc.update(&buf); }
        let crc = crc.sum();
        let mut dd = Vec::new();
        dd.extend_from_slice(&0x08074b50u32.to_le_bytes());
        dd.extend_from_slice(&crc.to_le_bytes());
        if zip64_size {
            dd.extend_from_slice(&size.to_le_bytes()); dd.extend_from_slice(&size.to_le_bytes());
        } else {
            dd.extend_from_slice(&(size as u32).to_le_bytes());
            dd.extend_from_slice(&(size as u32).to_le_bytes());
        }
        w.write_all(&dd).map_err(err)?;
        central.push(Central { name: fname, crc, size, offset: off, zip64_size, zip64_offset });
        off += (lh.len() as u64) + size + (dd.len() as u64);
      }
      let central_off = off;
      for c in &central {
        let mut extra = Vec::new();
        if c.zip64_size || c.zip64_offset {
            let mut body = Vec::new();
            if c.zip64_size { body.extend_from_slice(&c.size.to_le_bytes()); body.extend_from_slice(&c.size.to_le_bytes()); }
            if c.zip64_offset { body.extend_from_slice(&c.offset.to_le_bytes()); }
            extra.extend_from_slice(&1u16.to_le_bytes());
            extra.extend_from_slice(&(body.len() as u16).to_le_bytes());
            extra.extend_from_slice(&body);
        }
        let mut ch = Vec::new();
        ch.extend_from_slice(&0x02014b50u32.to_le_bytes());
        let version = if c.zip64_size || c.zip64_offset { 45u16 } else { 20u16 };
        ch.extend_from_slice(&((3u16 << 8) | version).to_le_bytes());
        ch.extend_from_slice(&version.to_le_bytes());
        ch.extend_from_slice(&8u16.to_le_bytes());
        ch.extend_from_slice(&0u16.to_le_bytes());
        ch.extend_from_slice(&time.to_le_bytes());
        ch.extend_from_slice(&date.to_le_bytes());
        ch.extend_from_slice(&c.crc.to_le_bytes());
        ch.extend_from_slice(&(if c.zip64_size { u32::MAX } else { c.size as u32 }).to_le_bytes());
        ch.extend_from_slice(&(if c.zip64_size { u32::MAX } else { c.size as u32 }).to_le_bytes());
        ch.extend_from_slice(&(c.name.len() as u16).to_le_bytes());
        ch.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        ch.extend_from_slice(&[0u8; 6]); // comment len, disk start, internal attr
        ch.extend_from_slice(&(0o600u32 << 16).to_le_bytes());
        ch.extend_from_slice(&(if c.zip64_offset { u32::MAX } else { c.offset as u32 }).to_le_bytes());
        ch.extend_from_slice(c.name.as_bytes()); ch.extend_from_slice(&extra);
        w.write_all(&ch).map_err(err)?; off += ch.len() as u64;
      }
    let central_size = off - central_off;
    let archive_zip64 = central_off >= u32::MAX as u64 || central_size >= u32::MAX as u64 || central.len() >= u16::MAX as usize;
    if archive_zip64 {
        let z64off = off;
        let mut z = Vec::new();
        z.extend_from_slice(&0x06064b50u32.to_le_bytes()); z.extend_from_slice(&44u64.to_le_bytes());
        z.extend_from_slice(&((3u16 << 8) | 45).to_le_bytes()); z.extend_from_slice(&45u16.to_le_bytes());
        z.extend_from_slice(&[0u8; 8]);
        z.extend_from_slice(&(central.len() as u64).to_le_bytes()); z.extend_from_slice(&(central.len() as u64).to_le_bytes());
        z.extend_from_slice(&central_size.to_le_bytes()); z.extend_from_slice(&central_off.to_le_bytes());
        w.write_all(&z).map_err(err)?;
        let mut loc = Vec::new(); loc.extend_from_slice(&0x07064b50u32.to_le_bytes());
        loc.extend_from_slice(&0u32.to_le_bytes()); loc.extend_from_slice(&z64off.to_le_bytes()); loc.extend_from_slice(&1u32.to_le_bytes());
        w.write_all(&loc).map_err(err)?;
    }
    let mut end = Vec::new();
    end.extend_from_slice(&0x06054b50u32.to_le_bytes());
    end.extend_from_slice(&[0u8; 4]);
    let count = if archive_zip64 { u16::MAX } else { central.len() as u16 };
    end.extend_from_slice(&count.to_le_bytes()); end.extend_from_slice(&count.to_le_bytes());
    end.extend_from_slice(&(if archive_zip64 { u32::MAX } else { central_size as u32 }).to_le_bytes());
    end.extend_from_slice(&(if archive_zip64 { u32::MAX } else { central_off as u32 }).to_le_bytes());
    end.extend_from_slice(&0u16.to_le_bytes());
    w.write_all(&end).map_err(err)?;
    w.flush().map_err(err)?;
    w.get_ref().sync_all().map_err(err)?;
    std::fs::rename(&tmp, path).map_err(err)?;
    Ok(())
    })();
    drop(w);
    if result.is_err() { let _ = std::fs::remove_file(&tmp); }
    result
}

/// Python `repr(float)` (shortest round-trip digits, fixed for -4 <= exp < 16).
pub fn py_float(x: f64) -> Result<String, String> {
    if !x.is_finite() {
        return Err(format!("non-finite float {} in report (allow_nan=False)", x));
    }
    if x == 0.0 {
        return Ok(if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() });
    }
    let e = format!("{:e}", x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    Ok(if (-4..16).contains(&exp) {
        let nd = digits.len() as i32;
        if exp >= nd - 1 {
            format!("{}{}{}.0", sign, digits, "0".repeat((exp - nd + 1) as usize))
        } else if exp >= 0 {
            let (a, b) = digits.split_at((exp + 1) as usize);
            format!("{}{}.{}", sign, a, b)
        } else {
            format!("{}0.{}{}", sign, "0".repeat((-exp - 1) as usize), digits)
        }
    } else {
        let (a, b) = digits.split_at(1);
        let m = if b.is_empty() { a.to_string() } else { format!("{}.{}", a, b) };
        format!("{}{}e{}{:02}", sign, m, if exp < 0 { '-' } else { '+' }, exp.abs())
    })
}

fn esc(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dump(value, sort_keys=True, indent=2, allow_nan=False)` + newline.
pub fn json(v: &J, ind: usize, out: &mut String) -> Result<(), String> {
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(i) => out.push_str(&i.to_string()),
        J::Float(f) => out.push_str(&py_float(*f)?),
        J::Str(s) => esc(s, out),
        J::List(xs) => {
            if xs.is_empty() {
                out.push_str("[]");
            } else {
                out.push('[');
                for (i, x) in xs.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    out.push_str(&" ".repeat(ind + 2));
                    json(x, ind + 2, out)?;
                }
                out.push('\n');
                out.push_str(&" ".repeat(ind));
                out.push(']');
            }
        }
        J::Obj(kv) => {
            if kv.is_empty() {
                out.push_str("{}");
            } else {
                let mut kv: Vec<&(String, J)> = kv.iter().collect();
                kv.sort_by(|a, b| a.0.cmp(&b.0));
                out.push('{');
                for (i, (k, x)) in kv.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    out.push_str(&" ".repeat(ind + 2));
                    esc(k, out);
                    out.push_str(": ");
                    json(x, ind + 2, out)?;
                }
                out.push('\n');
                out.push_str(&" ".repeat(ind));
                out.push('}');
            }
        }
    }
    Ok(())
}

pub fn write_json(path: &std::path::Path, v: &J) -> Result<(), String> {
    let mut s = String::new();
    json(v, 0, &mut s)?;
    s.push('\n');
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|e| format!("{}: {}", path.display(), e))?;
    f.write_all(s.as_bytes()).map_err(|e| format!("{}: {}", path.display(), e))
}

pub fn fsync(path: &std::path::Path) -> Result<(), String> {
    std::fs::File::open(path).and_then(|f| f.sync_all())
        .map_err(|e| format!("{}: {}", path.display(), e))
}
