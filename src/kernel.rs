//! The k-mer kernel: encode -> rolling canonical k-mer -> splitmix -> threshold.
//!
//! Ported from the shared `rustcore/src/kernel.rs`, which is left untouched. The recurrence,
//! the LUT and `splitmix` are bit-identical to it and to the Python oracle's
//! `sub_ref.dense_hash` (which calls the installed `rustcore.hash_track`). That identity is
//! the whole basis of the oracle comparison, so it is asserted in the fixtures rather than
//! assumed here.
//!
//! What is NEW relative to the shared kernel, and why:
//!
//!   * `forward_codes` -- the 2-bit forward encoding of every window, which the shared crate
//!     does not expose. The Python index persists it for T2 orientation, so the compiled
//!     index has to produce it or the exported index would be incomplete.
//!   * `run_local_minima` -- landmark selection restricted to a maximal run of VALID k-mer
//!     starts (I6). The shared `local_minima` runs over the whole track, where MASK64
//!     sentinels tie as minima and manufacture cuts inside ambiguous sequence.

/// splitmix64, bit-identical to the installed `rustcore` and to the Python oracle.
#[inline(always)]
pub fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub const MASK64: u64 = u64::MAX;

const fn make_lut() -> [i8; 256] {
    let mut t = [-1i8; 256];
    t[b'A' as usize] = 0;
    t[b'a' as usize] = 0;
    t[b'C' as usize] = 1;
    t[b'c' as usize] = 1;
    t[b'G' as usize] = 2;
    t[b'g' as usize] = 2;
    t[b'T' as usize] = 3;
    t[b't' as usize] = 3;
    t
}

/// ASCII -> 2-bit code, -1 for anything that is not ACGT. Case-insensitive (R10e).
pub static LUT: [i8; 256] = make_lut();

#[inline(always)]
pub fn kmask(k: usize) -> u64 {
    if k >= 32 {
        u64::MAX
    } else {
        (1u64 << (2 * k)) - 1
    }
}

/// Every position's canonical-k-mer hash, `MASK64` where the window is invalid.
pub fn hash_track(seq: &[u8], k: usize) -> Vec<u64> {
    let n = seq.len();
    if k == 0 || n < k {
        return Vec::new();
    }
    let nk = n - k + 1;
    let mask = kmask(k);
    let sh = 2 * (k - 1);
    let mut out = Vec::with_capacity(nk);
    let (mut f, mut r, mut run) = (0u64, 0u64, 0usize);
    for i in 0..n {
        let c = LUT[seq[i] as usize];
        let ok = c >= 0;
        let cs = if ok { c as u64 } else { 0 };
        f = ((f << 2) | cs) & mask;
        r = (r >> 2) | ((3 - cs) << sh);
        run = if ok { run + 1 } else { 0 };
        if i + 1 >= k {
            out.push(if run >= k {
                splitmix(if f <= r { f } else { r })
            } else {
                MASK64
            });
        }
    }
    out
}

/// Per k-mer START: are all k bases of the window ACGT (R10c + R10d).
pub fn valid_starts(seq: &[u8], k: usize) -> Vec<bool> {
    let n = seq.len();
    if k == 0 || n < k {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - k + 1);
    let mut run = 0usize;
    for i in 0..n {
        run = if LUT[seq[i] as usize] >= 0 { run + 1 } else { 0 };
        if i + 1 >= k {
            out.push(run >= k);
        }
    }
    out
}

/// Forward 2-bit encoding of every window; `MASK64` where the window is invalid.
///
/// Matches `sub_ref.forward_code`, including the `k > 32` degenerate case where the oracle
/// returns all zeros because a 2-bit code no longer fits in a u64.
pub fn forward_codes(seq: &[u8], k: usize) -> Vec<u64> {
    let n = seq.len();
    if k == 0 || n < k {
        return Vec::new();
    }
    let nk = n - k + 1;
    if k > 32 {
        return vec![0u64; nk];        // a 2-bit code no longer fits a u64; oracle returns zeros
    }
    let mask = kmask(k);
    let mut out = Vec::with_capacity(nk);
    let (mut f, mut run) = (0u64, 0usize);
    for i in 0..n {
        let c = LUT[seq[i] as usize];
        let ok = c >= 0;
        let cs = if ok { c as u64 } else { 0 };
        f = ((f << 2) | cs) & mask;
        run = if ok { run + 1 } else { 0 };
        if i + 1 >= k {
            out.push(if run >= k { f } else { MASK64 });
        }
    }
    out
}

/// One fused pass over a read: emit `(start, canonical_kmer)` for every SELECTED valid window.
///
/// Selection is the strict `hash < thresh`, so invalid windows are excluded here rather than
/// marked -- they carry `MASK64`, which is never below a threshold this code can produce.
/// Emission is in ascending start order and duplicates are NOT collapsed: a repeated
/// occurrence of the same key is a separate observation (S3).
#[inline]
pub fn scan_select<F: FnMut(usize, u64)>(seq: &[u8], k: usize, thresh: u64, mut emit: F) {
    let n = seq.len();
    if k == 0 || n < k {
        return;
    }
    let mask = kmask(k);
    let sh = 2 * (k - 1);
    let (mut f, mut r, mut run) = (0u64, 0u64, 0usize);
    for i in 0..n {
        let c = LUT[seq[i] as usize];
        let ok = c >= 0;
        let cs = if ok { c as u64 } else { 0 };
        f = ((f << 2) | cs) & mask;
        r = (r >> 2) | ((3 - cs) << sh);
        run = if ok { run + 1 } else { 0 };
        if i + 1 >= k && run >= k {
            let canon = if f <= r { f } else { r };
            let h = splitmix(canon);
            if h < thresh {
                emit(i + 1 - k, h);
            }
        }
    }
}

/// Landmark positions inside ONE maximal run of valid starts (I6).
///
/// `h` is the run's slice of the hash track, all entries valid by construction. Fires at every
/// `m` in `[r, ln-1-r]` whose value equals the minimum over the closed neighbourhood of radius
/// `r`; ties all fire, matching `sub_ref.landmark_cuts` and the shared `local_minima`. A run of
/// length `<= 2r` contributes NOTHING: the window is never shortened near a gap to manufacture
/// a cut.
pub fn run_local_minima(h: &[u64], r: usize) -> Vec<usize> {
    let nk = h.len();
    let mut out = Vec::new();
    if nk == 0 || nk <= 2 * r {
        return out;
    }
    let mut dq: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut next = 0usize;
    for m in r..=(nk - 1 - r) {
        while next <= m + r {
            while let Some(&b) = dq.back() {
                if h[b] >= h[next] {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(next);
            next += 1;
        }
        while *dq.front().unwrap() + r < m {
            dq.pop_front();
        }
        if h[*dq.front().unwrap()] == h[m] {
            out.push(m);
        }
    }
    out
}

/// The FracMinHash threshold for a sampling rate `c`.
///
/// `c <= 1` means "every k-mer is selected": 2^64 overflows a u64, so it is clamped to the
/// maximum. Clamping rather than special-casing keeps invalid windows excluded, because they
/// carry `MASK64` and the comparison is a strict `<`.
pub fn threshold(c: u64) -> u64 {
    if c <= 1 {
        u64::MAX
    } else {
        ((1u128 << 64) / c as u128) as u64
    }
}
