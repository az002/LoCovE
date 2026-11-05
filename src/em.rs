//! OPT-IN local depth attribution from the EXISTING scaled-Poisson abundance EM (2026-10-05).
//!
//! Model, initialization and stopping rule are those of benchmarks/typing/models.py (`fit` with
//! arm scaled_poisson_em), ported, not changed:
//!   * columns = genomes; identical columns form one GROUP (never split); rank and per-group
//!     identifiability (column outside the span of the others) of the grouped matrix;
//!   * private initialization: per genome, mean raw count of distinct anchor keys present in
//!     that genome only with multiplicity 1 (floored at 0; NaN if none); group start =
//!     max(nansum(members), 1e-6) where the group has exposure, else 0;
//!   * EM on identical-row-compressed sufficient statistics: lambda = base.theta;
//!     theta <- theta * (y/lambda).base / exposure; stop when max|delta| < tolerance (1e-5
//!     approved; 1e-9 the development comparison), at most 100000 iterations; positive counts
//!     without modeled support are an error;
//!   * reported per-genome theta: NaN unless the genome is a single-member identifiable group;
//!     group theta: NaN unless identifiable.
//! Rank/identifiability are the EXACT RATIONAL ones of the integer exposure matrix, and each is
//! PROVEN by a certificate (`certify`): a nonzero minor mod p for the lower bound, and n - r
//! integer null vectors verified A w = 0 in exact arithmetic for the upper bound. Without a
//! complete certificate the run refuses. (Correction 2026-10-05: the first candidate,
//! 9882fe3c1b93699b, only required two modular ranks to agree, which is not a proof -- both
//! primes can divide the same minor. See GATE.md.) numpy uses an SVD tolerance; the exact
//! rank can differ from it on near-singular matrices.
//!
//! ATTRIBUTION at the fitted theta, on the ORIGINAL segment rows j (no compression):
//!   expected e_jG = A_{j,rep(G)} theta_G,  lambda_j = sum_G e_jG,
//!   responsibility r_jG = e_jG / lambda_j  (a fraction; NaN for non-identifiable groups and
//!   where lambda_j = 0),  attributed support Z_jG = Y_j r_jG  (counts; exactly 0 where Y_j = 0:
//!   redistribution never creates support at an observed zero).
//! Own-genome local estimate: Z_{j,own group} / A_{j,own} with A_{j,own} = n_j (exposure file
//! rule), i.e. pooled raw-count total * r / n_j, then the SAME sample scalars as the pooled
//! variant (edge_global, 1 - novelty), applied once. Defined for the raw-count variants only
//! (raw_mean, geom_global, geom_global_novelty); the per-read variants use the read-weighted
//! channel, which the raw-count EM does not model (blocker reported, nothing invented).
//! SOURCE-ATTRIBUTED and model-dependent; the pooled outputs are unchanged and kept alongside.

use crate::exposure::Exposures;
use crate::product::ProductIndex;
use crate::reduce::{obj, Sample, J};
use std::collections::HashMap;

pub const MAX_ITER: usize = 100000;
pub const TOLERANCES: [f64; 2] = [1e-5, 1e-9];
/// Primes for the modular eliminations (2^61 - 1 and the three largest primes below 2^62).
const PRIMES: [u64; 4] = [2305843009213693951, 4611686018427387847, 4611686018427387817,
                          4611686018427387787];
pub const NO_SUPPORT: &str = "positive observed counts without modeled reference support";

pub struct Structure {
    pub groups: Vec<Vec<usize>>,
    pub group_of: Vec<usize>,
    pub rank: usize,
    pub identifiable: Vec<bool>,
    pub certificate: Certificate,
}

pub struct EmFit {
    pub theta: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_max_delta: f64,
    pub compressed_rows: usize,
    pub original_rows: usize,
}

/// Sparse rows of a nonnegative integer matrix: row i = cols[off[i]..off[i+1]] (ascending).
pub struct Rows {
    pub off: Vec<usize>,
    pub col: Vec<u32>,
    pub val: Vec<u64>,
    pub ncols: usize,
}

impl Rows {
    pub fn row(&self, i: usize) -> (&[u32], &[u64]) {
        (&self.col[self.off[i]..self.off[i + 1]], &self.val[self.off[i]..self.off[i + 1]])
    }
    pub fn nrows(&self) -> usize {
        self.off.len() - 1
    }
}

fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}
fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}

/// Full RREF of the Gram matrix over GF(p): (pivot columns, reduced rows).
fn rref_mod(gram: &[Vec<u128>], p: u64) -> (Vec<usize>, Vec<Vec<u64>>) {
    let n = gram.len();
    let mut m: Vec<Vec<u64>> = gram.iter().map(|r| r.iter().map(|&x| (x % p as u128) as u64)
                                                   .collect()).collect();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for col in 0..n {
        let Some(i) = (r..n).find(|&i| m[i][col] != 0) else { continue };
        m.swap(r, i);
        let inv = powmod(m[r][col], p - 2, p);
        for x in m[r].iter_mut() {
            *x = mulmod(*x, inv, p);
        }
        for i in 0..n {
            if i != r && m[i][col] != 0 {
                let f = m[i][col];
                for c in 0..n {
                    let sub = mulmod(f, m[r][c], p);
                    m[i][c] = (m[i][c] + p - sub) % p;
                }
            }
        }
        pivots.push(col);
        r += 1;
    }
    (pivots, m)
}

/// Rational reconstruction (Wang): the unique a/b with |a|, b <= sqrt(m/2) and a = u b (mod m),
/// if it exists.
fn ratrec(u: u128, m: u128) -> Option<(i128, i128)> {
    let bound = isqrt(m / 2) as i128;
    let (mut r0, mut r1) = (m as i128, u as i128);
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 > bound {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if t1 == 0 || t1.abs() > bound {
        return None;
    }
    let (a, b) = if t1 < 0 { (-r1, -t1) } else { (r1, t1) };
    if gcd(a.unsigned_abs(), b as u128) != 1 {
        return None;
    }
    Some((a, b))
}

fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = (n as f64).sqrt() as u128;
    while x.checked_mul(x).map_or(true, |xx| xx > n) {
        x -= 1;
    }
    while (x + 1).checked_mul(x + 1).map_or(false, |xx| xx <= n) {
        x += 1;
    }
    x
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// How the exact rational rank of the grouped matrix was proven (see `certify`).
pub struct Certificate {
    pub primes_used: Vec<u64>,
    pub null_vectors: Vec<Vec<i128>>,
}

/// PROVEN rational rank and identifiability of the grouped integer matrix (columns =
/// `group_rep`, rows = `rows`), whose exact Gram matrix is `gram`.
///   lower bound: the Gram matrix has rank r over GF(p), so some r x r minor is nonzero mod p,
///     hence a nonzero integer, hence rank_Q >= r;
///   upper bound: for every free column of that RREF, the candidate null vector is lifted to Q
///     (CRT over up to two primes sharing the pivot set, then rational reconstruction), scaled
///     to integers and CHECKED A w = 0 in exact integer arithmetic on every original row. The
///     n - r verified vectors are independent (unit pattern on the free columns), so
///     rank_Q <= r and they span the rational null space.
///   identifiable(G) (models.structure: deleting G lowers the rank) <=> no null vector touches G.
/// Any failure (reconstruction, overflow, a vector that is not null) returns an error naming
/// the step; nothing is asserted without the certificate. Several primes guard against one
/// prime dividing a minor (an "unlucky" prime understates the rank; verification would refuse
/// it, the next prime avoids it).
fn certify(rows: &Rows, group_rep: &[usize], gram: &[Vec<u128>], primes: &[u64])
    -> Result<(usize, Vec<bool>, Certificate), String> {
    let n = gram.len();
    let red: Vec<(Vec<usize>, Vec<Vec<u64>>)> = primes.iter().map(|&p| rref_mod(gram, p)).collect();
    let rmax = red.iter().map(|(pv, _)| pv.len()).max().unwrap_or(0);
    let lead = red.iter().position(|(pv, _)| pv.len() == rmax).unwrap();
    let pivots = red[lead].0.clone();
    let use_p: Vec<usize> = (0..primes.len()).filter(|&i| red[i].0 == pivots).take(2).collect();
    let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
    let modulus: u128 = use_p.iter().map(|&i| primes[i] as u128).product();
    let col_pos: HashMap<usize, usize> = group_rep.iter().enumerate().map(|(gi, &s)| (s, gi))
        .collect();
    let mut nulls: Vec<Vec<i128>> = Vec::new();
    for &f in &free {
        // residue of entry pc of the null vector with v_f = 1: -m[row][f]
        let mut frac: Vec<(i128, i128)> = vec![(0, 1); n];
        frac[f] = (1, 1);
        for (row, &pc) in pivots.iter().enumerate() {
            let res: Vec<u128> = use_p.iter().map(|&i| {
                let p = primes[i];
                ((p - red[i].1[row][f] % p) % p) as u128
            }).collect();
            let x = if use_p.len() == 1 { res[0] } else {
                let (p0, p1) = (primes[use_p[0]] as u128, primes[use_p[1]] as u128);
                let inv = powmod((p0 % p1) as u64, (p1 - 2) as u64, p1 as u64) as u128;
                let k = ((res[1] + p1 - res[0] % p1) % p1) * inv % p1;
                res[0] + p0 * k
            };
            frac[pc] = ratrec(x, modulus).ok_or_else(|| format!("rank certificate: rational \
reconstruction failed for null vector of free group {} (entries exceed the {}-prime bound); \
structure NOT certified", f, use_p.len()))?;
        }
        let mut l: i128 = 1;
        for &(_, b) in &frac {
            let g = gcd(l as u128, b as u128) as i128;
            l = (l / g).checked_mul(b).ok_or("rank certificate: denominator overflow")?;
        }
        let w: Vec<i128> = frac.iter().map(|&(a, b)| a.checked_mul(l / b))
            .collect::<Option<_>>().ok_or("rank certificate: numerator overflow")?;
        for i in 0..rows.nrows() {
            let (c, v) = rows.row(i);
            let mut s: i128 = 0;
            for (&cc, &vv) in c.iter().zip(v) {
                if let Some(&gi) = col_pos.get(&(cc as usize)) {
                    s = (vv as i128).checked_mul(w[gi]).and_then(|t| s.checked_add(t))
                        .ok_or("rank certificate: overflow while verifying a null vector")?;
                }
            }
            if s != 0 {
                return Err(format!("rank certificate: candidate null vector for free group {} is \
not in the rational null space (row {}); modular rank understated the exact rank for every \
prime tried; structure NOT certified", f, i));
            }
        }
        nulls.push(w);
    }
    let ident: Vec<bool> = (0..n).map(|g| nulls.iter().all(|w| w[g] == 0)).collect();
    Ok((rmax, ident, Certificate {
        primes_used: use_p.iter().map(|&i| primes[i]).collect(), null_vectors: nulls }))
}

/// models.structure: groups of identical columns (in column order), exact rank and
/// identifiability of the grouped matrix. `rows` are the ORIGINAL segment rows.
pub fn structure(rows: &Rows) -> Result<Structure, String> {
    let g = rows.ncols;
    let mut cols: Vec<Vec<(u32, u64)>> = vec![Vec::new(); g];
    for i in 0..rows.nrows() {
        let (c, v) = rows.row(i);
        for (&cc, &vv) in c.iter().zip(v) {
            cols[cc as usize].push((i as u32, vv));
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut by_col: HashMap<&Vec<(u32, u64)>, usize> = HashMap::new();
    let mut group_of = vec![0usize; g];
    for s in 0..g {
        match by_col.get(&cols[s]) {
            Some(&gi) => {
                groups[gi].push(s);
                group_of[s] = gi;
            }
            None => {
                by_col.insert(&cols[s], groups.len());
                group_of[s] = groups.len();
                groups.push(vec![s]);
            }
        }
    }
    let ng = groups.len();
    let rep_group: HashMap<usize, usize> = groups.iter().enumerate().map(|(gi, m)| (m[0], gi))
        .collect();
    let mut gram = vec![vec![0u128; ng]; ng];
    for i in 0..rows.nrows() {
        let (c, v) = rows.row(i);
        let e: Vec<(usize, u128)> = c.iter().zip(v).filter_map(|(&cc, &vv)|
            rep_group.get(&(cc as usize)).map(|&gi| (gi, vv as u128))).collect();
        for &(a, x) in &e {
            for &(b, y) in &e {
                gram[a][b] = x.checked_mul(y).and_then(|t| gram[a][b].checked_add(t))
                    .ok_or("rank certificate: Gram matrix overflows 128 bits")?;
            }
        }
    }
    let reps: Vec<usize> = groups.iter().map(|m| m[0]).collect();
    let (rank, identifiable, certificate) = certify(rows, &reps, &gram, &PRIMES)?;
    Ok(Structure { groups, group_of, rank, identifiable, certificate })
}

/// models.private on distinct anchor keys (slots).
pub fn private_init(exp: &Exposures, slot_count: &[i64]) -> Vec<f64> {
    let g = exp.n_genomes;
    let (mut sum, mut cnt) = (vec![0i64; g], vec![0u64; g]);
    for s in 0..exp.n_slots {
        let (a, b) = (exp.key_off[s] as usize, exp.key_off[s + 1] as usize);
        if b - a == 1 && exp.key_mult[a] == 1 {
            let gg = exp.key_genome[a] as usize;
            sum[gg] += slot_count[s];
            cnt[gg] += 1;
        }
    }
    (0..g).map(|s| if cnt[s] == 0 { f64::NAN }
                   else { (sum[s] as f64 / cnt[s] as f64).max(0.0) }).collect()
}

/// models.em: compressed identical rows (restricted to group representatives), same updates.
/// Returns (fit, inverse row index, unique rows) so lambda can be recovered per original row.
pub fn em(rows: &Rows, y_rows: &[f64], st: &Structure, initial: &[f64], tol: f64)
    -> Result<(EmFit, Vec<usize>, Rows, Vec<f64>), String> {
    let ng = st.groups.len();
    let rep_group: HashMap<usize, usize> = st.groups.iter().enumerate().map(|(gi, m)| (m[0], gi))
        .collect();
    let mut index: HashMap<Vec<(u32, u64)>, usize> = HashMap::new();
    let mut inverse = vec![0usize; rows.nrows()];
    let mut uniq: Vec<Vec<(u32, u64)>> = Vec::new();
    let mut freq: Vec<u64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for i in 0..rows.nrows() {
        let (c, v) = rows.row(i);
        let r: Vec<(u32, u64)> = c.iter().zip(v).filter_map(|(&cc, &vv)|
            rep_group.get(&(cc as usize)).map(|&gi| (gi as u32, vv))).collect();
        let u = match index.get(&r) {
            Some(&u) => u,
            None => {
                index.insert(r.clone(), uniq.len());
                uniq.push(r);
                freq.push(0);
                y.push(0.0);
                uniq.len() - 1
            }
        };
        inverse[i] = u;
        freq[u] += 1;
        y[u] += y_rows[i];
    }
    let mut base = Rows { off: vec![0], col: Vec::new(), val: Vec::new(), ncols: ng };
    for r in &uniq {
        for &(c, v) in r {
            base.col.push(c);
            base.val.push(v);
        }
        base.off.push(base.col.len());
    }
    let mut exposure = vec![0f64; ng];
    for u in 0..base.nrows() {
        let (c, v) = base.row(u);
        for (&cc, &vv) in c.iter().zip(v) {
            exposure[cc as usize] += freq[u] as f64 * vv as f64;
        }
    }
    let mut theta: Vec<f64> = (0..ng).map(|gi| if exposure[gi] > 0.0 {
        let s: f64 = st.groups[gi].iter().map(|&m| initial[m]).filter(|x| !x.is_nan()).sum();
        s.max(1e-6)
    } else { 0.0 }).collect();
    let nu = base.nrows();
    let lam_of = |theta: &[f64]| -> Vec<f64> {
        (0..nu).map(|u| {
            let (c, v) = base.row(u);
            let mut l = 0.0f64;
            for (&cc, &vv) in c.iter().zip(v) {
                l += vv as f64 * theta[cc as usize];
            }
            l
        }).collect()
    };
    let (mut it, mut delta) = (0usize, f64::INFINITY);
    for iteration in 1..=MAX_ITER {
        it = iteration;
        let lam = lam_of(&theta);
        if (0..nu).any(|u| lam[u] == 0.0 && y[u] > 0.0) {
            return Err(NO_SUPPORT.into());
        }
        let mut num = vec![0f64; ng];
        for u in 0..nu {
            if lam[u] > 0.0 {
                let ratio = y[u] / lam[u];
                let (c, v) = base.row(u);
                for (&cc, &vv) in c.iter().zip(v) {
                    num[cc as usize] += ratio * vv as f64;
                }
            }
        }
        let new: Vec<f64> = (0..ng).map(|gi| if exposure[gi] > 0.0 {
            theta[gi] * num[gi] / exposure[gi] } else { 0.0 }).collect();
        delta = (0..ng).map(|gi| (new[gi] - theta[gi]).abs()).fold(0.0, f64::max);
        theta = new;
        if delta < tol {
            break;
        }
    }
    let lam = lam_of(&theta);
    Ok((EmFit { theta, iterations: it, converged: delta < tol, final_max_delta: delta,
                compressed_rows: nu, original_rows: rows.nrows() }, inverse, base, lam))
}

/// Everything the opt-in output writes.
pub struct Attribution {
    pub st: Structure,
    pub fit: EmFit,
    pub initial: Vec<f64>,
    pub tolerance: f64,
    pub genome_theta: Vec<f64>,
    pub group_theta: Vec<f64>,
    // per segment
    pub own_genome: Vec<i64>,
    pub own_group: Vec<i64>,
    pub own_group_size: Vec<i64>,
    pub own_group_identifiable: Vec<i64>,
    pub own_exposure: Vec<i64>,
    pub observed_support: Vec<i64>,
    pub modeled_support: Vec<f64>,
    pub own_responsibility: Vec<f64>,
    pub own_attributed_support: Vec<f64>,
    pub nonidentifiable_support: Vec<f64>,
    pub attributed: Vec<(String, Vec<f64>)>,
    pub attributed_info: J,
    // sparse, every segment row with modeled entries
    pub sp_off: Vec<i64>,
    pub sp_group: Vec<i64>,
    pub sp_exposure: Vec<i64>,
    pub sp_expected: Vec<f64>,
    pub sp_resp: Vec<f64>,
    pub sp_support: Vec<f64>,
}

/// Segment rows A_j (genome-sparse) from the exposures and the index postings, the raw totals
/// Y_j and the per-slot counts. `raw` = per-anchor raw counts after counting.
pub fn rows_from_index(idx: &ProductIndex, exp: &Exposures, raw: &[i64])
    -> Result<(Rows, Vec<i64>, Vec<i64>, Vec<i64>), String> {
    let ns = idx.st.len();
    let na = idx.anchor_seg.len();
    let nslot = exp.n_slots;
    let mut slot_of = vec![0u32; na];
    let mut slot_count = vec![0i64; nslot];
    for s in 0..nslot {
        let (a, b) = (idx.aoff[s] as usize, idx.aoff[s + 1] as usize);
        slot_count[s] = raw[idx.aidx[a] as usize];
        for i in a..b {
            let an = idx.aidx[i] as usize;
            if raw[an] != slot_count[s] {
                return Err("anchors of one key carry different counts".into());
            }
            slot_of[an] = s as u32;
        }
    }
    // anchors of each segment, in anchor order
    let mut n = vec![0i64; ns];
    for &sg in &idx.anchor_seg {
        n[sg as usize] += 1;
    }
    let mut so = vec![0usize; ns + 1];
    for j in 0..ns {
        so[j + 1] = so[j] + n[j] as usize;
    }
    let mut fill = so.clone();
    let mut by_seg = vec![0usize; na];
    for (a, &sg) in idx.anchor_seg.iter().enumerate() {
        by_seg[fill[sg as usize]] = a;
        fill[sg as usize] += 1;
    }
    let g = exp.n_genomes;
    let mut scratch = vec![0u64; g];
    let mut touched: Vec<u32> = Vec::new();
    let mut rows = Rows { off: vec![0], col: Vec::new(), val: Vec::new(), ncols: g };
    let mut ytot = vec![0i64; ns];
    for j in 0..ns {
        for &a in &by_seg[so[j]..so[j + 1]] {
            ytot[j] += raw[a];
            let s = slot_of[a] as usize;
            for i in exp.key_off[s] as usize..exp.key_off[s + 1] as usize {
                let gg = exp.key_genome[i] as usize;
                if scratch[gg] == 0 {
                    touched.push(gg as u32);
                }
                scratch[gg] += exp.key_mult[i] as u64;
            }
        }
        touched.sort_unstable();
        for &gg in &touched {
            rows.col.push(gg);
            rows.val.push(scratch[gg as usize]);
            scratch[gg as usize] = 0;
        }
        touched.clear();
        rows.off.push(rows.col.len());
    }
    Ok((rows, ytot, n, slot_count))
}

pub fn attribute(idx: &ProductIndex, exp: &Exposures, raw: &[i64], sm: &Sample, tol: f64)
    -> Result<Attribution, String> {
    if !TOLERANCES.contains(&tol) {
        return Err(format!("EM tolerance {} is not an approved value (1e-5, or the 1e-9 \
development comparison)", tol));
    }
    let (rows, ytot, n, slot_count) = rows_from_index(idx, exp, raw)?;
    let ns = rows.nrows();
    let st = structure(&rows)?;
    let initial = private_init(exp, &slot_count);
    let yf: Vec<f64> = ytot.iter().map(|&v| v as f64).collect();
    let (fit, inverse, base, lam_u) = em(&rows, &yf, &st, &initial, tol)?;
    let ng = st.groups.len();
    let group_theta: Vec<f64> = (0..ng).map(|gi| if st.identifiable[gi] { fit.theta[gi] }
                                                 else { f64::NAN }).collect();
    let mut genome_theta = vec![f64::NAN; exp.n_genomes];
    for (gi, m) in st.groups.iter().enumerate() {
        if m.len() == 1 && st.identifiable[gi] {
            genome_theta[m[0]] = fit.theta[gi];
        }
    }
    let mut a = Attribution {
        own_genome: vec![0; ns], own_group: vec![0; ns], own_group_size: vec![0; ns],
        own_group_identifiable: vec![0; ns], own_exposure: vec![0; ns],
        observed_support: ytot.clone(), modeled_support: vec![f64::NAN; ns],
        own_responsibility: vec![f64::NAN; ns], own_attributed_support: vec![f64::NAN; ns],
        nonidentifiable_support: vec![f64::NAN; ns], attributed: Vec::new(),
        attributed_info: J::Null, sp_off: vec![0], sp_group: Vec::new(), sp_exposure: Vec::new(),
        sp_expected: Vec::new(), sp_resp: Vec::new(), sp_support: Vec::new(),
        initial, tolerance: tol, genome_theta, group_theta, st, fit,
    };
    for j in 0..ns {
        let own = idx.rec_genome[idx.seg_rec[j] as usize] as usize;
        let og = a.st.group_of[own];
        a.own_genome[j] = own as i64;
        a.own_group[j] = og as i64;
        a.own_group_size[j] = a.st.groups[og].len() as i64;
        a.own_group_identifiable[j] = a.st.identifiable[og] as i64;
        let u = inverse[j];
        let lam = lam_u[u];
        let yj = yf[j];
        let (c, v) = base.row(u);
        let mut nonid = 0.0f64;
        let mut own_e = None;
        for (&gi, &vv) in c.iter().zip(v) {
            let gi = gi as usize;
            let e = vv as f64 * a.fit.theta[gi];
            let ident = a.st.identifiable[gi];
            let r = if lam > 0.0 && ident { e / lam } else { f64::NAN };
            let z = if yj == 0.0 { 0.0 } else { yj * r };
            if !ident && lam > 0.0 {
                nonid += e / lam;
            }
            if gi == og {
                own_e = Some((vv, r, z));
            }
            a.sp_group.push(gi as i64);
            a.sp_exposure.push(vv as i64);
            a.sp_expected.push(e);
            a.sp_resp.push(r);
            a.sp_support.push(z);
        }
        a.sp_off.push(a.sp_group.len() as i64);
        a.modeled_support[j] = lam;
        a.nonidentifiable_support[j] = if yj == 0.0 { 0.0 } else if lam > 0.0 { yj * nonid }
                                       else { f64::NAN };
        if n[j] == 0 {
            a.own_exposure[j] = 0;
            continue;                           // no eligible anchor: everything stays NaN
        }
        let Some((vv, r, z)) = own_e else {
            return Err(format!("segment {} has anchors but no own-genome exposure", j));
        };
        if vv as i64 != n[j] {
            return Err(format!("segment {}: own-genome exposure {} != anchor count {}", j, vv,
                               n[j]));
        }
        a.own_exposure[j] = vv as i64;
        a.own_responsibility[j] = r;
        a.own_attributed_support[j] = z;
    }
    // attributed local depth: Z_own / A_own, then the pooled variant's own sample scalars
    let raw_attr: Vec<f64> = (0..ns).map(|j| if n[j] > 0 {
        a.own_attributed_support[j] / (n[j].max(1) as f64) } else { f64::NAN }).collect();
    let mut infos = Vec::new();
    for (name, nov) in [("raw_mean", false), ("geom_global", false), ("geom_global_novelty", true)] {
        let mut d = raw_attr.clone();
        let mut info = vec![("pooled_variant", J::Str(format!("depth_{}", name)))];
        if name != "raw_mean" {
            match sm.edge_global.filter(|&e| e != 0.0) {
                Some(e) if sm.geometry_defined => {
                    for x in d.iter_mut() {
                        *x /= e;
                    }
                    info.push(("edge", J::Float(e)));
                }
                _ => {
                    d = vec![f64::NAN; ns];
                    info.push(("arm_undefined_cause", J::Str(sm.geo_cause.clone())));
                }
            }
        }
        if nov {
            match sm.novelty {
                Some(v) if 1.0 - v != 0.0 => {
                    for x in d.iter_mut() {
                        *x /= 1.0 - v;
                    }
                    info.push(("novelty_used", J::Float(v)));
                }
                other => {
                    d = vec![f64::NAN; ns];
                    info.push(("novelty_correction_cause", J::Str(if other.is_none() {
                        "zero selected occurrences" } else { "novelty equals 1" }.into())));
                }
            }
        }
        a.attributed.push((format!("attributed_depth_{}", name), d));
        infos.push((name.to_string(), obj(info)));
    }
    infos.push(("geom_per_read".into(), J::Str(PER_READ_BLOCKER.into())));
    infos.push(("geom_per_read_novelty".into(), J::Str(PER_READ_BLOCKER.into())));
    a.attributed_info = J::Obj(infos);
    Ok(a)
}

pub const PER_READ_BLOCKER: &str = "not produced: the per-read variants use the read-weighted \
channel (per-read L/(L-k+1) weights), while the existing EM is fitted on RAW counts; attributing \
weighted support with raw-count responsibilities would assume source-independent read weights, \
which is not an approved rule";

/// Scalars that turn raw-unit values into each variant (for the fitted-theta comparison).
pub fn variant_scale(sm: &Sample, name: &str) -> Option<f64> {
    match name {
        "raw_mean" => Some(1.0),
        "geom_global" => sm.edge_global.filter(|&e| e != 0.0 && sm.geometry_defined),
        "geom_global_novelty" => {
            let e = sm.edge_global.filter(|&e| e != 0.0 && sm.geometry_defined)?;
            let v = sm.novelty.filter(|&v| 1.0 - v != 0.0)?;
            Some(e * (1.0 - v))
        }
        _ => None,
    }
}

fn fj(x: f64) -> J {
    if x.is_finite() { J::Float(x) } else { J::Null }
}

/// Per-genome length-weighted summaries over each genome's OWN segments: weights = scored ACGT
/// bases (index dom_bases), mask = eligible anchors > 0 AND the attributed value defined. The
/// pooled variant is averaged over the SAME mask; the fitted theta in the variant's units is
/// reported beside it (they are different quantities and need not agree).
pub fn genome_summaries(idx: &ProductIndex, a: &Attribution, pooled: &[(String, Vec<f64>)],
                        sm: &Sample, names: &[String]) -> J {
    let ng = idx.n_genomes;
    let ns = idx.st.len();
    let mut out = Vec::new();
    let mut by_genome: Vec<Vec<usize>> = vec![Vec::new(); ng];
    for j in 0..ns {
        by_genome[a.own_genome[j] as usize].push(j);
    }
    for s in 0..ng {
        let gi = a.st.group_of[s];
        let own = &by_genome[s];
        let mut variants = Vec::new();
        for (vname, d) in &a.attributed {
            let base = vname.trim_start_matches("attributed_depth_");
            let p = &pooled.iter().find(|(n, _)| n == &format!("depth_{}", base)).unwrap().1;
            let (mut sw, mut swd, mut swp, mut nd) = (0f64, 0f64, 0f64, 0usize);
            for &j in own {
                if a.own_exposure[j] > 0 && d[j].is_finite() {
                    let w = idx.dom_bases[j] as f64;
                    sw += w;
                    swd += w * d[j];
                    swp += w * p[j];
                    nd += 1;
                }
            }
            let theta_v = variant_scale(sm, base).map(|sc| a.fit.theta[gi] / sc)
                .filter(|_| a.st.identifiable[gi]).unwrap_or(f64::NAN);
            variants.push((base.to_string(), obj(vec![
                ("defined_segments", J::Int(nd as i64)),
                ("weight_sum_scored_acgt_bases", J::Float(sw)),
                ("attributed_weighted_mean", fj(if sw > 0.0 { swd / sw } else { f64::NAN })),
                ("pooled_weighted_mean_same_mask", fj(if sw > 0.0 { swp / sw } else { f64::NAN })),
                ("fitted_group_theta_in_variant_units", fj(theta_v))])));
        }
        out.push(obj(vec![
            ("genome", J::Int(s as i64)), ("name", J::Str(names[s].clone())),
            ("group", J::Int(gi as i64)), ("group_size", J::Int(a.st.groups[gi].len() as i64)),
            ("group_identifiable", J::Bool(a.st.identifiable[gi])),
            ("genome_theta_raw", fj(a.genome_theta[s])),
            ("own_segments", J::Int(own.len() as i64)),
            ("own_segments_without_anchors", J::Int(own.iter().filter(|&&j|
                a.own_exposure[j] == 0).count() as i64)),
            ("variants", J::Obj(variants.into_iter().map(|(k, v)| (k, v)).collect())),
        ]));
    }
    J::List(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(dense: &[Vec<u64>]) -> Rows {
        let mut r = Rows { off: vec![0], col: vec![], val: vec![], ncols: dense[0].len() };
        for row in dense {
            for (c, &v) in row.iter().enumerate() {
                if v > 0 {
                    r.col.push(c as u32);
                    r.val.push(v);
                }
            }
            r.off.push(r.col.len());
        }
        r
    }

    #[test]
    fn structure_groups_and_identifiability() {
        // columns: 0, 1 identical; 2 = 0 + 3 -> {0,1}, 2, 3 groups; groups 1 (col 2) and
        // 2 (col 3) and 0 all in one dependency -> none identifiable; rank 2
        let m = vec![vec![1, 1, 1, 0], vec![0, 0, 2, 2], vec![3, 3, 3, 0]];
        let st = structure(&rows(&m)).unwrap();
        assert_eq!(st.groups, vec![vec![0, 1], vec![2], vec![3]]);
        assert_eq!(st.rank, 2);
        assert_eq!(st.identifiable, vec![false, false, false]);
        let m2 = vec![vec![1, 0], vec![0, 1], vec![1, 1]];
        let st2 = structure(&rows(&m2)).unwrap();
        assert_eq!((st2.rank, st2.identifiable.clone()), (2, vec![true, true]));
        let m3 = vec![vec![0, 1], vec![0, 2]];
        let st3 = structure(&rows(&m3)).unwrap();
        assert_eq!((st3.rank, st3.identifiable.clone()), (1, vec![false, true]));
    }

    fn column(entries: &[u64]) -> Vec<Vec<u64>> {
        entries.iter().map(|&x| vec![x]).collect()
    }
    // sum of squares = 2^61 - 1 = PRIMES[0]; and = PRIMES[0] * PRIMES[1]
    const NORM_P0: [u64; 5] = [1518500249, 54778, 51, 8, 1];
    const NORM_P0P1: [u64; 9] = [3260954456333195532, 1218736432, 37496, 99, 5, 2, 1, 1, 1];

    #[test]
    fn certificate_survives_primes_that_divide_the_gram_minor() {
        // Gram = [p0 p1]: rank 0 mod BOTH of the first two primes (the old two-prime
        // agreement rule would have accepted rank 0); the exact rank is 1
        let gram = vec![vec![PRIMES[0] as u128 * PRIMES[1] as u128]];
        assert_eq!((rref_mod(&gram, PRIMES[0]).0.len(), rref_mod(&gram, PRIMES[1]).0.len()), (0, 0));
        let st = structure(&rows(&column(&NORM_P0P1))).unwrap();
        assert_eq!((st.rank, st.identifiable.clone()), (1, vec![true]));
        assert!(st.certificate.null_vectors.is_empty());
        assert!(!st.certificate.primes_used.contains(&PRIMES[0]));
        // two columns: the unlucky prime loses only one pivot
        let mut m = column(&NORM_P0P1).into_iter().map(|r| vec![r[0], 0]).collect::<Vec<_>>();
        m.push(vec![0, 1]);
        let st = structure(&rows(&m)).unwrap();
        assert_eq!((st.rank, st.identifiable.clone()), (2, vec![true, true]));
    }

    #[test]
    fn certificate_refuses_when_every_prime_understates() {
        let r = rows(&column(&NORM_P0));
        let gram = vec![vec![PRIMES[0] as u128]];
        let e = certify(&r, &[0], &gram, &PRIMES[..1]).err().unwrap();
        assert!(e.contains("not in the rational null space"), "{}", e);
        // the full prime list certifies it
        let (rk, id, c) = certify(&r, &[0], &gram, &PRIMES).unwrap();
        assert_eq!((rk, id, c.null_vectors.len()), (1, vec![true], 0));
    }

    #[test]
    fn certificate_rational_null_vector() {
        // 2 c2 = 3 c0 + c1 (non-unit rational coefficients); c3 independent
        let m = vec![vec![2, 0, 3, 0], vec![0, 2, 1, 0], vec![2, 2, 4, 0], vec![0, 0, 0, 7]];
        let st = structure(&rows(&m)).unwrap();
        assert_eq!(st.rank, 3);
        assert_eq!(st.identifiable, vec![false, false, false, true]);
        assert_eq!(st.certificate.null_vectors.len(), 1);
        let w = &st.certificate.null_vectors[0];
        let s = if w[0] < 0 { -1 } else { 1 };
        assert_eq!(w.iter().map(|x| x * s).collect::<Vec<_>>(), vec![3, 1, -2, 0]);
        let p = PRIMES[0] as u128;
        assert_eq!(ratrec(p - 2, p), Some((-2, 1)));
        assert_eq!(ratrec((p + 1) / 2, p), Some((1, 2)));
    }

    #[test]
    fn no_support_error_matches_python_message() {
        let m = vec![vec![1, 0], vec![0, 0]];
        let r = rows(&m);
        let st = structure(&r).unwrap();
        let e = em(&r, &[1.0, 5.0], &st, &[1.0, f64::NAN], 1e-5).err().unwrap();
        assert_eq!(e, NO_SUPPORT);
    }

    #[test]
    fn single_source_mle_is_ratio_of_sums() {
        let m = vec![vec![2], vec![3], vec![5]];
        let r = rows(&m);
        let st = structure(&r).unwrap();
        let (f, _, _, _) = em(&r, &[4.0, 9.0, 7.0], &st, &[1.0], 1e-9).unwrap();
        assert!((f.theta[0] - 20.0 / 10.0).abs() < 1e-8, "{}", f.theta[0]);
    }
}
