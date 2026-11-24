// Host-side reference for one immutable (header, A, B) commitment. All host
// buffers are contiguous. A and B can be released after preparation.
use pearl_blake3::MerkleTree;
use std::{ptr, slice, sync::Arc};
use zk_pow::api::{
    proof::{
        IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
        SeedDerivation,
    },
    proof_utils::{compute_jackpot_hash, CompiledPublicParams},
    sanity_checks::check_jackpot_against_nbits,
    verify::verify_plain_proof,
};
use zk_pow::circuit::{
    chip::compute_jackpot,
    pearl_noise::{compute_noise, compute_noise_for_indices},
};
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, MatrixMerkleProof, PlainProof};

const K: usize = 2048;
const RANK: usize = 128;
struct WeightData {
    // Template for a single header, pattern, N and immutable committed B. A
    // request fills m and hash_a before deriving its own activation seed.
    template: PublicProofParams,
    key: [u8; 32],
    tree: MerkleTree,
    col_starts: Vec<u32>,
    row_multiple: usize,
}
pub struct Weight(Arc<WeightData>);

pub struct Reference {
    params: PublicProofParams,
    a: MerkleTree,
    weight: Arc<WeightData>,
    a_seed: [u8; 32],
    row_starts: Vec<u32>,
}

impl Reference {
    fn tickets(&self) -> usize {
        self.row_starts.len() * self.weight.col_starts.len()
    }

    fn witness(&self, ticket: usize) -> PlainProof {
        let conf = &self.params.mining_config;
        let row = self.row_starts[ticket / self.weight.col_starts.len()];
        let col = self.weight.col_starts[ticket % self.weight.col_starts.len()];
        let open = |tree: &MerkleTree, size, pattern: &PeriodicPattern, offset| {
            let row_indices: Vec<_> = pattern
                .indices_with_offset(offset)
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let leaves = MerkleTree::compute_leaf_indices_from_rows(&row_indices, (size, K));
            MatrixMerkleProof {
                proof: tree.get_multileaf_proof(&leaves),
                row_indices,
            }
        };
        let m = self.params.m as usize;
        let n = self.params.n as usize;
        PlainProof {
            m,
            n,
            k: K,
            noise_rank: RANK,
            a: open(&self.a, m, &conf.rows_pattern, row),
            bt: open(&self.weight.tree, n, &conf.cols_pattern, col),
            moe: None,
        }
    }
}

fn valid_weight_shape(n: i32, compact: i32) -> bool {
    (compact == 0 || compact == 1)
        && n > 0
        && n <= (1 << 24)
        && n % (if compact == 1 { 16 } else { 256 }) == 0
}
fn valid_rows(m: i32, n: u32, row_multiple: usize) -> bool {
    m > 0
        && m <= (1 << 24)
        && m as usize % row_multiple == 0
        && i64::from(m) * i64::from(n) <= i64::from(i32::MAX)
}
unsafe fn valid_signal_tree(src: *const i8, rows: usize, key: [u8; 32]) -> Option<MerkleTree> {
    let values = slice::from_raw_parts(src, rows * K);
    if values.iter().all(|v| (-64..=64).contains(v)) {
        Some(MerkleTree::new(
            slice::from_raw_parts(src.cast(), rows * K),
            key,
        ))
    } else {
        None
    }
}
// The CPU verifier operates on plain host memory. No CUDA functions run here.
unsafe fn prepare_weight(
    header: *const u8,
    bt: *const i8,
    nb: *mut i8,
    n: i32,
    compact: i32,
) -> Option<Weight> {
    if header.is_null() || bt.is_null() || nb.is_null() || !valid_weight_shape(n, compact) {
        return None;
    }
    let header = IncompleteBlockHeader::from_bytes(slice::from_raw_parts(header, 76)).ok()?;
    let compact = compact == 1;
    let rows: &[u32] = if compact { &[0, 1] } else { &[0, 8] };
    let cols: Vec<u32> = if compact {
        (0..16).collect()
    } else {
        (0..256).step_by(8).flat_map(|i| [i, i + 1]).collect()
    };
    let conf = MiningConfiguration {
        common_dim: K as u32,
        rank: RANK as u16,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(rows).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        moe: None,
    };
    let mut template = PublicProofParams {
        block_header: header,
        seed_derivation: SeedDerivation::Salted,
        mining_config: conf,
        m: 0,
        n: n as u32,
        t_rows: 0,
        t_cols: 0,
        hash_a: [0; 32],
        hash_b: [0; 32],
        hash_jackpot: [0; 32],
        moe: None,
    };
    let key = template.job_key();
    let tree = valid_signal_tree(bt, n as usize, key)?;
    template.hash_b = tree.root();
    // V3 binds B's root only to N. The call receives no A rows.
    let noise = compute_noise_for_indices(
        K,
        RANK,
        template.commitment_hash(key),
        &[],
        &(0..n as usize).collect::<Vec<_>>(),
    );
    for (i, col) in noise.b.iter().enumerate() {
        ptr::copy_nonoverlapping(col.as_ptr(), nb.add(i * K), K);
    }
    let col_starts = (0..n as u32)
        .filter(|&c| conf.cols_pattern.offset_is_valid(c))
        .collect();
    Some(Weight(Arc::new(WeightData {
        template,
        key,
        tree,
        col_starts,
        row_multiple: if compact { 2 } else { 16 },
    })))
}
impl Weight {
    unsafe fn prepare(&self, a: *const i8, na: *mut i8, m: i32) -> Option<Reference> {
        if a.is_null() || na.is_null() || !valid_rows(m, self.0.template.n, self.0.row_multiple) {
            return None;
        }
        let mut params = self.0.template.clone();
        params.m = m as u32;
        let a = valid_signal_tree(a, m as usize, self.0.key)?;
        params.hash_a = a.root();
        let seeds = params.commitment_hash(self.0.key);
        let noise =
            compute_noise_for_indices(K, RANK, seeds, &(0..m as usize).collect::<Vec<_>>(), &[]);
        for (i, row) in noise.a.iter().enumerate() {
            ptr::copy_nonoverlapping(row.as_ptr(), na.add(i * K), K);
        }
        let row_starts = (0..m as u32)
            .filter(|&r| params.mining_config.rows_pattern.offset_is_valid(r))
            .collect();
        Some(Reference {
            params,
            a,
            weight: Arc::clone(&self.0),
            a_seed: seeds.1,
            row_starts,
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_weight_new(
    header: *const u8,
    bt: *const i8,
    nb: *mut i8,
    n: i32,
    compact: i32,
) -> *mut Weight {
    prepare_weight(header, bt, nb, n, compact)
        .map_or(ptr::null_mut(), |w| Box::into_raw(Box::new(w)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_weight_prepare(
    weight: *const Weight,
    a: *const i8,
    na: *mut i8,
    m: i32,
) -> *mut Reference {
    weight
        .as_ref()
        .and_then(|w| w.prepare(a, na, m))
        .map_or(ptr::null_mut(), |r| Box::into_raw(Box::new(r)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_weight_free(weight: *mut Weight) {
    if !weight.is_null() {
        drop(Box::from_raw(weight));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_ref_new(
    header: *const u8,
    a: *const i8,
    bt: *const i8,
    na: *mut i8,
    nb: *mut i8,
    m: i32,
    n: i32,
    compact: i32,
) -> *mut Reference {
    if a.is_null()
        || na.is_null()
        || !valid_weight_shape(n, compact)
        || !valid_rows(m, n as u32, if compact == 1 { 2 } else { 16 })
    {
        return ptr::null_mut();
    }
    prepare_weight(header, bt, nb, n, compact)
        .and_then(|w| w.prepare(a, na, m))
        .map_or(ptr::null_mut(), |r| Box::into_raw(Box::new(r)))
}
// stats = total hits, first hit index + 1, first miss index + 1, reference checks.
// Rebuild complete transcript words from the verifier's Merkle openings on both
// target decisions and at fixed intervals across the matrix.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_ref_scan(
    ctx: *const Reference,
    words: *const [u32; 16],
    count: usize,
    target_bits: u32,
    stats: *mut u64,
) -> i32 {
    if ctx.is_null() || words.is_null() || stats.is_null() {
        return -1;
    }
    let ctx = &*ctx;
    if count != ctx.tickets() {
        return -2;
    }
    let words = slice::from_raw_parts(words, count);
    let stats = slice::from_raw_parts_mut(stats, 4);
    stats.fill(0);
    let mut params = ctx.params.clone();
    for (i, word) in words.iter().enumerate() {
        params.hash_jackpot = compute_jackpot_hash(word, ctx.a_seed);
        let hit = check_jackpot_against_nbits(&params, Some(target_bits)).is_ok();
        stats[0] += u64::from(hit);
        let slot = if hit { 1 } else { 2 };
        let first = stats[slot] == 0;
        if first {
            stats[slot] = (i + 1) as u64;
        }
        if first || i == count - 1 || i % (count / 8).max(1) == 0 {
            let proof = ctx.witness(i);
            if check_cert_version_eligible(3, &proof).is_err() {
                return -3;
            }
            let Ok((private, public)) =
                proof.parse_proof(params.block_header, SeedDerivation::Salted)
            else {
                return -4;
            };
            let compiled = CompiledPublicParams::from(&public);
            if *word
                != compute_jackpot(
                    &compiled,
                    &private.s_a,
                    &private.s_b,
                    &compute_noise(&compiled),
                )
            {
                return -5;
            }
            let result = verify_plain_proof(
                &params.block_header,
                &proof,
                Some(target_bits),
                SeedDerivation::Salted,
            );
            match (hit, result) {
                (true, Ok(())) => {}
                (false, Err(e)) if e.to_string().contains("Jackpot condition not satisfied") => {}
                _ => return -6,
            }
            stats[3] += 1;
        }
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_ref_proof(
    ctx: *const Reference,
    ticket: usize,
    bits: u32,
    dst: *mut u8,
    cap: usize,
) -> isize {
    if ctx.is_null() {
        return -1;
    }
    let ctx = &*ctx;
    if ticket >= ctx.tickets() {
        return -2;
    }
    let proof = ctx.witness(ticket);
    if check_cert_version_eligible(3, &proof).is_err()
        || verify_plain_proof(
            &ctx.params.block_header,
            &proof,
            Some(bits),
            SeedDerivation::Salted,
        )
        .is_err()
    {
        return -3;
    }
    let Ok(bytes) = bincode::serialize(&proof) else {
        return -4;
    };
    if dst.is_null() {
        return bytes.len() as isize;
    }
    if bytes.len() > cap {
        return -5;
    }
    ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    bytes.len() as isize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_ref_free(ctx: *mut Reference) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}
