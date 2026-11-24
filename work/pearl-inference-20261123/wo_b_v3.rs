//! Offline prefill and decode certificate checks for one TP4 shard of DeepSeek V4.1 Flash wo_b.
//! The CPU reference miner generates synthetic operands. Decode cases use a nonsaturating target.
//! Verifier source tree: Pearl v1.5.8 zk-pow/src, Git tree 712bec0665ee11b5d0be429867fb6c3f4ad5b3e7.
//! To replay on this host from the benchmark repository:
//!     deps=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/deps
//!     probe=$(mktemp -d)
//!     rustc +1.97.0 --edition=2021 -O -C panic=abort -L "dependency=$deps" --extern zk_pow="$deps/libzk_pow-fc153e5b815e44e1.rlib" --extern bincode="$deps/libbincode-6e76495728f8471e.rlib" work/pearl-inference-20261123/wo_b_v3.rs -o "$probe/check"
//!     RAYON_NUM_THREADS=2 "$probe/check"

use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation,
};
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::mine::mine;
use zk_pow::ffi::plain_proof::check_cert_version_eligible;

fn check_shape(label: &str, m: usize, rows: &[u32], columns: &[u32], nbits: u32) {
    let (n, k) = (5120, 2048);
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x6666_6666,
        nbits,
    };

    let config = MiningConfiguration {
        common_dim: k as u32,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(rows).unwrap(),
        cols_pattern: PeriodicPattern::from_list(columns).unwrap(),
        moe: None,
    };
    let proof = mine(m, n, k, header, config, None, false, SeedDerivation::Salted).unwrap();
    check_cert_version_eligible(3, &proof).unwrap();
    verify_plain_proof(&header, &proof, Some(header.nbits), SeedDerivation::Salted).unwrap();
    println!(
        "Verified {label} salted v3, M={m} N={n} K={k}, rank={}, opened A/B rows={}/{}, proof bytes={}",
        proof.noise_rank, proof.a.row_indices.len(), proof.bt.row_indices.len(),
        bincode::serialized_size(&proof).unwrap()
    );
}

fn main() {
    // Released vLLM prefill pattern.
    let prefill_columns: Vec<_> = (0..256).step_by(8).flat_map(|i| [i, i + 1]).collect();
    check_shape("prefill", 2048, &[0, 8], &prefill_columns, 0x207f_ffff);

    // Adjacent rows can come from the same target verification pass or a natural decode batch.
    // At 2 × 16 × 2048 MACs per ticket, this compact target scales to 2^254.
    let decode_columns: Vec<_> = (0..16).collect();
    check_shape("two-row decode", 2, &[0, 1], &decode_columns, 0x1e40_0000);
    check_shape("six-row DSPARK shape", 6, &[0, 1], &decode_columns, 0x1e40_0000);
}
