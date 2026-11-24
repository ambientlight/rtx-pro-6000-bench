//! CPU-only, synthetic single-token wo_b on one TP4 shard. All corrected outputs
//! and sampled v3 tickets come from the same two-row noised integer product.
//! FP8 calibration, GPU execution and F2Pool acceptance require separate checks.
//! Replay from the benchmark repo with the pinned Pearl v1.5.8-compatible build.
//!     deps=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/deps
//!     probe=$(mktemp -d)
//!     rustc +1.97.0 --edition=2021 -O -C panic=abort -L "dependency=$deps" --extern zk_pow="$deps/libzk_pow-fc153e5b815e44e1.rlib" --extern pearl_blake3="$deps/libpearl_blake3-2e850ac2b7919e97.rlib" --extern bincode="$deps/libbincode-6e76495728f8471e.rlib" work/pearl-inference-20261123/singleton_v3.rs -o "$probe/check"
//!     RAYON_NUM_THREADS=2 "$probe/check"

use pearl_blake3::MerkleTree;
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
    SeedDerivation,
};
use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
use zk_pow::api::sanity_checks::check_jackpot_against_nbits;
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::circuit::chip::compute_jackpot;
use zk_pow::circuit::pearl_noise::{compute_noise, compute_noise_for_indices};
use zk_pow::circuit::pearl_program::JACKPOT_SIZE;
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, MatrixMerkleProof, PlainProof};

const M: usize = 2;
const N: usize = 5120;
const K: usize = 2048;
const RANK: usize = 128;
const TILE_COLS: usize = 16;

fn tree(values: &[i8], job_key: [u8; 32]) -> MerkleTree {
    let bytes: Vec<_> = values.iter().map(|&value| value as u8).collect();
    // Each row is an exact multiple of Pearl's 1024-byte Merkle chunk.
    MerkleTree::new(&bytes, job_key)
}

fn open_rows(tree: &MerkleTree, total_rows: usize, row_indices: Vec<usize>) -> MatrixMerkleProof {
    let leaves = MerkleTree::compute_leaf_indices_from_rows(&row_indices, (total_rows, K));
    MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaves),
        row_indices,
    }
}

fn main() {
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x6666_6666,
        nbits: 0x1e40_0000, // Adjusted by 2 * 16 * 2048 to 2^254.
    };
    let cols: Vec<_> = (0..TILE_COLS as u32).collect();
    let mut params = PublicProofParams {
        block_header: header,
        seed_derivation: SeedDerivation::Salted,
        mining_config: MiningConfiguration {
            common_dim: K as u32,
            rank: RANK as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 1]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
            moe: None,
        },
        m: M as u32,
        n: N as u32,
        t_rows: 0,
        t_cols: 0,
        hash_a: [0; 32],
        hash_b: [0; 32],
        hash_jackpot: [0; 32],
        moe: None,
    };

    // Every input value in [-64,64] occurs, including negative nonmultiples of 8.
    let x: Vec<i8> = (0..K)
        .map(|i| (((i * 37) % 129) as i16 - 64) as i8)
        .collect();
    let mut activation = Vec::with_capacity(M * K);
    activation.extend(x.iter().map(|v| v.div_euclid(8)));
    activation.extend(x.iter().map(|v| v.rem_euclid(8)));
    let mut state = 0x24ac_2351u32;
    let weights: Vec<i8> = (0..N * K)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (((state >> 16) % 129) as i16 - 64) as i8
        })
        .collect();

    let job_key = params.job_key();
    let a_tree = tree(&activation, job_key);
    let bt_tree = tree(&weights, job_key);
    params.hash_a = a_tree.root();
    params.hash_b = bt_tree.root();
    let seeds = params.commitment_hash(job_key);
    let noise = compute_noise_for_indices(K, RANK, seeds, &[0, 1], &(0..N).collect::<Vec<_>>());

    // One checkpoint per transcript word, so the rotation acts on the initial zero.
    assert_eq!(K / RANK, JACKPOT_SIZE);
    let mut hits = 0;
    let mut checked = [false; 2];
    for first_col in (0..N).step_by(TILE_COLS) {
        let mut transcript = [0u32; JACKPOT_SIZE];
        for col in first_col..first_col + TILE_COLS {
            let weight_row = &weights[col * K..(col + 1) * K];
            let mut planes = [0i32; M];
            for row in 0..M {
                let mut noised_product = 0i32;
                let mut correction = 0i32;
                for (checkpoint, end) in (RANK..=K).step_by(RANK).enumerate() {
                    for i in end - RANK..end {
                        let a = i32::from(activation[row * K + i]);
                        let b = i32::from(weight_row[i]);
                        let na = i32::from(noise.a[row][i]);
                        let nb = i32::from(noise.b[col][i]);
                        noised_product += (a + na) * (b + nb);
                        correction += na * (b + nb) + a * nb;
                    }
                    transcript[checkpoint] ^= noised_product as u32;
                }
                planes[row] = noised_product - correction;
            }
            let expected: i32 = x
                .iter()
                .zip(weight_row)
                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                .sum();
            assert_eq!(8 * planes[0] + planes[1], expected, "output column {col}");
        }
        let digest = compute_jackpot_hash(&transcript, seeds.1);
        params.hash_jackpot = digest;
        params.t_cols = first_col as u32;
        let hit = check_jackpot_against_nbits(&params, Some(header.nbits)).is_ok();
        hits += usize::from(hit);
        if !checked[usize::from(hit)] {
            let proof = PlainProof {
                m: M,
                n: N,
                k: K,
                noise_rank: RANK,
                moe: None,
                a: open_rows(&a_tree, M, vec![0, 1]),
                bt: open_rows(&bt_tree, N, (first_col..first_col + TILE_COLS).collect()),
            };
            check_cert_version_eligible(3, &proof).unwrap();
            let (private, public) = proof.parse_proof(header, SeedDerivation::Salted).unwrap();
            let compiled = CompiledPublicParams::from(&public);
            let reference = compute_jackpot(
                &compiled,
                &private.s_a,
                &private.s_b,
                &compute_noise(&compiled),
            );
            assert_eq!(transcript, reference);
            let verified =
                verify_plain_proof(&header, &proof, Some(header.nbits), SeedDerivation::Salted);
            if hit {
                verified.unwrap();
            } else {
                assert!(verified
                    .unwrap_err()
                    .to_string()
                    .contains("Jackpot condition not satisfied"));
            }
            checked[usize::from(hit)] = true;
            println!(
                "first_col={first_col}, hit={hit}, witness_bytes={}",
                bincode::serialized_size(&proof).unwrap()
            );
        }
    }
    assert_eq!(checked, [true, true]);
    println!(
        "one synthetic token, {N} corrected outputs matched, {hits}/{} tickets met the fixture target",
        N / TILE_COLS
    );
}
