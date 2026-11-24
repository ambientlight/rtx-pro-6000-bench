use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation};
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::mine::mine;
use zk_pow::ffi::plain_proof::check_cert_version_eligible;
fn main() {
    let (m,n,k)=(2048,5120,2048);
    let header = IncompleteBlockHeader {
        version: 0, prev_block: [1;32], merkle_root: [2;32], timestamp: 0x66666666, nbits: 0x207fffff
    };
    let config = MiningConfiguration {
        common_dim: k as u32, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0,8,64,72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0,1,8,9,32,33,40,41]).unwrap(), moe:None
    };
    let proof = mine(m,n,k,header,config,None,false,SeedDerivation::Salted).unwrap();
    check_cert_version_eligible(3, &proof).unwrap();
    verify_plain_proof(&header,&proof,Some(header.nbits),SeedDerivation::Salted).unwrap();
    println!("Salted v3 verified, rank {}, M={} N={} K={}, opened A rows {} B rows {}, serialized {} bytes",
        proof.noise_rank,proof.m,proof.n,proof.k,proof.a.row_indices.len(),proof.bt.row_indices.len(),
        bincode::serialized_size(&proof).unwrap());
}
