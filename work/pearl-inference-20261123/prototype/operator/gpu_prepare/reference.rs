// Extend the same pinned Rust reference ABI with readonly weight metadata and an
// independently owned proof-opening handle. Compile in place of the original cdylib.
include!("../reference.rs");

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_weight_material(weight: *const Weight, out: *mut u8) -> i32 {
    let Some(weight) = weight.as_ref() else { return -1 };
    if out.is_null() { return -1; }
    ptr::copy_nonoverlapping(weight.0.key.as_ptr(), out, 32);
    ptr::copy_nonoverlapping(weight.0.template.hash_b.as_ptr(), out.add(32), 32);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_weight_retain(weight: *const Weight) -> *mut Weight {
    weight.as_ref().map_or(ptr::null_mut(), |w| Box::into_raw(Box::new(Weight(Arc::clone(&w.0)))))
}

// Rebuild the activation Merkle openings on the proof worker. This never generates
// dense noise. The GPU root/seed are checked before this handle can emit any proof.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_activation_open(weight: *const Weight, src: *const i8,
        m: i32, expected: *const u8) -> *mut Reference {
    let Some(w) = weight.as_ref() else { return ptr::null_mut() };
    if src.is_null() || expected.is_null() || !valid_rows(m, w.0.template.n, w.0.row_multiple) {
        return ptr::null_mut();
    }
    let Some(a) = valid_signal_tree(src, m as usize, w.0.key) else { return ptr::null_mut() };
    let mut params = w.0.template.clone();
    params.m = m as u32;
    params.hash_a = a.root();
    let a_seed = params.commitment_hash(w.0.key).1;
    if params.hash_a != slice::from_raw_parts(expected, 32) || a_seed != slice::from_raw_parts(expected.add(32), 32) {
        return ptr::null_mut();
    }
    let row_starts = (0..m as u32).filter(|&r| params.mining_config.rows_pattern.offset_is_valid(r)).collect();
    Box::into_raw(Box::new(Reference {params, a, a_seed, row_starts, weight: Arc::clone(&w.0)}))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pearl_ref_activation_material(reference: *const Reference, out: *mut u8) -> i32 {
    let Some(r) = reference.as_ref() else { return -1 };
    if out.is_null() { return -1; }
    ptr::copy_nonoverlapping(r.params.hash_a.as_ptr(), out, 32);
    ptr::copy_nonoverlapping(r.a_seed.as_ptr(), out.add(32), 32);
    0
}
