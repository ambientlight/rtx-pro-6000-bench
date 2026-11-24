#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All buffers are 4-byte-aligned contiguous device memory and must live through stream completion.
// A and noise_a are [m,k], weight-transpose and noise_bt [n,k].
// Output is [m,n] int32. Transcripts are [(m/2)*(n/(compact?16:64)),16] uint32.
// compact=0: m a multiple of 16 and n a multiple of 256 (released prefill pattern).
// compact=1: m even and n a multiple of 16 (adjacent row pairs for decode).
// k=2048, noise rank 128. Return CUDA launch status. Does not allocate or synchronize.
cudaError_t pearl_wo_b_i7(const int8_t* a, const int8_t* bt,
                          const int8_t* noise_a, const int8_t* noise_bt,
                          int32_t* output, uint32_t* transcripts,
                          int m, int n, int k, int compact, cudaStream_t stream);

// Plain per-rank INT7 result [m,n] = A[m,2048] times transpose(BT[n,2048]).
// Any positive m and n are supported within the launch/index limits. No external
// padding, noise, transcript storage, allocations, or synchronization.
cudaError_t pearl_wo_b_i7_plain(const int8_t* a, const int8_t* bt, int32_t* output,
                                int m, int n, int k, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
