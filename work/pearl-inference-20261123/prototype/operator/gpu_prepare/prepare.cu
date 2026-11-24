// Graph-capturable preparation of Pearl v3 activation noise for K=2048 and noise_rank=128.
// All pointers are device addresses. The caller owns every buffer and the stream.
#include <cstdint>
#include <cuda_runtime.h>
#include "cutlass/device_kernel.h"
#include "blake3/blake3.cuh"
#include "tensor_hash/commitment_hash_from_merkle_roots_kernel.hpp"

namespace activation {
constexpr int k = 2048, noise_rank = 128, leaf_bytes = 1024;
constexpr int max_rows = 1 << 20;

__device__ auto digest(const uint32_t* src) {
    auto cv = cute::make_tensor<uint32_t>(cute::Shape<cute::_8>{});
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv(i) = src[i];
    return cv;
}

__global__ void leaves(const uint32_t* input, const uint32_t* key, uint32_t* layer, int count) {
    const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= count) return;
    auto cv = digest(key);
    const uint32_t* data = input + size_t(leaf) * (leaf_bytes / 4);
    for (int block = 0; block < 16; ++block) {
        auto msg = cute::make_tensor<uint32_t>(cute::Shape<cute::_16>{});
        #pragma unroll
        for (int i = 0; i < 16; ++i) msg(i) = data[block * 16 + i];
        const uint32_t flags = blake3::KEYED_HASH | (block == 0 ? blake3::CHUNK_START : 0)
                                                 | (block == 15 ? blake3::CHUNK_END : 0);
        blake3::compress_msg_block_u32(msg, cv, {uint64_t(leaf), 64, flags});
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) layer[leaf * 8 + i] = cv(i);
}

__global__ void parents(const uint32_t* layer, const uint32_t* key, uint32_t* next, int count) {
    const int parent = blockIdx.x * blockDim.x + threadIdx.x;
    if (parent >= (count + 1) / 2) return;
    const uint32_t* children = layer + parent * 16;
    auto cv = digest(key);
    if (2 * parent + 1 < count) {
        auto msg = cute::make_tensor<uint32_t>(cute::Shape<cute::_16>{});
        #pragma unroll
        for (int i = 0; i < 16; ++i) msg(i) = children[i];
        blake3::compress_msg_block_u32(msg, cv, count == 2 ? blake3::COMPRESS_PARAMS_ROOT
                                                         : blake3::COMPRESS_PARAMS_INNER_NODE);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv(i) = children[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) next[parent * 8 + i] = cv(i);
}

// One keyed hash supplies 32 basis components or 8 permutation pairs.
__global__ void noise_factors(const uint32_t* seed, uint32_t* basis, uint16_t* pairs, int dense_hashes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dense_hashes + k / 8) return;
    const bool dense = tid < dense_hashes;
    const int i = dense ? tid : tid - dense_hashes;
    auto cv = digest(seed);
    auto msg = cute::make_tensor<uint32_t>(cute::Shape<cute::_16>{});
    cute::clear(msg);
    msg(dense ? 0 : 1) = i + 1;
    msg(8) = 0x65745f41u; // A_te
    msg(9) = 0x726f736eu; // nsor
    blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t word = cv(j);
        if (dense) {
            basis[i * 8 + j] = word & 0x3f3f3f3fu;
        } else {
            uint32_t pos = word & 127u;
            uint32_t neg = pos ^ (1u + __umulhi(127u, word));
            pairs[i * 8 + j] = uint16_t(pos | neg << 8);
        }
    }
}

__global__ void expand(const uint32_t* basis, const uint16_t* pairs, uint32_t* noise) {
    __shared__ uint32_t row[noise_rank / 4];
    int tid = threadIdx.x;
    if (tid < noise_rank / 4) row[tid] = basis[blockIdx.x * (noise_rank / 4) + tid];
    __syncthreads();
    auto bytes = reinterpret_cast<const uint8_t*>(row);
    for (int group = tid; group < k / 4; group += blockDim.x) {
        uint32_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint16_t pair = pairs[4 * group + j];
            packed |= uint32_t(uint8_t(int(bytes[pair & 255]) - int(bytes[pair >> 8]))) << (8 * j);
        }
        noise[size_t(blockIdx.x) * (k / 4) + group] = packed;
    }
}

size_t tree_bytes(int m) {
    size_t size = 0;
    // Root is returned separately to the caller.
    for (int count = 2 * m; count > 1; count = (count + 1) / 2) size += 32ull * count;
    return size;
}
}

using namespace activation;

extern "C" size_t pearl_activation_scratch(int m) {
    return m > 0 && m <= max_rows ? tree_bytes(m) + noise_rank * size_t(m) + 2 * k + 32 : 0;
}

extern "C" int pearl_activation_prepare(const void* a, const void* key, const void* root_b,
        void* root_a, void* seed_a, void* noise, void* scratch, size_t bytes,
        int m, int n, cudaStream_t stream) {
    if (!a || !key || !root_b || !root_a || !seed_a || !noise || !scratch ||
        m <= 0 || m > max_rows || n <= 0 || bytes < pearl_activation_scratch(m)) return cudaErrorInvalidValue;
    uintptr_t pointers = uintptr_t(a) | uintptr_t(key) | uintptr_t(root_b) | uintptr_t(root_a) |
                         uintptr_t(seed_a) | uintptr_t(noise) | uintptr_t(scratch);
    if (pointers & 3) return cudaErrorInvalidValue;
    int count = 2 * m;
    auto layer = static_cast<uint32_t*>(scratch);
    auto job_key = static_cast<const uint32_t*>(key);
    leaves<<<(count + 31) / 32, 32, 0, stream>>>(static_cast<const uint32_t*>(a), job_key, layer, count);
    while (count > 1) {
        auto next = count == 2 ? static_cast<uint32_t*>(root_a) : layer + count * 8;
        int next_count = (count + 1) / 2;
        parents<<<(next_count + 63) / 64, 64, 0, stream>>>(layer, job_key, next, count);
        layer = next;
        count = next_count;
    }
    auto basis = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(scratch) + tree_bytes(m));
    auto pairs = reinterpret_cast<uint16_t*>(basis + size_t(m) * (noise_rank / 4));
    auto seed_b = reinterpret_cast<uint8_t*>(pairs + k);
    using Commit = pearl::CommitmentHashFromMerkleRootsKernel;
    Commit::Arguments args{static_cast<const uint8_t*>(root_a), static_cast<const uint8_t*>(root_b),
        static_cast<const uint8_t*>(key), static_cast<uint8_t*>(seed_a), seed_b, nullptr, nullptr,
        true, uint32_t(m), uint32_t(n)};
    cutlass::device_kernel<Commit><<<1, 1, 0, stream>>>(Commit::to_underlying_arguments(args));
    int dense_hashes = m * (noise_rank / 32);
    noise_factors<<<(dense_hashes + k / 8 + 63) / 64, 64, 0, stream>>>(
        static_cast<const uint32_t*>(seed_a), basis, pairs, dense_hashes);
    expand<<<m, 256, 0, stream>>>(basis, pairs, static_cast<uint32_t*>(noise));
    return cudaGetLastError();
}
