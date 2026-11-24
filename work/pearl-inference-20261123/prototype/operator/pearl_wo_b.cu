#include "pearl_wo_b.h"
#include <cuda_runtime.h>

namespace {
constexpr int kDim = 2048;
constexpr int rank = 128;
constexpr int checkpoints = kDim / rank;

__device__ __forceinline__ void mma(int (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One warp computes 16 rows by 8 output columns. All of the tensor-core
// accumulators feed both the checkpoint reductions and the corrected result.
template<bool Compact>
__global__ void projection(const unsigned* a, const unsigned* bt, const unsigned* na,
                           const unsigned* nb, int* out, unsigned* transcript,
                           int m, int n) {
    constexpr int warps = Compact ? 2 : 32;
    constexpr int tile_cols = 8 * warps;
    __shared__ unsigned partial[Compact ? 128 : 1024];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int group = lane / 4, sub = lane % 4;
    const int row0 = blockIdx.y * 16, col0 = blockIdx.x * tile_cols;
    const int acol = col0 + warp * 8 + group;
    int product[4] = {}, correction[4] = {};

    for (int cp = 0; cp < checkpoints; ++cp) {
        #pragma unroll
        for (int step = 0; step < rank / 32; ++step) {
            unsigned orig_a[4], noise_a[4], mix_a[4], mix_b[2], noise_b[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int offset = cp * rank / 4 + step * 8 + h * 4 + sub;
                const int boff = acol * (kDim / 4) + offset;
                noise_b[h] = nb[boff];
                mix_b[h] = __vadd4(bt[boff], noise_b[h]);
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = row0 + group + 8 * r;
                    const int aoff = row * (kDim / 4) + offset;
                    const int f = 2 * h + r;
                    orig_a[f] = row < m ? a[aoff] : 0;
                    noise_a[f] = row < m ? na[aoff] : 0;
                    mix_a[f] = __vadd4(orig_a[f], noise_a[f]);
                }
            }
            mma(product, mix_a, mix_b);
            // (A+NA)(B+NB) - NA(B+NB) - A*NB = A*B.
            mma(correction, noise_a, mix_b);
            mma(correction, orig_a, noise_b);
        }

        if constexpr (Compact) {
            const int paircol = warp * 4 + sub;
            partial[group * 8 + paircol] = unsigned(product[0] ^ product[1]);
            partial[(group + 8) * 8 + paircol] = unsigned(product[2] ^ product[3]);
            __syncthreads();
            const int row_pair = warp * 4 + lane / 8;
            const int slot = (2 * row_pair) * 8 + lane % 8;
            unsigned word = partial[slot] ^ partial[slot + 8];
            #pragma unroll
            for (int delta = 4; delta > 0; delta /= 2)
                word ^= __shfl_xor_sync(0xffffffff, word, delta, 8);
            if (lane % 8 == 0 && row0 + 2 * row_pair < m) {
                const int ticket = (row0 / 2 + row_pair) * (n / 16) + col0 / 16;
                transcript[ticket * checkpoints + cp] = word;
            }
        } else {
            // Rows [g,g+8], columns 2*sub + {0,1,8,9,...,248,249}.
            partial[warp * 32 + lane] = unsigned(product[0] ^ product[1] ^ product[2] ^ product[3]);
            __syncthreads();
            const unsigned word = __reduce_xor_sync(0xffffffff, partial[lane * 32 + warp]);
            if (lane == 0) {
                const int ticket = (row0 / 2 + warp / 4) * (n / 64) + col0 / 64 + warp % 4;
                transcript[ticket * checkpoints + cp] = word;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + group + 8 * (i / 2);
        const int col = col0 + warp * 8 + 2 * sub + i % 2;
        if (row < m) out[row * n + col] = product[i] - correction[i];
    }
}
// The plain path uses the same signed tensor-core arithmetic and fragment
// layout. Predicated loads cover partial tiles, including a single real row.
template<int Warps>
__global__ void plain_projection(const unsigned* a, const unsigned* bt, int* out, int m, int n) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int group = lane / 4, sub = lane % 4;
    const int row0 = blockIdx.y * 16 + group, col0 = 8 * (blockIdx.x * Warps + warp);
    const int b_row = col0 + group;
    int product[4] = {};

    for (int k = 0; k < kDim / 4; k += 8) {
        unsigned af[4], bf[2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int offset = k + 4 * h + sub;
            bf[h] = b_row < n ? bt[b_row * (kDim / 4) + offset] : 0;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + 8 * r;
                af[2 * h + r] = row < m ? a[row * (kDim / 4) + offset] : 0;
            }
        }
        mma(product, af, bf);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + 8 * (i / 2), col = col0 + 2 * sub + i % 2;
        if (row < m && col < n) out[row * n + col] = product[i];
    }
}
} // namespace

extern "C" cudaError_t pearl_wo_b_i7(const int8_t* a, const int8_t* bt, const int8_t* na,
                                     const int8_t* nb, int32_t* out, uint32_t* words,
                                     int m, int n, int k, int compact, cudaStream_t stream) {
    if (!a || !bt || !na || !nb || !out || !words || k != kDim || m <= 0 || n <= 0 ||
        (compact != 0 && compact != 1) || m % (compact ? 2 : 16) || n % (compact ? 16 : 256) ||
        m > 65535 * 16 || int64_t(m) * n > INT32_MAX || int64_t(n) * k > INT32_MAX ||
        int64_t(m) * k > INT32_MAX) return cudaErrorInvalidValue;
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(bt)
                             | reinterpret_cast<uintptr_t>(na) | reinterpret_cast<uintptr_t>(nb)
                             | reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(words);
    if (pointers & 3) return cudaErrorInvalidValue;
    auto args_a = reinterpret_cast<const unsigned*>(a);
    auto args_b = reinterpret_cast<const unsigned*>(bt);
    auto args_na = reinterpret_cast<const unsigned*>(na);
    auto args_nb = reinterpret_cast<const unsigned*>(nb);
    const dim3 grid(n / (compact ? 16 : 256), (m + 15) / 16);
    if (compact)
        projection<true><<<grid, 64, 0, stream>>>(args_a, args_b, args_na, args_nb, out, words, m, n);
    else
        projection<false><<<grid, 1024, 0, stream>>>(args_a, args_b, args_na, args_nb, out, words, m, n);
    return cudaGetLastError();
}

extern "C" cudaError_t pearl_wo_b_i7_plain(const int8_t* a, const int8_t* bt, int32_t* out,
                                           int m, int n, int k, cudaStream_t stream) {
    if (!a || !bt || !out || k != kDim || m <= 0 || n <= 0 || m > 65535 * 16 ||
        int64_t(m) * n > INT32_MAX || int64_t(n) * k > INT32_MAX || int64_t(m) * k > INT32_MAX)
        return cudaErrorInvalidValue;
    const uintptr_t addresses = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(bt)
                              | reinterpret_cast<uintptr_t>(out);
    if (addresses & 3) return cudaErrorInvalidValue;
    const auto av = reinterpret_cast<const unsigned*>(a);
    const auto bv = reinterpret_cast<const unsigned*>(bt);
    // Small decode batches need enough CTAs to spread the weight reads across SMs.
    if (m <= 16)
        plain_projection<1><<<dim3((n + 7) / 8, 1), 32, 0, stream>>>(av, bv, out, m, n);
    else
        plain_projection<4><<<dim3((n + 31) / 32, (m + 15) / 16), 128, 0, stream>>>(av, bv, out, m, n);
    return cudaGetLastError();
}
