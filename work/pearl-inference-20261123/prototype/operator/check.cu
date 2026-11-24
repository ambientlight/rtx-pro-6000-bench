// Full-shard correctness test. Host reference products use all original elements.
// Time only the already-prepared GPU work separately from the host proof oracle.
#include "pearl_wo_b.h"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
void* pearl_ref_new(const uint8_t*, const int8_t*, const int8_t*, int8_t*, int8_t*, int, int, int);
void pearl_ref_free(void*);
int pearl_ref_scan(const void*, const uint32_t*, size_t, uint32_t, uint64_t*);
ptrdiff_t pearl_ref_proof(const void*, size_t, uint32_t, uint8_t*, size_t);
}

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc)); std::exit(1); }
}
template<class T> struct Gpu {
    T* data = nullptr;
    explicit Gpu(size_t count) { check(cudaMalloc(&data,count*sizeof(T))); }
    ~Gpu() { cudaFree(data); }
    void upload(const std::vector<T>& v) { check(cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    void download(std::vector<T>& v) { check(cudaMemcpy(v.data(),data,v.size()*sizeof(T),cudaMemcpyDeviceToHost)); }
};

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "prefill" && std::string(argv[1]) != "decode")) return 2;
    const bool compact = std::string(argv[1]) == "decode";
    const int m = compact ? 6 : 2048, n = 5120, k = 2048;
    const int tickets = (m/2)*(n/(compact ? 16 : 64));
    // 2x16 and 2x64 tickets both have adjusted test threshold 2^254.
    const uint32_t bits = compact ? 0x1e400000 : 0x1e100000;
    std::array<uint8_t,76> header{};
    for (int i = 0; i < 32; ++i) { header[4+i] = 1; header[36+i] = 2; }
    for (int i = 0; i < 4; ++i) { header[68+i] = 0x66; header[72+i] = bits >> (8*i); }
    std::vector<int8_t> a(m*k), bt(n*k), na(m*k), nb(n*k);
    std::vector<int32_t> out(m*n);
    std::vector<uint32_t> words(tickets*16);
    uint32_t seed = 73421;
    for (auto* data : {&a, &bt}) for (auto& v : *data) {
        seed = seed*1664525+1013904223;
        v = int((seed >> 16) % 129) - 64;
    }
    void* ref = pearl_ref_new(header.data(),a.data(),bt.data(),na.data(),nb.data(),m,n,compact);
    if (!ref) { std::fprintf(stderr, "could not prepare reference\n"); return 1; }
    Gpu<int8_t> ga(a.size()), gb(bt.size()), gna(na.size()), gnb(nb.size());
    Gpu<int32_t> go(out.size());
    Gpu<uint32_t> gw(words.size());
    ga.upload(a); gb.upload(bt); gna.upload(na); gnb.upload(nb);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t start,end;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    std::vector<float> samples;
    constexpr int warmup = 5, repetitions = 32;
    for (int trial=0; trial<warmup+repetitions; ++trial) {
        check(cudaEventRecord(start,stream));
        check(pearl_wo_b_i7(ga.data,gb.data,gna.data,gnb.data,go.data,gw.data,m,n,k,compact,stream));
        check(cudaEventRecord(end,stream));
        check(cudaEventSynchronize(end));
        float elapsed; check(cudaEventElapsedTime(&elapsed,start,end));
        if (trial >= warmup) samples.push_back(elapsed);
    }
    auto sorted = samples;
    std::sort(sorted.begin(),sorted.end());
    go.download(out); gw.download(words);
    // CUDA graph capture is part of the decode path. Reuse preallocated buffers
    // and check that the captured launch writes the same output and transcripts.
    if (compact) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        check(pearl_wo_b_i7(ga.data,gb.data,gna.data,gnb.data,go.data,gw.data,m,n,k,compact,stream));
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiateWithFlags(&exec,graph,0));
        check(cudaMemsetAsync(go.data,0x5a,out.size()*sizeof(int32_t),stream));
        check(cudaMemsetAsync(gw.data,0x5a,words.size()*sizeof(uint32_t),stream));
        check(cudaGraphLaunch(exec,stream)); check(cudaStreamSynchronize(stream));
        std::vector<int32_t> replay_out(out.size());
        std::vector<uint32_t> replay_words(words.size());
        go.download(replay_out); gw.download(replay_words);
        if (replay_out != out || replay_words != words) { std::fprintf(stderr,"graph replay mismatch\n"); return 1; }
        check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    }
    // Check every checkpoint word using a scalar-layout host accumulation. Pre-add
    // exact signed-byte noise once on CPU, independent of the device fragment mapping.
    std::vector<int8_t> mix_a(a.size()), mix_bt(bt.size());
    std::transform(a.begin(),a.end(),na.begin(),mix_a.begin(),[](int x,int y){ return x+y; });
    std::transform(bt.begin(),bt.end(),nb.begin(),mix_bt.begin(),[](int x,int y){ return x+y; });
    size_t output_errors = 0, word_errors = 0;
    const int sampled_cols = compact ? 16 : 64;
    #pragma omp parallel for reduction(+:output_errors,word_errors)
    for (int ticket = 0; ticket < tickets; ++ticket) {
        const int ir = ticket / (n / sampled_cols), ic = ticket % (n / sampled_cols);
        const int first_row = compact ? 2*ir : (ir/8)*16 + ir%8;
        const int first_col = compact ? 16*ic : (ic/4)*256 + 2*(ic%4);
        std::array<uint32_t,16> expected_words{};
        for (int r = 0; r < 2; ++r) {
            const int row = first_row + (compact ? r : 8*r);
            const int8_t* ar = a.data() + row*k;
            const int8_t* mixed_ar = mix_a.data() + row*k;
            for (int c = 0; c < sampled_cols; ++c) {
                const int col = first_col + (compact ? c : (c/2)*8 + c%2);
                const int8_t* br = bt.data() + col*k;
                const int8_t* mixed_br = mix_bt.data() + col*k;
                int32_t expected = 0, noisy = 0;
                for (int cp = 0; cp < 16; ++cp) {
                    #pragma omp simd reduction(+:expected,noisy)
                    for (int i = cp*128; i < (cp+1)*128; ++i) {
                        expected += int(ar[i])*int(br[i]);
                        noisy += int(mixed_ar[i])*int(mixed_br[i]);
                    }
                    expected_words[cp] ^= static_cast<uint32_t>(noisy);
                }
                output_errors += (expected != out[row*n+col]);
            }
        }
        for (int cp = 0; cp < 16; ++cp)
            word_errors += (expected_words[cp] != words[ticket*16+cp]);
    }
    if (output_errors || word_errors) {
        std::fprintf(stderr,"output mismatches: %zu, transcript word mismatches: %zu\n",output_errors,word_errors);
        return 1;
    }
    uint64_t stats[4]{};
    int rc = pearl_ref_scan(ref,words.data(),tickets,bits,stats);
    if (rc != 0 || !stats[1] || !stats[2]) {
        std::fprintf(stderr,"reference check failed: %d, hits %llu\n",rc,(unsigned long long)stats[0]); return 1;
    }
    std::vector<uint8_t> proof(200000);
    const ptrdiff_t bytes = pearl_ref_proof(ref,stats[1]-1,bits,proof.data(),proof.size());
    if (bytes <= 0) return 1;
    std::printf("{\"phase\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"exact_outputs\":%zu,"
                "\"gpu_tickets\":%d,\"exact_checkpoint_words\":%zu,\"hits\":%llu,\"first_hit\":%llu,\"first_miss\":%llu,"
                "\"official_checks\":%llu,\"witness_bytes\":%td,\"graph_replay_checked\":%s,"
                "\"warmups\":%d,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"max_ms\":%.6f,\"samples_ms\":[",
                argv[1],m,n,k,out.size(),tickets,words.size(),(unsigned long long)stats[0],(unsigned long long)(stats[1]-1),
                (unsigned long long)(stats[2]-1),(unsigned long long)stats[3],bytes,compact?"true":"false",
                warmup,(sorted[15]+sorted[16])/2,sorted[30],sorted.back());
    for (size_t i=0; i<samples.size(); ++i) std::printf("%s%.6f",i?",":"",samples[i]);
    std::puts("]}");
    pearl_ref_free(ref);
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end)); check(cudaStreamDestroy(stream));
}
