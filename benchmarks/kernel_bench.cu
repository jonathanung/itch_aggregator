// Kernel-level timings via cudaEvents: H2D transfer, decode, aggregate,
// snapshot, and the combined per-batch hot path. Batches come from the
// fixture (realistic symbol skew) or are synthesized.
//
// Usage: kernel_bench [--file data/fixture.itch.gz] [--batches N]

#include "cuda/kernels.cuh"
#include "itch_builder.hpp"
#include "parser/itch_parser.hpp"
#include "parser/order_hash.hpp"
#include "pipeline/batch_assembler.hpp"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace itch;
using namespace itch::gpu;

namespace {

struct HostBatches {
    std::vector<uint8_t*> pinned;   // BATCH_MSGS*SLOT_BYTES each
    std::vector<uint32_t> n_msgs;
    ~HostBatches() { for (auto* p : pinned) cudaFreeHost(p); }
};

void load_batches(const std::string& file, size_t max_batches, HostBatches& out) {
    std::unique_ptr<ByteSource> src;
    std::vector<uint8_t> synth;
    if (!file.empty()) {
        src = make_gz_source(file);
    } else {
        test::StreamBuilder b;
        std::mt19937_64 rng(7);
        for (size_t i = 0; i < max_batches * BATCH_MSGS; ++i) {
            const uint16_t locate = (uint16_t)(1 + rng() % 2000);
            if (rng() % 3 == 0)
                b.trade(locate, i, 0, (rng() & 1) ? 'B' : 'S',
                        (uint32_t)(1 + rng() % 500), "SYM",
                        (uint32_t)(10000 + rng() % 4000000), i);
            else
                b.add_order(locate, i, i + 1, (rng() & 1) ? 'B' : 'S',
                            (uint32_t)(1 + rng() % 500), "SYM",
                            (uint32_t)(10000 + rng() % 4000000));
        }
        synth = b.bytes();
        src = make_mem_source(synth.data(), synth.size());
    }

    ItchParser parser(std::move(src));
    OrderHash orders;
    BatchAssembler assembler(parser, orders);
    while (out.pinned.size() < max_batches) {
        uint8_t* buf = nullptr;
        CUDA_CHECK(cudaMallocHost(&buf, BATCH_MSGS * SLOT_BYTES));
        BatchMeta m = assembler.fill(buf);
        if (m.n_msgs == 0) { cudaFreeHost(buf); break; }
        out.pinned.push_back(buf);
        out.n_msgs.push_back(m.n_msgs);
    }
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(a, s)); }
    float stop_ms(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string file;
    size_t n_batches = 256;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--file" && i + 1 < argc) file = argv[++i];
        else if (a == "--batches" && i + 1 < argc) n_batches = std::strtoull(argv[++i], nullptr, 10);
    }

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::fprintf(stderr, "preparing %zu batches%s...\n", n_batches,
                 file.empty() ? " (synthetic)" : file.c_str());
    HostBatches hb;
    load_batches(file, n_batches, hb);
    if (hb.pinned.empty()) { std::fprintf(stderr, "no batches\n"); return 1; }
    std::fprintf(stderr, "loaded %zu batches\n", hb.pinned.size());

    uint8_t* d_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw, BATCH_MSGS * SLOT_BYTES));
    DecodedBatch soa = alloc_decoded();
    DeviceAccum acc = alloc_accum();
    launch_reset(acc, MAX_SYMBOLS, nullptr);
    Snapshot* d_snap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_snap, MAX_SYMBOLS * sizeof(Snapshot)));
    CUDA_CHECK(cudaDeviceSynchronize());

    Timer t;
    double h2d_ms = 0, dec_ms = 0, agg_ms = 0, snap_ms = 0, hot_ms = 0;
    uint64_t total_msgs = 0, total_bytes = 0;

    // Warmup.
    for (int w = 0; w < 8; ++w) {
        const size_t i = w % hb.pinned.size();
        CUDA_CHECK(cudaMemcpyAsync(d_raw, hb.pinned[i],
                                   (size_t)hb.n_msgs[i] * SLOT_BYTES,
                                   cudaMemcpyHostToDevice, nullptr));
        launch_decode(d_raw, (int)hb.n_msgs[i], soa, nullptr);
        launch_aggregate(soa, (int)hb.n_msgs[i], acc, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < hb.pinned.size(); ++i) {
        const size_t bytes = (size_t)hb.n_msgs[i] * SLOT_BYTES;
        total_msgs += hb.n_msgs[i];
        total_bytes += bytes;

        t.start(nullptr);
        CUDA_CHECK(cudaMemcpyAsync(d_raw, hb.pinned[i], bytes,
                                   cudaMemcpyHostToDevice, nullptr));
        h2d_ms += t.stop_ms(nullptr);

        t.start(nullptr);
        launch_decode(d_raw, (int)hb.n_msgs[i], soa, nullptr);
        dec_ms += t.stop_ms(nullptr);

        t.start(nullptr);
        launch_aggregate(soa, (int)hb.n_msgs[i], acc, nullptr);
        agg_ms += t.stop_ms(nullptr);
    }

    // Snapshot cost (amortized once per interval in production).
    for (int i = 0; i < 32; ++i) {
        t.start(nullptr);
        launch_snapshot(acc, d_snap, MAX_SYMBOLS, nullptr);
        snap_ms += t.stop_ms(nullptr);
    }
    snap_ms /= 32.0;

    // Hot path: H2D + decode + aggregate measured as one chain per batch.
    for (size_t i = 0; i < hb.pinned.size(); ++i) {
        const size_t bytes = (size_t)hb.n_msgs[i] * SLOT_BYTES;
        t.start(nullptr);
        CUDA_CHECK(cudaMemcpyAsync(d_raw, hb.pinned[i], bytes,
                                   cudaMemcpyHostToDevice, nullptr));
        launch_decode(d_raw, (int)hb.n_msgs[i], soa, nullptr);
        launch_aggregate(soa, (int)hb.n_msgs[i], acc, nullptr);
        hot_ms += t.stop_ms(nullptr);
    }

    const double nb = (double)hb.pinned.size();
    std::printf("batches=%zu msgs/batch=%.0f\n", hb.pinned.size(), total_msgs / nb);
    std::printf("H2D      : %8.2f us/batch  %7.2f GB/s\n",
                1e3 * h2d_ms / nb, total_bytes / (h2d_ms / 1e3) / 1e9);
    std::printf("decode   : %8.2f us/batch  %7.2fM msg/s\n",
                1e3 * dec_ms / nb, total_msgs / (dec_ms / 1e3) / 1e6);
    std::printf("aggregate: %8.2f us/batch  %7.2fM msg/s\n",
                1e3 * agg_ms / nb, total_msgs / (agg_ms / 1e3) / 1e6);
    std::printf("snapshot : %8.2f us (16384 symbols)\n", 1e3 * snap_ms);
    std::printf("hot path : %8.2f us/batch (H2D+decode+aggregate)  %7.2fM msg/s\n",
                1e3 * hot_ms / nb, total_msgs / (hot_ms / 1e3) / 1e6);

    cudaFree(d_snap);
    free_accum(acc);
    free_decoded(soa);
    cudaFree(d_raw);
    return 0;
}
