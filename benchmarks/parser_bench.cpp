// Parser throughput, isolated from gzip inflate: pre-decompress the input
// into memory, then time framing + dispatch + order-hash + batch assembly.
//
// Usage:
//   parser_bench --file data/fixture.itch.gz   (decompressed to RAM first)
//   parser_bench --file data/fixture.itch      (raw ITCH, mmap-less read)
//   parser_bench --synthetic 20000000          (generated in RAM)

#include "itch_builder.hpp"
#include "parser/itch_parser.hpp"
#include "parser/order_hash.hpp"
#include "pipeline/batch_assembler.hpp"

#include <zlib.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace itch;

namespace {

std::vector<uint8_t> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<uint8_t> data((size_t)f.tellg());
    f.seekg(0);
    f.read((char*)data.data(), (std::streamsize)data.size());
    return data;
}

std::vector<uint8_t> read_gz(const std::string& path) {
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    gzbuffer(f, 1u << 20);
    std::vector<uint8_t> data;
    std::vector<uint8_t> chunk(16u << 20);
    int r;
    while ((r = gzread(f, chunk.data(), (unsigned)chunk.size())) > 0)
        data.insert(data.end(), chunk.data(), chunk.data() + r);
    gzclose(f);
    return data;
}

std::vector<uint8_t> synthesize(uint64_t n) {
    test::StreamBuilder b;
    std::mt19937_64 rng(7);
    uint64_t ref = 1;
    std::vector<uint64_t> live;
    for (uint64_t i = 0; i < n; ++i) {
        const uint16_t locate = (uint16_t)(1 + rng() % 2000);
        const uint32_t price = (uint32_t)(10000 + rng() % 4000000);
        const uint32_t shares = (uint32_t)(1 + rng() % 1000);
        switch (rng() % 10) {  // roughly ITCH-like mix: adds/deletes dominate
            case 0: case 1: case 2: case 3:
                b.add_order(locate, i, ref, (rng() & 1) ? 'B' : 'S', shares,
                            "SYM", price);
                live.push_back(ref++);
                break;
            case 4: case 5: case 6:
                if (!live.empty()) {
                    size_t k = rng() % live.size();
                    b.del(locate, i, live[k]);
                    live[k] = live.back();
                    live.pop_back();
                }
                break;
            case 7:
                if (!live.empty()) {
                    size_t k = rng() % live.size();
                    b.replace(locate, i, live[k], ref, shares, price);
                    live[k] = ref++;
                }
                break;
            case 8:
                if (!live.empty())
                    b.cancel(locate, i, live[rng() % live.size()], 1 + (uint32_t)(rng() % 50));
                break;
            case 9:
                b.trade(locate, i, 0, (rng() & 1) ? 'B' : 'S', shares, "SYM",
                        price, i);
                break;
        }
    }
    return b.bytes();
}

}  // namespace

int main(int argc, char** argv) {
    std::string file;
    uint64_t synthetic = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--file" && i + 1 < argc) file = argv[++i];
        else if (a == "--synthetic" && i + 1 < argc) synthetic = std::strtoull(argv[++i], nullptr, 10);
    }
    if (file.empty() && !synthetic) {
        std::fprintf(stderr, "usage: %s --file PATH | --synthetic N\n", argv[0]);
        return 2;
    }

    std::vector<uint8_t> data;
    if (!file.empty()) {
        const bool gz = file.size() > 3 && file.compare(file.size() - 3, 3, ".gz") == 0;
        std::fprintf(stderr, "loading %s into RAM%s...\n", file.c_str(),
                     gz ? " (decompressing)" : "");
        data = gz ? read_gz(file) : read_raw(file);
    } else {
        std::fprintf(stderr, "synthesizing %" PRIu64 " messages...\n", synthetic);
        data = synthesize(synthetic);
    }
    std::fprintf(stderr, "input: %.1f MB raw\n", data.size() / 1e6);

    // Pass 1: framing only (parser ceiling).
    {
        ItchParser p(make_mem_source(data.data(), data.size()));
        uint16_t len = 0;
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t n = 0, bytes = 0;
        while (p.next(&len)) { ++n; bytes += len; }
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("framing-only: %" PRIu64 " msgs in %.3fs = %.2fM msg/s, %.2f GB/s\n",
                    n, dt, n / dt / 1e6, bytes / dt / 1e9);
    }

    // Pass 2: framing + order hash + batch assembly (production CPU path).
    {
        ItchParser p(make_mem_source(data.data(), data.size()));
        OrderHash orders;
        BatchAssembler assembler(p, orders);
        std::vector<uint8_t> batch(BATCH_MSGS * SLOT_BYTES);
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t batches = 0, shipped = 0;
        for (;;) {
            BatchMeta m = assembler.fill(batch.data());
            if (m.n_msgs == 0) break;
            ++batches;
            shipped += m.n_msgs;
        }
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("full-assembly: %" PRIu64 " msgs (%" PRIu64 " batches) in %.3fs = %.2fM msg/s\n",
                    p.total_messages(), batches, dt, p.total_messages() / dt / 1e6);
        std::printf("shipped-to-gpu: %" PRIu64 " msgs (%.1f%%)\n", shipped,
                    100.0 * (double)shipped / (double)p.total_messages());
    }
    return 0;
}
