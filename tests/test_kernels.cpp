// CPU reference semantics against hand-computed values, then GPU kernels
// against the CPU reference, bit-exactly, over randomized synthetic batches.
// GPU cases skip (WARN) when no CUDA device is present.

#include "doctest/doctest.h"

#include "cuda/kernels.cuh"
#include "itch_builder.hpp"
#include "parser/itch_parser.hpp"
#include "parser/order_hash.hpp"
#include "pipeline/batch_assembler.hpp"
#include "pipeline/cpu_reference.hpp"

#include <random>
#include <vector>

using namespace itch;
using test::StreamBuilder;

namespace {

// Assemble a full stream into consecutive batch buffers.
struct Batches {
    std::vector<std::vector<uint8_t>> bufs;
    std::vector<BatchMeta> metas;
};

Batches assemble_all(const std::vector<uint8_t>& bytes) {
    ItchParser parser(make_mem_source(bytes.data(), bytes.size()));
    OrderHash orders(1 << 16);
    BatchAssembler assembler(parser, orders);
    Batches out;
    for (;;) {
        std::vector<uint8_t> buf(BATCH_MSGS * SLOT_BYTES);
        BatchMeta m = assembler.fill(buf.data());
        if (m.n_msgs == 0) break;
        out.bufs.push_back(std::move(buf));
        out.metas.push_back(m);
    }
    return out;
}

bool have_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace

TEST_CASE("cpu reference: VWAP, imbalance, window quotes, cancels") {
    StreamBuilder b;
    // Quotes: bids 12.30/12.31, asks 12.40/12.35 -> window 12.31 x 12.35.
    b.add_order(7, 1, 100, 'B', 50, "AAPL", 123000);
    b.add_order(7, 2, 101, 'B', 50, "AAPL", 123100);
    b.add_order(7, 3, 102, 'S', 50, "AAPL", 124000);
    b.add_order(7, 4, 103, 'S', 50, "AAPL", 123500);
    // Trades: 100 @ 12.34 buy-side, 50 @ 12.36 sell-side, 200 @ 12.35 cross.
    b.trade(7, 5, 0, 'B', 100, "AAPL", 123400, 1);
    b.trade(7, 6, 0, 'S', 50, "AAPL", 123600, 2);
    b.cross(7, 7, 200, "AAPL", 123500, 3, 'O');
    // Cancel 30 shares of order 100.
    b.cancel(7, 8, 100, 30);

    Batches bs = assemble_all(b.bytes());
    REQUIRE(bs.bufs.size() == 1);

    HostAccum acc;
    cpu_aggregate(bs.bufs[0].data(), bs.metas[0].n_msgs, acc);

    CHECK(acc.vwap_num[7] == 100ull * 123400 + 50ull * 123600 + 200ull * 123500);
    CHECK(acc.vwap_den[7] == 350);
    CHECK(acc.buy_vol[7] == 100);
    CHECK(acc.sell_vol[7] == 50);
    CHECK(acc.cancel_shares[7] == 30);
    CHECK(acc.trade_count[7] == 3);
    CHECK(acc.add_count[7] == 4);
    CHECK(acc.win_bid[7] == 123100);
    CHECK(acc.win_ask[7] == 123500);

    std::vector<Snapshot> snap(MAX_SYMBOLS);
    cpu_snapshot(acc, snap.data(), MAX_SYMBOLS);
    CHECK(snap[7].vwap == doctest::Approx(12.348571428571).epsilon(1e-9));
    CHECK(snap[7].volume == 350);
    CHECK(snap[7].imbalance == doctest::Approx((100.0 - 50.0) / 150.0));
    CHECK(snap[7].bid == 123100);
    CHECK(snap[7].ask == 123500);
    // Snapshot consumed the window.
    CHECK(acc.win_bid[7] == BID_EMPTY);
    CHECK(acc.win_ask[7] == ASK_EMPTY);
    // Untouched symbol stays clean.
    CHECK(snap[8].volume == 0);
    CHECK(snap[8].bid == BID_EMPTY);
    CHECK(snap[8].ask == ASK_EMPTY);
}

TEST_CASE("cpu reference: replace contributes via resolved side tag") {
    StreamBuilder b;
    b.add_order(7, 1, 100, 'B', 50, "AAPL", 123000);
    b.replace(7, 2, 100, 200, 50, 123200);  // bid improves via U
    Batches bs = assemble_all(b.bytes());
    HostAccum acc;
    cpu_aggregate(bs.bufs[0].data(), bs.metas[0].n_msgs, acc);
    CHECK(acc.win_bid[7] == 123200);
    CHECK(acc.add_count[7] == 2);
}

TEST_CASE("gpu kernels match cpu reference bit-exactly on random batches") {
    if (!have_gpu()) {
        WARN("no CUDA device - skipping GPU test");
        return;
    }

    // Random but valid stream: ~6 full batches across 50 symbols.
    StreamBuilder b;
    std::mt19937_64 rng(42);
    uint64_t next_ref = 1;
    std::vector<uint64_t> live;
    for (int i = 0; i < 50000; ++i) {
        const uint16_t locate = (uint16_t)(1 + rng() % 50);
        const uint32_t price = (uint32_t)(10000 + rng() % 4000000);
        const uint32_t shares = (uint32_t)(1 + rng() % 1000);
        const uint64_t ts = 34200000000000ull + i;
        switch (rng() % 7) {
            case 0:
            case 1:
                b.add_order(locate, ts, next_ref, (rng() & 1) ? 'B' : 'S',
                            shares, "SYM", price, rng() % 4 == 0);
                live.push_back(next_ref++);
                break;
            case 2:
                if (!live.empty())
                    b.cancel(locate, ts, live[rng() % live.size()],
                             (uint32_t)(1 + rng() % 100));
                break;
            case 3:
                if (!live.empty()) {
                    size_t k = rng() % live.size();
                    b.del(locate, ts, live[k]);
                    live[k] = live.back();
                    live.pop_back();
                }
                break;
            case 4:
                if (!live.empty()) {
                    size_t k = rng() % live.size();
                    b.replace(locate, ts, live[k], next_ref, shares, price);
                    live[k] = next_ref++;
                }
                break;
            case 5:
                b.trade(locate, ts, 0, (rng() & 1) ? 'B' : 'S', shares,
                        "SYM", price, i);
                break;
            case 6:
                b.cross(locate, ts, shares, "SYM", price, i, 'O');
                break;
        }
        if (i == 25000) b.sys_event(ts, 'O');  // mid-stream VWAP reset
    }

    Batches bs = assemble_all(b.bytes());
    REQUIRE(bs.bufs.size() >= 2);

    // CPU reference over all batches.
    HostAccum cpu;
    for (size_t i = 0; i < bs.bufs.size(); ++i) {
        if (bs.metas[i].reset_vwap) cpu.reset();
        cpu_aggregate(bs.bufs[i].data(), bs.metas[i].n_msgs, cpu);
    }

    // GPU over the same batches, single stream.
    using namespace itch::gpu;
    DeviceAccum acc = alloc_accum();
    launch_reset(acc, MAX_SYMBOLS, nullptr);
    uint8_t* d_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw, BATCH_MSGS * SLOT_BYTES));
    DecodedBatch soa = alloc_decoded();
    for (size_t i = 0; i < bs.bufs.size(); ++i) {
        if (bs.metas[i].reset_vwap) launch_reset(acc, MAX_SYMBOLS, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(d_raw, bs.bufs[i].data(),
                                   (size_t)bs.metas[i].n_msgs * SLOT_BYTES,
                                   cudaMemcpyHostToDevice, nullptr));
        launch_decode(d_raw, (int)bs.metas[i].n_msgs, soa, nullptr);
        launch_aggregate(soa, (int)bs.metas[i].n_msgs, acc, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto check_u64 = [&](const char* name, uint64_t* dev,
                         const std::vector<uint64_t>& ref) {
        std::vector<uint64_t> host(MAX_SYMBOLS);
        CUDA_CHECK(cudaMemcpy(host.data(), dev,
                              MAX_SYMBOLS * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        for (int s = 0; s < MAX_SYMBOLS; ++s) {
            INFO(name << " symbol " << s);
            REQUIRE(host[s] == ref[s]);
        }
    };
    auto check_u32 = [&](const char* name, uint32_t* dev,
                         const std::vector<uint32_t>& ref) {
        std::vector<uint32_t> host(MAX_SYMBOLS);
        CUDA_CHECK(cudaMemcpy(host.data(), dev,
                              MAX_SYMBOLS * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        for (int s = 0; s < MAX_SYMBOLS; ++s) {
            INFO(name << " symbol " << s);
            REQUIRE(host[s] == ref[s]);
        }
    };

    check_u64("vwap_num", acc.vwap_num, cpu.vwap_num);
    check_u64("vwap_den", acc.vwap_den, cpu.vwap_den);
    check_u64("buy_vol", acc.buy_vol, cpu.buy_vol);
    check_u64("sell_vol", acc.sell_vol, cpu.sell_vol);
    check_u64("cancel_shares", acc.cancel_shares, cpu.cancel_shares);
    check_u32("trade_count", acc.trade_count, cpu.trade_count);
    check_u32("add_count", acc.add_count, cpu.add_count);
    check_u32("win_bid", acc.win_bid, cpu.win_bid);
    check_u32("win_ask", acc.win_ask, cpu.win_ask);

    // Snapshot kernel vs CPU snapshot (bit-exact doubles: same operations).
    Snapshot* d_snap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_snap, MAX_SYMBOLS * sizeof(Snapshot)));
    launch_snapshot(acc, d_snap, MAX_SYMBOLS, nullptr);
    std::vector<Snapshot> gpu_snap(MAX_SYMBOLS);
    CUDA_CHECK(cudaMemcpy(gpu_snap.data(), d_snap,
                          MAX_SYMBOLS * sizeof(Snapshot),
                          cudaMemcpyDeviceToHost));
    std::vector<Snapshot> cpu_snap(MAX_SYMBOLS);
    cpu_snapshot(cpu, cpu_snap.data(), MAX_SYMBOLS);
    for (int s = 0; s < MAX_SYMBOLS; ++s) {
        INFO("snapshot symbol " << s);
        REQUIRE(gpu_snap[s].vwap == cpu_snap[s].vwap);
        REQUIRE(gpu_snap[s].volume == cpu_snap[s].volume);
        REQUIRE(gpu_snap[s].buy_vol == cpu_snap[s].buy_vol);
        REQUIRE(gpu_snap[s].sell_vol == cpu_snap[s].sell_vol);
        REQUIRE(gpu_snap[s].cancel_shares == cpu_snap[s].cancel_shares);
        REQUIRE(gpu_snap[s].bid == cpu_snap[s].bid);
        REQUIRE(gpu_snap[s].ask == cpu_snap[s].ask);
        REQUIRE(gpu_snap[s].trades == cpu_snap[s].trades);
        REQUIRE(gpu_snap[s].adds == cpu_snap[s].adds);
        REQUIRE(gpu_snap[s].imbalance == cpu_snap[s].imbalance);
    }

    cudaFree(d_snap);
    free_decoded(soa);
    cudaFree(d_raw);
    free_accum(acc);
}
