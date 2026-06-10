#pragma once
// Device-side data structures and kernel launch wrappers. Included by .cu
// translation units and by host-only code (main.cpp), so everything here is
// plain CUDA runtime API - kernels themselves live in the .cu files.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "parser/message_types.hpp"
#include "pipeline/snapshot.hpp"

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err__ = (expr);                                           \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                         cudaGetErrorName(err__), __FILE__, __LINE__,        \
                         cudaGetErrorString(err__));                         \
            std::exit(1);                                                    \
        }                                                                     \
    } while (0)

namespace itch::gpu {

using itch::ASK_EMPTY;
using itch::BID_EMPTY;
using itch::Snapshot;

// Per-symbol accumulators, SoA device arrays of length MAX_SYMBOLS.
// All atomically updated by the aggregate kernel; persistent across batches.
struct DeviceAccum {
    uint64_t* vwap_num;       // sum(price * shares) over trades
    uint64_t* vwap_den;       // sum(shares) over trades
    uint64_t* buy_vol;        // P-side 'B' shares
    uint64_t* sell_vol;       // P-side 'S' shares
    uint64_t* cancel_shares;  // X canceled shares
    uint32_t* trade_count;
    uint32_t* add_count;      // A/F/U placements
    uint32_t* win_bid;        // window best bid (atomicMax), reset by snapshot
    uint32_t* win_ask;        // window best ask (atomicMin), reset by snapshot
};

// Decoded batch, SoA device arrays of capacity BATCH_MSGS. qty is u64 because
// Q (cross trade) carries 64-bit shares; everything else fits trivially.
struct DecodedBatch {
    uint8_t*  type;
    uint8_t*  side;    // 'B'/'S' for A/F/P; resolved tag for U; 0 otherwise
    uint16_t* locate;
    uint32_t* price;
    uint64_t* qty;
    uint64_t* ts;
};

// ---------------------------------------------------------------------------
// Allocation helpers (host)
// ---------------------------------------------------------------------------
inline DeviceAccum alloc_accum() {
    DeviceAccum a{};
    auto u64 = [](uint64_t*& p) { CUDA_CHECK(cudaMalloc(&p, MAX_SYMBOLS * sizeof(uint64_t))); };
    auto u32 = [](uint32_t*& p) { CUDA_CHECK(cudaMalloc(&p, MAX_SYMBOLS * sizeof(uint32_t))); };
    u64(a.vwap_num); u64(a.vwap_den); u64(a.buy_vol); u64(a.sell_vol);
    u64(a.cancel_shares);
    u32(a.trade_count); u32(a.add_count); u32(a.win_bid); u32(a.win_ask);
    return a;
}

inline void free_accum(DeviceAccum& a) {
    cudaFree(a.vwap_num); cudaFree(a.vwap_den); cudaFree(a.buy_vol);
    cudaFree(a.sell_vol); cudaFree(a.cancel_shares); cudaFree(a.trade_count);
    cudaFree(a.add_count); cudaFree(a.win_bid); cudaFree(a.win_ask);
    a = DeviceAccum{};
}

inline DecodedBatch alloc_decoded() {
    DecodedBatch d{};
    CUDA_CHECK(cudaMalloc(&d.type,   BATCH_MSGS * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d.side,   BATCH_MSGS * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d.locate, BATCH_MSGS * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d.price,  BATCH_MSGS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.qty,    BATCH_MSGS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.ts,     BATCH_MSGS * sizeof(uint64_t)));
    return d;
}

inline void free_decoded(DecodedBatch& d) {
    cudaFree(d.type); cudaFree(d.side); cudaFree(d.locate);
    cudaFree(d.price); cudaFree(d.qty); cudaFree(d.ts);
    d = DecodedBatch{};
}

// ---------------------------------------------------------------------------
// Kernel launch wrappers (defined in the .cu files)
// ---------------------------------------------------------------------------
// Byteswap + field extraction: raw 48-byte slots -> SoA.
void launch_decode(const uint8_t* d_raw, int n_msgs, DecodedBatch out,
                   cudaStream_t stream);

// Per-symbol atomic accumulation.
void launch_aggregate(DecodedBatch in, int n_msgs, DeviceAccum acc,
                      cudaStream_t stream);

// Final per-symbol metrics; also resets window bid/ask to sentinels.
void launch_snapshot(DeviceAccum acc, Snapshot* d_out, int n_symbols,
                     cudaStream_t stream);

// Zero VWAP/volume/count accumulators and window bid/ask (market open, and
// initial state after alloc_accum).
void launch_reset(DeviceAccum acc, int n_symbols, cudaStream_t stream);

}  // namespace itch::gpu
