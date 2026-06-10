#pragma once
// Per-symbol output record and quote-window sentinels. CUDA-free so the CPU
// reference path builds without the toolkit headers; kernels.cuh re-exports
// these for device code.

#include <cstdint>

namespace itch {

constexpr uint32_t BID_EMPTY = 0u;           // win_bid sentinel (atomicMax)
constexpr uint32_t ASK_EMPTY = 0xFFFFFFFFu;  // win_ask sentinel (atomicMin)

struct Snapshot {
    double   vwap;           // dollars (price units / 10000)
    uint64_t volume;
    uint64_t buy_vol;
    uint64_t sell_vol;
    uint64_t cancel_shares;
    uint32_t bid;            // raw 1/10000 USD; BID_EMPTY if no window bid
    uint32_t ask;            // raw 1/10000 USD; ASK_EMPTY if no window ask
    uint32_t trades;
    uint32_t adds;
    float    imbalance;      // (buy-sell)/(buy+sell), 0 if no sided flow
    uint32_t _pad;
};
static_assert(sizeof(Snapshot) == 64);

}  // namespace itch
