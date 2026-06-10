#pragma once
// CPU twin of the GPU decode + aggregate + snapshot path. Used by the
// --cpu-only pipeline mode and by tests that require the GPU results to be
// bit-exactly equal to this implementation. All arithmetic is integer until
// snapshot time, mirroring the kernels exactly.

#include "parser/message_types.hpp"
#include "pipeline/snapshot.hpp"

#include <cstdint>
#include <vector>

namespace itch {

struct HostAccum {
    std::vector<uint64_t> vwap_num, vwap_den, buy_vol, sell_vol, cancel_shares;
    std::vector<uint32_t> trade_count, add_count, win_bid, win_ask;

    HostAccum()
        : vwap_num(MAX_SYMBOLS), vwap_den(MAX_SYMBOLS), buy_vol(MAX_SYMBOLS),
          sell_vol(MAX_SYMBOLS), cancel_shares(MAX_SYMBOLS),
          trade_count(MAX_SYMBOLS), add_count(MAX_SYMBOLS),
          win_bid(MAX_SYMBOLS), win_ask(MAX_SYMBOLS) {
        reset();
    }

    // Mirrors the GPU reset kernel: zero everything, sentinel the window.
    void reset() {
        std::fill(vwap_num.begin(), vwap_num.end(), 0);
        std::fill(vwap_den.begin(), vwap_den.end(), 0);
        std::fill(buy_vol.begin(), buy_vol.end(), 0);
        std::fill(sell_vol.begin(), sell_vol.end(), 0);
        std::fill(cancel_shares.begin(), cancel_shares.end(), 0);
        std::fill(trade_count.begin(), trade_count.end(), 0);
        std::fill(add_count.begin(), add_count.end(), 0);
        std::fill(win_bid.begin(), win_bid.end(), BID_EMPTY);
        std::fill(win_ask.begin(), win_ask.end(), ASK_EMPTY);
    }
};

// Aggregate one batch of 48-byte slots (decode folded in - the GPU's separate
// decode stage is an implementation detail of the device memory layout).
void cpu_aggregate(const uint8_t* slots, uint32_t n_msgs, HostAccum& acc);

// Mirrors the GPU snapshot kernel, including consuming the quote window.
void cpu_snapshot(HostAccum& acc, Snapshot* out, int n_symbols);

}  // namespace itch
