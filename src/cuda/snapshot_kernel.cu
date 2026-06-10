// Snapshot kernel: one thread per symbol. Reads accumulators, computes final
// metrics, and consumes the bid/ask window (resets it to sentinels) so each
// snapshot reports quote extremes for its own interval.

#include "cuda/kernels.cuh"

namespace itch::gpu {

__global__ void snapshot_kernel(DeviceAccum acc, Snapshot* __restrict__ out,
                                int n_symbols) {
    const int sym = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym >= n_symbols) return;

    const uint64_t num  = acc.vwap_num[sym];
    const uint64_t den  = acc.vwap_den[sym];
    const uint64_t buy  = acc.buy_vol[sym];
    const uint64_t sell = acc.sell_vol[sym];

    Snapshot s;
    // Price units are 1/10000 USD; emit dollars.
    s.vwap          = den ? (double)num / (double)den / 10000.0 : 0.0;
    s.volume        = den;
    s.buy_vol       = buy;
    s.sell_vol      = sell;
    s.cancel_shares = acc.cancel_shares[sym];
    s.trades        = acc.trade_count[sym];
    s.adds          = acc.add_count[sym];
    s.bid           = acc.win_bid[sym];
    s.ask           = acc.win_ask[sym];
    s.imbalance     = (buy + sell)
        ? (float)((double)((int64_t)buy - (int64_t)sell) / (double)(buy + sell))
        : 0.0f;
    s._pad = 0;
    out[sym] = s;

    // Consume the quote window.
    acc.win_bid[sym] = BID_EMPTY;
    acc.win_ask[sym] = ASK_EMPTY;
}

void launch_snapshot(DeviceAccum acc, Snapshot* d_out, int n_symbols,
                     cudaStream_t stream) {
    const int block = 256;
    const int grid = (n_symbols + block - 1) / block;
    snapshot_kernel<<<grid, block, 0, stream>>>(acc, d_out, n_symbols);
}

}  // namespace itch::gpu
