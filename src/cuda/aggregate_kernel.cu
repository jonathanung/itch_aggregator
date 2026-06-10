// Aggregation kernel: one thread per decoded message, atomics on per-symbol
// accumulators. Atomics are commutative so concurrent batches (two streams)
// may aggregate simultaneously. Warp-level pre-aggregation for hot symbols is
// a planned optimization once correctness is locked (see README).

#include "cuda/kernels.cuh"

namespace itch::gpu {

namespace {
__device__ inline void atom_add64(uint64_t* p, uint64_t v) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p),
              (unsigned long long)v);
}
}  // namespace

__global__ void aggregate_kernel(DecodedBatch in, int n, DeviceAccum acc) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint8_t  type   = in.type[idx];
    const uint8_t  side   = in.side[idx];
    const uint16_t locate = in.locate[idx];
    const uint32_t price  = in.price[idx];
    const uint64_t qty    = in.qty[idx];

    if (locate >= MAX_SYMBOLS) return;

    switch (type) {
        case MSG_ADD_ORDER:
        case MSG_ADD_ORDER_MPID:
        case MSG_ORDER_REPLACE:
            atomicAdd(&acc.add_count[locate], 1u);
            // Window-scoped best bid/ask from live add/replace flow. A 'U'
            // with unresolved side (0) contributes to neither.
            if (side == 'B') atomicMax(&acc.win_bid[locate], price);
            else if (side == 'S') atomicMin(&acc.win_ask[locate], price);
            break;
        case MSG_ORDER_CANCEL:
            atom_add64(&acc.cancel_shares[locate], qty);
            break;
        case MSG_TRADE:
            if (qty == 0) break;
            atom_add64(&acc.vwap_num[locate], (uint64_t)price * qty);
            atom_add64(&acc.vwap_den[locate], qty);
            atomicAdd(&acc.trade_count[locate], 1u);
            if (side == 'B') atom_add64(&acc.buy_vol[locate], qty);
            else if (side == 'S') atom_add64(&acc.sell_vol[locate], qty);
            break;
        case MSG_TRADE_CROSS:
            // Crosses carry no side: VWAP/volume only, imbalance-neutral.
            if (qty == 0) break;
            atom_add64(&acc.vwap_num[locate], (uint64_t)price * qty);
            atom_add64(&acc.vwap_den[locate], qty);
            atomicAdd(&acc.trade_count[locate], 1u);
            break;
        default:  // 'D': no measurable contribution (no shares on the wire)
            break;
    }
}

__global__ void reset_kernel(DeviceAccum acc, int n_symbols) {
    const int sym = blockIdx.x * blockDim.x + threadIdx.x;
    if (sym >= n_symbols) return;
    acc.vwap_num[sym] = 0;
    acc.vwap_den[sym] = 0;
    acc.buy_vol[sym] = 0;
    acc.sell_vol[sym] = 0;
    acc.cancel_shares[sym] = 0;
    acc.trade_count[sym] = 0;
    acc.add_count[sym] = 0;
    acc.win_bid[sym] = BID_EMPTY;
    acc.win_ask[sym] = ASK_EMPTY;
}

void launch_aggregate(DecodedBatch in, int n_msgs, DeviceAccum acc,
                      cudaStream_t stream) {
    if (n_msgs <= 0) return;
    const int block = 256;
    const int grid = (n_msgs + block - 1) / block;
    aggregate_kernel<<<grid, block, 0, stream>>>(in, n_msgs, acc);
}

void launch_reset(DeviceAccum acc, int n_symbols, cudaStream_t stream) {
    const int block = 256;
    const int grid = (n_symbols + block - 1) / block;
    reset_kernel<<<grid, block, 0, stream>>>(acc, n_symbols);
}

}  // namespace itch::gpu
