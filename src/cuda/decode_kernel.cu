// Decode kernel: one thread per message. Reads a raw big-endian 48-byte slot,
// byteswaps and extracts the fields the aggregation pass needs, writes SoA.
// The switch diverges within a warp, but the kernel is memory-bound on the
// slot reads, so divergence is not the limiter (verified in profiling phase).

#include "cuda/kernels.cuh"

namespace itch::gpu {

__global__ void decode_kernel(const uint8_t* __restrict__ raw, int n,
                              DecodedBatch out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint8_t* slot = raw + (size_t)idx * SLOT_BYTES;
    const uint8_t type = slot[ofs::TYPE];

    uint8_t  side  = 0;
    uint32_t price = 0;
    uint64_t qty   = 0;

    switch (type) {
        case MSG_ADD_ORDER:
        case MSG_ADD_ORDER_MPID:
            side  = slot[ofs::add::SIDE];
            qty   = load_be32(slot + ofs::add::SHARES);
            price = load_be32(slot + ofs::add::PRICE);
            break;
        case MSG_ORDER_REPLACE:
            side  = slot[SLOT_SIDE_TAG];  // resolved by CPU; 0 if unknown
            qty   = load_be32(slot + ofs::replace::SHARES);
            price = load_be32(slot + ofs::replace::PRICE);
            break;
        case MSG_ORDER_CANCEL:
            qty = load_be32(slot + ofs::cancel::CANCELED);
            break;
        case MSG_TRADE:
            side  = slot[ofs::trade::SIDE];
            qty   = load_be32(slot + ofs::trade::SHARES);
            price = load_be32(slot + ofs::trade::PRICE);
            break;
        case MSG_TRADE_CROSS:
            qty   = load_be64(slot + ofs::cross::SHARES64);
            price = load_be32(slot + ofs::cross::PRICE);
            break;
        default:  // 'D' and anything unexpected: header only
            break;
    }

    out.type[idx]   = type;
    out.side[idx]   = side;
    out.locate[idx] = load_be16(slot + ofs::LOCATE);
    out.price[idx]  = price;
    out.qty[idx]    = qty;
    out.ts[idx]     = load_be48(slot + ofs::TIMESTAMP);
}

void launch_decode(const uint8_t* d_raw, int n_msgs, DecodedBatch out,
                   cudaStream_t stream) {
    if (n_msgs <= 0) return;
    const int block = 256;
    const int grid = (n_msgs + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(d_raw, n_msgs, out);
}

}  // namespace itch::gpu
