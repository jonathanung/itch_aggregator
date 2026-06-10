#pragma once
// Nasdaq ITCH 5.0 message layout constants and big-endian field helpers.
// Shared by the CPU parser and the CUDA decode kernel, so everything here
// must be host/device-safe: no virtuals, no std containers, constexpr only.
//
// Reference: Nasdaq TotalView-ITCH 5.0 specification. All fields are
// big-endian, packed, no padding. Prices are integers in 1/10000 USD.

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define ITCH_HD __host__ __device__
#else
#define ITCH_HD
#endif

namespace itch {

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------
enum MsgType : uint8_t {
    MSG_SYSTEM_EVENT   = 'S',  // 12 B - market open/close etc.
    MSG_STOCK_DIR      = 'R',  // 39 B - locate -> ticker mapping
    MSG_ADD_ORDER      = 'A',  // 36 B
    MSG_ADD_ORDER_MPID = 'F',  // 40 B - A + 4-byte attribution
    MSG_ORDER_CANCEL   = 'X',  // 23 B - partial cancel
    MSG_ORDER_DELETE   = 'D',  // 19 B
    MSG_ORDER_REPLACE  = 'U',  // 35 B
    MSG_TRADE          = 'P',  // 44 B - non-cross, hidden-order execution
    MSG_TRADE_CROSS    = 'Q',  // 40 B - open/close cross; NOTE shares are u64
};

// Length of each message in bytes, INCLUDING the type byte. 0 = not in scope
// for the GPU pipeline (still skipped correctly by the framing layer, which
// uses the wire length prefix, never this table).
ITCH_HD constexpr int msg_length(uint8_t type) {
    switch (type) {
        case MSG_SYSTEM_EVENT:   return 12;
        case MSG_STOCK_DIR:      return 39;
        case MSG_ADD_ORDER:      return 36;
        case MSG_ADD_ORDER_MPID: return 40;
        case MSG_ORDER_CANCEL:   return 23;
        case MSG_ORDER_DELETE:   return 19;
        case MSG_ORDER_REPLACE:  return 35;
        case MSG_TRADE:          return 44;
        case MSG_TRADE_CROSS:    return 40;
        default:                 return 0;
    }
}

// Does this type get copied into a batch slot and shipped to the GPU?
ITCH_HD constexpr bool gpu_relevant(uint8_t type) {
    switch (type) {
        case MSG_ADD_ORDER:
        case MSG_ADD_ORDER_MPID:
        case MSG_ORDER_CANCEL:
        case MSG_ORDER_DELETE:
        case MSG_ORDER_REPLACE:
        case MSG_TRADE:
        case MSG_TRADE_CROSS:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Field offsets (bytes from start of message, i.e. from the type byte)
// ---------------------------------------------------------------------------
namespace ofs {
// Common header on every ITCH 5.0 message.
constexpr int TYPE      = 0;   // u8
constexpr int LOCATE    = 1;   // u16 - stock locate code, our symbol index
constexpr int TRACKING  = 3;   // u16
constexpr int TIMESTAMP = 5;   // u48 - nanoseconds since midnight ET

// A / F (Add Order). F appends MPID at 36.
namespace add {
constexpr int REF    = 11;  // u64 order reference
constexpr int SIDE   = 19;  // char 'B' / 'S'
constexpr int SHARES = 20;  // u32
constexpr int STOCK  = 24;  // char[8], space-padded
constexpr int PRICE  = 32;  // u32
}
// X (Order Cancel, partial)
namespace cancel {
constexpr int REF      = 11;  // u64
constexpr int CANCELED = 19;  // u32 shares being canceled
}
// D (Order Delete)
namespace del {
constexpr int REF = 11;  // u64
}
// U (Order Replace). Side/stock inherited from the original order.
namespace replace {
constexpr int ORIG_REF = 11;  // u64
constexpr int NEW_REF  = 19;  // u64
constexpr int SHARES   = 27;  // u32
constexpr int PRICE    = 31;  // u32
}
// P (Trade, non-cross)
namespace trade {
constexpr int REF    = 11;  // u64
constexpr int SIDE   = 19;  // char - side of the NON-DISPLAY resting order
constexpr int SHARES = 20;  // u32
constexpr int STOCK  = 24;  // char[8]
constexpr int PRICE  = 32;  // u32
constexpr int MATCH  = 36;  // u64
}
// Q (Cross Trade)
namespace cross {
constexpr int SHARES64   = 11;  // u64 (!)
constexpr int STOCK      = 19;  // char[8]
constexpr int PRICE      = 27;  // u32
constexpr int MATCH      = 31;  // u64
constexpr int CROSS_TYPE = 39;  // char 'O'/'C'/'H'
}
// S (System Event)
namespace sys {
constexpr int EVENT = 11;  // char; 'O' = start of market hours -> VWAP reset
}
// R (Stock Directory) - only the ticker matters to us
namespace dir {
constexpr int STOCK = 11;  // char[8]
}
}  // namespace ofs

// ---------------------------------------------------------------------------
// Batch slot layout
// ---------------------------------------------------------------------------
// Raw messages are copied into fixed 48-byte slots (max in-scope message is
// P at 44 B; 48 = 3 x 16 so a slot is three aligned uint4 loads on the GPU).
// Bytes past the message length are zeroed, except:
constexpr int SLOT_BYTES    = 48;
constexpr int SLOT_SIDE_TAG = 44;  // for 'U': resolved side ('B'/'S', 0 if unknown)
                                   // stamped by the CPU from the order hash

constexpr int BATCH_MSGS  = 8192;
constexpr int MAX_SYMBOLS = 16384;  // locate codes are ~1..9000 in practice

// ---------------------------------------------------------------------------
// Big-endian loads/stores - host/device safe (plain byte arithmetic; both
// nvcc and gcc reduce these to bswap/PRMT under -O2+)
// ---------------------------------------------------------------------------
ITCH_HD inline uint16_t load_be16(const uint8_t* p) {
    return (uint16_t)((uint32_t)p[0] << 8 | p[1]);
}
ITCH_HD inline uint32_t load_be32(const uint8_t* p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 |
           (uint32_t)p[2] << 8  | (uint32_t)p[3];
}
ITCH_HD inline uint64_t load_be48(const uint8_t* p) {
    return (uint64_t)p[0] << 40 | (uint64_t)p[1] << 32 |
           (uint64_t)p[2] << 24 | (uint64_t)p[3] << 16 |
           (uint64_t)p[4] << 8  | (uint64_t)p[5];
}
ITCH_HD inline uint64_t load_be64(const uint8_t* p) {
    return (uint64_t)load_be32(p) << 32 | load_be32(p + 4);
}

// Host-only big-endian stores, for building test fixtures and synthetic data.
inline void put_be16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v;
}
inline void put_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}
inline void put_be48(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v >> 40); p[1] = (uint8_t)(v >> 32);
    p[2] = (uint8_t)(v >> 24); p[3] = (uint8_t)(v >> 16);
    p[4] = (uint8_t)(v >> 8);  p[5] = (uint8_t)v;
}
inline void put_be64(uint8_t* p, uint64_t v) {
    put_be32(p, (uint32_t)(v >> 32));
    put_be32(p + 4, (uint32_t)v);
}

}  // namespace itch
