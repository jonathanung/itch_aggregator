#include "pipeline/batch_assembler.hpp"

#include <cstring>

namespace itch {

BatchMeta BatchAssembler::fill(uint8_t* dst) {
    BatchMeta meta;
    meta.reset_vwap = pending_reset_;
    pending_reset_ = false;

    uint16_t len = 0;
    while (meta.n_msgs < BATCH_MSGS) {
        const uint8_t* msg = parser_.next(&len);
        if (!msg) break;

        const uint8_t type = msg[0];

        if (type == MSG_STOCK_DIR) {
            symbols_.update(msg);
            continue;
        }
        if (type == MSG_SYSTEM_EVENT) {
            if (msg[ofs::sys::EVENT] == 'O') {
                // Market open: VWAP reset must land between batches. If this
                // batch already has messages, close it and flag the next one.
                if (meta.n_msgs == 0) meta.reset_vwap = true;
                else { pending_reset_ = true; break; }
            }
            continue;
        }
        if (!gpu_relevant(type)) continue;
        if (len > SLOT_BYTES) continue;  // malformed: in-scope types are <= 44 B

        uint8_t side_tag = 0;  // only meaningful for 'U'

        switch (type) {
            case MSG_ADD_ORDER:
            case MSG_ADD_ORDER_MPID: {
                orders_.insert(load_be64(msg + ofs::add::REF),
                               load_be16(msg + ofs::LOCATE),
                               msg[ofs::add::SIDE],
                               load_be32(msg + ofs::add::SHARES),
                               load_be32(msg + ofs::add::PRICE));
                break;
            }
            case MSG_ORDER_CANCEL: {
                if (!orders_.reduce(load_be64(msg + ofs::cancel::REF),
                                    load_be32(msg + ofs::cancel::CANCELED)))
                    ++unknown_cancels_;
                break;
            }
            case MSG_ORDER_DELETE: {
                if (!orders_.erase(load_be64(msg + ofs::del::REF)))
                    ++unknown_cancels_;
                break;
            }
            case MSG_ORDER_REPLACE: {
                const uint64_t orig = load_be64(msg + ofs::replace::ORIG_REF);
                if (const OrderInfo* o = orders_.find(orig)) {
                    side_tag = o->side;
                    orders_.insert(load_be64(msg + ofs::replace::NEW_REF),
                                   o->locate, o->side,
                                   load_be32(msg + ofs::replace::SHARES),
                                   load_be32(msg + ofs::replace::PRICE));
                    orders_.erase(orig);
                } else {
                    ++unresolved_replaces_;  // original predates the stream
                }
                break;
            }
            default:
                break;  // P, Q: no order-hash interaction (hidden/cross)
        }

        uint8_t* slot = dst + (size_t)meta.n_msgs * SLOT_BYTES;
        std::memcpy(slot, msg, len);
        std::memset(slot + len, 0, SLOT_BYTES - len);
        if (type == MSG_ORDER_REPLACE) slot[SLOT_SIDE_TAG] = side_tag;

        const uint64_t ts = load_be48(msg + ofs::TIMESTAMP);
        if (meta.n_msgs == 0) meta.first_ts = ts;
        meta.last_ts = ts;
        ++meta.n_msgs;
    }
    return meta;
}

}  // namespace itch
