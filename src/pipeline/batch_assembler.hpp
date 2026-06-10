#pragma once
// Pulls messages from the parser, resolves order references against the order
// hash, and packs GPU-relevant messages into fixed 48-byte slots in a
// caller-provided buffer (pinned in production, plain memory in tests).

#include "parser/itch_parser.hpp"
#include "parser/message_types.hpp"
#include "parser/order_hash.hpp"

#include <cstdint>

namespace itch {

struct BatchMeta {
    uint32_t n_msgs = 0;
    bool reset_vwap = false;   // market-open ('S'/'O') precedes this batch
    uint64_t first_ts = 0;     // ns since midnight, from copied messages
    uint64_t last_ts = 0;
};

class BatchAssembler {
public:
    BatchAssembler(ItchParser& parser, OrderHash& orders)
        : parser_(parser), orders_(orders) {}

    // Fill dst (capacity >= BATCH_MSGS * SLOT_BYTES) with up to BATCH_MSGS
    // messages. The batch is cut early at a market-open System Event so the
    // VWAP reset lands between batches. n_msgs == 0 means stream exhausted.
    BatchMeta fill(uint8_t* dst);

    const SymbolTable& symbols() const { return symbols_; }
    uint64_t unresolved_replaces() const { return unresolved_replaces_; }
    uint64_t unknown_cancels() const { return unknown_cancels_; }

private:
    ItchParser& parser_;
    OrderHash& orders_;
    SymbolTable symbols_;
    bool pending_reset_ = false;
    uint64_t unresolved_replaces_ = 0;  // U whose original ref was unknown
    uint64_t unknown_cancels_ = 0;      // X/D whose ref was unknown
};

}  // namespace itch
