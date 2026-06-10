#pragma once
// Open-addressing hash map: order reference -> {locate, side, price, shares}.
//
// Purpose: ITCH Cancel/Delete/Replace messages carry only the order reference,
// not the symbol or side. The CPU resolves them against this table before
// shipping batches to the GPU (a Replace inherits its side from the original
// order; see SLOT_SIDE_TAG in message_types.hpp).
//
// Linear probing with backward-shift deletion (no tombstones - a full trading
// day does ~10^8 deletes and tombstone rot would destroy probe lengths).
// Order references are Nasdaq-assigned, dense-ish integers, so they are mixed
// with a splitmix64 finalizer before bucketing.

#include <cstdint>
#include <vector>

namespace itch {

struct OrderInfo {
    uint64_t ref    = 0;   // 0 = empty slot (Nasdaq refs start at 1)
    uint32_t shares = 0;
    uint32_t price  = 0;
    uint16_t locate = 0;
    uint8_t  side   = 0;   // 'B' or 'S'
    uint8_t  _pad   = 0;
};
static_assert(sizeof(OrderInfo) == 24);

class OrderHash {
public:
    // capacity must be a power of two. Default 2^24 slots = 402 MB, comfortably
    // above Nasdaq's single-digit-millions peak of simultaneously live orders.
    explicit OrderHash(uint64_t capacity = 1ull << 24)
        : mask_(capacity - 1), slots_(capacity) {}

    bool insert(uint64_t ref, uint16_t locate, uint8_t side,
                uint32_t shares, uint32_t price) {
        if (size_ + 1 > (mask_ + 1) / 2) return false;  // refuse >50% load
        uint64_t i = bucket(ref);
        while (slots_[i].ref != 0) {
            if (slots_[i].ref == ref) return false;  // duplicate ref: feed error
            i = (i + 1) & mask_;
        }
        slots_[i] = {ref, shares, price, locate, side, 0};
        ++size_;
        return true;
    }

    const OrderInfo* find(uint64_t ref) const {
        uint64_t i = bucket(ref);
        while (slots_[i].ref != 0) {
            if (slots_[i].ref == ref) return &slots_[i];
            i = (i + 1) & mask_;
        }
        return nullptr;
    }

    // Partial cancel: decrement shares; erase the order when it hits zero.
    // Returns the order's info (pre-decrement) or nullptr if unknown.
    bool reduce(uint64_t ref, uint32_t canceled) {
        uint64_t i = bucket(ref);
        while (slots_[i].ref != 0) {
            if (slots_[i].ref == ref) {
                if (slots_[i].shares <= canceled) erase_at(i);
                else slots_[i].shares -= canceled;
                return true;
            }
            i = (i + 1) & mask_;
        }
        return false;
    }

    bool erase(uint64_t ref) {
        uint64_t i = bucket(ref);
        while (slots_[i].ref != 0) {
            if (slots_[i].ref == ref) { erase_at(i); return true; }
            i = (i + 1) & mask_;
        }
        return false;
    }

    uint64_t size() const { return size_; }

private:
    uint64_t bucket(uint64_t ref) const { return mix(ref) & mask_; }

    static uint64_t mix(uint64_t x) {  // splitmix64 finalizer
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    // Backward-shift deletion: walk forward from the hole and move back any
    // entry whose home bucket means it is reachable through the hole.
    void erase_at(uint64_t hole) {
        --size_;
        uint64_t i = hole;
        for (;;) {
            slots_[hole].ref = 0;
            for (;;) {
                i = (i + 1) & mask_;
                if (slots_[i].ref == 0) return;
                uint64_t home = bucket(slots_[i].ref);
                // Is `hole` in the cyclic probe interval [home, i)?
                bool reachable = (home <= i)
                    ? (home <= hole && hole < i)
                    : (home <= hole || hole < i);
                if (reachable) break;
            }
            slots_[hole] = slots_[i];
            hole = i;
        }
    }

    uint64_t mask_;
    uint64_t size_ = 0;
    std::vector<OrderInfo> slots_;
};

}  // namespace itch
