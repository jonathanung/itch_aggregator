#pragma once
// Builds framed ITCH 5.0 byte streams for tests and synthetic benchmarks:
// each message is appended as [u16 BE length][payload], exactly as on disk.

#include "parser/message_types.hpp"

#include <cstring>
#include <vector>

namespace itch::test {

class StreamBuilder {
public:
    const std::vector<uint8_t>& bytes() const { return bytes_; }

    void add_order(uint16_t locate, uint64_t ts, uint64_t ref, char side,
                   uint32_t shares, const char* stock, uint32_t price,
                   bool mpid = false) {
        uint8_t m[40] = {};
        header(m, mpid ? MSG_ADD_ORDER_MPID : MSG_ADD_ORDER, locate, ts);
        put_be64(m + ofs::add::REF, ref);
        m[ofs::add::SIDE] = (uint8_t)side;
        put_be32(m + ofs::add::SHARES, shares);
        put_stock(m + ofs::add::STOCK, stock);
        put_be32(m + ofs::add::PRICE, price);
        if (mpid) std::memcpy(m + 36, "TEST", 4);
        frame(m, mpid ? 40 : 36);
    }

    void cancel(uint16_t locate, uint64_t ts, uint64_t ref, uint32_t canceled) {
        uint8_t m[23] = {};
        header(m, MSG_ORDER_CANCEL, locate, ts);
        put_be64(m + ofs::cancel::REF, ref);
        put_be32(m + ofs::cancel::CANCELED, canceled);
        frame(m, 23);
    }

    void del(uint16_t locate, uint64_t ts, uint64_t ref) {
        uint8_t m[19] = {};
        header(m, MSG_ORDER_DELETE, locate, ts);
        put_be64(m + ofs::del::REF, ref);
        frame(m, 19);
    }

    void replace(uint16_t locate, uint64_t ts, uint64_t orig, uint64_t nref,
                 uint32_t shares, uint32_t price) {
        uint8_t m[35] = {};
        header(m, MSG_ORDER_REPLACE, locate, ts);
        put_be64(m + ofs::replace::ORIG_REF, orig);
        put_be64(m + ofs::replace::NEW_REF, nref);
        put_be32(m + ofs::replace::SHARES, shares);
        put_be32(m + ofs::replace::PRICE, price);
        frame(m, 35);
    }

    void trade(uint16_t locate, uint64_t ts, uint64_t ref, char side,
               uint32_t shares, const char* stock, uint32_t price,
               uint64_t match) {
        uint8_t m[44] = {};
        header(m, MSG_TRADE, locate, ts);
        put_be64(m + ofs::trade::REF, ref);
        m[ofs::trade::SIDE] = (uint8_t)side;
        put_be32(m + ofs::trade::SHARES, shares);
        put_stock(m + ofs::trade::STOCK, stock);
        put_be32(m + ofs::trade::PRICE, price);
        put_be64(m + ofs::trade::MATCH, match);
        frame(m, 44);
    }

    void cross(uint16_t locate, uint64_t ts, uint64_t shares64,
               const char* stock, uint32_t price, uint64_t match,
               char cross_type) {
        uint8_t m[40] = {};
        header(m, MSG_TRADE_CROSS, locate, ts);
        put_be64(m + ofs::cross::SHARES64, shares64);
        put_stock(m + ofs::cross::STOCK, stock);
        put_be32(m + ofs::cross::PRICE, price);
        put_be64(m + ofs::cross::MATCH, match);
        m[ofs::cross::CROSS_TYPE] = (uint8_t)cross_type;
        frame(m, 40);
    }

    void sys_event(uint64_t ts, char code) {
        uint8_t m[12] = {};
        header(m, MSG_SYSTEM_EVENT, 0, ts);
        m[ofs::sys::EVENT] = (uint8_t)code;
        frame(m, 12);
    }

    void stock_dir(uint16_t locate, uint64_t ts, const char* stock) {
        uint8_t m[39] = {};
        header(m, MSG_STOCK_DIR, locate, ts);
        put_stock(m + ofs::dir::STOCK, stock);
        frame(m, 39);
    }

    // End-of-session marker: zero length prefix.
    void end_marker() {
        bytes_.push_back(0);
        bytes_.push_back(0);
    }

private:
    static void header(uint8_t* m, uint8_t type, uint16_t locate, uint64_t ts) {
        m[ofs::TYPE] = type;
        put_be16(m + ofs::LOCATE, locate);
        put_be16(m + ofs::TRACKING, 0);
        put_be48(m + ofs::TIMESTAMP, ts);
    }

    static void put_stock(uint8_t* dst, const char* stock) {
        std::memset(dst, ' ', 8);
        std::memcpy(dst, stock, std::strlen(stock) > 8 ? 8 : std::strlen(stock));
    }

    void frame(const uint8_t* m, uint16_t len) {
        uint8_t pre[2];
        put_be16(pre, len);
        bytes_.insert(bytes_.end(), pre, pre + 2);
        bytes_.insert(bytes_.end(), m, m + len);
    }

    std::vector<uint8_t> bytes_;
};

}  // namespace itch::test
