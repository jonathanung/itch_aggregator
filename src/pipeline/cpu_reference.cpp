#include "pipeline/cpu_reference.hpp"

namespace itch {

void cpu_aggregate(const uint8_t* slots, uint32_t n_msgs, HostAccum& acc) {
    for (uint32_t i = 0; i < n_msgs; ++i) {
        const uint8_t* slot = slots + (size_t)i * SLOT_BYTES;
        const uint8_t type = slot[ofs::TYPE];
        const uint16_t locate = load_be16(slot + ofs::LOCATE);
        if (locate >= MAX_SYMBOLS) continue;

        switch (type) {
            case MSG_ADD_ORDER:
            case MSG_ADD_ORDER_MPID: {
                const uint8_t side = slot[ofs::add::SIDE];
                const uint32_t price = load_be32(slot + ofs::add::PRICE);
                ++acc.add_count[locate];
                if (side == 'B') {
                    if (price > acc.win_bid[locate]) acc.win_bid[locate] = price;
                } else if (side == 'S') {
                    if (price < acc.win_ask[locate]) acc.win_ask[locate] = price;
                }
                break;
            }
            case MSG_ORDER_REPLACE: {
                const uint8_t side = slot[SLOT_SIDE_TAG];
                const uint32_t price = load_be32(slot + ofs::replace::PRICE);
                ++acc.add_count[locate];
                if (side == 'B') {
                    if (price > acc.win_bid[locate]) acc.win_bid[locate] = price;
                } else if (side == 'S') {
                    if (price < acc.win_ask[locate]) acc.win_ask[locate] = price;
                }
                break;
            }
            case MSG_ORDER_CANCEL:
                acc.cancel_shares[locate] += load_be32(slot + ofs::cancel::CANCELED);
                break;
            case MSG_TRADE: {
                const uint64_t qty = load_be32(slot + ofs::trade::SHARES);
                if (qty == 0) break;
                const uint32_t price = load_be32(slot + ofs::trade::PRICE);
                acc.vwap_num[locate] += (uint64_t)price * qty;
                acc.vwap_den[locate] += qty;
                ++acc.trade_count[locate];
                const uint8_t side = slot[ofs::trade::SIDE];
                if (side == 'B') acc.buy_vol[locate] += qty;
                else if (side == 'S') acc.sell_vol[locate] += qty;
                break;
            }
            case MSG_TRADE_CROSS: {
                const uint64_t qty = load_be64(slot + ofs::cross::SHARES64);
                if (qty == 0) break;
                const uint32_t price = load_be32(slot + ofs::cross::PRICE);
                acc.vwap_num[locate] += (uint64_t)price * qty;
                acc.vwap_den[locate] += qty;
                ++acc.trade_count[locate];
                break;
            }
            default:  // 'D': no measurable contribution
                break;
        }
    }
}

void cpu_snapshot(HostAccum& acc, Snapshot* out, int n_symbols) {
    for (int sym = 0; sym < n_symbols; ++sym) {
        const uint64_t num = acc.vwap_num[sym];
        const uint64_t den = acc.vwap_den[sym];
        const uint64_t buy = acc.buy_vol[sym];
        const uint64_t sell = acc.sell_vol[sym];

        Snapshot& s = out[sym];
        s.vwap = den ? (double)num / (double)den / 10000.0 : 0.0;
        s.volume = den;
        s.buy_vol = buy;
        s.sell_vol = sell;
        s.cancel_shares = acc.cancel_shares[sym];
        s.trades = acc.trade_count[sym];
        s.adds = acc.add_count[sym];
        s.bid = acc.win_bid[sym];
        s.ask = acc.win_ask[sym];
        s.imbalance = (buy + sell)
            ? (float)((double)((int64_t)buy - (int64_t)sell) / (double)(buy + sell))
            : 0.0f;
        s._pad = 0;

        acc.win_bid[sym] = BID_EMPTY;
        acc.win_ask[sym] = ASK_EMPTY;
    }
}

}  // namespace itch
