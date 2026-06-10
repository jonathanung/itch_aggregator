#include "doctest/doctest.h"

#include "itch_builder.hpp"
#include "parser/itch_parser.hpp"
#include "parser/order_hash.hpp"
#include "pipeline/batch_assembler.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace itch;
using test::StreamBuilder;

namespace {
struct Harness {
    explicit Harness(const std::vector<uint8_t>& bytes, size_t parser_buf = 4u << 20)
        : parser(make_mem_source(bytes.data(), bytes.size()), parser_buf),
          assembler(parser, orders) {}
    ItchParser parser;
    OrderHash orders{1 << 12};
    BatchAssembler assembler;
    std::vector<uint8_t> buf = std::vector<uint8_t>(BATCH_MSGS * SLOT_BYTES);
    BatchMeta fill() { return assembler.fill(buf.data()); }
    const uint8_t* slot(uint32_t i) const { return buf.data() + (size_t)i * SLOT_BYTES; }
};
}  // namespace

TEST_CASE("messages land in 48-byte slots, zero padded") {
    StreamBuilder b;
    b.del(7, 11, 42);  // 19 bytes, shortest in-scope type
    Harness h(b.bytes());
    BatchMeta m = h.fill();
    REQUIRE(m.n_msgs == 1);
    CHECK(h.slot(0)[0] == MSG_ORDER_DELETE);
    CHECK(load_be64(h.slot(0) + ofs::del::REF) == 42);
    for (int i = msg_length(MSG_ORDER_DELETE); i < SLOT_BYTES; ++i)
        CHECK(h.slot(0)[i] == 0);
    CHECK(m.first_ts == 11);
    CHECK(m.last_ts == 11);
}

TEST_CASE("adds populate the order hash; R and S are not shipped") {
    StreamBuilder b;
    b.stock_dir(7, 1, "AAPL");
    b.sys_event(2, 'Q');  // non-open system event: skipped entirely
    b.add_order(7, 3, 100, 'B', 50, "AAPL", 1234500);
    Harness h(b.bytes());
    BatchMeta m = h.fill();
    CHECK(m.n_msgs == 1);
    CHECK(m.reset_vwap == false);
    CHECK(std::string(h.assembler.symbols().name(7)) == "AAPL");
    const OrderInfo* o = h.orders.find(100);
    REQUIRE(o != nullptr);
    CHECK(o->side == 'B');
    CHECK(o->price == 1234500);
}

TEST_CASE("replace inherits side from original and stamps the slot tag") {
    StreamBuilder b;
    b.add_order(7, 1, 100, 'S', 50, "AAPL", 1240000);
    b.replace(7, 2, 100, 200, 80, 1238000);
    Harness h(b.bytes());
    BatchMeta m = h.fill();
    REQUIRE(m.n_msgs == 2);
    CHECK(h.slot(1)[0] == MSG_ORDER_REPLACE);
    CHECK(h.slot(1)[SLOT_SIDE_TAG] == 'S');
    CHECK(h.orders.find(100) == nullptr);       // original erased
    const OrderInfo* o = h.orders.find(200);    // replacement live
    REQUIRE(o != nullptr);
    CHECK(o->side == 'S');
    CHECK(o->locate == 7);
    CHECK(o->shares == 80);
    CHECK(o->price == 1238000);
    CHECK(h.assembler.unresolved_replaces() == 0);
}

TEST_CASE("replace of unknown ref ships with side tag 0") {
    StreamBuilder b;
    b.replace(7, 1, 999, 1000, 10, 100);
    Harness h(b.bytes());
    BatchMeta m = h.fill();
    REQUIRE(m.n_msgs == 1);
    CHECK(h.slot(0)[SLOT_SIDE_TAG] == 0);
    CHECK(h.assembler.unresolved_replaces() == 1);
    CHECK(h.orders.find(1000) == nullptr);  // nothing inserted blindly
}

TEST_CASE("delete and cancel-to-zero erase from the hash") {
    StreamBuilder b;
    b.add_order(7, 1, 100, 'B', 50, "AAPL", 1234500);
    b.add_order(7, 2, 101, 'B', 30, "AAPL", 1234600);
    b.cancel(7, 3, 100, 20);   // partial
    b.cancel(7, 4, 100, 30);   // to zero -> erased
    b.del(7, 5, 101);
    b.del(7, 6, 555);          // unknown
    Harness h(b.bytes());
    BatchMeta m = h.fill();
    CHECK(m.n_msgs == 6);      // all six are GPU-relevant and shipped
    CHECK(h.orders.find(100) == nullptr);
    CHECK(h.orders.find(101) == nullptr);
    CHECK(h.orders.size() == 0);
    CHECK(h.assembler.unknown_cancels() == 1);
}

TEST_CASE("market-open event cuts the batch and flags the next one") {
    StreamBuilder b;
    b.add_order(7, 1, 100, 'B', 50, "AAPL", 1234500);
    b.sys_event(2, 'O');
    b.add_order(7, 3, 101, 'S', 60, "AAPL", 1236000);
    Harness h(b.bytes());

    BatchMeta m1 = h.fill();
    CHECK(m1.n_msgs == 1);
    CHECK(m1.reset_vwap == false);

    BatchMeta m2 = h.fill();
    CHECK(m2.n_msgs == 1);
    CHECK(m2.reset_vwap == true);
    CHECK(h.slot(0)[0] == MSG_ADD_ORDER);
    CHECK(load_be64(h.slot(0) + ofs::add::REF) == 101);

    BatchMeta m3 = h.fill();
    CHECK(m3.n_msgs == 0);
}

TEST_CASE("market-open at stream start flags the first batch directly") {
    StreamBuilder b;
    b.sys_event(1, 'O');
    b.add_order(7, 2, 100, 'B', 50, "AAPL", 1234500);
    Harness h(b.bytes());
    BatchMeta m = h.fill();
    CHECK(m.n_msgs == 1);
    CHECK(m.reset_vwap == true);
}

TEST_CASE("batch cuts at BATCH_MSGS") {
    StreamBuilder b;
    for (uint32_t i = 0; i < (uint32_t)BATCH_MSGS + 1; ++i)
        b.add_order(7, i, 1000 + i, 'B', 10, "AAPL", 1000000 + i);
    Harness h(b.bytes());
    BatchMeta m1 = h.fill();
    CHECK(m1.n_msgs == (uint32_t)BATCH_MSGS);
    CHECK(m1.first_ts == 0);
    CHECK(m1.last_ts == (uint64_t)BATCH_MSGS - 1);
    BatchMeta m2 = h.fill();
    CHECK(m2.n_msgs == 1);
    CHECK(m2.first_ts == (uint64_t)BATCH_MSGS);
}
