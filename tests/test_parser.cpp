#include "doctest/doctest.h"

#include "itch_builder.hpp"
#include "parser/itch_parser.hpp"
#include "parser/message_types.hpp"

using namespace itch;
using test::StreamBuilder;

TEST_CASE("be load/store helpers round-trip") {
    uint8_t buf[8];
    put_be16(buf, 0xBEEF);
    CHECK(load_be16(buf) == 0xBEEF);
    CHECK(buf[0] == 0xBE);  // genuinely big-endian on the wire
    put_be32(buf, 0xDEADBEEFu);
    CHECK(load_be32(buf) == 0xDEADBEEFu);
    put_be48(buf, 0x123456789ABCull);
    CHECK(load_be48(buf) == 0x123456789ABCull);
    put_be64(buf, 0x0123456789ABCDEFull);
    CHECK(load_be64(buf) == 0x0123456789ABCDEFull);
}

TEST_CASE("message length table matches built messages") {
    StreamBuilder b;
    b.sys_event(1, 'O');
    b.stock_dir(7, 2, "AAPL");
    b.add_order(7, 3, 100, 'B', 50, "AAPL", 1234500);
    b.add_order(7, 4, 101, 'S', 50, "AAPL", 1235500, /*mpid=*/true);
    b.cancel(7, 5, 100, 10);
    b.del(7, 6, 100);
    b.replace(7, 7, 101, 102, 60, 1235000);
    b.trade(7, 8, 0, 'B', 25, "AAPL", 1235000, 9001);
    b.cross(7, 9, 1000, "AAPL", 1235000, 9002, 'O');

    ItchParser p(make_mem_source(b.bytes().data(), b.bytes().size()));
    uint16_t len = 0;
    int n = 0;
    while (const uint8_t* msg = p.next(&len)) {
        CHECK(len == msg_length(msg[0]));
        ++n;
    }
    CHECK(n == 9);
    CHECK(p.total_messages() == 9);
    CHECK_FALSE(p.error());
}

TEST_CASE("field extraction at documented offsets") {
    StreamBuilder b;
    b.add_order(1234, 0x123456789ABCull, 0xCAFEBABE1234ull, 'B', 777,
                "MSFT", 1507500);
    ItchParser p(make_mem_source(b.bytes().data(), b.bytes().size()));
    uint16_t len = 0;
    const uint8_t* m = p.next(&len);
    REQUIRE(m != nullptr);
    CHECK(m[ofs::TYPE] == MSG_ADD_ORDER);
    CHECK(load_be16(m + ofs::LOCATE) == 1234);
    CHECK(load_be48(m + ofs::TIMESTAMP) == 0x123456789ABCull);
    CHECK(load_be64(m + ofs::add::REF) == 0xCAFEBABE1234ull);
    CHECK(m[ofs::add::SIDE] == 'B');
    CHECK(load_be32(m + ofs::add::SHARES) == 777);
    CHECK(std::memcmp(m + ofs::add::STOCK, "MSFT    ", 8) == 0);
    CHECK(load_be32(m + ofs::add::PRICE) == 1507500);
}

TEST_CASE("framing across refill boundaries (tiny buffer)") {
    // 200 trades of 44+2 bytes through a 64-byte parser buffer forces a
    // compact-and-refill in the middle of nearly every message.
    StreamBuilder b;
    for (uint32_t i = 0; i < 200; ++i)
        b.trade(7, 1000 + i, i, (i & 1) ? 'S' : 'B', 10 + i, "AAPL",
                1000000 + i, i);

    ItchParser p(make_mem_source(b.bytes().data(), b.bytes().size()), 64);
    uint16_t len = 0;
    uint32_t i = 0;
    while (const uint8_t* m = p.next(&len)) {
        REQUIRE(len == 44);
        CHECK(load_be48(m + ofs::TIMESTAMP) == 1000 + i);
        CHECK(load_be32(m + ofs::trade::SHARES) == 10 + i);
        CHECK(load_be32(m + ofs::trade::PRICE) == 1000000 + i);
        ++i;
    }
    CHECK(i == 200);
    CHECK_FALSE(p.error());
}

TEST_CASE("zero length prefix terminates the stream") {
    StreamBuilder b;
    b.del(7, 1, 42);
    b.end_marker();
    b.del(7, 2, 43);  // must never be seen

    ItchParser p(make_mem_source(b.bytes().data(), b.bytes().size()));
    uint16_t len = 0;
    CHECK(p.next(&len) != nullptr);
    CHECK(p.next(&len) == nullptr);
    CHECK(p.total_messages() == 1);
    CHECK_FALSE(p.error());
}

TEST_CASE("truncated message flags an error") {
    StreamBuilder b;
    b.trade(7, 1, 0, 'B', 10, "AAPL", 1000000, 1);
    auto bytes = b.bytes();
    bytes.resize(bytes.size() - 5);  // chop the tail

    ItchParser p(make_mem_source(bytes.data(), bytes.size()));
    uint16_t len = 0;
    CHECK(p.next(&len) == nullptr);
    CHECK(p.error());
}

TEST_CASE("per-type counts") {
    StreamBuilder b;
    b.add_order(1, 1, 1, 'B', 1, "A", 1);
    b.add_order(1, 2, 2, 'B', 1, "A", 1);
    b.del(1, 3, 1);
    ItchParser p(make_mem_source(b.bytes().data(), b.bytes().size()));
    uint16_t len = 0;
    while (p.next(&len)) {}
    CHECK(p.counts_by_type()[(uint8_t)'A'] == 2);
    CHECK(p.counts_by_type()[(uint8_t)'D'] == 1);
    CHECK(p.counts_by_type()[(uint8_t)'P'] == 0);
}

TEST_CASE("symbol table from stock directory") {
    StreamBuilder b;
    b.stock_dir(42, 1, "NVDA");
    ItchParser p(make_mem_source(b.bytes().data(), b.bytes().size()));
    uint16_t len = 0;
    const uint8_t* m = p.next(&len);
    REQUIRE(m != nullptr);
    SymbolTable t;
    t.update(m);
    CHECK(std::string(t.name(42)) == "NVDA");
    CHECK(std::string(t.name(43)) == "");
}
