#include "doctest/doctest.h"

#include "parser/order_hash.hpp"

#include <random>

using itch::OrderHash;
using itch::OrderInfo;

TEST_CASE("insert / find / erase") {
    OrderHash h(1 << 10);
    CHECK(h.insert(42, 7, 'B', 100, 1234500));
    const OrderInfo* o = h.find(42);
    REQUIRE(o != nullptr);
    CHECK(o->locate == 7);
    CHECK(o->side == 'B');
    CHECK(o->shares == 100);
    CHECK(o->price == 1234500);
    CHECK(h.size() == 1);

    CHECK(h.erase(42));
    CHECK(h.find(42) == nullptr);
    CHECK(h.size() == 0);
    CHECK_FALSE(h.erase(42));
}

TEST_CASE("duplicate insert is rejected") {
    OrderHash h(1 << 10);
    CHECK(h.insert(1, 1, 'B', 1, 1));
    CHECK_FALSE(h.insert(1, 2, 'S', 2, 2));
    CHECK(h.find(1)->locate == 1);
}

TEST_CASE("reduce decrements and erases at zero") {
    OrderHash h(1 << 10);
    h.insert(5, 1, 'S', 100, 999);
    CHECK(h.reduce(5, 30));
    CHECK(h.find(5)->shares == 70);
    CHECK(h.reduce(5, 70));
    CHECK(h.find(5) == nullptr);
    CHECK_FALSE(h.reduce(5, 1));  // already gone
}

TEST_CASE("load factor cap refuses inserts past 50%") {
    OrderHash h(8);
    int ok = 0;
    for (uint64_t r = 1; r <= 8; ++r) ok += h.insert(r, 1, 'B', 1, 1);
    CHECK(ok == 4);
}

TEST_CASE("backward-shift deletion keeps probe chains intact under churn") {
    // Small table + many keys = heavy collisions; interleaved erases would
    // break lookups if deletion left holes inside probe chains.
    OrderHash h(1 << 12);
    std::mt19937_64 rng(1234);
    std::vector<uint64_t> live;

    for (int round = 0; round < 200; ++round) {
        for (int i = 0; i < 10; ++i) {
            uint64_t ref = rng() | 1;  // nonzero
            if (h.insert(ref, (uint16_t)(ref & 0xFFF), 'B', 10, 100))
                live.push_back(ref);
        }
        for (int i = 0; i < 5 && !live.empty(); ++i) {
            size_t k = rng() % live.size();
            CHECK(h.erase(live[k]));
            live[k] = live.back();
            live.pop_back();
        }
        for (uint64_t ref : live) {
            const OrderInfo* o = h.find(ref);
            REQUIRE(o != nullptr);
            CHECK(o->ref == ref);
        }
    }
    CHECK(h.size() == live.size());
}
