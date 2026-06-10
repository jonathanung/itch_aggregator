#!/usr/bin/env python3
"""Generate a synthetic but valid ITCH 5.0 BinaryFILE stream, gzipped.

Lets the whole pipeline (parse -> assemble -> GPU/CPU aggregate -> snapshot)
and the validation cross-check run without downloading a 3.5 GB day file. The
output is wire-identical in framing and field layout to a real feed, so the
C++ parser, validate_counts.py, and make_fixture.py all accept it.

The stream is internally consistent: every Cancel/Delete/Replace references a
live order reference, so the pipeline's unknown_cancels / unresolved_replaces
stay at zero (a real feed has a few from orders that predate the session).

Usage: make_synthetic.py OUT.gz [n_messages=2000000] [--seed N]
"""

import gzip
import random
import struct
import sys

# Field offsets mirror src/parser/message_types.hpp exactly.
SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "NVDA", "META", "NFLX",
    "AMD", "INTC", "SPY", "QQQ", "JPM", "BAC", "XOM", "WMT",
]


def be(width: int, value: int) -> bytes:
    return value.to_bytes(width, "big")


def header(buf: bytearray, mtype: str, locate: int, ts: int) -> None:
    buf += mtype.encode("ascii")          # TYPE   @0
    buf += be(2, locate)                  # LOCATE @1
    buf += be(2, 0)                       # TRACK  @3
    buf += be(6, ts)                      # TS     @5 (u48)


def stock_field(sym: str) -> bytes:
    return sym.encode("ascii").ljust(8)[:8]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out_path = sys.argv[1]
    n_target = 2_000_000
    seed = 1
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--seed":
            seed = int(sys.argv[i + 1])
            i += 2
        else:
            n_target = int(sys.argv[i])
            i += 1

    rng = random.Random(seed)
    out = bytearray()

    def emit(msg: bytes) -> None:
        out.extend(be(2, len(msg)))
        out.extend(msg)

    ts = 34_200_000_000_000  # 09:30:00 ET in ns since midnight

    # Stock Directory ('R'): locate code -> ticker. Locate codes are 1-based.
    for loc, sym in enumerate(SYMBOLS, start=1):
        b = bytearray()
        header(b, "R", loc, ts)
        b += stock_field(sym)             # STOCK @11
        b += bytes(39 - len(b))           # pad to the 39-byte R layout
        emit(bytes(b))

    # System Event 'O' (start of market hours) -> VWAP reset boundary.
    b = bytearray()
    header(b, "S", 0, ts)
    b += b"O"                             # EVENT @11
    emit(bytes(b))

    n_meta = len(SYMBOLS) + 1
    # Live resting orders as a flat list of [ref, locate, side, shares, price].
    # Random access by index and swap-pop removal keep every op O(1), so the
    # whole stream generates in O(n) rather than O(n^2).
    live: list[list] = []
    next_ref = 1
    match = 1
    produced = 0
    while produced < n_target - n_meta:
        ts += rng.randint(1, 5000)
        locate = rng.randint(1, len(SYMBOLS))
        price = rng.randint(10_000, 4_000_000)   # $1.00 - $400.00 in 1/10000
        shares = rng.randint(1, 1000)
        roll = rng.random()

        if roll < 0.45 or not live:               # Add Order (A or F)
            side = "B" if rng.random() < 0.5 else "S"
            mpid = rng.random() < 0.1
            b = bytearray()
            header(b, "F" if mpid else "A", locate, ts)
            b += be(8, next_ref)                  # REF    @11
            b += side.encode()                    # SIDE   @19
            b += be(4, shares)                    # SHARES @20
            b += stock_field(SYMBOLS[locate - 1]) # STOCK  @24
            b += be(4, price)                     # PRICE  @32
            if mpid:
                b += b"MPID"                       # MPID   @36
            emit(bytes(b))
            live.append([next_ref, locate, side, shares, price])
            next_ref += 1
        elif roll < 0.60:                          # Order Cancel (X)
            k = rng.randrange(len(live))
            ref, loc, _, sh, _ = live[k]
            canceled = rng.randint(1, sh)
            b = bytearray()
            header(b, "X", loc, ts)
            b += be(8, ref)                       # REF      @11
            b += be(4, canceled)                  # CANCELED @19
            emit(bytes(b))
            if canceled >= sh:
                live[k] = live[-1]                # swap-pop: full cancel
                live.pop()
            else:
                live[k][3] = sh - canceled        # partial cancel
        elif roll < 0.75:                          # Order Delete (D)
            k = rng.randrange(len(live))
            ref, loc = live[k][0], live[k][1]
            b = bytearray()
            header(b, "D", loc, ts)
            b += be(8, ref)                       # REF @11
            emit(bytes(b))
            live[k] = live[-1]
            live.pop()
        elif roll < 0.85:                          # Order Replace (U)
            k = rng.randrange(len(live))
            ref, loc, side = live[k][0], live[k][1], live[k][2]
            b = bytearray()
            header(b, "U", loc, ts)
            b += be(8, ref)                       # ORIG_REF @11
            b += be(8, next_ref)                  # NEW_REF  @19
            b += be(4, shares)                    # SHARES   @27
            b += be(4, price)                     # PRICE    @31
            emit(bytes(b))
            live[k] = [next_ref, loc, side, shares, price]
            next_ref += 1
        elif roll < 0.97:                          # Trade non-cross (P)
            side = "B" if rng.random() < 0.5 else "S"
            b = bytearray()
            header(b, "P", locate, ts)
            b += be(8, 0)                         # REF    @11 (hidden)
            b += side.encode()                    # SIDE   @19
            b += be(4, shares)                    # SHARES @20
            b += stock_field(SYMBOLS[locate - 1]) # STOCK  @24
            b += be(4, price)                     # PRICE  @32
            b += be(8, match)                     # MATCH  @36
            emit(bytes(b))
            match += 1
        else:                                      # Cross Trade (Q)
            b = bytearray()
            header(b, "Q", locate, ts)
            b += be(8, shares)                    # SHARES64   @11
            b += stock_field(SYMBOLS[locate - 1]) # STOCK      @19
            b += be(4, price)                     # PRICE      @27
            b += be(8, match)                     # MATCH      @31
            b += b"O"                             # CROSS_TYPE @39
            emit(bytes(b))
            match += 1
        produced += 1

    # System Event 'C' (end of market hours) then end-of-session marker.
    b = bytearray()
    header(b, "S", 0, ts)
    b += b"C"
    emit(bytes(b))
    out.extend(be(2, 0))  # zero-length frame terminates the stream

    with gzip.open(out_path, "wb", compresslevel=6) as f:
        f.write(out)

    total = n_meta + produced + 1
    print(f"wrote {total:,} messages ({len(out) / 1e6:.1f} MB raw) to {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
