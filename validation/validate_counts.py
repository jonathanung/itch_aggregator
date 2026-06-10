#!/usr/bin/env python3
"""Independent ITCH 5.0 cross-check, stdlib only.

Streams a gzipped ITCH file and computes, with a completely separate
implementation from the C++ pipeline:
  - message counts by type
  - per-symbol traded shares and VWAP numerator (P and Q messages)
  - top-20 symbols by traded shares

Output is JSON. With --against STATS.json (from `itch_pipeline --stats-json`),
exits non-zero if per-type counts disagree.

Usage: validate_counts.py FILE.gz [--against STATS.json] [--max-msgs N]
"""

import gzip
import json
import struct
import sys
from collections import defaultdict

U16 = struct.Struct(">H")
U32 = struct.Struct(">I")
U64 = struct.Struct(">Q")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    against = None
    max_msgs = 0
    i = 1
    while i < len(args):
        if args[i] == "--against":
            against = args[i + 1]
            i += 2
        elif args[i] == "--max-msgs":
            max_msgs = int(args[i + 1])
            i += 2
        else:
            print(f"unknown arg {args[i]}", file=sys.stderr)
            return 2

    counts: dict[str, int] = defaultdict(int)
    shares_by_locate: dict[int, int] = defaultdict(int)
    num_by_locate: dict[int, int] = defaultdict(int)
    names: dict[int, str] = {}
    total = 0

    with gzip.open(path, "rb") as f:
        read = f.read
        while True:
            pre = read(2)
            if len(pre) < 2:
                break
            (length,) = U16.unpack(pre)
            if length == 0:
                break
            msg = read(length)
            if len(msg) < length:
                print("truncated message at end of stream", file=sys.stderr)
                return 1
            t = chr(msg[0])
            counts[t] += 1
            total += 1

            if t == "R":
                (locate,) = U16.unpack(msg[1:3])
                names[locate] = msg[11:19].decode("ascii").strip()
            elif t == "P":
                (locate,) = U16.unpack(msg[1:3])
                (shares,) = U32.unpack(msg[20:24])
                (price,) = U32.unpack(msg[32:36])
                shares_by_locate[locate] += shares
                num_by_locate[locate] += shares * price
            elif t == "Q":
                (locate,) = U16.unpack(msg[1:3])
                (shares,) = U64.unpack(msg[11:19])
                (price,) = U32.unpack(msg[27:31])
                shares_by_locate[locate] += shares
                num_by_locate[locate] += shares * price

            if max_msgs and total >= max_msgs:
                break

    top = sorted(shares_by_locate.items(), key=lambda kv: -kv[1])[:20]
    result = {
        "total_messages": total,
        "counts_by_type": dict(sorted(counts.items())),
        "top20_by_traded_shares": [
            {
                "locate": loc,
                "symbol": names.get(loc, "?"),
                "shares": sh,
                "vwap": round(num_by_locate[loc] / sh / 10000.0, 4) if sh else 0.0,
            }
            for loc, sh in top
        ],
    }
    json.dump(result, sys.stdout, indent=2)
    print()

    if against:
        with open(against) as f:
            stats = json.load(f)
        theirs = stats.get("counts_by_type", {})
        ok = True
        for t, n in sorted(counts.items()):
            if theirs.get(t, 0) != n:
                print(f"MISMATCH type {t}: python={n} pipeline={theirs.get(t, 0)}",
                      file=sys.stderr)
                ok = False
        for t, n in sorted(theirs.items()):
            if n and t not in counts:
                print(f"MISMATCH type {t}: python=0 pipeline={n}", file=sys.stderr)
                ok = False
        if stats.get("total_messages") != total:
            print(f"MISMATCH total: python={total} "
                  f"pipeline={stats.get('total_messages')}", file=sys.stderr)
            ok = False
        print("counts match" if ok else "counts DIFFER", file=sys.stderr)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
