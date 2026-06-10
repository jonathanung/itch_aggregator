#!/usr/bin/env python3
"""Carve a development fixture from a full ITCH 5.0 day file.

Streams the gzipped day file, copies the first N whole framed messages, and
re-gzips them. Frame boundaries are respected, so the fixture is itself a
valid ITCH 5.0 BinaryFILE stream.

Usage: make_fixture.py IN.gz OUT.gz [n_messages=20000000]
"""

import gzip
import struct
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src_path, dst_path = sys.argv[1], sys.argv[2]
    n_target = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000_000

    n = 0
    with gzip.open(src_path, "rb") as src, gzip.open(dst_path, "wb", compresslevel=6) as dst:
        while n < n_target:
            pre = src.read(2)
            if len(pre) < 2:
                break
            (length,) = struct.unpack(">H", pre)
            if length == 0:
                break
            payload = src.read(length)
            if len(payload) < length:
                break
            dst.write(pre)
            dst.write(payload)
            n += 1
            if n % 1_000_000 == 0:
                print(f"\r{n:,} messages", end="", file=sys.stderr, flush=True)
    print(f"\nwrote {n:,} messages to {dst_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
