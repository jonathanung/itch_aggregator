# itch_aggregator

GPU-accelerated Nasdaq ITCH 5.0 market-data pipeline. Streams a gzipped full-day
TotalView-ITCH feed, parses it on the CPU, ships batches to an NVIDIA GPU over
pinned memory, and maintains per-symbol **VWAP**, **order imbalance**, and
**quote-window bid/ask spread** with CUDA kernels, emitting periodic CSV snapshots.

```
[ITCH .gz day file]
        |
        v
[CPU] gzFile stream  ->  framing (2-byte BE length)  ->  type dispatch
        |                  - 'R' builds locate->ticker table
        |                  - 'S'/'O' (market open) cuts the batch, schedules VWAP reset
        |                  - order hash resolves U/X/D order references
        v
[CPU] BatchAssembler: raw messages -> fixed 48-byte slots in pinned memory
        |
   cudaMemcpyAsync H2D (2 streams, 4-slot ring, event-gated recycling)
        |
        v
[GPU] decode kernel    : byteswap + field extraction -> SoA (1 thread/msg)
[GPU] aggregate kernel : per-symbol atomics (VWAP num/den, buy/sell vol,
                         cancel shares, window best bid/ask)
[GPU] snapshot kernel  : per-symbol metrics every N batches
        |
   cudaMemcpyAsync D2H (pinned, double-buffered)
        |
        v
[CPU] CSV rows for active symbols
```

## Build

Requires CUDA 12.x, GCC 13+, CMake >= 3.25, zlib. Inside this repo's Zone
container all of that is provisioned by `zone.toml` (CUDA 12.6, sm_89).

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Data

```sh
curl -o data/12302019.NASDAQ_ITCH50.gz \
  "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz"   # 3.5 GB
python3 tools/make_fixture.py data/12302019.NASDAQ_ITCH50.gz data/fixture.itch.gz 20000000
```

The day file is always stream-decompressed; nothing larger than the fixture is
ever materialized on disk.

## Run

```sh
./build/itch_pipeline --file data/fixture.itch.gz > snapshots.csv
./build/itch_pipeline --file data/12302019.NASDAQ_ITCH50.gz --stats-json stats.json --no-csv
./build/itch_pipeline --file data/fixture.itch.gz --cpu-only        # CPU reference path
```

CSV columns: `ts_ns,symbol,vwap,volume,buy_vol,sell_vol,imbalance,bid,ask,spread,trades,adds,cancel_shares`.
VWAP/volume/imbalance are cumulative since market open (System Event `O`);
bid/ask are the best quotes *added* during the snapshot window.

## Validation

Three independent layers:

1. **Unit tests** (`ctest`): framing across buffer boundaries, field offsets vs
   hand-built big-endian fixtures, order-hash backward-shift deletion, batch
   cutting at market open.
2. **Bit-exactness**: the GPU accumulators must equal the CPU reference
   implementation (`src/pipeline/cpu_reference.cpp`) exactly, over randomized
   synthetic streams (part of `ctest`) and over the real fixture.
3. **Cross-implementation**: `validation/validate_counts.py` re-parses the
   file in pure Python and diffs per-type message counts against
   `itch_pipeline --stats-json`:

   ```sh
   ./build/itch_pipeline --file data/fixture.itch.gz --no-csv --stats-json stats.json
   python3 validation/validate_counts.py data/fixture.itch.gz --against stats.json
   ```

## Benchmarks

```sh
./build/parser_bench --file data/fixture.itch.gz   # CPU parse throughput (inflate excluded)
./build/kernel_bench --file data/fixture.itch.gz   # per-kernel us + GB/s via cudaEvents
```

Results on RTX 4080 + 16-core host (TODO: fill in after M8):

| Metric | Value |
|---|---|
| Parser framing-only | TBD M msg/s |
| Parser + hash + assembly | TBD M msg/s |
| H2D bandwidth | TBD GB/s |
| Decode kernel | TBD M msg/s |
| Aggregate kernel | TBD M msg/s |
| Per-batch hot path (H2D+decode+aggregate) | TBD us |
| End-to-end full day | TBD s |

## Design decisions and honest limitations

- **Fixed 48-byte slots, not packed.** Max in-scope message is 44 B; a fixed
  stride removes per-message offset bookkeeping and makes each slot three
  aligned 16-byte loads. The ~25% pinned-bandwidth waste is ~100 KB/batch.
- **Big-endian conversion happens on the GPU.** The single-core CPU parser is
  the pipeline bottleneck; it only inspects the type byte, the locate code and
  order references. Field byteswaps ride on thousands of GPU threads.
- **Bid/ask is window-scoped, not an order book.** Best bid/ask are the
  extremes of add/replace flow within a snapshot window (atomicMax/atomicMin),
  reset at each snapshot. Cancels cannot lower the best bid without per-price
  book state, so a window's "spread" can be stale or crossed. A real spread
  needs a full book; this is documented scope, not an oversight.
- **Imbalance uses P-message sides only.** The `B`/`S` indicator on a non-cross
  trade describes the resting non-display order, so this is *resting-side*
  imbalance, not aggressor imbalance (which would need E/C executions, out of
  scope). Cross trades (Q) count toward VWAP/volume but are imbalance-neutral.
- **Order hash on the CPU.** Open addressing, linear probing, backward-shift
  deletion, 2^24 slots (~400 MB). Replaces whose original order predates the
  stream are shipped with an unknown-side tag and counted
  (`unresolved_replaces` in stats).
- **Atomic contention.** Aggregation uses one global atomic per message-field;
  a planned optimization (after profiling) is warp-level pre-aggregation with
  `__reduce_add_sync` for hot symbols.

## Layout

```
src/parser/    message_types.hpp (wire layout), itch_parser (framing), order_hash
src/pipeline/  batch_assembler, ring_buffer (pinned slot ring), cpu_reference, snapshot.hpp
src/cuda/      kernels.cuh, decode_kernel.cu, aggregate_kernel.cu, snapshot_kernel.cu
src/main.cpp   CLI + double-buffered stream loop
tests/         doctest suites + StreamBuilder fixture builder
benchmarks/    parser_bench (CPU), kernel_bench (cudaEvent timings)
validation/    validate_counts.py (independent Python cross-check)
tools/         make_fixture.py
```
