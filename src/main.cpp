// itch_pipeline: stream a gzipped ITCH 5.0 day file, aggregate per-symbol
// VWAP / order imbalance / quote-window spread on the GPU (or on the CPU
// reference path with --cpu-only), emit periodic CSV snapshots.

#include "cuda/kernels.cuh"
#include "parser/itch_parser.hpp"
#include "parser/order_hash.hpp"
#include "pipeline/batch_assembler.hpp"
#include "pipeline/cpu_reference.hpp"
#include "pipeline/ring_buffer.hpp"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string file;
    std::string stats_json;
    long long snapshot_interval = 128;  // batches between snapshots
    unsigned long long max_msgs = 0;    // 0 = unlimited
    bool cpu_only = false;
    bool no_csv = false;                // suppress row output (benchmark runs)
};

void usage(const char* argv0) {
    std::printf(
        "Usage: %s --file <path.gz> [options]\n"
        "  --file PATH            gzipped ITCH 5.0 input (required)\n"
        "  --snapshot-interval N  batches between snapshots (default 128)\n"
        "  --max-msgs N           stop after N parsed messages (default: all)\n"
        "  --stats-json PATH      write run statistics as JSON\n"
        "  --cpu-only             use the CPU reference aggregator (no GPU)\n"
        "  --no-csv               skip CSV row output (throughput runs)\n",
        argv0);
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--file") {
            const char* v = val("--file"); if (!v) return false; o.file = v;
        } else if (a == "--snapshot-interval") {
            const char* v = val("--snapshot-interval"); if (!v) return false;
            o.snapshot_interval = std::atoll(v);
            if (o.snapshot_interval <= 0) return false;
        } else if (a == "--max-msgs") {
            const char* v = val("--max-msgs"); if (!v) return false;
            o.max_msgs = std::strtoull(v, nullptr, 10);
        } else if (a == "--stats-json") {
            const char* v = val("--stats-json"); if (!v) return false;
            o.stats_json = v;
        } else if (a == "--cpu-only") {
            o.cpu_only = true;
        } else if (a == "--no-csv") {
            o.no_csv = true;
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
    }
    return !o.file.empty();
}

void print_csv_header() {
    std::printf("ts_ns,symbol,vwap,volume,buy_vol,sell_vol,imbalance,"
                "bid,ask,spread,trades,adds,cancel_shares\n");
}

void print_snapshot_csv(uint64_t ts, const itch::Snapshot* snap,
                        const itch::SymbolTable& symbols) {
    using itch::ASK_EMPTY;
    using itch::BID_EMPTY;
    for (int sym = 0; sym < itch::MAX_SYMBOLS; ++sym) {
        const itch::Snapshot& s = snap[sym];
        const bool dirty = s.volume || s.trades || s.adds || s.cancel_shares ||
                           s.bid != BID_EMPTY || s.ask != ASK_EMPTY;
        if (!dirty) continue;

        char bid[24] = "", ask[24] = "", spread[24] = "";
        if (s.bid != BID_EMPTY)
            std::snprintf(bid, sizeof bid, "%.4f", s.bid / 10000.0);
        if (s.ask != ASK_EMPTY)
            std::snprintf(ask, sizeof ask, "%.4f", s.ask / 10000.0);
        if (s.bid != BID_EMPTY && s.ask != ASK_EMPTY)
            std::snprintf(spread, sizeof spread, "%.4f",
                          ((int64_t)s.ask - (int64_t)s.bid) / 10000.0);

        std::printf("%" PRIu64 ",%s,%.4f,%" PRIu64 ",%" PRIu64 ",%" PRIu64
                    ",%.6f,%s,%s,%s,%u,%u,%" PRIu64 "\n",
                    ts, symbols.name((uint16_t)sym), s.vwap, s.volume,
                    s.buy_vol, s.sell_vol, (double)s.imbalance, bid, ask,
                    spread, s.trades, s.adds, s.cancel_shares);
    }
}

struct RunStats {
    uint64_t batches = 0;
    uint64_t shipped_msgs = 0;   // messages copied into batch slots
    uint64_t snapshots = 0;
    double wall_seconds = 0.0;
};

void write_stats(const Options& opt, const itch::ItchParser& parser,
                 const itch::BatchAssembler& assembler, const RunStats& rs) {
    const auto& c = parser.counts_by_type();
    const double mps = rs.wall_seconds > 0
        ? (double)parser.total_messages() / rs.wall_seconds : 0.0;

    std::fprintf(stderr,
                 "# parsed=%" PRIu64 " shipped=%" PRIu64 " batches=%" PRIu64
                 " snapshots=%" PRIu64 " wall=%.2fs rate=%.2fM msg/s"
                 " unresolved_replaces=%" PRIu64 " unknown_cancels=%" PRIu64 "\n",
                 parser.total_messages(), rs.shipped_msgs, rs.batches,
                 rs.snapshots, rs.wall_seconds, mps / 1e6,
                 assembler.unresolved_replaces(), assembler.unknown_cancels());

    if (opt.stats_json.empty()) return;
    std::ofstream f(opt.stats_json);
    f << "{\n  \"total_messages\": " << parser.total_messages()
      << ",\n  \"shipped_messages\": " << rs.shipped_msgs
      << ",\n  \"batches\": " << rs.batches
      << ",\n  \"snapshots\": " << rs.snapshots
      << ",\n  \"wall_seconds\": " << rs.wall_seconds
      << ",\n  \"messages_per_sec\": " << (uint64_t)mps
      << ",\n  \"unresolved_replaces\": " << assembler.unresolved_replaces()
      << ",\n  \"unknown_cancels\": " << assembler.unknown_cancels()
      << ",\n  \"counts_by_type\": {";
    bool first = true;
    for (int t = 0; t < 256; ++t) {
        if (!c[t]) continue;
        if (!first) f << ",";
        first = false;
        f << "\n    \"" << (char)t << "\": " << c[t];
    }
    f << "\n  }\n}\n";
}

// ---------------------------------------------------------------------------
// CPU reference pipeline
// ---------------------------------------------------------------------------
int run_cpu(const Options& opt) {
    itch::ItchParser parser(itch::make_gz_source(opt.file));
    itch::OrderHash orders;
    itch::BatchAssembler assembler(parser, orders);

    std::vector<uint8_t> batch(itch::BATCH_MSGS * itch::SLOT_BYTES);
    itch::HostAccum acc;
    std::vector<itch::Snapshot> snap(itch::MAX_SYMBOLS);

    RunStats rs;
    if (!opt.no_csv) print_csv_header();
    const auto t0 = std::chrono::steady_clock::now();

    uint64_t last_ts = 0;
    for (;;) {
        itch::BatchMeta meta = assembler.fill(batch.data());
        if (meta.n_msgs == 0) break;
        if (meta.reset_vwap) acc.reset();
        cpu_aggregate(batch.data(), meta.n_msgs, acc);
        last_ts = meta.last_ts;
        rs.shipped_msgs += meta.n_msgs;
        ++rs.batches;
        if (rs.batches % (uint64_t)opt.snapshot_interval == 0) {
            cpu_snapshot(acc, snap.data(), itch::MAX_SYMBOLS);
            ++rs.snapshots;
            if (!opt.no_csv) print_snapshot_csv(last_ts, snap.data(), assembler.symbols());
        }
        if (opt.max_msgs && parser.total_messages() >= opt.max_msgs) break;
    }
    cpu_snapshot(acc, snap.data(), itch::MAX_SYMBOLS);
    ++rs.snapshots;
    if (!opt.no_csv) print_snapshot_csv(last_ts, snap.data(), assembler.symbols());

    rs.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    write_stats(opt, parser, assembler, rs);
    return parser.error() ? 1 : 0;
}

// ---------------------------------------------------------------------------
// GPU pipeline: double-buffered, 2 streams
// ---------------------------------------------------------------------------
int run_gpu(const Options& opt) {
    using namespace itch::gpu;

    itch::ItchParser parser(itch::make_gz_source(opt.file));
    itch::OrderHash orders;
    itch::BatchAssembler assembler(parser, orders);

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    SlotRing ring(4);
    DeviceAccum acc = alloc_accum();
    launch_reset(acc, itch::MAX_SYMBOLS, streams[0]);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    Snapshot* d_snap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_snap, itch::MAX_SYMBOLS * sizeof(Snapshot)));
    Snapshot* h_snap[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMallocHost(&h_snap[0], itch::MAX_SYMBOLS * sizeof(Snapshot)));
    CUDA_CHECK(cudaMallocHost(&h_snap[1], itch::MAX_SYMBOLS * sizeof(Snapshot)));
    cudaEvent_t snap_done[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&snap_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&snap_done[1], cudaEventDisableTiming));

    // Last aggregate event recorded on each stream, for cross-stream ordering
    // of reset and snapshot (which read/write whole accumulator arrays).
    cudaEvent_t last_agg[2] = {nullptr, nullptr};

    // Reset and snapshot mutate accumulators; batches issued afterwards on the
    // OTHER stream must not aggregate before they finish. Re-recorded after
    // each reset/snapshot launch; every batch waits on the latest recording.
    cudaEvent_t barrier_ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&barrier_ev, cudaEventDisableTiming));
    bool barrier_set = false;

    int pending = -1;            // h_snap index with an in-flight D2H
    uint64_t pending_ts = 0;

    RunStats rs;
    if (!opt.no_csv) print_csv_header();
    const auto t0 = std::chrono::steady_clock::now();

    uint64_t last_ts = 0;
    for (;;) {
        BatchSlot& slot = ring.acquire();
        itch::BatchMeta meta = assembler.fill(slot.h_raw);
        if (meta.n_msgs == 0) break;

        const int si = (int)(rs.batches & 1);
        cudaStream_t s = streams[si];

        if (barrier_set) CUDA_CHECK(cudaStreamWaitEvent(s, barrier_ev));

        if (meta.reset_vwap) {
            if (last_agg[1 - si])
                CUDA_CHECK(cudaStreamWaitEvent(s, last_agg[1 - si]));
            launch_reset(acc, itch::MAX_SYMBOLS, s);
            CUDA_CHECK(cudaEventRecord(barrier_ev, s));
            barrier_set = true;
        }

        CUDA_CHECK(cudaMemcpyAsync(slot.d_raw, slot.h_raw,
                                   (size_t)meta.n_msgs * itch::SLOT_BYTES,
                                   cudaMemcpyHostToDevice, s));
        launch_decode(slot.d_raw, (int)meta.n_msgs, slot.soa, s);
        launch_aggregate(slot.soa, (int)meta.n_msgs, acc, s);
        CUDA_CHECK(cudaEventRecord(slot.done, s));
        last_agg[si] = slot.done;

        last_ts = meta.last_ts;
        rs.shipped_msgs += meta.n_msgs;
        ++rs.batches;

        if (rs.batches % (uint64_t)opt.snapshot_interval == 0) {
            // Snapshot must observe both streams' aggregation up to here.
            if (last_agg[1 - si])
                CUDA_CHECK(cudaStreamWaitEvent(s, last_agg[1 - si]));
            const int bi = pending == 0 ? 1 : 0;
            if (pending >= 0) {  // drain the previous snapshot first
                CUDA_CHECK(cudaEventSynchronize(snap_done[pending]));
                if (!opt.no_csv)
                    print_snapshot_csv(pending_ts, h_snap[pending], assembler.symbols());
            }
            launch_snapshot(acc, d_snap, itch::MAX_SYMBOLS, s);
            CUDA_CHECK(cudaEventRecord(barrier_ev, s));
            barrier_set = true;
            CUDA_CHECK(cudaMemcpyAsync(h_snap[bi], d_snap,
                                       itch::MAX_SYMBOLS * sizeof(Snapshot),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaEventRecord(snap_done[bi], s));
            pending = bi;
            pending_ts = last_ts;
            ++rs.snapshots;
        }
        if (opt.max_msgs && parser.total_messages() >= opt.max_msgs) break;
    }

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    if (pending >= 0 && !opt.no_csv)
        print_snapshot_csv(pending_ts, h_snap[pending], assembler.symbols());

    // Final snapshot of whatever accumulated since the last interval.
    launch_snapshot(acc, d_snap, itch::MAX_SYMBOLS, streams[0]);
    CUDA_CHECK(cudaMemcpyAsync(h_snap[0], d_snap,
                               itch::MAX_SYMBOLS * sizeof(Snapshot),
                               cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    ++rs.snapshots;
    if (!opt.no_csv) print_snapshot_csv(last_ts, h_snap[0], assembler.symbols());

    rs.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    write_stats(opt, parser, assembler, rs);

    cudaEventDestroy(barrier_ev);
    cudaEventDestroy(snap_done[0]);
    cudaEventDestroy(snap_done[1]);
    cudaFreeHost(h_snap[0]);
    cudaFreeHost(h_snap[1]);
    cudaFree(d_snap);
    free_accum(acc);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    return parser.error() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage(argv[0]);
        return 2;
    }
    try {
        return opt.cpu_only ? run_cpu(opt) : run_gpu(opt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
