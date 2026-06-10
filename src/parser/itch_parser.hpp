#pragma once
// Streaming ITCH 5.0 BinaryFILE parser: 2-byte big-endian length prefix per
// message, message type at payload[0]. Pull API - the caller drains messages
// one at a time; returned pointers are valid until the next call.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace itch {

// Minimal byte stream abstraction so the framing layer is testable from
// memory and the production path can stream-decompress a .gz day file.
class ByteSource {
public:
    virtual ~ByteSource() = default;
    // Read up to n bytes into dst. Returns bytes read, 0 at EOF, -1 on error.
    virtual long read(uint8_t* dst, size_t n) = 0;
};

std::unique_ptr<ByteSource> make_gz_source(const std::string& path);
std::unique_ptr<ByteSource> make_mem_source(const uint8_t* data, size_t n);

class ItchParser {
public:
    // buf_bytes: internal refill buffer; must exceed the largest message (50 B),
    // sized in MB so gzread amortizes. Small values are used by boundary tests.
    explicit ItchParser(std::unique_ptr<ByteSource> src, size_t buf_bytes = 4u << 20);

    // Next message, or nullptr at end of stream. *len excludes the 2-byte
    // length prefix. A zero length prefix terminates the stream (BinaryFILE
    // end-of-session marker).
    const uint8_t* next(uint16_t* len);

    bool error() const { return error_; }
    uint64_t total_messages() const { return total_; }
    const std::array<uint64_t, 256>& counts_by_type() const { return counts_; }

private:
    // Ensure >= want unconsumed bytes buffered; false at EOF/error.
    bool ensure(size_t want);

    std::unique_ptr<ByteSource> src_;
    std::vector<uint8_t> buf_;
    size_t head_ = 0;   // first unconsumed byte
    size_t tail_ = 0;   // one past last valid byte
    bool eof_ = false;
    bool error_ = false;
    uint64_t total_ = 0;
    std::array<uint64_t, 256> counts_{};
};

// locate code -> ticker, built from Stock Directory ('R') messages.
class SymbolTable {
public:
    SymbolTable();
    void update(const uint8_t* r_msg);          // an 'R' message
    const char* name(uint16_t locate) const;    // NUL-terminated, "" if unknown

private:
    std::vector<std::array<char, 9>> names_;
};

}  // namespace itch
