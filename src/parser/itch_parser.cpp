#include "parser/itch_parser.hpp"
#include "parser/message_types.hpp"

#include <zlib.h>

#include <cstring>
#include <stdexcept>

namespace itch {

// ---------------------------------------------------------------------------
// Byte sources
// ---------------------------------------------------------------------------
namespace {

class GzSource final : public ByteSource {
public:
    explicit GzSource(const std::string& path) {
        f_ = gzopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("gzopen failed: " + path);
        gzbuffer(f_, 1u << 20);  // 1 MB inflate window per syscall
    }
    ~GzSource() override { if (f_) gzclose(f_); }

    long read(uint8_t* dst, size_t n) override {
        int r = gzread(f_, dst, (unsigned)n);
        return r;  // gzread returns <0 on error, 0 at EOF
    }

private:
    gzFile f_ = nullptr;
};

class MemSource final : public ByteSource {
public:
    MemSource(const uint8_t* data, size_t n) : data_(data), left_(n) {}

    long read(uint8_t* dst, size_t n) override {
        size_t take = n < left_ ? n : left_;
        std::memcpy(dst, data_, take);
        data_ += take;
        left_ -= take;
        return (long)take;
    }

private:
    const uint8_t* data_;
    size_t left_;
};

}  // namespace

std::unique_ptr<ByteSource> make_gz_source(const std::string& path) {
    return std::make_unique<GzSource>(path);
}
std::unique_ptr<ByteSource> make_mem_source(const uint8_t* data, size_t n) {
    return std::make_unique<MemSource>(data, n);
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------
ItchParser::ItchParser(std::unique_ptr<ByteSource> src, size_t buf_bytes)
    : src_(std::move(src)), buf_(buf_bytes) {
    if (buf_bytes < 64) throw std::runtime_error("parser buffer too small");
}

bool ItchParser::ensure(size_t want) {
    if (tail_ - head_ >= want) return true;
    if (eof_ || error_) return false;
    // Compact: move the unconsumed tail to the front, then top up.
    if (head_ > 0) {
        std::memmove(buf_.data(), buf_.data() + head_, tail_ - head_);
        tail_ -= head_;
        head_ = 0;
    }
    while (tail_ - head_ < want) {
        long r = src_->read(buf_.data() + tail_, buf_.size() - tail_);
        if (r < 0) { error_ = true; return false; }
        if (r == 0) { eof_ = true; return false; }
        tail_ += (size_t)r;
    }
    return true;
}

const uint8_t* ItchParser::next(uint16_t* len) {
    if (!ensure(2)) return nullptr;
    uint16_t n = load_be16(buf_.data() + head_);
    if (n == 0) { eof_ = true; return nullptr; }  // end-of-session marker
    if (!ensure(2u + n)) {
        // Length prefix promised more bytes than the stream holds.
        if (!error_ && tail_ - head_ > 2) error_ = true;
        return nullptr;
    }
    const uint8_t* msg = buf_.data() + head_ + 2;
    head_ += 2u + n;
    ++total_;
    ++counts_[msg[0]];
    *len = n;
    return msg;
}

// ---------------------------------------------------------------------------
// Symbol table
// ---------------------------------------------------------------------------
SymbolTable::SymbolTable() : names_(MAX_SYMBOLS) {
    for (auto& a : names_) a[0] = '\0';
}

void SymbolTable::update(const uint8_t* r_msg) {
    uint16_t locate = load_be16(r_msg + ofs::LOCATE);
    if (locate >= names_.size()) return;
    auto& dst = names_[locate];
    int n = 0;
    for (; n < 8; ++n) {
        char c = (char)r_msg[ofs::dir::STOCK + n];
        if (c == ' ' || c == '\0') break;
        dst[n] = c;
    }
    dst[n] = '\0';
}

const char* SymbolTable::name(uint16_t locate) const {
    return locate < names_.size() ? names_[locate].data() : "";
}

}  // namespace itch
