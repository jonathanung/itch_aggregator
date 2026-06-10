#pragma once
// Fixed ring of batch slots for the double-buffered pipeline. Each slot owns
// a pinned host buffer (DMA-able for cudaMemcpyAsync), a device raw buffer,
// a decoded SoA, and a cudaEvent marking when the GPU is done with it.
// acquire() blocks (cudaEventSynchronize) until the slot's previous use has
// drained, which back-pressures the CPU parser naturally.

#include "cuda/kernels.cuh"

#include <vector>

namespace itch::gpu {

struct BatchSlot {
    uint8_t* h_raw = nullptr;  // pinned, BATCH_MSGS * SLOT_BYTES
    uint8_t* d_raw = nullptr;
    DecodedBatch soa{};
    cudaEvent_t done{};        // recorded after this slot's aggregate kernel
};

class SlotRing {
public:
    explicit SlotRing(int n_slots = 4) : slots_(n_slots) {
        for (auto& s : slots_) {
            CUDA_CHECK(cudaMallocHost(&s.h_raw, BATCH_MSGS * SLOT_BYTES));
            CUDA_CHECK(cudaMalloc(&s.d_raw, BATCH_MSGS * SLOT_BYTES));
            s.soa = alloc_decoded();
            // Never-recorded events report complete, so first acquire is free.
            CUDA_CHECK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
        }
    }

    ~SlotRing() {
        for (auto& s : slots_) {
            cudaEventDestroy(s.done);
            free_decoded(s.soa);
            cudaFree(s.d_raw);
            cudaFreeHost(s.h_raw);
        }
    }

    SlotRing(const SlotRing&) = delete;
    SlotRing& operator=(const SlotRing&) = delete;

    BatchSlot& acquire() {
        BatchSlot& s = slots_[next_];
        next_ = (next_ + 1) % slots_.size();
        CUDA_CHECK(cudaEventSynchronize(s.done));
        return s;
    }

private:
    std::vector<BatchSlot> slots_;
    size_t next_ = 0;
};

}  // namespace itch::gpu
