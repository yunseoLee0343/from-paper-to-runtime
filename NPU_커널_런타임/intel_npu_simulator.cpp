#include "intel_npu_defs.hpp"

#include <sstream>

namespace intel_npu {

LocalSramAllocator::LocalSramAllocator(HardwareCaps caps)
    : caps_(caps),
      bank_size_(caps.sram_bank_count == 0 ? 0 : caps.local_sram_bytes / caps.sram_bank_count),
      reservations_(caps.sram_bank_count),
      cursors_(caps.sram_bank_count, 0) {
    if (caps_.sram_bank_count == 0) {
        throw std::runtime_error("sram_bank_count must be greater than zero");
    }
    if (caps_.dma_alignment == 0) {
        throw std::runtime_error("dma_alignment must be greater than zero");
    }
}

LocalAlloc LocalSramAllocator::allocate(
    const std::string& name,
    u64 bytes,
    u64 alignment,
    std::optional<u64> preferred_bank) {
    if (bytes == 0) {
        throw std::runtime_error("LOCAL_SRAM allocation bytes must be > 0");
    }
    if (alignment == 0) {
        throw std::runtime_error("LOCAL_SRAM allocation alignment must be > 0");
    }

    std::vector<u64> bank_order;
    if (preferred_bank.has_value()) {
        if (*preferred_bank >= caps_.sram_bank_count) {
            throw std::runtime_error("preferred LOCAL_SRAM bank out of range");
        }
        bank_order.push_back(*preferred_bank);
    } else {
        for (u64 bank_id = 0; bank_id < caps_.sram_bank_count; ++bank_id) {
            bank_order.push_back(bank_id);
        }
        std::stable_sort(bank_order.begin(), bank_order.end(), [&](u64 lhs, u64 rhs) {
            return cursors_[lhs] < cursors_[rhs];
        });
    }

    for (u64 bank_id : bank_order) {
        const u64 offset = align_up(cursors_[bank_id], alignment);
        LocalAlloc candidate{name, MemorySpace::LOCAL_SRAM, bank_id, offset, bytes, alignment};
        if (candidate.end() > bank_size_) {
            continue;
        }
        if (conflicts(candidate, bank_id)) {
            continue;
        }
        reservations_[bank_id].push_back(candidate);
        cursors_[bank_id] = candidate.end();
        return candidate;
    }

    throw std::runtime_error("LOCAL_SRAM allocation failed for " + name + ": all banks exhausted");
}

const std::vector<LocalAlloc>& LocalSramAllocator::reservations(u64 bank_id) const {
    if (bank_id >= reservations_.size()) {
        throw std::runtime_error("bank_id out of range");
    }
    return reservations_[bank_id];
}

bool LocalSramAllocator::conflicts(const LocalAlloc& candidate, u64 bank_id) const {
    for (const auto& existing : reservations_[bank_id]) {
        const bool overlap = !(candidate.end() <= existing.offset || existing.end() <= candidate.offset);
        if (overlap) {
            return true;
        }
    }
    return false;
}

IntelNPUSimulator::IntelNPUSimulator(HardwareCaps caps, LevelZeroMockHandle handle)
    : caps_(caps),
      handle_(std::move(handle)),
      allocator_(caps_),
      banks_(caps_.sram_bank_count, std::vector<u8>(allocator_.bank_size(), 0)) {}

LocalAlloc IntelNPUSimulator::allocate_local(
    const std::string& name,
    u64 bytes,
    u64 alignment,
    std::optional<u64> preferred_bank) {
    return allocator_.allocate(name, bytes, alignment, preferred_bank);
}

void IntelNPUSimulator::exec_dma(const DmaDescriptor& desc) {
    if (desc.bytes == 0) {
        throw std::runtime_error("DMA bytes must be > 0");
    }
    if (desc.host_ptr == nullptr) {
        throw std::runtime_error("DMA host_ptr must not be null");
    }

    validate_dma_alignment(desc);
    validate_bank_bounds(desc.bank_id, desc.bank_offset, desc.bytes);

    auto* host = static_cast<u8*>(desc.host_ptr);
    auto& bank = banks_[desc.bank_id];
    auto* local = bank.data() + desc.bank_offset;

    if (desc.direction == DmaDirection::HostToLocal) {
        std::memcpy(local, host, static_cast<std::size_t>(desc.bytes));
        stats_.dma_load_bytes += desc.bytes;
    } else {
        std::memcpy(host, local, static_cast<std::size_t>(desc.bytes));
        stats_.dma_store_bytes += desc.bytes;
    }
}

std::vector<u8> IntelNPUSimulator::read_local(u64 bank_id, u64 offset, u64 bytes) const {
    validate_bank_bounds(bank_id, offset, bytes);
    const auto& bank = banks_[bank_id];
    return std::vector<u8>(bank.begin() + static_cast<std::ptrdiff_t>(offset),
                           bank.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
}

void IntelNPUSimulator::write_local(u64 bank_id, u64 offset, const void* src, u64 bytes) {
    if (src == nullptr) {
        throw std::runtime_error("write_local src must not be null");
    }
    validate_bank_bounds(bank_id, offset, bytes);
    std::memcpy(banks_[bank_id].data() + offset, src, static_cast<std::size_t>(bytes));
}

void IntelNPUSimulator::validate_dma_alignment(const DmaDescriptor& desc) const {
    const auto require_aligned = [&](u64 value, const char* field) {
        if (value % caps_.dma_alignment != 0) {
            throw std::runtime_error(std::string("DMA alignment violation in ") + field);
        }
    };

    require_aligned(desc.bytes, "bytes");
    require_aligned(desc.bank_offset, "bank_offset");

    const auto host_addr = reinterpret_cast<std::uintptr_t>(desc.host_ptr);
    if (host_addr % caps_.dma_alignment != 0) {
        throw std::runtime_error("DMA alignment violation in host_ptr");
    }
}

void IntelNPUSimulator::validate_bank_bounds(u64 bank_id, u64 offset, u64 bytes) const {
    if (bank_id >= banks_.size()) {
        throw std::runtime_error("LOCAL_SRAM bank_id out of range");
    }
    const auto& bank = banks_[bank_id];
    if (offset + bytes > bank.size()) {
        std::ostringstream oss;
        oss << "LOCAL_SRAM access out of bounds: bank=" << bank_id
            << " offset=" << offset
            << " bytes=" << bytes;
        throw std::runtime_error(oss.str());
    }
}

}  // namespace intel_npu
