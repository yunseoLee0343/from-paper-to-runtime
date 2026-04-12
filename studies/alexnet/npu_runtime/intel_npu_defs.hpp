#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace intel_npu {

/// Unsigned 8-bit integer used for byte-addressable storage.
using u8 = std::uint8_t;
/// Signed 32-bit integer used for tensor extents and tile bounds.
using i32 = std::int32_t;
/// Unsigned 32-bit integer used for ids and hardware ordinals.
using u32 = std::uint32_t;
/// Unsigned 64-bit integer used for addresses and byte sizes.
using u64 = std::uint64_t;

/**
 * @brief Align a byte count or address upward to the next alignment boundary.
 *
 * Intel NPU DMA engines typically assume fixed burst granularity. This helper
 * is used by both host buffer allocation and SRAM planning so descriptors land
 * on hardware-friendly boundaries.
 *
 * @param value Value to round upward.
 * @param alignment Power-of-two style alignment requirement in bytes.
 * @return Rounded-up value.
 */
inline u64 align_up(u64 value, u64 alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

/**
 * @brief Logical memory domains visible to the simulator.
 */
enum class MemorySpace {
    /// CPU-owned memory used for staging and tests.
    HOST,
    /// External DRAM / DDR region visible to DMA.
    DDR,
    /// On-chip shared SRAM partitioned into banks.
    LOCAL_SRAM,
};

/**
 * @brief Direction of a DMA transaction relative to LOCAL_SRAM.
 */
enum class DmaDirection {
    /// Read from host/DDR staging memory into on-chip SRAM.
    HostToLocal,
    /// Flush on-chip SRAM contents back to host/DDR staging memory.
    LocalToHost,
};

/**
 * @brief Element type abstraction used for memory occupancy estimation.
 */
enum class DataType {
    FP32,
    FP16,
    BF16,
    INT8,
};

/**
 * @brief Activation or clamp style post-op fused onto a producer op.
 */
enum class PostOpKind {
    NONE,
    RELU,
    CLAMP,
};

inline const char* to_string(PostOpKind post_op) {
    switch (post_op) {
        case PostOpKind::NONE: return "NONE";
        case PostOpKind::RELU: return "RELU";
        case PostOpKind::CLAMP: return "CLAMP";
    }
    throw std::runtime_error("Unknown post-op");
}

/**
 * @brief Return the byte size for a single tensor element.
 *
 * @param dtype Element type.
 * @return Element size in bytes.
 */
inline u32 dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
    }
    throw std::runtime_error("Unknown dtype");
}

/**
 * @brief Compute total byte footprint from element count and dtype.
 *
 * @param elements Number of tensor elements.
 * @param dtype Element type stored in memory.
 * @return Total byte footprint.
 */
inline u64 tensor_bytes(u64 elements, DataType dtype) {
    return elements * static_cast<u64>(dtype_size(dtype));
}

/**
 * @brief Static hardware contract used by the Intel NPU simulator.
 *
 * The fields model the hardware properties Phase 1 depends on: total SRAM
 * capacity, number of SRAM banks, DMA burst alignment, and tile size limits
 * later reused by the scheduler and overlap pipeline.
 */
struct HardwareCaps {
    /// Total LOCAL_SRAM capacity shared by all banks.
    u64 local_sram_bytes = 4 * 1024 * 1024;
    /// DMA descriptor alignment requirement in bytes.
    u64 dma_alignment = 64;
    /// Number of SRAM banks available for bank-aware allocation and ping-pong.
    u64 sram_bank_count = 2;
    /// Maximum number of concurrent DMA operations the runtime may model.
    u32 max_dma_inflight = 4;
    /// Maximum output tile height supported by the simulated DPU.
    i32 max_tile_h = 32;
    /// Maximum output tile width supported by the simulated DPU.
    i32 max_tile_w = 32;
    /// Maximum output tile channel block supported by the simulated DPU.
    i32 max_tile_c = 128;
};

/**
 * @brief Minimal Level Zero style device/queue handle used by tests.
 */
struct LevelZeroMockHandle {
    u32 ordinal = 0;
    u32 queue_index = 0;
    std::string name = "intel-npu-ze-mock";
};

/**
 * @brief One reserved SRAM segment inside a specific LOCAL_SRAM bank.
 *
 * Offsets are bank-local rather than global, which matches the simulator's
 * bank vector layout and makes conflict detection explicit.
 */
struct LocalAlloc {
    std::string name;
    MemorySpace mem_space = MemorySpace::LOCAL_SRAM;
    u64 bank_id = 0;
    u64 offset = 0;
    u64 bytes = 0;
    u64 alignment = 64;

    /**
     * @brief Compute the first byte after this allocation.
     *
     * @return Exclusive end offset in the selected bank.
     */
    u64 end() const {
        return offset + bytes;
    }
};

/**
 * @brief DMA command descriptor submitted to the simulator.
 */
struct DmaDescriptor {
    DmaDirection direction = DmaDirection::HostToLocal;
    void* host_ptr = nullptr;
    u64 bank_id = 0;
    u64 bank_offset = 0;
    u64 bytes = 0;
    u64 alignment = 64;
    std::string tag;
};

/**
 * @brief Counters accumulated by the Phase 1 runtime.
 */
struct RuntimeStats {
    u64 dma_load_bytes = 0;
    u64 dma_store_bytes = 0;
};

/**
 * @brief RAII wrapper for host-visible aligned buffers.
 *
 * Tests use this class to guarantee 64-byte DMA legality before issuing
 * `exec_dma()` commands.
 */
class AlignedHostBuffer {
public:
    /**
     * @brief Allocate a zero-initialized aligned host buffer.
     *
     * @param bytes Requested payload size.
     * @param alignment Alignment requirement in bytes.
     */
    AlignedHostBuffer(std::size_t bytes, std::size_t alignment)
        : bytes_(align_up(bytes, alignment)), alignment_(alignment) {
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes_, alignment_);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, alignment_, bytes_) != 0) {
            ptr = nullptr;
        }
        if (!ptr) {
            throw std::bad_alloc();
        }
#endif
        data_.reset(static_cast<u8*>(ptr));
        std::memset(data_.get(), 0, bytes_);
    }

    /// @return Mutable pointer to aligned storage.
    u8* data() { return data_.get(); }
    /// @return Read-only pointer to aligned storage.
    const u8* data() const { return data_.get(); }
    /// @return Total allocated bytes after alignment rounding.
    std::size_t size() const { return bytes_; }

private:
    struct Deleter {
        void operator()(u8* ptr) const {
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    };

    std::size_t bytes_;
    std::size_t alignment_;
    std::unique_ptr<u8, Deleter> data_;
};

class LocalSramAllocator {
public:
    /**
     * @brief Create a bank-aware SRAM allocator from hardware capabilities.
     *
     * @param caps Hardware contract describing total SRAM and bank count.
     */
    explicit LocalSramAllocator(HardwareCaps caps);

    /**
     * @brief Reserve a conflict-free segment in LOCAL_SRAM.
     *
     * Allocations are aligned, checked against bank capacity, and compared
     * against all previous reservations in the same bank.
     *
     * @param name Human-readable allocation label.
     * @param bytes Requested size in bytes.
     * @param alignment Required alignment in bytes.
     * @param preferred_bank Optional bank affinity.
     * @return Reserved SRAM segment.
     */
    LocalAlloc allocate(
        const std::string& name,
        u64 bytes,
        u64 alignment = 64,
        std::optional<u64> preferred_bank = std::nullopt);

    /**
     * @brief Inspect all reservations assigned to a bank.
     *
     * @param bank_id Bank index.
     * @return Immutable list of reservations in the bank.
     */
    const std::vector<LocalAlloc>& reservations(u64 bank_id) const;
    /// @return Capacity of a single SRAM bank in bytes.
    u64 bank_size() const { return bank_size_; }

private:
    bool conflicts(const LocalAlloc& candidate, u64 bank_id) const;

    HardwareCaps caps_;
    u64 bank_size_;
    std::vector<std::vector<LocalAlloc>> reservations_;
    std::vector<u64> cursors_;
};

class IntelNPUSimulator {
public:
    /**
     * @brief Construct a simple Intel NPU simulator with banked SRAM storage.
     *
     * @param caps Hardware contract for the simulated device.
     * @param handle Level Zero style device/queue identity used by tests.
     */
    explicit IntelNPUSimulator(HardwareCaps caps, LevelZeroMockHandle handle = {});

    /// @return Associated mock Level Zero handle.
    const LevelZeroMockHandle& handle() const { return handle_; }
    /// @return Immutable hardware capabilities used by this simulator.
    const HardwareCaps& caps() const { return caps_; }
    /// @return Current DMA counters.
    RuntimeStats stats() const { return stats_; }

    /**
     * @brief Allocate a LOCAL_SRAM segment through the bank-aware allocator.
     *
     * @param name Allocation label.
     * @param bytes Requested bytes.
     * @param alignment Required alignment.
     * @param preferred_bank Optional target bank.
     * @return Reserved SRAM allocation.
     */
    LocalAlloc allocate_local(
        const std::string& name,
        u64 bytes,
        u64 alignment = 64,
        std::optional<u64> preferred_bank = std::nullopt);

    /**
     * @brief Execute a DMA transfer after alignment and bounds validation.
     *
     * The simulator enforces the same 64-byte style contract expected by Intel
     * NPU DMA paths: host pointer, bank offset, and transfer length must all be
     * aligned before the copy is allowed to proceed.
     *
     * @param desc DMA descriptor describing direction, bank, and payload.
     */
    void exec_dma(const DmaDescriptor& desc);

    /**
     * @brief Read a byte range back from a specific SRAM bank.
     *
     * @param bank_id Bank index.
     * @param offset Bank-local start offset.
     * @param bytes Number of bytes to copy.
     * @return Snapshot of the requested SRAM range.
     */
    std::vector<u8> read_local(u64 bank_id, u64 offset, u64 bytes) const;
    /**
     * @brief Write raw bytes into a bank-local SRAM range.
     *
     * @param bank_id Bank index.
     * @param offset Bank-local start offset.
     * @param src Source pointer.
     * @param bytes Number of bytes to copy.
     */
    void write_local(u64 bank_id, u64 offset, const void* src, u64 bytes);

private:
    void validate_dma_alignment(const DmaDescriptor& desc) const;
    void validate_bank_bounds(u64 bank_id, u64 offset, u64 bytes) const;

    HardwareCaps caps_;
    LevelZeroMockHandle handle_;
    LocalSramAllocator allocator_;
    std::vector<std::vector<u8>> banks_;
    RuntimeStats stats_;
};

}  // namespace intel_npu
