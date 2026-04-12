#include "intel_npu_defs.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

using namespace intel_npu;

int main() {
    HardwareCaps caps;
    caps.local_sram_bytes = 512;
    caps.sram_bank_count = 2;
    caps.dma_alignment = 64;

    LevelZeroMockHandle handle{0, 0, "ze-intel-npu-mock"};
    IntelNPUSimulator sim(caps, handle);

    const auto bank0_alloc = sim.allocate_local("input_tile_bank0", 64, 64, 0);
    const auto bank1_alloc = sim.allocate_local("output_tile_bank1", 64, 64, 1);

    assert(bank0_alloc.bank_id == 0);
    assert(bank1_alloc.bank_id == 1);
    assert(bank0_alloc.end() <= caps.local_sram_bytes / caps.sram_bank_count);
    assert(bank1_alloc.end() <= caps.local_sram_bytes / caps.sram_bank_count);

    AlignedHostBuffer host_in(64, caps.dma_alignment);
    AlignedHostBuffer host_out(64, caps.dma_alignment);

    for (std::size_t i = 0; i < host_in.size(); ++i) {
        host_in.data()[i] = static_cast<u8>(i);
    }

    sim.exec_dma(DmaDescriptor{
        DmaDirection::HostToLocal,
        host_in.data(),
        bank0_alloc.bank_id,
        bank0_alloc.offset,
        64,
        caps.dma_alignment,
        "load_bank0"
    });

    const auto bank0_bytes = sim.read_local(bank0_alloc.bank_id, bank0_alloc.offset, 64);
    assert(std::equal(bank0_bytes.begin(), bank0_bytes.end(), host_in.data()));

    std::fill(host_out.data(), host_out.data() + host_out.size(), static_cast<u8>(0));
    sim.write_local(bank1_alloc.bank_id, bank1_alloc.offset, host_in.data(), 64);

    sim.exec_dma(DmaDescriptor{
        DmaDirection::LocalToHost,
        host_out.data(),
        bank1_alloc.bank_id,
        bank1_alloc.offset,
        64,
        caps.dma_alignment,
        "store_bank1"
    });

    assert(std::equal(host_out.data(), host_out.data() + host_out.size(), host_in.data()));

    bool caught_alignment_error = false;
    try {
        sim.exec_dma(DmaDescriptor{
            DmaDirection::HostToLocal,
            host_in.data(),
            bank0_alloc.bank_id,
            bank0_alloc.offset,
            48,
            caps.dma_alignment,
            "misaligned_dma"
        });
    } catch (const std::runtime_error&) {
        caught_alignment_error = true;
    }
    assert(caught_alignment_error);

    const auto stats = sim.stats();
    assert(stats.dma_load_bytes == 64);
    assert(stats.dma_store_bytes == 64);

    std::cout << "Intel NPU Phase 1 test passed on handle: " << sim.handle().name << "\n";
    return 0;
}
