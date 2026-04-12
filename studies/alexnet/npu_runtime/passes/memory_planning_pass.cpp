#include "../intel_npu_scheduler.hpp"
#include "../ir/npu_ir.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace intel_npu {

namespace {

u64 tile_elements(const TileRegion& region) {
    return static_cast<u64>(region.n_extent()) *
           static_cast<u64>(region.c_extent()) *
           static_cast<u64>(region.h_extent()) *
           static_cast<u64>(region.w_extent());
}

bool overlaps_spatial(const TileRegion& a, const TileRegion& b) {
    const bool h_overlap = !(a.h1 <= b.h0 || b.h1 <= a.h0);
    const bool w_overlap = !(a.w1 <= b.w0 || b.w1 <= a.w0);
    return h_overlap && w_overlap;
}

bool overlaps_channels(const TileRegion& a, const TileRegion& b) {
    return !(a.c1 <= b.c0 || b.c1 <= a.c0);
}

TileMemoryPlan make_tile_memory_plan(const ConvTileContract& tile, const HardwareCaps& hw) {
    const u64 bank_count = std::max<u64>(1, hw.sram_bank_count);
    const u64 bank_size = hw.local_sram_bytes / bank_count;
    const u64 bank_id = static_cast<u64>(tile.task.pipeline_phase % static_cast<i32>(bank_count));

    TileMemoryPlan plan;
    plan.bank_id = bank_id;
    plan.lifetime_begin = tile.task.seq_no;
    plan.lifetime_end = tile.task.seq_no;

    u64 offset = 0;
    plan.input_tile = {"input_tile", MemorySpace::LOCAL_SRAM, bank_id, align_up(offset, hw.dma_alignment), tile_elements(tile.input_region) * 2, hw.dma_alignment};
    offset = plan.input_tile.end();
    plan.weight_tile = {"weight_tile", MemorySpace::LOCAL_SRAM, bank_id, align_up(offset, hw.dma_alignment), tile_elements(tile.weight_region) * 2, hw.dma_alignment};
    offset = plan.weight_tile.end();
    plan.output_tile = {"output_tile", MemorySpace::LOCAL_SRAM, bank_id, align_up(offset, hw.dma_alignment), tile_elements(tile.output_region) * 2, hw.dma_alignment};

    if (plan.output_tile.end() > bank_size) {
        throw std::runtime_error("Tile memory plan exceeds SRAM bank size");
    }
    return plan;
}

}  // namespace

bool run_memory_planning_pass(std::vector<ConvTileContract>& tiles, const HardwareCaps& hw, PassLog* log) {
    bool changed = false;
    std::map<u32, std::size_t> tile_index;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        tile_index[tiles[i].task.global_tile_id] = i;
    }

    for (auto& tile : tiles) {
        tile.mem_plan = make_tile_memory_plan(tile, hw);
        if (!tile.consumer_tile_ids.empty()) {
            i32 lifetime_end = tile.task.seq_no;
            for (u32 consumer_id : tile.consumer_tile_ids) {
                const auto it = tile_index.find(consumer_id);
                if (it != tile_index.end()) {
                    lifetime_end = std::max(lifetime_end, tiles[it->second].task.seq_no);
                }
            }
            tile.mem_plan.lifetime_end = lifetime_end;
        }
        changed = true;
    }

    for (auto& tile : tiles) {
        for (u32 producer_id : tile.producer_tile_ids) {
            const auto it = tile_index.find(producer_id);
            if (it == tile_index.end()) {
                continue;
            }
            auto& producer = tiles[it->second];
            if (!overlaps_spatial(producer.output_region, tile.input_region) ||
                !overlaps_channels(producer.output_region, tile.input_region)) {
                continue;
            }

            tile.mem_plan.input_tile = producer.mem_plan.output_tile;
            tile.mem_plan.input_tile.name = "reused_input_tile";
            tile.mem_plan.reuses_producer_sram = true;
            tile.mem_plan.reused_from_tile_id = producer.task.global_tile_id;
            tile.mem_plan.bank_id = producer.mem_plan.bank_id;
            tile.task.pipeline_phase = producer.task.pipeline_phase;
            producer.mem_plan.lifetime_end = std::max(producer.mem_plan.lifetime_end, tile.task.seq_no);
            if (log != nullptr) {
                log->append(
                    "MemoryPlanningPass",
                    "tile " + std::to_string(tile.task.global_tile_id) + " reuses SRAM from tile " + std::to_string(producer.task.global_tile_id));
            }
            break;
        }
    }

    return changed;
}

}  // namespace intel_npu
