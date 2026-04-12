FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    WORKSPACE=/intel_npu_workspace \
    INTEL_NPU_SIM_ROOT=/opt/intel/npu_sim \
    CMAKE_GENERATOR=Ninja

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gpg-agent \
        lsb-release \
        software-properties-common \
        wget \
    && mkdir -p /etc/apt/keyrings \
    && wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --dearmor -o /etc/apt/keyrings/intel-gpu.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/intel-gpu.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
        > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        git \
        intel-level-zero-gpu \
        level-zero-devel \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir openvino-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
    "${WORKSPACE}" \
    "${INTEL_NPU_SIM_ROOT}/include/intel_npu/arch" \
    "${INTEL_NPU_SIM_ROOT}/include/intel_npu/runtime" \
    "${INTEL_NPU_SIM_ROOT}/include/intel_npu/tile" \
    "${INTEL_NPU_SIM_ROOT}/include/intel_npu/vpu" \
    "${INTEL_NPU_SIM_ROOT}/src" \
    "${WORKSPACE}/build" \
    "${WORKSPACE}/include" \
    "${WORKSPACE}/src"

RUN cat <<'EOF' > "${INTEL_NPU_SIM_ROOT}/include/intel_npu/arch/tile_config.hpp"
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace intel_npu::arch {

struct TileShape {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t channels;
};

struct TileMemoryBudget {
    std::size_t sram_bytes;
    std::size_t dma_burst_bytes;
    std::uint32_t dma_channels;
};

struct TileExecutionPolicy {
    std::uint32_t tile_count;
    std::uint32_t compute_engines_per_tile;
    bool overlap_dma_with_compute;
};

inline constexpr TileShape kDefaultTileShape{64, 64, 32};
inline constexpr TileMemoryBudget kDefaultTileBudget{2U * 1024U * 1024U, 256U, 2U};
inline constexpr TileExecutionPolicy kDefaultExecutionPolicy{2U, 1U, true};

}  // namespace intel_npu::arch
EOF

RUN cat <<'EOF' > "${INTEL_NPU_SIM_ROOT}/include/intel_npu/runtime/dma_engine.hpp"
#pragma once

#include <cstddef>
#include <cstdint>

namespace intel_npu::runtime {

enum class DmaDirection : std::uint8_t {
    kHostToSram,
    kSramToHost,
    kSramToTile,
    kTileToSram,
};

struct DmaTransferDesc {
    DmaDirection direction;
    std::size_t bytes;
    std::uint32_t src_tile;
    std::uint32_t dst_tile;
    bool fence;
};

}  // namespace intel_npu::runtime
EOF

RUN cat <<'EOF' > "${INTEL_NPU_SIM_ROOT}/include/intel_npu/vpu/sram_layout.hpp"
#pragma once

#include <cstddef>
#include <cstdint>

namespace intel_npu::vpu {

struct SramRegion {
    const char* name;
    std::size_t offset;
    std::size_t size;
};

inline constexpr SramRegion kWeightsRegion{"weights", 0U, 512U * 1024U};
inline constexpr SramRegion kActivationRegion{"activations", 512U * 1024U, 1024U * 1024U};
inline constexpr SramRegion kCommandRegion{"commands", 1536U * 1024U, 512U * 1024U};

}  // namespace intel_npu::vpu
EOF

RUN cat <<'EOF' > "${WORKSPACE}/CMakeLists.txt"
cmake_minimum_required(VERSION 3.20)
project(intel_npu_workspace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(INTEL_NPU_ENABLE_TILE_PROFILING "Enable tile-oriented instrumentation hooks" ON)

add_library(intel_npu_sim INTERFACE)
target_include_directories(intel_npu_sim INTERFACE /opt/intel/npu_sim/include)
target_compile_definitions(
    intel_npu_sim
    INTERFACE
        INTEL_NPU_TILE_COUNT=2
        INTEL_NPU_SRAM_BYTES=2097152
        INTEL_NPU_DMA_CHANNELS=2
)

if(INTEL_NPU_ENABLE_TILE_PROFILING)
    target_compile_definitions(intel_npu_sim INTERFACE INTEL_NPU_ENABLE_TILE_PROFILING=1)
endif()

add_executable(tile_runtime_probe src/tile_runtime_probe.cpp)
target_link_libraries(tile_runtime_probe PRIVATE intel_npu_sim)
EOF

RUN cat <<'EOF' > "${WORKSPACE}/src/tile_runtime_probe.cpp"
#include <iostream>

#include "intel_npu/arch/tile_config.hpp"
#include "intel_npu/runtime/dma_engine.hpp"
#include "intel_npu/vpu/sram_layout.hpp"

int main() {
    using namespace intel_npu;

    std::cout << "Tile count: " << arch::kDefaultExecutionPolicy.tile_count << '\n';
    std::cout << "Tile SRAM bytes: " << arch::kDefaultTileBudget.sram_bytes << '\n';
    std::cout << "DMA channels: " << arch::kDefaultTileBudget.dma_channels << '\n';
    std::cout << "Activation SRAM offset: " << vpu::kActivationRegion.offset << '\n';
    return 0;
}
EOF

WORKDIR ${WORKSPACE}

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cmake -S "${WORKSPACE}" -B "${WORKSPACE}/build"' \
    'cmake --build "${WORKSPACE}/build"' \
    > /usr/local/bin/configure_intel_npu_workspace \
    && chmod +x /usr/local/bin/configure_intel_npu_workspace

ENV CPLUS_INCLUDE_PATH=${INTEL_NPU_SIM_ROOT}/include:${CPLUS_INCLUDE_PATH} \
    OpenVINO_DIR=/usr/local/lib/python3.10/dist-packages/openvino

CMD ["/bin/bash"]
