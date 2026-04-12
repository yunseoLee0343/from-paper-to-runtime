cmake_minimum_required(VERSION 3.20)
project(alexnet_npu_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(alexnet_npu_runtime_core STATIC
    intel_npu_simulator.cpp
    intel_npu_scheduler.cpp
    intel_npu_pipeline.cpp
    ir/npu_ir.cpp
    passes/strip_dropout_pass.cpp
    passes/fuse_activation_pass.cpp
    passes/decompose_unsupported_pass.cpp
    passes/canonicalize_pass.cpp
    passes/memory_planning_pass.cpp
    passes/emit_descriptors_pass.cpp
    runtime/descriptor_serializer.cpp
)

target_include_directories(alexnet_npu_runtime_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ir
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

add_executable(alexnet_npu_runtime_demo alexnet_npu_runtime_demo.cpp)
target_link_libraries(alexnet_npu_runtime_demo PRIVATE alexnet_npu_runtime_core)

add_executable(intel_npu_test_p1 tests/intel_npu_test_p1.cpp)
target_link_libraries(intel_npu_test_p1 PRIVATE alexnet_npu_runtime_core)
add_test(NAME intel_npu_test_p1 COMMAND intel_npu_test_p1)

add_executable(intel_npu_test_p2 tests/intel_npu_test_p2.cpp)
target_link_libraries(intel_npu_test_p2 PRIVATE alexnet_npu_runtime_core)
add_test(NAME intel_npu_test_p2 COMMAND intel_npu_test_p2)

add_executable(intel_npu_test_p3 tests/intel_npu_test_p3.cpp)
target_link_libraries(intel_npu_test_p3 PRIVATE alexnet_npu_runtime_core)
add_test(NAME intel_npu_test_p3 COMMAND intel_npu_test_p3)

add_executable(intel_npu_test_passes tests/intel_npu_test_passes.cpp)
target_link_libraries(intel_npu_test_passes PRIVATE alexnet_npu_runtime_core)
add_test(NAME intel_npu_test_passes COMMAND intel_npu_test_passes)

add_executable(intel_npu_test_descriptors tests/intel_npu_test_descriptors.cpp)
target_link_libraries(intel_npu_test_descriptors PRIVATE alexnet_npu_runtime_core)
add_test(NAME intel_npu_test_descriptors COMMAND intel_npu_test_descriptors)
