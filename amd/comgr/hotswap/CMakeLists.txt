cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# When this CMakeLists is the top-level project we build the gtest-based
# transpiler_tests binary. When pulled in by a parent project (e.g. COMGR)
# via add_subdirectory we only build the library.
if(NOT DEFINED HOTSWAP_TRANSPILER_BUILD_TOOLS)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOTSWAP_TRANSPILER_BUILD_TOOLS ON)
  else()
    set(HOTSWAP_TRANSPILER_BUILD_TOOLS OFF)
  endif()
endif()

# --- LLVM ---------------------------------------------------------------------
# The MC stack uses AMDGPU target-private headers (SIDefines.h, SIInstrInfo.h,
# AMDGPUBaseInfo.h) which transitively include TableGen-generated
# AMDGPUGen*.inc files. Those .inc files are produced by llvm-tblgen during
# the LLVM build and are not copied by `ninja install`, so we require an
# LLVM *build tree* rather than an install tree.
if(TARGET LLVMSupport AND DEFINED LLVM_MAIN_SRC_DIR)
  set(LLVM_BUILD_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}")
else()
  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

if(NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR OR NOT LLVM_BUILD_MAIN_SRC_DIR OR
   NOT DEFINED LLVM_BINARY_DIR OR NOT LLVM_BINARY_DIR)
  message(FATAL_ERROR
    "hotswap-transpiler requires an LLVM build tree (not an install tree). "
    "The install tree does not expose AMDGPU target-private headers "
    "(SIDefines.h, AMDGPUBaseInfo.h, AMDGPUGen*.inc). "
    "Pass -DLLVM_DIR=<llvm-build>/lib/cmake/llvm.")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# --- Library ------------------------------------------------------------------
add_library(hotswap-transpiler STATIC
  raiser.cpp
  raise_failure.cpp
  code_object_utils.cpp
  mc_state.cpp
  canonical_op.cpp
  opcode_map.cpp
  decode.cpp
  raise_context.cpp
  reg_file.cpp
  kernarg_layout.cpp
  user_sgpr_layout.cpp
  wave_projection.cpp
  setpc_analysis.cpp
  handle_sopp.cpp
  handle_sopc.cpp
  canonical_op_attrs.cpp
  handle_sop1.cpp
  handle_sop2.cpp
  pipeline.cpp
  translation_cache.cpp
  handle_smem.cpp
  handle_sopk.cpp
  flat_addr.cpp
  mubuf_addr.cpp
  handle_flat.cpp
  handle_mubuf.cpp
  handle_ds.cpp
  wave_size_obstruction.cpp
  wmma_lowering.cpp
  mxfp4_dequant.cpp
  handle_valu.cpp
  handle_valu_cross_lane.cpp
  handle_valu_small_ops.cpp
  handle_valu_vcmp.cpp
  handle_valu_vop3p.cpp
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-transpiler)

set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# AMDGPU target-private headers (not installed by `ninja install`).
target_include_directories(hotswap-transpiler PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)

# Public include root so consumers can `#include "hotswap/raiser.hpp"`.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# `llc`, `llvm-mc`, `ld.lld` etc. live in the build tree's bin/. The
# pipeline driver (pipeline.cpp) shells out to them via the
# `LLVM_TOOLS_DIR` macro at runtime.
set(HOTSWAP_TRANSPILER_LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}" CACHE PATH
  "Directory containing llc, llvm-mc, ld.lld, llvm-objdump used by Hotswap")
target_compile_definitions(hotswap-transpiler PRIVATE
  LLVM_TOOLS_DIR="${HOTSWAP_TRANSPILER_LLVM_TOOLS_DIR}"
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    LLVMCore
    LLVMSupport
    LLVMTargetParser
    LLVMObject
    LLVMBinaryFormat
    LLVMMC
    LLVMMCDisassembler
    LLVMMCParser
    LLVMTarget
    LLVMAMDGPUDisassembler
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser
    LLVMAMDGPUDesc
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
)

# --- Tests --------------------------------------------------------------------
if(HOTSWAP_TRANSPILER_BUILD_TOOLS)
  find_package(GTest REQUIRED)
  enable_testing()
  include(GoogleTest)

  add_executable(transpiler_tests
    tests/test_main.cpp
    tests/raiser_empty_module_test.cpp
    tests/code_object_utils_test.cpp
    tests/mc_context_inline_srcmgr_test.cpp
    tests/opcode_map_test.cpp
    tests/kernarg_layout_test.cpp
    tests/wave_projection_test.cpp
  
    tests/translation_cache_test.cpp
  
    tests/mxfp4_dequant_test.cpp
  )
  llvm_update_compile_flags(transpiler_tests)
  target_link_libraries(transpiler_tests
    PRIVATE
      hotswap-transpiler
      GTest::GTest
  )
  # Tests that include transpiler internals need the same AMDGPU
  # target-private include paths the library uses.
  target_include_directories(transpiler_tests PRIVATE
    "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
    "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
    "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
  )
  gtest_discover_tests(transpiler_tests
    PROPERTIES TIMEOUT 60 LABELS "transpiler"
  )
endif()
