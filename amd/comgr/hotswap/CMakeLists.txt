cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# When this CMakeLists is the top-level project we build the raise_cli driver
# and the gtest-based transpiler_tests binary. When it is pulled in by a
# parent project (e.g. ROCR) via add_subdirectory we only build the library.
# A parent may still opt in explicitly by setting HOTSWAP_TRANSPILER_BUILD_TOOLS
# before add_subdirectory.
if(NOT DEFINED HOTSWAP_TRANSPILER_BUILD_TOOLS)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOTSWAP_TRANSPILER_BUILD_TOOLS ON)
  else()
    set(HOTSWAP_TRANSPILER_BUILD_TOOLS OFF)
  endif()
endif()

# --- Paths to LLVM ------------------------------------------------------------
# Hotswap needs an LLVM *build tree* (not an install tree) because the raiser
# uses AMDGPU target-private headers (SIDefines.h, SIInstrInfo.h,
# AMDGPUBaseInfo.h) which transitively #include TableGen-generated
# AMDGPUGen*.inc files. Those .inc files are produced by llvm-tblgen during
# the LLVM build and are not copied by `ninja install`.
#
# We support three configurations:
#
#   1. In-tree LLVM build. Hotswap was added via add_subdirectory() from
#      COMGR's CMake via add_subdirectory(). LLVM has
#      already populated LLVM_MAIN_SRC_DIR / LLVM_BINARY_DIR and the LLVM
#      library targets; AddLLVM.cmake is loaded. No find_package needed.
#
#   2. add_subdirectory() from another out-of-tree project (comgr, ROCR,
#      ...) that already called find_package(LLVM). Variables are
#      populated; the find_package call below becomes a no-op.
#
#   3. Standalone (cmake -S amd/comgr/hotswap -B build). We need
#      find_package(LLVM) to resolve LLVM_DIR. Since hotswap now lives at
#      <llvm-project>/amd/comgr/hotswap, default LLVM_DIR to the conventional
#      <llvm-project>/build tree if the user did not pass one explicitly.
if(TARGET LLVMSupport AND DEFINED LLVM_MAIN_SRC_DIR)
  # Case 1. Mirror the variable name find_package(LLVM) would have set
  # against a build tree, so the rest of this file does not have to
  # branch.
  set(LLVM_BUILD_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}")
else()
  # Cases 2 and 3.
  if(NOT DEFINED LLVM_DIR OR NOT LLVM_DIR)
    # amd/comgr/hotswap/ -> amd/comgr/ -> amd/ -> <llvm-project>/
    set(_hotswap_default_llvm_build
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/cmake/llvm")
    if(EXISTS "${_hotswap_default_llvm_build}/LLVMConfig.cmake")
      get_filename_component(_hotswap_default_llvm_build
                             "${_hotswap_default_llvm_build}" ABSOLUTE)
      set(LLVM_DIR "${_hotswap_default_llvm_build}" CACHE PATH
          "Path to LLVMConfig.cmake (auto-detected from <llvm-project>/build)")
      message(STATUS
        "LLVM_DIR not set; defaulting to ${LLVM_DIR}")
    endif()
    unset(_hotswap_default_llvm_build)
  endif()
  find_package(LLVM REQUIRED CONFIG)
endif()

if(NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR OR NOT LLVM_BUILD_MAIN_SRC_DIR OR
   NOT DEFINED LLVM_BINARY_DIR OR NOT LLVM_BINARY_DIR)
  message(FATAL_ERROR
    "hotswap-transpiler requires an LLVM build tree (not an install tree). "
    "The install tree does not expose AMDGPU target-private headers "
    "(SIDefines.h, AMDGPUBaseInfo.h, AMDGPUGen*.inc). "
    "Pass -DLLVM_DIR=<llvm-build>/lib/cmake/llvm or build LLVM at "
    "<llvm-project>/build so it can be auto-detected.")
endif()

if(DEFINED LLVM_DIR AND LLVM_DIR)
  message(STATUS "Using LLVM from: ${LLVM_DIR}")
endif()
message(STATUS "LLVM source tree: ${LLVM_BUILD_MAIN_SRC_DIR}")
message(STATUS "LLVM build tree:  ${LLVM_BINARY_DIR}")
if(DEFINED LLVM_PACKAGE_VERSION)
  message(STATUS "LLVM version:     ${LLVM_PACKAGE_VERSION}")
endif()

# In the in-tree case (1), AddLLVM is already loaded by the parent LLVM
# build and llvm_update_compile_flags is available. Only pull it in
# explicitly for cases (2) and (3).
if(NOT COMMAND llvm_update_compile_flags)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

# --- Include paths ------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})

# --- Link paths ---------------------------------------------------------------
link_directories(${LLVM_LIBRARY_DIRS})

# --- LLVM definitions ---------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

# --- TDM emulation runtime helper (HIP -> bitcode -> embedded array) --------
# When `hipcc` is available we build `runtime/tdm.hip` to LLVM bitcode for
# gfx942 and embed the bytes via `cmake/EmbedBinary.cmake`. The transpiler
# parses and link-merges the bitcode into each raised IR module that uses
# the cross-target VIMAGE TENSOR helper (see `tdm_runtime.{hpp,cpp}` and
# `raiser.cpp` Phase 6.7). When hipcc is missing the embedded blob is
# empty (one-byte stub in tdm_runtime.cpp), `tdmRuntimeAvailable()`
# returns false, and the cross-target VIMAGE handler keeps today's loud
# refusal. This preserves the no-hipcc build behaviour exactly.
find_program(HIPCC hipcc)
set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP "")
if(HIPCC)
  set(_tdm_src "${CMAKE_CURRENT_SOURCE_DIR}/runtime/tdm.hip")
  set(_tdm_bc  "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime.gfx942.bc")
  set(_tdm_cpp "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime_blob.cpp")

  add_custom_command(
    OUTPUT ${_tdm_bc}
    COMMAND ${HIPCC}
            --offload-arch=gfx942
            -O2 -emit-llvm -c
            -fgpu-rdc
            --cuda-device-only
            -o ${_tdm_bc} ${_tdm_src}
    DEPENDS ${_tdm_src}
    COMMENT "Building TDM emulation runtime bitcode (gfx942)"
    VERBATIM)

  add_custom_command(
    OUTPUT ${_tdm_cpp}
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=${_tdm_bc}
            -DOUTPUT=${_tdm_cpp}
            -DSYMBOL=tdm_runtime_bc
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    DEPENDS ${_tdm_bc} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    COMMENT "Embedding TDM runtime bitcode into C++ blob"
    VERBATIM)

  set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP ${_tdm_cpp})
  set(HOTSWAP_HAVE_TDM_RUNTIME 1)
  message(STATUS "TDM runtime helper enabled (hipcc=${HIPCC})")
else()
  set(HOTSWAP_HAVE_TDM_RUNTIME 0)
  message(STATUS "TDM runtime helper DISABLED (hipcc not found); "
                 "cross-target gfx1250->gfx942 TENSOR ops will keep refusing")
endif()

# --- Raiser + Pipeline library ------------------------------------------------
add_library(hotswap-transpiler STATIC
  raiser.cpp
  raise_context.cpp
  reg_file.cpp
  decode.cpp
  wave_projection.cpp
  wave_size_obstruction.cpp
  rewrite_cross_lane_divergent.cpp
  c5_predicate_chain_classifier.cpp
  sem_op_attrs.cpp
  raise_failure.cpp
  mc_state.cpp
  opcode_map.cpp
  semop.cpp
  setpc_analysis.cpp
  handle_sopp.cpp
  handle_smem.cpp
  handle_sopc.cpp
  handle_sop1.cpp
  handle_sopk.cpp
  handle_sop2.cpp
  handle_valu.cpp
  handle_valu_cross_lane.cpp
  handle_valu_small_ops.cpp
  handle_valu_vcmp.cpp
  handle_valu_vop3p.cpp
  flat_addr.cpp
  mubuf_addr.cpp
  handle_flat.cpp
  handle_ds.cpp
  handle_mubuf.cpp
  handle_mfma.cpp
  handle_vopd.cpp
  handle_vimage.cpp
  wmma_lowering.cpp
  mxfp4_dequant.cpp
  pipeline.cpp
  translation_cache.cpp
  code_object_utils.cpp
  kernarg_layout.cpp
  user_sgpr_layout.cpp
  tdm_runtime.cpp
  ${HOTSWAP_TDM_RUNTIME_GENERATED_CPP}
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

target_compile_definitions(hotswap-transpiler PRIVATE
  HOTSWAP_HAVE_TDM_RUNTIME=${HOTSWAP_HAVE_TDM_RUNTIME}
)

# Match LLVM's compile flags (no-rtti, no-exceptions, etc.). LLVM is built
# with `LLVM_ENABLE_RTTI=OFF` and `LLVM_ENABLE_EH=OFF` by default, and every
# public LLVM type we consume — Value, Type, Instruction, MCInst, ... — relies
# on LLVM's hand-rolled RTTI (`isa<>` / `cast<>` / `dyn_cast<>`) rather than
# C++ `typeid` / `dynamic_cast`. Mixing an RTTI-on TU with RTTI-off LLVM libs
# produces undefined-reference errors for `typeinfo` symbols on any class with
# a virtual method. `llvm_update_compile_flags` (from AddLLVM) is the canonical
# way to stay in lockstep with libLLVM: it reads `LLVM_ENABLE_RTTI` /
# `LLVM_ENABLE_EH` from the LLVM CMake package and propagates the matching
# `-fno-rtti` / `-fno-exceptions` (or MSVC equivalents) to the target.
llvm_update_compile_flags(hotswap-transpiler)

# AMDGPU target-private headers (not installed by `ninja install`).
# SIDefines.h, SIInstrInfo.h → source tree Target/AMDGPU/{,Utils}
# AMDGPUGenInstrInfo.inc etc.  → build tree Target/AMDGPU
# These are PRIVATE because no public header (pipeline.hpp, transpiler.hpp)
# exposes AMDGPU-private LLVM types, so consumers (e.g. ROCR) do not need
# these paths to compile against us.
target_include_directories(hotswap-transpiler PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)

# Allow this static library to be linked into a shared library
# (libhsa-runtime64.so when consumed from ROCR).
set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public headers consumed by parent projects (pipeline.hpp, transpiler.hpp).
# Use the hotswap/ root so consumers can write `#include "hotswap/transpiler/pipeline.hpp"`
# or `#include "hotswap/transpiler.hpp"`, matching existing ROCR usage.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# `llc`, `llvm-mc`, `llvm-objdump` etc. live in the build tree's bin/.
# ROCR's trap-handler subproject may require a ROCm Clang package even when
# Hotswap itself is validated against a local LLVM build, so keep the tool path
# overrideable rather than deriving it unconditionally from the LLVM package
# used for in-process linkage.
set(HOTSWAP_TRANSPILER_LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}" CACHE PATH
  "Directory containing llc, llvm-mc, llvm-objdump, and related LLVM tools used by Hotswap")
target_compile_definitions(hotswap-transpiler PRIVATE
  LLVM_TOOLS_DIR="${HOTSWAP_TRANSPILER_LLVM_TOOLS_DIR}"
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    # LLVM AMDGPU
    LLVMAMDGPUDisassembler
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser
    LLVMAMDGPUDesc
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    # LLVM CodeGen
    LLVMCodeGen
    LLVMCodeGenTypes
    LLVMTarget
    LLVMAsmPrinter
    LLVMSelectionDAG
    LLVMGlobalISel
    LLVMMIRParser
    # LLVM MC
    LLVMMCDisassembler
    LLVMMC
    LLVMMCParser
    # LLVM Core + Support
    LLVMCore
    LLVMSupport
    LLVMObject
    LLVMBinaryFormat
    LLVMRemarks
    LLVMBitReader
    LLVMBitWriter
    LLVMBitstreamReader
    LLVMScalarOpts
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMIRReader
    LLVMAsmParser
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMInstrumentation
    LLVMVectorize
    LLVMPasses
    LLVMProfileData
    LLVMDebugInfoDWARF
    LLVMDebugInfoCodeView
    LLVMCFGuard
    LLVMSymbolize
    LLVMDebugInfoMSF
    LLVMDebugInfoPDB
    LLVMDemangle
    LLVMTextAPI
    LLVMFrontendOffloading
    LLVMFrontendOpenMP
    LLVMLinker
    LLVMObjCARCOpts
)

if(HOTSWAP_TRANSPILER_BUILD_TOOLS)

# --- Per-file raiser CLI (kerneldex coverage contract) ------------------------
# A small standalone binary that raises every kernel in a single .co/.hsaco
# file, fork-isolated, and emits the OK/FAIL line protocol that
# `kerneldex coverage --tool` parses. No gtest dependency so it can live
# outside the test harness and be invoked directly as a coverage tool.
add_executable(raise_cli raise_cli.cpp)
target_link_libraries(raise_cli PRIVATE hotswap-transpiler)
llvm_update_compile_flags(raise_cli)

# --- GoogleTest-based unified test binary -------------------------------------
find_package(GTest REQUIRED)
enable_testing()
include(GoogleTest)

set(TEST_SOURCES
  tests/test_main.cpp
  tests/c5_predicate_chain_test.cpp
  tests/kernarg_layout_test.cpp
  tests/mc_context_inline_srcmgr_test.cpp
  tests/mxfp4_dequant_test.cpp
  tests/opcode_map_test.cpp
  tests/translation_cache_test.cpp
  tests/wave_projection_test.cpp
)

add_executable(transpiler_tests ${TEST_SOURCES})
llvm_update_compile_flags(transpiler_tests)

target_compile_definitions(transpiler_tests PRIVATE
  LLVM_TOOLS_DIR="${LLVM_TOOLS_BINARY_DIR}"
)

target_link_libraries(transpiler_tests
  PRIVATE
    -Wl,--exclude-libs,ALL
    hotswap-transpiler
    GTest::GTest
)

# AMDGPU target-private header paths — same as the library target
# (see the PRIVATE `target_include_directories` block further up).
# Tests that include transpiler headers pulling in
# `amdgpu_formats.hpp` / `decoded_inst.hpp` (e.g.
# `tests/wave_projection_test.cpp`) need these paths to find
# `SIDefines.h` / `AMDGPUGen*.inc` during compilation.  The library
# itself links these in PRIVATE because no public header exposes
# the AMDGPU-private types, but test TUs that reach into the full
# transpiler graph need them at compile time.
target_include_directories(transpiler_tests PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)

gtest_discover_tests(transpiler_tests
  PROPERTIES TIMEOUT 120 LABELS "transpiler"
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lit_tests/CMakeLists.txt)
  add_subdirectory(lit_tests)
endif()

endif() # HOTSWAP_TRANSPILER_BUILD_TOOLS
