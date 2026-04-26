# Salmon transpiler lit + FileCheck harness.
#
# The suite is hermetic: fixtures are checked-in AMDGPU `.s` sources
# that each RUN line assembles on the fly. Each test's RUN line invokes
# `%llvm_mc` and `%ld_lld` directly to turn the `.s` into an amdhsa
# code object, then hands the hsaco to `%raise_cli` for lifting; the
# raiser itself has no assembly path. There is no HIP toolchain
# dependency at configure time or at test time; the only external
# requirements are LLVM's `FileCheck`, `not`, `llvm-mc`, `ld.lld`, and
# `llvm-lit`, all of which already ship alongside the LLVM build tree
# the parent project links against.

set(LIT_FILECHECK_EXE "${LLVM_BINARY_DIR}/bin/FileCheck")
if(NOT EXISTS "${LIT_FILECHECK_EXE}")
  find_program(LIT_FILECHECK_EXE FileCheck HINTS "${LLVM_BINARY_DIR}/bin")
endif()
set(LIT_NOT_EXE "${LLVM_BINARY_DIR}/bin/not")
if(NOT EXISTS "${LIT_NOT_EXE}")
  find_program(LIT_NOT_EXE NAMES not HINTS "${LLVM_BINARY_DIR}/bin")
endif()
set(LIT_LLVM_MC_EXE "${LLVM_BINARY_DIR}/bin/llvm-mc")
if(NOT EXISTS "${LIT_LLVM_MC_EXE}")
  find_program(LIT_LLVM_MC_EXE llvm-mc HINTS "${LLVM_BINARY_DIR}/bin")
endif()
set(LIT_LD_LLD_EXE "${LLVM_BINARY_DIR}/bin/ld.lld")
if(NOT EXISTS "${LIT_LD_LLD_EXE}")
  find_program(LIT_LD_LLD_EXE ld.lld HINTS "${LLVM_BINARY_DIR}/bin")
endif()
if(NOT LIT_FILECHECK_EXE OR NOT LIT_NOT_EXE
    OR NOT LIT_LLVM_MC_EXE OR NOT LIT_LD_LLD_EXE)
  message(STATUS "transpiler lit tests skipped: FileCheck / not / llvm-mc / ld.lld not found (LLVM tools)")
  return()
endif()

set(LIT_LLVM_LIT_EXE "${LLVM_BINARY_DIR}/bin/llvm-lit")
if(NOT EXISTS "${LIT_LLVM_LIT_EXE}")
  find_program(LIT_LLVM_LIT_EXE llvm-lit HINTS "${LLVM_BINARY_DIR}/bin")
endif()
if(NOT LIT_LLVM_LIT_EXE)
  message(STATUS "transpiler lit tests skipped: llvm-lit not found")
  return()
endif()

set(LIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRANSPILER_BUILD_DIR "${CMAKE_BINARY_DIR}")
# Single-config Ninja: raise_cli lands next to this project in the build tree.
set(LIT_TOOL_RAISE_CLI "${CMAKE_BINARY_DIR}/raise_cli")
set(LIT_TOOL_FILECHECK "${LIT_FILECHECK_EXE}")
set(LIT_TOOL_NOT "${LIT_NOT_EXE}")
set(LIT_TOOL_LLVM_MC "${LIT_LLVM_MC_EXE}")
set(LIT_TOOL_LD_LLD "${LIT_LD_LLD_EXE}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)

# Canonical LLVM convention for `check-*` targets: `-sv` = succinct
# (only per-test results on failure) + verbose-on-failure. Matches
# check-llvm / check-clang / check-mlir so CI logs stay readable.
add_custom_target(
  check-transpiler-lit
  COMMAND "${LIT_LLVM_LIT_EXE}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS raise_cli
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Running transpiler lit tests")

# CTest registration. The `transpiler-lit` test only invokes llvm-lit;
# `raise_cli` is the sole build dependency (the checked-in `.s`
# fixtures are source-tree artefacts). We pair the test with a
# FIXTURES_SETUP test that runs `cmake --build` for raise_cli, letting
# `ctest -R transpiler-lit` just work on a clean tree. The setup test
# is idempotent — if the binary already exists, the build is a no-op.
add_test(
  NAME transpiler-lit-build-raise-cli
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target raise_cli)
set_tests_properties(
  transpiler-lit-build-raise-cli
  PROPERTIES FIXTURES_SETUP transpiler-lit-raise-cli)

add_test(
  NAME transpiler-lit
  COMMAND "${LIT_LLVM_LIT_EXE}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(
  transpiler-lit
  PROPERTIES FIXTURES_REQUIRED transpiler-lit-raise-cli)
