//===- comgr-hotswap-transpile.cpp - ISA transpilation via LLVM IR --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// COMgr entry point for the hotswap transpiler. Where the byte-level
/// `amd_comgr_hotswap_rewrite` path patches a small set of stepping-specific
/// instruction encodings in place, this entry point hands the whole code
/// object to the hotswap pipeline - every kernel is disassembled, raised to
/// LLVM IR, re-lowered through the stock AMDGPU backend for the target ISA,
/// and re-linked into a single merged HSACO via
/// `transpiler::runPipelineAllKernels` (see amd/comgr/hotswap/pipeline.hpp and
/// amd/comgr/hotswap/raise_cli.cpp for the standalone driver this entry point
/// mirrors).
///
/// Failure is loud: any per-kernel raise failure surfaced by the hotswap
/// pipeline turns into `AMD_COMGR_STATUS_ERROR`. The hotswap library logs
/// the offending kernel and mnemonic on stderr (use hotswap's CLI with the
/// `--write-hsaco` mode for the same output).
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"

#include "hotswap/code_object_utils.hpp"
#include "hotswap/pipeline.hpp"
#include "hotswap/translation_cache.hpp"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

using namespace COMGR;

namespace {

using TimingClock = std::chrono::steady_clock;

double secondsBetween(TimingClock::time_point start, TimingClock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

TimingClock::time_point timingStart(bool collectTimings) {
  return collectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool collectTimings, TimingClock::time_point start) {
  return collectTimings ? secondsBetween(start, TimingClock::now()) : 0.0;
}

bool HotSwapTimingEnabled() {
  const char *value = std::getenv("HSA_HOTSWAP_TIMING");
  return value && value[0] && std::strcmp(value, "0") != 0;
}

struct HotswapComgrTimings {
  double totalSeconds = 0.0;
  double inputCopySeconds = 0.0;
  double listKernelsSeconds = 0.0;
  double cacheLookupTotalSeconds = 0.0;
  double cacheLookupKeyBuildSeconds = 0.0;
  double cacheLookupKeySourceHashSeconds = 0.0;
  double cacheLookupKeyElfHeaderSeconds = 0.0;
  double cacheLookupKeyRulesHashSeconds = 0.0;
  double cacheLookupKeyLoadedImageIdentitySeconds = 0.0;
  double cacheLookupKeyLlvmToolIdentitySeconds = 0.0;
  double cacheLookupKeyKernelNamesSeconds = 0.0;
  double cacheLookupKeyMaterialBuildSeconds = 0.0;
  double cacheLookupKeyHashSeconds = 0.0;
  double cacheLookupStatSeconds = 0.0;
  double cacheLookupObjectReadSeconds = 0.0;
  double cacheLookupObjectHashSeconds = 0.0;
  double cacheLookupMetadataReadSeconds = 0.0;
  double cacheLookupMetadataParseSeconds = 0.0;
  double cacheLookupMetadataValidateSeconds = 0.0;
  double pipelineTotalSeconds = 0.0;
  double pipelineListKernelsSeconds = 0.0;
  double pipelineExtractTextSeconds = 0.0;
  double pipelineCreateTempDirSeconds = 0.0;
  double pipelineRaiseSeconds = 0.0;
  double pipelineWriteIrSeconds = 0.0;
  double pipelineLlcSeconds = 0.0;
  double pipelineReadAsmSeconds = 0.0;
  double pipelineLlvmMcSeconds = 0.0;
  double pipelineLinkSeconds = 0.0;
  double pipelineReadHsacoSeconds = 0.0;
  double pipelineCollectMetadataSeconds = 0.0;
  double cacheWriteTotalSeconds = 0.0;
  double cacheWriteKeyBuildSeconds = 0.0;
  double cacheWriteKeySourceHashSeconds = 0.0;
  double cacheWriteKeyElfHeaderSeconds = 0.0;
  double cacheWriteKeyRulesHashSeconds = 0.0;
  double cacheWriteKeyLoadedImageIdentitySeconds = 0.0;
  double cacheWriteKeyLlvmToolIdentitySeconds = 0.0;
  double cacheWriteKeyKernelNamesSeconds = 0.0;
  double cacheWriteKeyMaterialBuildSeconds = 0.0;
  double cacheWriteKeyHashSeconds = 0.0;
  double cacheWriteCreateDirectorySeconds = 0.0;
  double cacheWriteObjectHashSeconds = 0.0;
  double cacheWriteObjectWriteSeconds = 0.0;
  double cacheWriteMetadataBuildSeconds = 0.0;
  double cacheWriteMetadataWriteSeconds = 0.0;
  double createOutputDataSeconds = 0.0;
};

std::string timingJson(const HotswapComgrTimings &timings) {
  llvm::json::Object object{
      {"total_seconds", timings.totalSeconds},
      {"input_copy_seconds", timings.inputCopySeconds},
      {"list_kernels_seconds", timings.listKernelsSeconds},
      {"cache_lookup_total_seconds", timings.cacheLookupTotalSeconds},
      {"cache_lookup_key_build_seconds", timings.cacheLookupKeyBuildSeconds},
      {"cache_lookup_key_source_hash_seconds",
       timings.cacheLookupKeySourceHashSeconds},
      {"cache_lookup_key_elf_header_seconds",
       timings.cacheLookupKeyElfHeaderSeconds},
      {"cache_lookup_key_rules_hash_seconds",
       timings.cacheLookupKeyRulesHashSeconds},
      {"cache_lookup_key_loaded_image_identity_seconds",
       timings.cacheLookupKeyLoadedImageIdentitySeconds},
      {"cache_lookup_key_llvm_tool_identity_seconds",
       timings.cacheLookupKeyLlvmToolIdentitySeconds},
      {"cache_lookup_key_kernel_names_seconds",
       timings.cacheLookupKeyKernelNamesSeconds},
      {"cache_lookup_key_material_build_seconds",
       timings.cacheLookupKeyMaterialBuildSeconds},
      {"cache_lookup_key_hash_seconds", timings.cacheLookupKeyHashSeconds},
      {"cache_lookup_stat_seconds", timings.cacheLookupStatSeconds},
      {"cache_lookup_object_read_seconds", timings.cacheLookupObjectReadSeconds},
      {"cache_lookup_object_hash_seconds", timings.cacheLookupObjectHashSeconds},
      {"cache_lookup_metadata_read_seconds",
       timings.cacheLookupMetadataReadSeconds},
      {"cache_lookup_metadata_parse_seconds",
       timings.cacheLookupMetadataParseSeconds},
      {"cache_lookup_metadata_validate_seconds",
       timings.cacheLookupMetadataValidateSeconds},
      {"pipeline_total_seconds", timings.pipelineTotalSeconds},
      {"pipeline_list_kernels_seconds", timings.pipelineListKernelsSeconds},
      {"pipeline_extract_text_seconds", timings.pipelineExtractTextSeconds},
      {"pipeline_create_temp_dir_seconds", timings.pipelineCreateTempDirSeconds},
      {"pipeline_raise_seconds", timings.pipelineRaiseSeconds},
      {"pipeline_write_ir_seconds", timings.pipelineWriteIrSeconds},
      {"pipeline_llc_seconds", timings.pipelineLlcSeconds},
      {"pipeline_read_asm_seconds", timings.pipelineReadAsmSeconds},
      {"pipeline_llvm_mc_seconds", timings.pipelineLlvmMcSeconds},
      {"pipeline_link_seconds", timings.pipelineLinkSeconds},
      {"pipeline_read_hsaco_seconds", timings.pipelineReadHsacoSeconds},
      {"pipeline_collect_metadata_seconds",
       timings.pipelineCollectMetadataSeconds},
      {"cache_write_total_seconds", timings.cacheWriteTotalSeconds},
      {"cache_write_key_build_seconds", timings.cacheWriteKeyBuildSeconds},
      {"cache_write_key_source_hash_seconds",
       timings.cacheWriteKeySourceHashSeconds},
      {"cache_write_key_elf_header_seconds",
       timings.cacheWriteKeyElfHeaderSeconds},
      {"cache_write_key_rules_hash_seconds",
       timings.cacheWriteKeyRulesHashSeconds},
      {"cache_write_key_loaded_image_identity_seconds",
       timings.cacheWriteKeyLoadedImageIdentitySeconds},
      {"cache_write_key_llvm_tool_identity_seconds",
       timings.cacheWriteKeyLlvmToolIdentitySeconds},
      {"cache_write_key_kernel_names_seconds",
       timings.cacheWriteKeyKernelNamesSeconds},
      {"cache_write_key_material_build_seconds",
       timings.cacheWriteKeyMaterialBuildSeconds},
      {"cache_write_key_hash_seconds", timings.cacheWriteKeyHashSeconds},
      {"cache_write_create_directory_seconds",
       timings.cacheWriteCreateDirectorySeconds},
      {"cache_write_object_hash_seconds", timings.cacheWriteObjectHashSeconds},
      {"cache_write_object_write_seconds", timings.cacheWriteObjectWriteSeconds},
      {"cache_write_metadata_build_seconds",
       timings.cacheWriteMetadataBuildSeconds},
      {"cache_write_metadata_write_seconds",
       timings.cacheWriteMetadataWriteSeconds},
      {"create_output_data_seconds", timings.createOutputDataSeconds},
  };
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::Value(std::move(object)).print(os);
  return out;
}

void addLookupTimings(HotswapComgrTimings &timings,
                      const transpiler::TranslationCacheLookupTimings &lookup) {
  timings.cacheLookupTotalSeconds += lookup.totalSeconds;
  timings.cacheLookupKeyBuildSeconds += lookup.keyBuildSeconds;
  timings.cacheLookupKeySourceHashSeconds += lookup.keyBuild.sourceHashSeconds;
  timings.cacheLookupKeyElfHeaderSeconds += lookup.keyBuild.elfHeaderSeconds;
  timings.cacheLookupKeyRulesHashSeconds += lookup.keyBuild.rulesHashSeconds;
  timings.cacheLookupKeyLoadedImageIdentitySeconds +=
      lookup.keyBuild.loadedImageIdentitySeconds;
  timings.cacheLookupKeyLlvmToolIdentitySeconds +=
      lookup.keyBuild.llvmToolIdentitySeconds;
  timings.cacheLookupKeyKernelNamesSeconds +=
      lookup.keyBuild.kernelNamesSeconds;
  timings.cacheLookupKeyMaterialBuildSeconds +=
      lookup.keyBuild.materialBuildSeconds;
  timings.cacheLookupKeyHashSeconds += lookup.keyBuild.keyHashSeconds;
  timings.cacheLookupStatSeconds += lookup.metadataObjectStatSeconds;
  timings.cacheLookupObjectReadSeconds += lookup.objectReadSeconds;
  timings.cacheLookupObjectHashSeconds += lookup.objectHashSeconds;
  timings.cacheLookupMetadataReadSeconds += lookup.metadataReadSeconds;
  timings.cacheLookupMetadataParseSeconds += lookup.metadataParseSeconds;
  timings.cacheLookupMetadataValidateSeconds += lookup.metadataValidateSeconds;
}

void addPipelineTimings(HotswapComgrTimings &timings,
                        const transpiler::PipelineTimings &pipeline) {
  timings.pipelineTotalSeconds += pipeline.totalSeconds;
  timings.pipelineListKernelsSeconds += pipeline.listKernelsSeconds;
  timings.pipelineExtractTextSeconds += pipeline.extractTextSeconds;
  timings.pipelineCreateTempDirSeconds += pipeline.createTempDirSeconds;
  timings.pipelineRaiseSeconds += pipeline.raiseSeconds;
  timings.pipelineWriteIrSeconds += pipeline.writeIrSeconds;
  timings.pipelineLlcSeconds += pipeline.llcSeconds;
  timings.pipelineReadAsmSeconds += pipeline.readAsmSeconds;
  timings.pipelineLlvmMcSeconds += pipeline.llvmMcSeconds;
  timings.pipelineLinkSeconds += pipeline.linkSeconds;
  timings.pipelineReadHsacoSeconds += pipeline.readHsacoSeconds;
  timings.pipelineCollectMetadataSeconds += pipeline.collectMetadataSeconds;
}

void addWriteTimings(HotswapComgrTimings &timings,
                     const transpiler::TranslationCacheWriteTimings &write) {
  timings.cacheWriteTotalSeconds += write.totalSeconds;
  timings.cacheWriteKeyBuildSeconds += write.keyBuildSeconds;
  timings.cacheWriteKeySourceHashSeconds += write.keyBuild.sourceHashSeconds;
  timings.cacheWriteKeyElfHeaderSeconds += write.keyBuild.elfHeaderSeconds;
  timings.cacheWriteKeyRulesHashSeconds += write.keyBuild.rulesHashSeconds;
  timings.cacheWriteKeyLoadedImageIdentitySeconds +=
      write.keyBuild.loadedImageIdentitySeconds;
  timings.cacheWriteKeyLlvmToolIdentitySeconds +=
      write.keyBuild.llvmToolIdentitySeconds;
  timings.cacheWriteKeyKernelNamesSeconds += write.keyBuild.kernelNamesSeconds;
  timings.cacheWriteKeyMaterialBuildSeconds +=
      write.keyBuild.materialBuildSeconds;
  timings.cacheWriteKeyHashSeconds += write.keyBuild.keyHashSeconds;
  timings.cacheWriteCreateDirectorySeconds += write.createDirectorySeconds;
  timings.cacheWriteObjectHashSeconds += write.objectHashSeconds;
  timings.cacheWriteObjectWriteSeconds += write.objectWriteSeconds;
  timings.cacheWriteMetadataBuildSeconds += write.metadataBuildSeconds;
  timings.cacheWriteMetadataWriteSeconds += write.metadataWriteSeconds;
}

struct HotswapTranspileResult {
  bool success = false;
  bool cacheHit = false;
  amd_comgr_hotswap_cache_lookup_status_t lookupStatus =
      AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  amd_comgr_hotswap_cache_write_status_t writeStatus =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  int64_t liftedCount = 0;
  int64_t totalCount = 0;
  std::string backend = "comgr";
  std::string sourceGfx;
  std::string targetGfx;
  std::string cacheKey;
  std::string cacheDetail;
  std::string cacheMetadataPath;
  std::string cacheObjectPath;
  std::string failReason;
  std::string failDetail;
  std::string timingJson;

  static HotswapTranspileResult *convert(
      amd_comgr_hotswap_transpile_result_t result) {
    return reinterpret_cast<HotswapTranspileResult *>(
        static_cast<uintptr_t>(result.handle));
  }

  static amd_comgr_hotswap_transpile_result_t convert(
      HotswapTranspileResult *result) {
    amd_comgr_hotswap_transpile_result_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result))};
    return handle;
  }
};

amd_comgr_status_t createDataObject(amd_comgr_data_kind_t kind,
                                    llvm::StringRef data,
                                    amd_comgr_data_t *output) {
  DataObject *Object = DataObject::allocate(kind);
  if (!Object)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (amd_comgr_status_t Status = Object->setData(data)) {
    Object->release();
    return Status;
  }

  *output = DataObject::convert(Object);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t createExecutableData(llvm::ArrayRef<uint8_t> hsaco,
                                        amd_comgr_data_t *output) {
  return createDataObject(
      AMD_COMGR_DATA_KIND_EXECUTABLE,
      llvm::StringRef(reinterpret_cast<const char *>(hsaco.data()),
                      hsaco.size()),
      output);
}

bool hasFlag(const amd_comgr_hotswap_transpile_options_t *options,
             amd_comgr_hotswap_transpile_option_flags_t flag) {
  return options && (options->flags & static_cast<uint64_t>(flag));
}

std::string pipelineFailReason(const transpiler::PipelineResult &pipeline) {
  if (!pipeline.failReason.empty())
    return pipeline.failReason;
  if (pipeline.hsaco.empty())
    return "empty_output";
  return "hotswap_pipeline_failed";
}

std::string pipelineFailDetail(const transpiler::PipelineResult &pipeline) {
  if (!pipeline.failDetail.empty())
    return pipeline.failDetail;
  if (!pipeline.failMnemonic.empty())
    return pipeline.failMnemonic;
  if (!pipeline.failKernel.empty())
    return pipeline.failKernel;
  return "hotswap pipeline did not produce a loadable HSACO";
}

amd_comgr_hotswap_cache_lookup_status_t
lookupStatusFromCacheStatus(transpiler::TranslationCacheStatus status) {
  switch (status) {
  case transpiler::TranslationCacheStatus::Disabled:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  case transpiler::TranslationCacheStatus::Bypassed:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_BYPASSED;
  case transpiler::TranslationCacheStatus::Miss:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_MISS;
  case transpiler::TranslationCacheStatus::Hit:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_HIT;
  case transpiler::TranslationCacheStatus::Invalid:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID;
  case transpiler::TranslationCacheStatus::WriteSuccess:
  case transpiler::TranslationCacheStatus::WriteFailed:
    break;
  }
  return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID;
}

amd_comgr_hotswap_cache_write_status_t
writeStatusFromCacheStatus(transpiler::TranslationCacheStatus status) {
  switch (status) {
  case transpiler::TranslationCacheStatus::WriteSuccess:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_SUCCESS;
  case transpiler::TranslationCacheStatus::WriteFailed:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_FAILED;
  case transpiler::TranslationCacheStatus::Disabled:
  case transpiler::TranslationCacheStatus::Bypassed:
  case transpiler::TranslationCacheStatus::Miss:
  case transpiler::TranslationCacheStatus::Hit:
  case transpiler::TranslationCacheStatus::Invalid:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  }
  return AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
}

void fillResult(HotswapTranspileResult &result, llvm::StringRef sourceGfx,
                llvm::StringRef targetGfx, bool success, bool cacheHit,
                amd_comgr_hotswap_cache_lookup_status_t lookupStatus,
                amd_comgr_hotswap_cache_write_status_t writeStatus,
                llvm::StringRef cacheDetail,
                const transpiler::PipelineResult *pipeline,
                llvm::StringRef cacheKey = "",
                llvm::StringRef cacheMetadataPath = "",
                llvm::StringRef cacheObjectPath = "",
                llvm::StringRef failReason = "",
                llvm::StringRef failDetail = "",
                llvm::StringRef timingJson = "") {
  result.sourceGfx = sourceGfx.str();
  result.targetGfx = targetGfx.str();
  result.success = success;
  result.cacheHit = cacheHit;
  result.lookupStatus = lookupStatus;
  result.writeStatus = writeStatus;
  result.cacheDetail = cacheDetail.str();
  result.cacheKey = cacheKey.str();
  result.cacheMetadataPath = cacheMetadataPath.str();
  result.cacheObjectPath = cacheObjectPath.str();
  result.failReason = failReason.str();
  result.failDetail = failDetail.str();
  result.timingJson = timingJson.str();
  if (pipeline) {
    result.liftedCount = pipeline->liftedCount;
    result.totalCount = pipeline->totalCount;
  }
}

amd_comgr_status_t returnResult(HotswapTranspileResult &&value,
                                amd_comgr_hotswap_transpile_result_t *result) {
  if (!result)
    return AMD_COMGR_STATUS_SUCCESS;
  HotswapTranspileResult *owned =
      new (std::nothrow) HotswapTranspileResult(std::move(value));
  if (!owned)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  *result = HotswapTranspileResult::convert(owned);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_with_options(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_transpile_options_t *options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result) {
  const bool CollectTimings = HotSwapTimingEnabled();
  auto totalStart = timingStart(CollectTimings);
  HotswapComgrTimings Timings;
  auto finalTimingJson = [&]() {
    if (!CollectTimings)
      return std::string();
    Timings.totalSeconds = timingElapsed(CollectTimings, totalStart);
    return timingJson(Timings);
  };
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE || !source_isa_name ||
      !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  if (options && options->size < sizeof(amd_comgr_hotswap_transpile_options_t))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate both ISA names through the same parser the byte-level
  // `amd_comgr_hotswap_rewrite` uses, so the public contract is identical:
  // malformed identifiers are rejected up-front and never reach the hotswap
  // pipeline. We do not gate on the processor name here — hotswap decides
  // per-kernel whether the source/target pair is supported, and surfaces
  // unsupported instructions as a pipeline failure (see
  // RaiseFailure::reason in amd/comgr/hotswap/raise_failure.hpp).
  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Hotswap takes the code object by `std::vector<uint8_t>` (see
  // pipeline.hpp). DataObject stores its bytes in a `char *` buffer; copy
  // into the hotswap-shaped container rather than reinterpret-casting the
  // pointer, since the pipeline reads through this buffer many times across
  // kernels and the temporary lifetime needs to be unambiguous.
  auto inputCopyStart = timingStart(CollectTimings);
  const auto *InputBegin = reinterpret_cast<const uint8_t *>(InputP->Data);
  std::vector<uint8_t> InputBytes(InputBegin, InputBegin + InputP->Size);
  Timings.inputCopySeconds =
      timingElapsed(CollectTimings, inputCopyStart);

  transpiler::TranslationCacheRequest CacheRequest;
  CacheRequest.sourceObject = llvm::ArrayRef<uint8_t>(InputBytes);
  CacheRequest.sourceGfx = SourceIdent.Processor.str();
  CacheRequest.targetGfx = TargetIdent.Processor.str();
  CacheRequest.sourceIsa = source_isa_name;
  CacheRequest.targetIsa = target_isa_name;
  CacheRequest.codeIsa = source_isa_name;
  CacheRequest.hotswapRulesPath =
      options && options->hotswap_rules_path ? options->hotswap_rules_path : "";
  CacheRequest.cacheDirectory =
      options && options->cache_directory ? options->cache_directory : "";
  CacheRequest.cacheSkipKernels =
      options && options->cache_skip_kernels ? options->cache_skip_kernels : "";
  CacheRequest.strictMode =
      hasFlag(options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_STRICT);
  CacheRequest.cacheDisabled =
      !options || hasFlag(options,
                          AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_DISABLE) ||
      CacheRequest.cacheDirectory.empty();
  CacheRequest.cacheReadonly =
      hasFlag(options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_READONLY);
  CacheRequest.collectTimings = CollectTimings;

  auto listKernelsStart = timingStart(CollectTimings);
  const std::vector<std::string> KernelNames =
      transpiler::listKernelNames(InputBytes);
  Timings.listKernelsSeconds =
      timingElapsed(CollectTimings, listKernelsStart);
  const std::string SkippedKernel =
      transpiler::skippedKernelForTranslationCache(
          KernelNames, CacheRequest.cacheSkipKernels);

  transpiler::TranslationCacheStatus CacheStatus =
      transpiler::TranslationCacheStatus::Disabled;
  std::string CacheDetail;
  std::string CacheKey;
  std::string CacheMetadataPath;
  std::string CacheObjectPath;
  bool CacheHit = false;

  transpiler::PipelineResult Pipeline;
  if (!SkippedKernel.empty()) {
    CacheStatus = transpiler::TranslationCacheStatus::Bypassed;
    CacheDetail = "kernel listed in HSA_HOTSWAP_CACHE_SKIP_KERNELS: " +
                  SkippedKernel;
  } else {
    transpiler::TranslationCacheLookup Lookup =
        transpiler::lookupTranslationCache(CacheRequest);
    addLookupTimings(Timings, Lookup.timings);
    CacheStatus = Lookup.status;
    CacheDetail = Lookup.reason;
    CacheKey = Lookup.key;
    CacheMetadataPath = Lookup.metadataPath;
    CacheObjectPath = Lookup.objectPath;

    if (Lookup.status == transpiler::TranslationCacheStatus::Invalid) {
      HotswapTranspileResult Result;
      fillResult(Result, CacheRequest.sourceGfx, CacheRequest.targetGfx, false,
                 false, lookupStatusFromCacheStatus(Lookup.status),
                 AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED, Lookup.reason,
                 nullptr, Lookup.key, Lookup.metadataPath, Lookup.objectPath,
                 "cache_invalid", Lookup.reason, finalTimingJson());
      if (amd_comgr_status_t ResultStatus =
              returnResult(std::move(Result), result))
        return ResultStatus;
      return AMD_COMGR_STATUS_ERROR;
    }

    if (Lookup.status == transpiler::TranslationCacheStatus::Hit) {
      Pipeline = std::move(Lookup.result);
      CacheHit = true;
    }
  }

  // Drive the same all-kernels merge path that raise_cli.cpp's --write-hsaco
  // mode falls back on for whole-file flows. We pass hotswap's defaults for
  // the writelane / wave-native toggles (both on, post-graduation) — the
  // public comgr surface intentionally hides those knobs since they are
  // either correctness-preserving rewrites (writelane) or projection
  // strategies (wave-native) that callers should not have to reason about.
  // If an opt-out is ever needed at the comgr boundary it should land as a
  // separate options struct rather than overloading this entry point.
  if (!CacheHit) {
    transpiler::ScopedStrictMode StrictMode(CacheRequest.strictMode);
    transpiler::PipelineOptions PipelineOptions;
    PipelineOptions.enableWritelaneRewrite =
        CacheRequest.enableWritelaneRewrite;
    PipelineOptions.enableWaveNative = CacheRequest.enableWaveNative;
    PipelineOptions.collectTimings = CollectTimings;
    Pipeline = transpiler::runPipelineAllKernels(InputBytes,
                                                 SourceIdent.Processor.str(),
                                                 TargetIdent.Processor.str(),
                                                 PipelineOptions);
    addPipelineTimings(Timings, Pipeline.timings);
  }

  if (!Pipeline.success || Pipeline.hsaco.empty()) {
    HotswapTranspileResult Result;
    fillResult(Result, CacheRequest.sourceGfx, CacheRequest.targetGfx, false,
               CacheHit, lookupStatusFromCacheStatus(CacheStatus),
               AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED, CacheDetail,
               &Pipeline, CacheKey, CacheMetadataPath, CacheObjectPath,
               pipelineFailReason(Pipeline), pipelineFailDetail(Pipeline),
               finalTimingJson());
    if (amd_comgr_status_t ResultStatus =
            returnResult(std::move(Result), result))
      return ResultStatus;
    return AMD_COMGR_STATUS_ERROR;
  }

  amd_comgr_hotswap_cache_write_status_t CacheWriteStatus =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  if (!CacheHit && CacheStatus == transpiler::TranslationCacheStatus::Miss) {
    transpiler::TranslationCacheWrite Write =
        transpiler::writeTranslationCache(CacheRequest, Pipeline);
    addWriteTimings(Timings, Write.timings);
    CacheWriteStatus = writeStatusFromCacheStatus(Write.status);
    if (!Write.key.empty())
      CacheKey = Write.key;
    if (!Write.metadataPath.empty())
      CacheMetadataPath = Write.metadataPath;
    if (!Write.objectPath.empty())
      CacheObjectPath = Write.objectPath;
    if (!Write.reason.empty())
      CacheDetail = Write.reason;
    if (Write.status == transpiler::TranslationCacheStatus::WriteFailed) {
      HotswapTranspileResult Result;
      fillResult(Result, CacheRequest.sourceGfx, CacheRequest.targetGfx, false,
                 false, lookupStatusFromCacheStatus(CacheStatus),
                 CacheWriteStatus, Write.reason, &Pipeline, Write.key,
                 Write.metadataPath, Write.objectPath, "cache_write_failed",
                 Write.reason, finalTimingJson());
      if (amd_comgr_status_t ResultStatus =
              returnResult(std::move(Result), result))
        return ResultStatus;
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  amd_comgr_data_t OutputData = {0};
  auto createOutputDataStart = timingStart(CollectTimings);
  if (auto Status = createExecutableData(Pipeline.hsaco, &OutputData))
    return Status;
  Timings.createOutputDataSeconds =
      timingElapsed(CollectTimings, createOutputDataStart);

  HotswapTranspileResult Result;
  fillResult(Result, CacheRequest.sourceGfx, CacheRequest.targetGfx, true,
             CacheHit, lookupStatusFromCacheStatus(CacheStatus),
             CacheWriteStatus, CacheDetail, &Pipeline, CacheKey,
             CacheMetadataPath, CacheObjectPath, "", "", finalTimingJson());
  if (amd_comgr_status_t ResultStatus =
          returnResult(std::move(Result), result)) {
    amd_comgr_release_data(OutputData);
    return ResultStatus;
  }

  *output = OutputData;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, amd_comgr_data_t *output) {
  return amd_comgr_hotswap_transpile_with_options(
      input, source_isa_name, target_isa_name, nullptr, output, nullptr);
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_destroy_hotswap_transpile_result(
    amd_comgr_hotswap_transpile_result_t result) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  delete Result;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_result_get_info(
    amd_comgr_hotswap_transpile_result_t result,
    amd_comgr_hotswap_transpile_result_info_t info, void *value) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result || !value)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  switch (info) {
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SUCCESS:
    *static_cast<bool *>(value) = Result->success;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT:
    *static_cast<bool *>(value) = Result->cacheHit;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_LOOKUP:
    *static_cast<amd_comgr_hotswap_cache_lookup_status_t *>(value) =
        Result->lookupStatus;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_WRITE:
    *static_cast<amd_comgr_hotswap_cache_write_status_t *>(value) =
        Result->writeStatus;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_LIFTED_COUNT:
    *static_cast<int64_t *>(value) = Result->liftedCount;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TOTAL_COUNT:
    *static_cast<int64_t *>(value) = Result->totalCount;
    return AMD_COMGR_STATUS_SUCCESS;
  }
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_result_get_string(
    amd_comgr_hotswap_transpile_result_t result,
    amd_comgr_hotswap_transpile_result_string_t field, size_t *size,
    char *value) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result || !size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  const std::string *Field = nullptr;
  switch (field) {
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_BACKEND:
    Field = &Result->backend;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SOURCE_GFX:
    Field = &Result->sourceGfx;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TARGET_GFX:
    Field = &Result->targetGfx;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_KEY:
    Field = &Result->cacheKey;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_DETAIL:
    Field = &Result->cacheDetail;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_METADATA_PATH:
    Field = &Result->cacheMetadataPath;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_OBJECT_PATH:
    Field = &Result->cacheObjectPath;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_REASON:
    Field = &Result->failReason;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL:
    Field = &Result->failDetail;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TIMING_JSON:
    Field = &Result->timingJson;
    break;
  }
  if (!Field)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  const size_t Required = Field->size() + 1;
  if (!value) {
    *size = Required;
    return AMD_COMGR_STATUS_SUCCESS;
  }
  if (*size < Required)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  std::memcpy(value, Field->c_str(), Required);
  *size = Required;
  return AMD_COMGR_STATUS_SUCCESS;
}
