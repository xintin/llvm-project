#ifndef HOTSWAP_TRANSPILER_TRANSLATION_CACHE_HPP
#define HOTSWAP_TRANSPILER_TRANSLATION_CACHE_HPP

#include "pipeline.hpp"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct TranslationCacheKeyBuildTimings {
  double sourceHashSeconds = 0.0;
  double elfHeaderSeconds = 0.0;
  double rulesHashSeconds = 0.0;
  double loadedImageIdentitySeconds = 0.0;
  double llvmToolIdentitySeconds = 0.0;
  double kernelNamesSeconds = 0.0;
  double materialBuildSeconds = 0.0;
  double keyHashSeconds = 0.0;
};

struct TranslationCacheLookupTimings {
  double totalSeconds = 0.0;
  double keyBuildSeconds = 0.0;
  TranslationCacheKeyBuildTimings keyBuild;
  double metadataObjectStatSeconds = 0.0;
  double objectReadSeconds = 0.0;
  double objectHashSeconds = 0.0;
  double metadataReadSeconds = 0.0;
  double metadataParseSeconds = 0.0;
  double metadataValidateSeconds = 0.0;
};

struct TranslationCacheWriteTimings {
  double totalSeconds = 0.0;
  double keyBuildSeconds = 0.0;
  TranslationCacheKeyBuildTimings keyBuild;
  double createDirectorySeconds = 0.0;
  double objectHashSeconds = 0.0;
  double objectWriteSeconds = 0.0;
  double metadataBuildSeconds = 0.0;
  double metadataWriteSeconds = 0.0;
};

struct TranslationCacheRequest {
  llvm::ArrayRef<uint8_t> sourceObject;
  std::string sourceGfx;
  std::string targetGfx;
  std::string sourceIsa;
  std::string targetIsa;
  std::string codeIsa;
  std::string hotswapRulesPath;
  std::string cacheDirectory;
  std::string cacheSkipKernels;
  int origMach = -1;
  bool enableWritelaneRewrite = true;
  bool enableWaveNative = true;
  bool strictMode = false;
  bool cacheDisabled = true;
  bool cacheReadonly = false;
  bool collectTimings = false;
};

enum class TranslationCacheStatus {
  Disabled,
  Bypassed,
  Miss,
  Hit,
  Invalid,
  WriteSuccess,
  WriteFailed,
};

struct TranslationCacheLookup {
  TranslationCacheStatus status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string metadataPath;
  std::string objectPath;
  std::string reason;
  TranslationCacheLookupTimings timings;
  PipelineResult result;
};

struct TranslationCacheWrite {
  TranslationCacheStatus status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string metadataPath;
  std::string objectPath;
  std::string reason;
  TranslationCacheWriteTimings timings;
};

const char *translationCacheStatusString(TranslationCacheStatus status);

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &request);

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &request, const PipelineResult &result);

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> kernelNames, llvm::StringRef skipList);

std::string sha256Hex(llvm::ArrayRef<uint8_t> data);

} // namespace transpiler

#endif
