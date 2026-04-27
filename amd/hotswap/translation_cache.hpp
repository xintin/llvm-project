#ifndef HOTSWAP_TRANSPILER_TRANSLATION_CACHE_HPP
#define HOTSWAP_TRANSPILER_TRANSLATION_CACHE_HPP

#include "pipeline.hpp"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct TranslationCacheRequest {
  llvm::ArrayRef<uint8_t> sourceObject;
  std::string sourceGfx;
  std::string targetGfx;
  std::string sourceIsa;
  std::string targetIsa;
  std::string codeIsa;
  std::string hotswapRulesPath;
  int origMach = -1;
  bool enableWritelaneRewrite = true;
  bool enableWaveNative = true;
  bool strictMode = false;
};

enum class TranslationCacheStatus {
  Disabled,
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
  PipelineResult result;
};

struct TranslationCacheWrite {
  TranslationCacheStatus status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string metadataPath;
  std::string objectPath;
  std::string reason;
};

const char *translationCacheStatusString(TranslationCacheStatus status);

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &request);

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &request, const PipelineResult &result);

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> kernelNames);

std::string sha256Hex(llvm::ArrayRef<uint8_t> data);

} // namespace transpiler

#endif
