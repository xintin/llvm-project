#include "translation_cache.hpp"

#include "code_object_utils.hpp"

#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <optional>
#include <string>
#include <sys/stat.h>

#define DEBUG_TYPE "translation-cache"

#ifndef LLVM_TOOLS_DIR
#define LLVM_TOOLS_DIR "/usr/bin"
#endif

namespace transpiler {
namespace {

constexpr int kCacheSchemaVersion = 1;

struct FileIdentity {
  std::string path;
  bool present = false;
  uint64_t size = 0;
  int64_t mtimeSec = 0;
  int64_t mtimeNsec = 0;
  std::string sha256;
  std::string error;
};

struct KeyData {
  std::string key;
  std::string sourceSha256;
  std::string rulesSha256;
  std::string rulesError;
  std::string buildIdentity;
  std::string llcIdentity;
  std::string llvmMcIdentity;
  std::string lldIdentity;
  std::string elfMachineHex;
  std::string elfFlagsHex;
  std::vector<std::string> kernelNames;
  std::string error;
};

std::string envString(const char *name) {
  const char *value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool envEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] && std::strcmp(value, "0") != 0;
}

std::string hexU32(uint32_t value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "0x" << llvm::format_hex_no_prefix(value, 0);
  return os.str();
}

bool readElfHeaderFields(llvm::ArrayRef<uint8_t> data, uint16_t &machine,
                         uint32_t &flags) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(data.data()),
                      data.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*buf);
  if (!objOrErr) {
    (void)llvm::toString(objOrErr.takeError());
    return false;
  }
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(objOrErr->get());
  if (!elf)
    return false;
  machine = elf->getEMachine();
  flags = elf->getPlatformFlags();
  return true;
}

std::string hashFile(llvm::StringRef path, std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = buffer.getError().message();
    return "";
  }
  llvm::StringRef contents = (*buffer)->getBuffer();
  return sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(contents.data()), contents.size()));
}

FileIdentity statIdentity(llvm::StringRef path) {
  FileIdentity id;
  id.path = path.str();
  struct stat st;
  if (::stat(id.path.c_str(), &st) != 0)
    return id;
  id.present = true;
  id.size = static_cast<uint64_t>(st.st_size);
#if defined(__linux__)
  id.mtimeSec = static_cast<int64_t>(st.st_mtim.tv_sec);
  id.mtimeNsec = static_cast<int64_t>(st.st_mtim.tv_nsec);
#else
  id.mtimeSec = static_cast<int64_t>(st.st_mtime);
  id.mtimeNsec = 0;
#endif
  id.sha256 = hashFile(id.path, id.error);
  return id;
}

std::string identityString(const FileIdentity &id) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << id.path << "|present=" << (id.present ? "1" : "0")
     << "|size=" << id.size << "|mtime=" << id.mtimeSec << "."
     << id.mtimeNsec << "|sha256=" << id.sha256;
  if (!id.error.empty())
    os << "|error=" << id.error;
  return os.str();
}

std::string loadedImageIdentity() {
  static const std::string identity = [] {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "compiled=" << __DATE__ << " " << __TIME__
       << "|llvm=" << LLVM_VERSION_STRING;
    Dl_info info;
    if (::dladdr(reinterpret_cast<void *>(&loadedImageIdentity), &info) &&
        info.dli_fname) {
      os << "|image=" << identityString(statIdentity(info.dli_fname));
    } else {
      os << "|image=<dladdr-unavailable>";
    }
    return os.str();
  }();
  return identity;
}

void appendKeyField(std::string &material, llvm::StringRef name,
                    llvm::StringRef value) {
  material.append(name.data(), name.size());
  material.push_back('\0');
  material += std::to_string(value.size());
  material.push_back(':');
  if (!value.empty())
    material.append(value.data(), value.size());
  material.push_back('\0');
}

void appendKeyField(std::string &material, llvm::StringRef name, bool value) {
  appendKeyField(material, name, llvm::StringRef(value ? "true" : "false"));
}

void appendKeyField(std::string &material, llvm::StringRef name, int value) {
  appendKeyField(material, name, std::to_string(value));
}

const FileIdentity &llcIdentity() {
  static const FileIdentity identity =
      statIdentity(std::string(LLVM_TOOLS_DIR) + "/llc");
  return identity;
}

const FileIdentity &llvmMcIdentity() {
  static const FileIdentity identity =
      statIdentity(std::string(LLVM_TOOLS_DIR) + "/llvm-mc");
  return identity;
}

const FileIdentity &lldIdentity() {
  static const FileIdentity identity =
      statIdentity(std::string(LLVM_TOOLS_DIR) + "/ld.lld");
  return identity;
}

KeyData buildKeyData(const TranslationCacheRequest &request) {
  KeyData data;
  if (request.sourceObject.empty()) {
    data.error = "empty source code object";
    return data;
  }
  if (request.sourceGfx.empty() || request.targetGfx.empty()) {
    data.error = "missing source or target gfx";
    return data;
  }

  data.sourceSha256 = sha256Hex(request.sourceObject);
  uint16_t machine = 0;
  uint32_t flags = 0;
  if (readElfHeaderFields(request.sourceObject, machine, flags)) {
    data.elfMachineHex = hexU32(machine);
    data.elfFlagsHex = hexU32(flags);
  } else {
    data.error = "source code object is not a 64-bit little-endian ELF";
    return data;
  }

  if (!request.hotswapRulesPath.empty()) {
    data.rulesSha256 = hashFile(request.hotswapRulesPath, data.rulesError);
    if (!data.rulesError.empty()) {
      data.error = "failed to hash HSA_HOTSWAP_RULES '" +
                   request.hotswapRulesPath + "': " + data.rulesError;
      return data;
    }
  }

  const std::string toolsDir = LLVM_TOOLS_DIR;
  const FileIdentity &llc = llcIdentity();
  const FileIdentity &llvmMc = llvmMcIdentity();
  const FileIdentity &lld = lldIdentity();
  if (!llc.present || !llvmMc.present || !lld.present || !llc.error.empty() ||
      !llvmMc.error.empty() || !lld.error.empty()) {
    data.error = "LLVM tool identity is incomplete under " + toolsDir;
    return data;
  }
  data.llcIdentity = identityString(llc);
  data.llvmMcIdentity = identityString(llvmMc);
  data.lldIdentity = identityString(lld);
  data.buildIdentity = loadedImageIdentity();
  data.kernelNames = listKernelNames(
      std::vector<uint8_t>(request.sourceObject.begin(),
                           request.sourceObject.end()));

  std::string material;
  appendKeyField(material, "schema", std::to_string(kCacheSchemaVersion));
  appendKeyField(material, "source_sha256", data.sourceSha256);
  appendKeyField(material, "source_gfx", request.sourceGfx);
  appendKeyField(material, "target_gfx", request.targetGfx);
  appendKeyField(material, "source_isa", request.sourceIsa);
  appendKeyField(material, "target_isa", request.targetIsa);
  appendKeyField(material, "code_isa", request.codeIsa);
  appendKeyField(material, "elf_machine", data.elfMachineHex);
  appendKeyField(material, "elf_flags", data.elfFlagsHex);
  appendKeyField(material, "orig_mach", request.origMach);
  appendKeyField(material, "rules_path", request.hotswapRulesPath);
  appendKeyField(material, "rules_sha256", data.rulesSha256);
  appendKeyField(material, "strict", request.strictMode);
  appendKeyField(material, "enable_writelane_rewrite",
                 request.enableWritelaneRewrite);
  appendKeyField(material, "enable_wave_native", request.enableWaveNative);
  appendKeyField(material, "salmon_build_identity", data.buildIdentity);
  appendKeyField(material, "llc_identity", data.llcIdentity);
  appendKeyField(material, "llvm_mc_identity", data.llvmMcIdentity);
  appendKeyField(material, "lld_identity", data.lldIdentity);
  data.key = sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(material.data()), material.size()));
  return data;
}

std::string cacheRoot() { return envString("HSA_SALMON_CACHE_DIR"); }

bool cacheDisabledByEnv() {
  return envEnabled("HSA_SALMON_CACHE_DISABLE") || cacheRoot().empty();
}

std::string cacheSubdir(llvm::StringRef key) {
  llvm::SmallString<256> path(cacheRoot());
  llvm::sys::path::append(path, key.substr(0, 2));
  return std::string(path);
}

std::string cacheObjectPath(llvm::StringRef key) {
  llvm::SmallString<256> path(cacheSubdir(key));
  llvm::sys::path::append(path, llvm::Twine(key) + ".hsaco");
  return std::string(path);
}

std::string cacheMetadataPath(llvm::StringRef key) {
  llvm::SmallString<256> path(cacheSubdir(key));
  llvm::sys::path::append(path, llvm::Twine(key) + ".json");
  return std::string(path);
}

bool exists(llvm::StringRef path) {
  return llvm::sys::fs::exists(path);
}

std::string jsonToString(llvm::json::Value value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  value.print(os);
  os << "\n";
  return os.str();
}

bool writeFileAtomic(llvm::StringRef path, llvm::StringRef contents,
                     std::string &error) {
  llvm::SmallString<256> model(path);
  model += ".tmp-%%%%%%";
  llvm::SmallString<256> tmpPath;
  int fd = -1;
  if (auto ec = llvm::sys::fs::createUniqueFile(model, fd, tmpPath)) {
    error = ec.message();
    return false;
  }
  {
    llvm::raw_fd_ostream os(fd, true);
    os << contents;
    if (os.has_error()) {
      error = os.error().message();
      os.clear_error();
      llvm::sys::fs::remove(tmpPath);
      return false;
    }
  }
  if (auto ec = llvm::sys::fs::rename(tmpPath, path)) {
    error = ec.message();
    llvm::sys::fs::remove(tmpPath);
    return false;
  }
  return true;
}

bool writeFileAtomic(llvm::StringRef path, llvm::ArrayRef<uint8_t> data,
                     std::string &error) {
  return writeFileAtomic(
      path, llvm::StringRef(reinterpret_cast<const char *>(data.data()),
                            data.size()),
      error);
}

std::optional<std::string> requireString(const llvm::json::Object &obj,
                                         llvm::StringRef field,
                                         std::string &reason) {
  auto value = obj.getString(field);
  if (!value) {
    reason = "metadata field '" + field.str() + "' missing or not a string";
    return std::nullopt;
  }
  return value->str();
}

std::optional<int64_t> requireInt(const llvm::json::Object &obj,
                                  llvm::StringRef field, std::string &reason) {
  auto value = obj.getInteger(field);
  if (!value) {
    reason = "metadata field '" + field.str() + "' missing or not an integer";
    return std::nullopt;
  }
  return *value;
}

std::optional<bool> requireBool(const llvm::json::Object &obj,
                                llvm::StringRef field, std::string &reason) {
  auto value = obj.getBoolean(field);
  if (!value) {
    reason = "metadata field '" + field.str() + "' missing or not a boolean";
    return std::nullopt;
  }
  return *value;
}

bool requireEqualString(const llvm::json::Object &obj, llvm::StringRef field,
                        llvm::StringRef expected, std::string &reason) {
  auto value = requireString(obj, field, reason);
  if (!value)
    return false;
  if (*value != expected) {
    reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualInt(const llvm::json::Object &obj, llvm::StringRef field,
                     int64_t expected, std::string &reason) {
  auto value = requireInt(obj, field, reason);
  if (!value)
    return false;
  if (*value != expected) {
    reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualBool(const llvm::json::Object &obj, llvm::StringRef field,
                      bool expected, std::string &reason) {
  auto value = requireBool(obj, field, reason);
  if (!value)
    return false;
  if (*value != expected) {
    reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

llvm::json::Array kernelArray(const std::vector<std::string> &kernelNames) {
  llvm::json::Array arr;
  for (llvm::StringRef name : kernelNames)
    arr.push_back(name);
  return arr;
}

bool validateKernelArray(const llvm::json::Object &obj,
                         const std::vector<std::string> &expected,
                         std::string &reason) {
  const llvm::json::Array *arr = obj.getArray("kernel_names");
  if (!arr) {
    reason = "metadata field 'kernel_names' missing or not an array";
    return false;
  }
  if (arr->size() != expected.size()) {
    reason = "metadata kernel_names size mismatch";
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    auto value = (*arr)[i].getAsString();
    if (!value || *value != expected[i]) {
      reason = "metadata kernel_names mismatch";
      return false;
    }
  }
  return true;
}

llvm::json::Object metadataObject(const TranslationCacheRequest &request,
                                  const KeyData &keyData,
                                  const PipelineResult &result,
                                  llvm::StringRef objectSha256) {
  return llvm::json::Object{
      {"schema_version", kCacheSchemaVersion},
      {"key", keyData.key},
      {"source_object_sha256", keyData.sourceSha256},
      {"source_gfx", request.sourceGfx},
      {"target_gfx", request.targetGfx},
      {"source_isa", request.sourceIsa},
      {"target_isa", request.targetIsa},
      {"code_isa", request.codeIsa},
      {"elf_machine", keyData.elfMachineHex},
      {"elf_flags", keyData.elfFlagsHex},
      {"orig_mach", request.origMach},
      {"hotswap_rules_path", request.hotswapRulesPath},
      {"hotswap_rules_sha256", keyData.rulesSha256},
      {"strict_mode", request.strictMode},
      {"enable_writelane_rewrite", request.enableWritelaneRewrite},
      {"enable_wave_native", request.enableWaveNative},
      {"salmon_build_identity", keyData.buildIdentity},
      {"llc_identity", keyData.llcIdentity},
      {"llvm_mc_identity", keyData.llvmMcIdentity},
      {"lld_identity", keyData.lldIdentity},
      {"kernel_count", static_cast<int64_t>(keyData.kernelNames.size())},
      {"kernel_names", kernelArray(keyData.kernelNames)},
      {"cached_object_sha256", objectSha256.str()},
      {"cached_object_size", static_cast<int64_t>(result.hsaco.size())},
      {"lifted_count", result.liftedCount},
      {"total_count", result.totalCount},
      {"c5_suppressed_count", result.c5SuppressedCount},
      {"c5_suppression_reason", result.c5SuppressionReason},
      {"uses_scratch_private_segment", result.usesScratchPrivateSegment},
      {"source_private_segment_fixed_size",
       static_cast<int64_t>(result.sourcePrivateSegmentFixedSize)},
      {"target_private_segment_fixed_size",
       static_cast<int64_t>(result.targetPrivateSegmentFixedSize)},
      {"target_enable_private_segment", result.targetEnablePrivateSegment},
  };
}

bool validateMetadata(const TranslationCacheRequest &request,
                      const KeyData &keyData, const llvm::json::Object &obj,
                      llvm::StringRef objectSha256, size_t objectSize,
                      PipelineResult &result, std::string &reason) {
  if (!requireEqualInt(obj, "schema_version", kCacheSchemaVersion, reason) ||
      !requireEqualString(obj, "key", keyData.key, reason) ||
      !requireEqualString(obj, "source_object_sha256", keyData.sourceSha256,
                          reason) ||
      !requireEqualString(obj, "source_gfx", request.sourceGfx, reason) ||
      !requireEqualString(obj, "target_gfx", request.targetGfx, reason) ||
      !requireEqualString(obj, "source_isa", request.sourceIsa, reason) ||
      !requireEqualString(obj, "target_isa", request.targetIsa, reason) ||
      !requireEqualString(obj, "code_isa", request.codeIsa, reason) ||
      !requireEqualString(obj, "elf_machine", keyData.elfMachineHex, reason) ||
      !requireEqualString(obj, "elf_flags", keyData.elfFlagsHex, reason) ||
      !requireEqualInt(obj, "orig_mach", request.origMach, reason) ||
      !requireEqualString(obj, "hotswap_rules_path", request.hotswapRulesPath,
                          reason) ||
      !requireEqualString(obj, "hotswap_rules_sha256", keyData.rulesSha256,
                          reason) ||
      !requireEqualBool(obj, "strict_mode", request.strictMode, reason) ||
      !requireEqualBool(obj, "enable_writelane_rewrite",
                        request.enableWritelaneRewrite, reason) ||
      !requireEqualBool(obj, "enable_wave_native", request.enableWaveNative,
                        reason) ||
      !requireEqualString(obj, "salmon_build_identity",
                          keyData.buildIdentity, reason) ||
      !requireEqualString(obj, "llc_identity", keyData.llcIdentity, reason) ||
      !requireEqualString(obj, "llvm_mc_identity", keyData.llvmMcIdentity,
                          reason) ||
      !requireEqualString(obj, "lld_identity", keyData.lldIdentity, reason) ||
      !requireEqualInt(obj, "kernel_count",
                       static_cast<int64_t>(keyData.kernelNames.size()),
                       reason) ||
      !validateKernelArray(obj, keyData.kernelNames, reason) ||
      !requireEqualString(obj, "cached_object_sha256", objectSha256, reason) ||
      !requireEqualInt(obj, "cached_object_size",
                       static_cast<int64_t>(objectSize), reason))
    return false;

  auto lifted = requireInt(obj, "lifted_count", reason);
  auto total = requireInt(obj, "total_count", reason);
  auto c5Count = requireInt(obj, "c5_suppressed_count", reason);
  auto c5Reason = requireString(obj, "c5_suppression_reason", reason);
  auto usesScratch = requireBool(obj, "uses_scratch_private_segment", reason);
  auto sourceScratch =
      requireInt(obj, "source_private_segment_fixed_size", reason);
  auto targetScratch =
      requireInt(obj, "target_private_segment_fixed_size", reason);
  auto targetEnable =
      requireBool(obj, "target_enable_private_segment", reason);
  if (!lifted || !total || !c5Count || !c5Reason || !usesScratch ||
      !sourceScratch || !targetScratch || !targetEnable)
    return false;

  result.success = true;
  result.liftedCount = static_cast<int>(*lifted);
  result.totalCount = static_cast<int>(*total);
  result.c5SuppressedCount = static_cast<int>(*c5Count);
  result.c5SuppressionReason = *c5Reason;
  result.usesScratchPrivateSegment = *usesScratch;
  result.sourcePrivateSegmentFixedSize =
      static_cast<uint32_t>(*sourceScratch);
  result.targetPrivateSegmentFixedSize =
      static_cast<uint32_t>(*targetScratch);
  result.targetEnablePrivateSegment = *targetEnable;
  return true;
}

} // namespace

const char *translationCacheStatusString(TranslationCacheStatus status) {
  switch (status) {
  case TranslationCacheStatus::Disabled:
    return "disabled";
  case TranslationCacheStatus::Miss:
    return "miss";
  case TranslationCacheStatus::Hit:
    return "hit";
  case TranslationCacheStatus::Invalid:
    return "invalid";
  case TranslationCacheStatus::WriteSuccess:
    return "write_success";
  case TranslationCacheStatus::WriteFailed:
    return "write_failed";
  }
  return "invalid";
}

std::string sha256Hex(llvm::ArrayRef<uint8_t> data) {
  auto digest = llvm::SHA256::hash(data);
  std::string out;
  llvm::raw_string_ostream os(out);
  for (uint8_t byte : digest)
    os << llvm::format_hex_no_prefix(byte, 2);
  return os.str();
}

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &request) {
  TranslationCacheLookup lookup;
  if (cacheDisabledByEnv())
    return lookup;

  KeyData keyData = buildKeyData(request);
  lookup.key = keyData.key;
  if (!keyData.error.empty()) {
    lookup.status = TranslationCacheStatus::Invalid;
    lookup.reason = keyData.error;
    return lookup;
  }
  lookup.metadataPath = cacheMetadataPath(keyData.key);
  lookup.objectPath = cacheObjectPath(keyData.key);

  const bool metadataExists = exists(lookup.metadataPath);
  const bool objectExists = exists(lookup.objectPath);
  if (!metadataExists && !objectExists) {
    lookup.status = TranslationCacheStatus::Miss;
    lookup.reason = "entry not present";
    return lookup;
  }
  if (metadataExists != objectExists) {
    lookup.status = TranslationCacheStatus::Invalid;
    lookup.reason = metadataExists ? "metadata exists without object"
                                   : "object exists without metadata";
    return lookup;
  }

  auto objectBuffer = llvm::MemoryBuffer::getFile(lookup.objectPath);
  if (!objectBuffer) {
    lookup.status = TranslationCacheStatus::Invalid;
    lookup.reason = "failed to read cached object: " +
                    objectBuffer.getError().message();
    return lookup;
  }
  llvm::StringRef objectBytes = (*objectBuffer)->getBuffer();
  std::string objectSha = sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(objectBytes.data()),
      objectBytes.size()));

  auto metadataBuffer = llvm::MemoryBuffer::getFile(lookup.metadataPath);
  if (!metadataBuffer) {
    lookup.status = TranslationCacheStatus::Invalid;
    lookup.reason = "failed to read cache metadata: " +
                    metadataBuffer.getError().message();
    return lookup;
  }
  auto parsed = llvm::json::parse((*metadataBuffer)->getBuffer());
  if (!parsed) {
    lookup.status = TranslationCacheStatus::Invalid;
    lookup.reason = "failed to parse cache metadata: " +
                    llvm::toString(parsed.takeError());
    return lookup;
  }
  const llvm::json::Object *obj = parsed->getAsObject();
  if (!obj) {
    lookup.status = TranslationCacheStatus::Invalid;
    lookup.reason = "cache metadata is not a JSON object";
    return lookup;
  }
  if (!validateMetadata(request, keyData, *obj, objectSha, objectBytes.size(),
                        lookup.result, lookup.reason)) {
    lookup.status = TranslationCacheStatus::Invalid;
    return lookup;
  }

  lookup.result.hsaco.assign(
      reinterpret_cast<const uint8_t *>(objectBytes.data()),
      reinterpret_cast<const uint8_t *>(objectBytes.data()) +
          objectBytes.size());
  lookup.status = TranslationCacheStatus::Hit;
  lookup.reason = "ok";
  return lookup;
}

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &request, const PipelineResult &result) {
  TranslationCacheWrite write;
  if (cacheDisabledByEnv() || envEnabled("HSA_SALMON_CACHE_READONLY"))
    return write;

  KeyData keyData = buildKeyData(request);
  write.key = keyData.key;
  if (!keyData.error.empty()) {
    write.status = TranslationCacheStatus::WriteFailed;
    write.reason = keyData.error;
    return write;
  }
  write.metadataPath = cacheMetadataPath(keyData.key);
  write.objectPath = cacheObjectPath(keyData.key);

  if (!result.success || result.hsaco.empty()) {
    write.status = TranslationCacheStatus::WriteFailed;
    write.reason = "refusing to cache unsuccessful or empty translation";
    return write;
  }

  std::string dir = cacheSubdir(keyData.key);
  if (auto ec = llvm::sys::fs::create_directories(dir)) {
    write.status = TranslationCacheStatus::WriteFailed;
    write.reason = "failed to create cache directory '" + dir + "': " +
                   ec.message();
    return write;
  }

  std::string objectSha = sha256Hex(result.hsaco);
  std::string error;
  if (!writeFileAtomic(write.objectPath, result.hsaco, error)) {
    write.status = TranslationCacheStatus::WriteFailed;
    write.reason = "failed to write cached object: " + error;
    return write;
  }

  llvm::json::Object meta =
      metadataObject(request, keyData, result, objectSha);
  if (!writeFileAtomic(write.metadataPath,
                       jsonToString(llvm::json::Value(std::move(meta))),
                       error)) {
    llvm::sys::fs::remove(write.objectPath);
    write.status = TranslationCacheStatus::WriteFailed;
    write.reason = "failed to write cache metadata: " + error;
    return write;
  }

  write.status = TranslationCacheStatus::WriteSuccess;
  write.reason = "ok";
  return write;
}

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> kernelNames) {
  const char *skipEnv = std::getenv("HSA_SALMON_CACHE_SKIP_KERNELS");
  if (!skipEnv || !skipEnv[0])
    return "";

  llvm::StringRef remaining(skipEnv);
  while (!remaining.empty()) {
    auto split = remaining.split(',');
    llvm::StringRef requested = split.first.trim();
    remaining = split.second;
    if (requested.empty())
      continue;
    for (llvm::StringRef kernelName : kernelNames) {
      if (requested == kernelName)
        return kernelName.str();
    }
  }
  return "";
}

} // namespace transpiler
