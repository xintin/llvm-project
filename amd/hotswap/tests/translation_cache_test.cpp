#include "test_common.hpp"

#include "../translation_cache.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TempDir {
  llvm::SmallString<128> path;
  bool valid = false;

  explicit TempDir(const char *prefix) {
    std::error_code ec = llvm::sys::fs::createUniqueDirectory(prefix, path);
    valid = !ec;
  }

  ~TempDir() {
    if (valid)
      llvm::sys::fs::remove_directories(path);
  }

  std::string file(const char *name) const {
    llvm::SmallString<256> p(path);
    llvm::sys::path::append(p, name);
    return std::string(p);
  }
};

struct ScopedEnv {
  std::string name;
  std::string oldValue;
  bool hadOldValue = false;

  ScopedEnv(const char *name, const std::string &value) : name(name) {
    if (const char *old = std::getenv(name)) {
      oldValue = old;
      hadOldValue = true;
    }
    setenv(name, value.c_str(), 1);
  }

  ~ScopedEnv() {
    if (hadOldValue)
      setenv(name.c_str(), oldValue.c_str(), 1);
    else
      unsetenv(name.c_str());
  }
};

std::vector<uint8_t> fakeAmdgpuElf() {
  std::vector<uint8_t> data(128, 0);
  data[0] = 0x7f;
  data[1] = 'E';
  data[2] = 'L';
  data[3] = 'F';
  data[4] = 2; // ELFCLASS64
  data[5] = 1; // little-endian
  data[6] = 1; // current ELF version
  const uint16_t machine = 224; // EM_AMDGPU
  const uint32_t flags = 0x49;
  std::memcpy(data.data() + 18, &machine, sizeof(machine));
  std::memcpy(data.data() + 48, &flags, sizeof(flags));
  return data;
}

void writeTextFile(const std::string &path, llvm::StringRef text) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  ASSERT_FALSE(ec) << "cannot write " << path << ": " << ec.message();
  os << text;
}

void writeBinaryFile(const std::string &path,
                     const std::vector<uint8_t> &bytes) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  ASSERT_FALSE(ec) << "cannot write " << path << ": " << ec.message();
  os.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

transpiler::TranslationCacheRequest makeRequest(
    const std::vector<uint8_t> &source, const std::string &rulesPath,
    const std::string &sourceGfx = "gfx1250",
    const std::string &targetGfx = "gfx942") {
  transpiler::TranslationCacheRequest request;
  request.sourceObject = llvm::ArrayRef<uint8_t>(source);
  request.sourceGfx = sourceGfx;
  request.targetGfx = targetGfx;
  request.sourceIsa = "amdgcn-amd-amdhsa--" + sourceGfx;
  request.targetIsa = "amdgcn-amd-amdhsa--" + targetGfx;
  request.codeIsa = "amdgcn-amd-amdhsa--gfx942";
  request.hotswapRulesPath = rulesPath;
  request.origMach = 0x49;
  request.enableWritelaneRewrite = true;
  request.enableWaveNative = true;
  request.strictMode = true;
  return request;
}

transpiler::PipelineResult makeSuccessfulResult(
    std::vector<uint8_t> hsaco = {0x7f, 'E', 'L', 'F', 1, 2, 3}) {
  transpiler::PipelineResult result;
  result.success = true;
  result.hsaco = std::move(hsaco);
  result.liftedCount = 7;
  result.totalCount = 7;
  return result;
}

} // namespace

TEST(TranslationCache, FirstRunMissWriteSecondRunHit) {
  TempDir temp("salmon_cache_test");
  ASSERT_TRUE(temp.valid);
  ScopedEnv cacheDir("HSA_SALMON_CACHE_DIR", temp.path.str().str());
  ScopedEnv noDisable("HSA_SALMON_CACHE_DISABLE", "0");
  ScopedEnv noReadonly("HSA_SALMON_CACHE_READONLY", "0");

  std::string rules = temp.file("rules.json");
  writeTextFile(rules, "{\"version\":1,\"rules\":[]}\n");
  auto source = fakeAmdgpuElf();
  auto request = makeRequest(source, rules);

  auto first = transpiler::lookupTranslationCache(request);
  EXPECT_EQ(first.status, transpiler::TranslationCacheStatus::Miss);

  auto result = makeSuccessfulResult();
  auto write = transpiler::writeTranslationCache(request, result);
  ASSERT_EQ(write.status, transpiler::TranslationCacheStatus::WriteSuccess)
      << write.reason;

  auto second = transpiler::lookupTranslationCache(request);
  ASSERT_EQ(second.status, transpiler::TranslationCacheStatus::Hit)
      << second.reason;
  EXPECT_EQ(second.result.hsaco, result.hsaco);
  EXPECT_EQ(second.result.liftedCount, result.liftedCount);
  EXPECT_EQ(second.result.totalCount, result.totalCount);
}

TEST(TranslationCache, ChangedInputHashCausesMiss) {
  TempDir temp("salmon_cache_test");
  ASSERT_TRUE(temp.valid);
  ScopedEnv cacheDir("HSA_SALMON_CACHE_DIR", temp.path.str().str());
  ScopedEnv noDisable("HSA_SALMON_CACHE_DISABLE", "0");
  ScopedEnv noReadonly("HSA_SALMON_CACHE_READONLY", "0");

  std::string rules = temp.file("rules.json");
  writeTextFile(rules, "{\"version\":1,\"rules\":[]}\n");
  auto source = fakeAmdgpuElf();
  auto request = makeRequest(source, rules);
  ASSERT_EQ(transpiler::writeTranslationCache(request, makeSuccessfulResult()).status,
            transpiler::TranslationCacheStatus::WriteSuccess);

  source[80] ^= 0x1;
  auto changed = makeRequest(source, rules);
  auto lookup = transpiler::lookupTranslationCache(changed);
  EXPECT_EQ(lookup.status, transpiler::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, ChangedIsaCausesMiss) {
  TempDir temp("salmon_cache_test");
  ASSERT_TRUE(temp.valid);
  ScopedEnv cacheDir("HSA_SALMON_CACHE_DIR", temp.path.str().str());
  ScopedEnv noDisable("HSA_SALMON_CACHE_DISABLE", "0");
  ScopedEnv noReadonly("HSA_SALMON_CACHE_READONLY", "0");

  std::string rules = temp.file("rules.json");
  writeTextFile(rules, "{\"version\":1,\"rules\":[]}\n");
  auto source = fakeAmdgpuElf();
  auto request = makeRequest(source, rules);
  ASSERT_EQ(transpiler::writeTranslationCache(request, makeSuccessfulResult()).status,
            transpiler::TranslationCacheStatus::WriteSuccess);

  auto changedSourceIsa = makeRequest(source, rules, "gfx1200", "gfx942");
  EXPECT_EQ(transpiler::lookupTranslationCache(changedSourceIsa).status,
            transpiler::TranslationCacheStatus::Miss);

  auto changedTargetIsa = makeRequest(source, rules, "gfx1250", "gfx950");
  EXPECT_EQ(transpiler::lookupTranslationCache(changedTargetIsa).status,
            transpiler::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, CorruptMetadataIsInvalid) {
  TempDir temp("salmon_cache_test");
  ASSERT_TRUE(temp.valid);
  ScopedEnv cacheDir("HSA_SALMON_CACHE_DIR", temp.path.str().str());
  ScopedEnv noDisable("HSA_SALMON_CACHE_DISABLE", "0");
  ScopedEnv noReadonly("HSA_SALMON_CACHE_READONLY", "0");

  std::string rules = temp.file("rules.json");
  writeTextFile(rules, "{\"version\":1,\"rules\":[]}\n");
  auto source = fakeAmdgpuElf();
  auto request = makeRequest(source, rules);
  auto write = transpiler::writeTranslationCache(request, makeSuccessfulResult());
  ASSERT_EQ(write.status, transpiler::TranslationCacheStatus::WriteSuccess);

  writeTextFile(write.metadataPath, "not-json\n");
  auto lookup = transpiler::lookupTranslationCache(request);
  EXPECT_EQ(lookup.status, transpiler::TranslationCacheStatus::Invalid);
  EXPECT_NE(lookup.reason.find("parse"), std::string::npos);
}

TEST(TranslationCache, CorruptObjectIsInvalid) {
  TempDir temp("salmon_cache_test");
  ASSERT_TRUE(temp.valid);
  ScopedEnv cacheDir("HSA_SALMON_CACHE_DIR", temp.path.str().str());
  ScopedEnv noDisable("HSA_SALMON_CACHE_DISABLE", "0");
  ScopedEnv noReadonly("HSA_SALMON_CACHE_READONLY", "0");

  std::string rules = temp.file("rules.json");
  writeTextFile(rules, "{\"version\":1,\"rules\":[]}\n");
  auto source = fakeAmdgpuElf();
  auto request = makeRequest(source, rules);
  auto write = transpiler::writeTranslationCache(request, makeSuccessfulResult());
  ASSERT_EQ(write.status, transpiler::TranslationCacheStatus::WriteSuccess);

  writeBinaryFile(write.objectPath, {1, 2, 3, 4});
  auto lookup = transpiler::lookupTranslationCache(request);
  EXPECT_EQ(lookup.status, transpiler::TranslationCacheStatus::Invalid);
  EXPECT_NE(lookup.reason.find("cached_object_sha256"), std::string::npos);
}

TEST(TranslationCache, SkipKernelListMatchesExactKernelName) {
  ScopedEnv skip("HSA_SALMON_CACHE_SKIP_KERNELS",
                 "other_kernel, target_kernel ,third_kernel");

  std::vector<std::string> kernels = {"first_kernel", "target_kernel"};
  EXPECT_EQ(transpiler::skippedKernelForTranslationCache(kernels),
            "target_kernel");
}

TEST(TranslationCache, SkipKernelListDoesNotUseSubstringMatching) {
  ScopedEnv skip("HSA_SALMON_CACHE_SKIP_KERNELS", "target");

  std::vector<std::string> kernels = {"target_kernel"};
  EXPECT_TRUE(transpiler::skippedKernelForTranslationCache(kernels).empty());
}
