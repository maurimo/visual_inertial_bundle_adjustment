/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Output function shared by Log.h and Checks.h.

#pragma once

#include <fmt/format.h>
#include <cstdio>
#include <string_view>

namespace viba_logging {

enum class Level { Debug, Info, Warning, Error };

inline void logMessage(Level level, std::string_view channel, std::string_view message) {
  static constexpr const char* kNames[] = {"DEBUG", "INFO", "WARNING", "ERROR"};
  std::FILE* out = level >= Level::Warning ? stderr : stdout;
  fmt::print(out, "[{}][{}]: {}\n", channel, kNames[static_cast<int>(level)], message);
  std::fflush(out);
}

} // namespace viba_logging
