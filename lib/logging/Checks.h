/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Minimal check macros, API-compatible with the subset of VRS' logging/Checks.h used in this
// project: XR_CHECK(cond, [fmt, args...]) and XR_CHECK_{EQ,NE,LT,LE,GT,GE}(a, b, [fmt, args...]).
// A failed check prints a message and aborts.

#pragma once

#include <logging/LogMessage.h>

#include <cstdlib>
#include <sstream>
#include <string>

namespace viba_logging {

[[noreturn]] inline void
checkFailed(const char* file, int line, const std::string& what, const std::string& msg) {
  logMessage(
      Level::Error,
      "CHECK",
      fmt::format("{}:{}: check failed: {}{}{}", file, line, what, msg.empty() ? "" : " ", msg));
  std::abort();
}

inline std::string checkMessage() {
  return {};
}

template <typename... Args>
std::string checkMessage(fmt::format_string<Args...> f, Args&&... args) {
  return fmt::format(f, std::forward<Args>(args)...);
}

template <typename A, typename B>
std::string describeOperands(const char* expr, const A& a, const B& b) {
  std::ostringstream os;
  os << expr << " (" << a << " vs. " << b << ")";
  return os.str();
}

} // namespace viba_logging

#define XR_CHECK(cond, ...)                                                       \
  do {                                                                            \
    if (!(cond)) {                                                                \
      ::viba_logging::checkFailed(                                                \
          __FILE__, __LINE__, #cond, ::viba_logging::checkMessage(__VA_ARGS__)); \
    }                                                                             \
  } while (0)

#define XR_CHECK_OP_(op, a, b, ...)                                          \
  do {                                                                       \
    const auto& xrCheckA_ = (a);                                             \
    const auto& xrCheckB_ = (b);                                             \
    if (!(xrCheckA_ op xrCheckB_)) {                                         \
      ::viba_logging::checkFailed(                                           \
          __FILE__,                                                          \
          __LINE__,                                                          \
          ::viba_logging::describeOperands(                                  \
              #a " " #op " " #b, xrCheckA_, xrCheckB_),                      \
          ::viba_logging::checkMessage(__VA_ARGS__));                        \
    }                                                                        \
  } while (0)

#define XR_CHECK_EQ(a, b, ...) XR_CHECK_OP_(==, a, b, ##__VA_ARGS__)
#define XR_CHECK_NE(a, b, ...) XR_CHECK_OP_(!=, a, b, ##__VA_ARGS__)
#define XR_CHECK_LT(a, b, ...) XR_CHECK_OP_(<, a, b, ##__VA_ARGS__)
#define XR_CHECK_LE(a, b, ...) XR_CHECK_OP_(<=, a, b, ##__VA_ARGS__)
#define XR_CHECK_GT(a, b, ...) XR_CHECK_OP_(>, a, b, ##__VA_ARGS__)
#define XR_CHECK_GE(a, b, ...) XR_CHECK_OP_(>=, a, b, ##__VA_ARGS__)
