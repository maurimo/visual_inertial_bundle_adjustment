/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Minimal logging macros, API-compatible with the subset of VRS' logging used in this project.
// Define DEFAULT_LOG_CHANNEL before including this header to set the channel name.

#pragma once

#include <logging/LogMessage.h>

#ifndef DEFAULT_LOG_CHANNEL
#define DEFAULT_LOG_CHANNEL "ViBa"
#endif

#define XR_LOG_IMPL_(level, channel, ...) \
  ::viba_logging::logMessage(::viba_logging::Level::level, channel, fmt::format(__VA_ARGS__))

#define XR_LOGD(...) XR_LOG_IMPL_(Debug, DEFAULT_LOG_CHANNEL, __VA_ARGS__)
#define XR_LOGI(...) XR_LOG_IMPL_(Info, DEFAULT_LOG_CHANNEL, __VA_ARGS__)
#define XR_LOGW(...) XR_LOG_IMPL_(Warning, DEFAULT_LOG_CHANNEL, __VA_ARGS__)
#define XR_LOGE(...) XR_LOG_IMPL_(Error, DEFAULT_LOG_CHANNEL, __VA_ARGS__)

#define XR_LOGCI(channel, ...) XR_LOG_IMPL_(Info, channel, __VA_ARGS__)
#define XR_LOGCI_IF(cond, channel, ...) \
  do {                                  \
    if (cond) {                         \
      XR_LOGCI(channel, __VA_ARGS__);   \
    }                                   \
  } while (0)
