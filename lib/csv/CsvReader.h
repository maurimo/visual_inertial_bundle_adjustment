/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace csv {

/// Minimal reader for comma separated files with a header line. Columns are selected by name,
/// in the order given to the constructor; extra columns in the file are ignored, missing ones are
/// an error. Fields are not quoted.
///
///   csv::Reader<3> reader(path, {"t_ns", "x", "y"});
///   int64_t t; double x, y;
///   while (reader.readRow(t, x, y)) { ... }
template <size_t N>
class Reader {
 public:
  Reader(const std::filesystem::path& path, const std::array<const char*, N>& columns)
      : name_(path.string()), owned_(std::make_unique<std::ifstream>(path)), is_(*owned_) {
    if (!*owned_) {
      throw std::runtime_error("Cannot open file: " + name_);
    }
    readHeader(columns);
  }

  Reader(std::string name, std::istream& is, const std::array<const char*, N>& columns)
      : name_(std::move(name)), is_(is) {
    readHeader(columns);
  }

  template <typename... Ts>
  bool readRow(Ts&... values) {
    static_assert(sizeof...(Ts) == N, "number of values must match number of columns");
    while (std::getline(is_, line_)) {
      lineNumber_++;
      stripLineEnd(line_);
      if (line_.empty()) {
        continue;
      }
      splitLine();
      size_t i = 0;
      (parseField(fields_.at(columnIndex_[i++]), values), ...);
      return true;
    }
    return false;
  }

 private:
  static void stripLineEnd(std::string& line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
      line.pop_back();
    }
  }

  static std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
      s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
      s.remove_suffix(1);
    }
    return s;
  }

  void splitLine() {
    fields_.clear();
    std::string_view rest(line_);
    while (true) {
      const size_t comma = rest.find(',');
      fields_.push_back(trim(rest.substr(0, comma)));
      if (comma == std::string_view::npos) {
        break;
      }
      rest.remove_prefix(comma + 1);
    }
    if (fields_.size() < numFileColumns_) {
      error("expected " + std::to_string(numFileColumns_) + " fields");
    }
  }

  void readHeader(const std::array<const char*, N>& columns) {
    if (!std::getline(is_, line_)) {
      error("missing header");
    }
    lineNumber_ = 1;
    stripLineEnd(line_);
    if (line_.rfind("\xEF\xBB\xBF", 0) == 0) { // UTF-8 BOM
      line_.erase(0, 3);
    }
    splitLine();
    numFileColumns_ = fields_.size();
    for (size_t i = 0; i < N; i++) {
      size_t j = 0;
      while (j < fields_.size() && fields_[j] != columns[i]) {
        j++;
      }
      if (j == fields_.size()) {
        error(std::string("missing column '") + columns[i] + "'");
      }
      columnIndex_[i] = j;
    }
  }

  template <typename T>
  void parseField(std::string_view field, T& value) {
    const std::string s(field);
    char* end = nullptr;
    if constexpr (std::is_same_v<T, std::string>) {
      value = s;
      return;
    } else if constexpr (std::is_floating_point_v<T>) {
      value = static_cast<T>(std::strtod(s.c_str(), &end));
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      value = static_cast<T>(std::strtoll(s.c_str(), &end, 10));
    } else if constexpr (std::is_integral_v<T>) {
      value = static_cast<T>(std::strtoull(s.c_str(), &end, 10));
    } else {
      static_assert(sizeof(T) == 0, "unsupported field type");
    }
    if (s.empty() || end != s.c_str() + s.size()) {
      error("cannot parse '" + s + "'");
    }
  }

  [[noreturn]] void error(const std::string& what) const {
    throw std::runtime_error(name_ + ":" + std::to_string(lineNumber_) + ": " + what);
  }

  std::string name_;
  std::unique_ptr<std::ifstream> owned_;
  std::istream& is_;
  std::array<size_t, N> columnIndex_{};
  size_t numFileColumns_ = 0;
  std::string line_;
  std::vector<std::string_view> fields_;
  int64_t lineNumber_ = 0;
};

} // namespace csv
