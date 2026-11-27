/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/loop_closure/MatchSource.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

namespace visual_inertial_ba::loop {

std::unique_ptr<CsvMatchSource> CsvMatchSource::load(const std::string& path, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    if (error) {
      *error = "cannot open " + path;
    }
    return nullptr;
  }
  auto src = std::make_unique<CsvMatchSource>();
  std::string line;
  int64_t lineNo = 0;
  while (std::getline(in, line)) {
    lineNo++;
    if (line.empty() || line[0] == '#' || !(std::isdigit(line[0]) || line[0] == '-')) {
      continue; // header or comment
    }
    std::stringstream ss(line);
    std::string item;
    double v[8];
    int n = 0;
    while (n < 8 && std::getline(ss, item, ',')) {
      char* end = nullptr;
      v[n] = std::strtod(item.c_str(), &end);
      if (end == item.c_str()) {
        break;
      }
      n++;
    }
    if (n < 8) {
      if (error) {
        *error = path + ":" + std::to_string(lineNo) + ": expected 8 numbers";
      }
      return nullptr;
    }
    int64_t tA = std::llround(v[0]), tB = std::llround(v[2]);
    int camA = int(v[1]), camB = int(v[3]);
    Vec2 pA(v[4], v[5]), pB(v[6], v[7]);
    if (tA > tB) {
      std::swap(tA, tB);
      std::swap(camA, camB);
      std::swap(pA, pB);
    }
    src->add(tA, camA, tB, camB, pA, pB);
  }
  return src;
}

void CsvMatchSource::add(
    int64_t tAUs,
    int camA,
    int64_t tBUs,
    int camB,
    const Vec2& pxA,
    const Vec2& pxB) {
  ImageMatches& m = data_[{tAUs, tBUs}][{camA, camB}];
  m.camA = camA;
  m.camB = camB;
  m.pxA.push_back(pxA);
  m.pxB.push_back(pxB);
  numCorrespondences_++;
}

std::vector<int64_t> CsvMatchSource::imageTimesUs() const {
  std::set<int64_t> s;
  for (const auto& [key, _] : data_) {
    s.insert(key.first);
    s.insert(key.second);
  }
  return {s.begin(), s.end()};
}

std::vector<int64_t> CsvMatchSource::partnersBefore(int64_t tBUs) const {
  std::vector<int64_t> out;
  for (const auto& [key, _] : data_) {
    if (key.second == tBUs) {
      out.push_back(key.first);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<ImageMatches> CsvMatchSource::matches(int64_t tAUs, int64_t tBUs) const {
  std::vector<ImageMatches> out;
  const auto it = data_.find({tAUs, tBUs});
  if (it == data_.end()) {
    return out;
  }
  for (const auto& [_, m] : it->second) {
    out.push_back(m);
  }
  return out;
}

bool loadLoopTracks(const std::string& path, std::vector<LoopTrackObs>& out, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    if (error) {
      *error = "cannot open " + path;
    }
    return false;
  }
  out.clear();
  std::string line;
  int64_t lineNo = 0;
  while (std::getline(in, line)) {
    lineNo++;
    if (line.empty() || !(std::isdigit(line[0]) || line[0] == '-')) {
      continue; // header or comment
    }
    std::stringstream ss(line);
    std::string item;
    double v[6];
    int n = 0;
    while (n < 6 && std::getline(ss, item, ',')) {
      char* end = nullptr;
      v[n] = std::strtod(item.c_str(), &end);
      if (end == item.c_str()) {
        break;
      }
      n++;
    }
    if (n < 6) {
      if (error) {
        *error = path + ":" + std::to_string(lineNo) + ": expected 6 numbers";
      }
      return false;
    }
    LoopTrackObs o;
    o.track = std::llround(v[0]);
    o.visit = int(v[1]);
    o.tUs = std::llround(v[2]);
    o.cam = int(v[3]);
    o.px = Vec2(v[4], v[5]);
    out.push_back(o);
  }
  return true;
}

std::unique_ptr<CsvMatchSource> CsvMatchSource::fromLoopTracks(
    const std::vector<LoopTrackObs>& obs,
    int maxPairsPerVisitPair) {
  auto src = std::make_unique<CsvMatchSource>();
  // per track, its observations in time order (ties: in the order given)
  std::map<int64_t, std::vector<const LoopTrackObs*>> tracks;
  for (const auto& o : obs) {
    tracks[o.track].push_back(&o);
  }
  for (auto& [_, rows] : tracks) {
    std::stable_sort(rows.begin(), rows.end(), [](const LoopTrackObs* a, const LoopTrackObs* b) {
      return a->tUs < b->tUs;
    });
    int maxVisit = 0;
    for (const auto* r : rows) {
      maxVisit = std::max(maxVisit, r->visit);
    }
    for (int va = 0; va < maxVisit; va++) {
      int n = 0;
      for (const auto* a : rows) {
        if (a->visit != va) {
          continue;
        }
        for (const auto* b : rows) {
          if (b->visit != va + 1) {
            continue;
          }
          if (maxPairsPerVisitPair > 0 && n >= maxPairsPerVisitPair) {
            break;
          }
          if (a->tUs <= b->tUs) {
            src->add(a->tUs, a->cam, b->tUs, b->cam, a->px, b->px);
          } else {
            src->add(b->tUs, b->cam, a->tUs, a->cam, b->px, a->px);
          }
          n++;
        }
      }
    }
  }
  return src;
}

} // namespace visual_inertial_ba::loop
